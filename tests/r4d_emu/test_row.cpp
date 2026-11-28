/* tests/r4d_emu/test_row.cpp -- mla_row_norm_rope_store (r4d_model_bf16.hip), Hy4-flash's latent
 * cache row, through the SIMT emulator, held to the three-launch chain it fuses bit for bit: the
 * norm over the row's first `latent` columns, rope over its last `rope_dim` (gptj, and imrope_gptj
 * with sections and [3, M] positions), then kv_store of the row as one KV head.
 *
 * The fused kernel writes the row and the cache slot out of the registers that produced each value,
 * where the chain wrote the row and copied it back; both are the chain's bytes or this test fails:
 * the row whole, at one pitch and at a padded one (its padding columns untouched), the cache whole
 * (its untouched blocks untouched), with a bf16 and an f32 gain, with a negative slot and a slot
 * past the blocks, at one token and at five.
 *
 * Also DFly2's chain (D-02, #88): one gather_rows and one rmsnorm over every round's rows against
 * a launch of each a round (hoist()).
 *
 * Also X-07 (#136), the int8 latent row: the three stores that write it, held to the chain above
 * byte for byte and then, for the 656 bytes' worth of row, to libref's own int8 store row, loaded
 * from RADIANCE_HOME (the CPU reference plugin, so the case needs no card). */
#include "ref_host.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> sh, std::vector<int64_t> st = {}) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)sh.size();
    int64_t n = 1;
    for (int i = (int)sh.size() - 1; i >= 0; --i) {
        t.shape[i] = sh[(size_t)i];
        t.stride[i] = st.empty() ? n : st[(size_t)i];
        if (st.empty()) n *= sh[(size_t)i];
    }
    return t;
}

RadParam pi(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}

RadParam pf(const char* k, double v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_F64;
    p.dval = v;
    return p;
}

RadParam ps(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

RadArgs mk(std::vector<RadTensor>& t, std::vector<RadParam>& p) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    return a;
}

struct Cfg {
    int L = 512, R = 64, M = 3, pitch = 576, bs = 8, blocks = 2;
    bool wf32 = false;           /* the gain's dtype, and so the kernel's template */
    bool plane = false;          /* [3, M] positions: imrope's three components */
    const char* mode = "gptj";
    std::vector<int> sections;   /* imrope's three, summing to rope_dim / 2 */
    std::vector<int> slot;       /* -1 is padding; >= blocks * bs is padding too */
    std::vector<int> pos;
    double theta = 5000000.0, eps = 1e-6, scale = 1.0;
    uint32_t seed = 1;
};

/* The chain: rmsnorm in place over the row's first L columns, rope over its last R (one head, the
 * way the row front calls it), kv_store of the row's halves as one KV head. */
int chain(const Cfg& c, std::vector<uint16_t>& row, std::vector<uint16_t>& kv,
          const void* gain, const RadTensor& pos, const RadTensor& sm) {
    const int W = c.L + c.R, half = W / 2, L = c.L, R = c.R;
    int rc;
    {
        std::vector<RadTensor> t{ tens(row.data(), RAD_BF16, { c.M, L }, { c.pitch, 1 }),
                                  tens(const_cast<void*>(gain), c.wf32 ? RAD_F32 : RAD_BF16, { L }),
                                  tens(row.data(), RAD_BF16, { c.M, L }, { c.pitch, 1 }) };
        std::vector<RadParam> p{ pi("M", c.M), pi("n", L), pf("eps", c.eps), ps("dtype", "bf16") };
        RadArgs a = mk(t, p);
        if ((rc = r4d_rad_rmsnorm(&a, nullptr)) != RAD_OK) return rc;
    }
    {
        std::vector<RadTensor> t{ tens((uint16_t*)row.data() + L, RAD_BF16, { c.M, R }, { c.pitch, 1 }),
                                  pos };
        std::vector<RadParam> p{ pi("M", c.M), pi("head_dim", R), pi("n_head", 1),
                                 pi("n_head_kv", 0), pf("theta", c.theta), pf("scale", c.scale),
                                 ps("mode", c.mode), pi("rotary_dim", R) };
        std::string sec;
        if (!c.sections.empty()) {
            for (size_t i = 0; i < c.sections.size(); ++i) {
                if (i) sec += ' ';
                sec += std::to_string(c.sections[i]);
            }
            p.push_back(ps("sections", sec.c_str()));
        }
        RadArgs a = mk(t, p);
        if ((rc = r4d_rad_rope(&a, nullptr)) != RAD_OK) return rc;
    }
    {
        std::vector<RadTensor> t{ tens(row.data(), RAD_BF16, { c.M, half }, { c.pitch, 1 }),
                                  tens((uint16_t*)row.data() + half, RAD_BF16, { c.M, half },
                                       { c.pitch, 1 }),
                                  sm, tens(kv.data(), RAD_BF16, { c.blocks, 1, c.bs, W }) };
        std::vector<RadParam> p{ pi("M", c.M), pi("head_dim", half), pi("n_head_kv", 1),
                                 pi("block_size", c.bs), ps("kv_dtype", "bf16") };
        RadArgs a = mk(t, p);
        if ((rc = r4d_rad_kv_store(&a, nullptr)) != RAD_OK) return rc;
    }
    return RAD_OK;
}

int one(const Cfg& c) {
    const int W = c.L + c.R, L = c.L, R = c.R;
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<uint16_t> row0((size_t)c.M * c.pitch), kv0((size_t)c.blocks * c.bs * W);
    for (auto& x : row0) x = f2b(u(rng));
    for (auto& x : kv0) x = f2b(u(rng));

    std::vector<uint16_t> gainb;
    std::vector<float> gainf;
    if (c.wf32) { gainf.resize((size_t)L); for (auto& g : gainf) g = u(rng); }
    else        { gainb.resize((size_t)L); for (auto& g : gainb) g = f2b(u(rng) * 2.0f); }
    const void* gain = c.wf32 ? (const void*)gainf.data() : (const void*)gainb.data();

    std::vector<int32_t> slot(c.slot.begin(), c.slot.end());
    std::vector<int32_t> pos(c.pos.begin(), c.pos.end());
    const int planes = c.plane ? 3 : 1;
    RadTensor t_sm = tens(slot.data(), RAD_I32, { c.M });
    RadTensor t_pos = c.plane ? tens(pos.data(), RAD_I32, { planes, c.M }, { c.M, 1 })
                              : tens(pos.data(), RAD_I32, { c.M });

    /* The fused op. */
    std::vector<uint16_t> rowF = row0, kvF = kv0;
    int rc1;
    {
        std::vector<RadTensor> t{ tens(rowF.data(), RAD_BF16, { c.M, W }, { c.pitch, 1 }),
                                  tens(const_cast<void*>(gain), c.wf32 ? RAD_F32 : RAD_BF16, { L }),
                                  t_pos, t_sm,
                                  tens(kvF.data(), RAD_BF16, { c.blocks, 1, c.bs, W }) };
        std::vector<RadParam> p{ pi("M", c.M), pi("latent", L), pi("rope_dim", R),
                                 pi("block_size", c.bs), pf("theta", c.theta), pf("eps", c.eps),
                                 pf("scale", c.scale), ps("mode", c.mode), ps("dtype", "bf16") };
        std::string sec;
        if (!c.sections.empty()) {
            for (size_t i = 0; i < c.sections.size(); ++i) {
                if (i) sec += ' ';
                sec += std::to_string(c.sections[i]);
            }
            p.push_back(ps("sections", sec.c_str()));
        }
        RadArgs a = mk(t, p);
        rc1 = r4d_rad_mla_row_norm_rope_store(&a, nullptr);
    }

    /* The chain it replaces. */
    std::vector<uint16_t> rowC = row0, kvC = kv0;
    const int rc2 = chain(c, rowC, kvC, gain, t_pos, t_sm);

    int bad = 0;
    if (rc1 != RAD_OK || rc2 != RAD_OK) {
        std::printf("row L%d R%d M%d: launch failed (fused %d, chain %d)\n", L, R, c.M, rc1, rc2);
        return 1;
    }
    for (size_t i = 0; i < rowF.size(); ++i) {
        if (rowF[i] != rowC[i]) {
            if (!bad++)
                std::printf("row L%d R%d M%d pitch%d: row byte %zu is %04x, the chain's %04x\n", L, R,
                            c.M, c.pitch, i, rowF[i], rowC[i]);
        }
    }
    for (size_t i = 0; i < kvF.size(); ++i) {
        if (kvF[i] != kvC[i]) {
            if (!bad++)
                std::printf("row L%d R%d M%d pitch%d: cache byte %zu is %04x, the chain's %04x\n",
                            L, R, c.M, c.pitch, i, kvF[i], kvC[i]);
        }
    }
    /* Neither launch may touch the row's padding columns, nor a block no slot names. */
    int pad = 0, blk_bad = 0;
    for (int m = 0; m < c.M; ++m) {
        for (int i = W; i < c.pitch; ++i) {
            const size_t o = (size_t)m * c.pitch + (size_t)i;
            pad += rowF[o] != row0[o];
            pad += rowC[o] != row0[o];
        }
    }
    std::vector<int> used((size_t)c.blocks, 0);
    for (int s : slot) if (s >= 0 && s < c.blocks * c.bs) used[(size_t)(s / c.bs)] = 1;
    for (int b = 0; b < c.blocks; ++b) {
        if (used[(size_t)b]) continue;
        for (size_t i = 0; i < (size_t)c.bs * W; ++i) {
            const size_t o = ((size_t)b * c.bs) * W + i;
            blk_bad += kvF[o] != kv0[o];
            blk_bad += kvC[o] != kv0[o];
        }
    }
    if (pad || blk_bad) {
        std::printf("row L%d R%d M%d pitch%d: %d padding byte(s), %d unnamed-block byte(s) written\n",
                    L, R, c.M, c.pitch, pad, blk_bad);
        ++bad;
    }
    std::printf("row L%-4d R%d M%d pitch%-4d %s%s: %s\n", L, R, c.M, c.pitch,
                c.mode, c.wf32 ? " f32 w" : " bf16 w", bad ? "FAIL" : "ok");
    return bad;
}

Cfg base() {
    Cfg c;
    c.slot = { 0, 9, -1 };             /* two blocks of eight slots: one padding, one in block 1 */
    c.pos = { 3, 7, 11 };
    c.sections = { 4, 6, 6 };          /* rope_dim / 2 = 32 */
    return c;
}

/* X-07 (#136): the int8 latent row. The cache's row is one KV head's
 *
 *     [L int8 codes][L/128 f32 scales][R rope columns as bf16]      L + 4 * (L / 128) + 2 * R
 *
 * and the three launch shapes that write it must write the same row: the fused
 * mla_row_norm_rope_store into an int8 cache, kv_store's int8 form fed the row it just wrote, and
 * kv_store_streamed's into a mirror and a ring. Held byte for byte, against the chain the fused op
 * stands for (rmsnorm, rope, kv_store), and its latent columns held to libref's own int8 store row
 * at one code step, with slots that are padding (a negative one, one past the blocks) and a block
 * whose slots none of them names. The row's byte count is a multiple of 16 at these geometries, so
 * every row of a 16-byte-aligned cache starts 16-byte aligned: the test also checks that, since a
 * row is what a wave's 16-byte loads read. */
/* libref's own store row (libref/ref_attn.cpp: `ref_kv_store` with `kv_dtype` "i8" and a
 * `rope_dim`), loaded the way tests/r4d_emu/test_mla_kv.cpp loads the plugin: the CPU reference
 * library, so this runs without a card. It is the latent columns' oracle, and this file does not
 * restate its form: libref builds the block's rotation as an explicit Hadamard matrix in double and
 * normalises it before the quantiser, so it shares no summation order with the kernel's wave
 * butterfly, and a change to either library shows up here. RADIANCE_HOME is looked at first, and
 * then the build tree this test runs out of (`<exe>/../radiance_home/kernels/libref.so`), since the
 * ctest case sets no environment of its own for this target. */
static const RadKernelInfo* libref_op(const char* op) {
    static void* h = nullptr;
    if (!h) {
        std::vector<std::string> where;
        if (const char* home = std::getenv("RADIANCE_HOME"); home && *home)
            where.push_back(std::string(home) + "/kernels/libref.so");
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) {
            exe[n] = '\0';
            std::string dir(exe);
            dir = dir.substr(0, dir.find_last_of('/'));
            where.push_back(dir + "/../radiance_home/kernels/libref.so");
        }
        for (const std::string& path : where)
            if ((h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL))) break;
        if (!h) return nullptr;
        auto open = (int (*)(void))dlsym(h, "rad_plugin_open");
        if (open && open() < 0) return nullptr;
    }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return nullptr;
    for (int i = 0; i < cnt(); ++i)
        if (at(i)->op && !std::strcmp(at(i)->op, op)) return at(i);
    return nullptr;
}

struct I8Cfg {
    int L = 512, R = 64, M = 3, pitch = 576, bs = 8, blocks = 3, slots = 2;
    std::vector<int> slot, pos, res;
    double theta = 5000000.0, eps = 1e-6, scale = 1.0;
    uint32_t seed = 7;
};

/* libref's int8 store's bytes for the rows in `k` and `v` ([M, pitch] bf16, a head_dim columns
 * each), over the cache filler `kv0` the kernel's runs started from. */
static bool libref_store(const uint16_t* k, const uint16_t* v, int64_t pitch, int64_t M,
                         int64_t hd, int64_t R, int64_t bs, int64_t blocks, int64_t rowb,
                         const std::vector<int32_t>& slot, const std::vector<uint8_t>& kv0,
                         std::vector<uint8_t>& out, std::string& why) {
    static const RadKernelInfo* ref = libref_op("kv_store");
    if (!ref) {
        why = "libref's kv_store not loadable (set RADIANCE_HOME)";
        return false;
    }
    out = kv0;
    std::vector<int32_t> sm = slot;
    std::vector<RadTensor> t{ tens(const_cast<uint16_t*>(k), RAD_BF16, { M, hd }, { pitch, 1 }),
                              tens(const_cast<uint16_t*>(v), RAD_BF16, { M, hd }, { pitch, 1 }),
                              tens(sm.data(), RAD_I32, { M }),
                              tens(out.data(), RAD_I8, { blocks, 1, bs, rowb }) };
    std::vector<RadParam> p{ pi("M", M), pi("head_dim", hd), pi("n_head_kv", 1),
                             pi("block_size", bs), ps("kv_dtype", "i8"), pi("rope_dim", R) };
    RadArgs a = mk(t, p);
    const int rc = ref->launch(&a, nullptr);
    if (rc != RAD_OK) {
        why = "libref's kv_store refused (" + std::to_string(rc) + ")";
        return false;
    }
    return true;
}

int i8(const I8Cfg& c) {
    const int W = c.L + c.R, L = c.L, R = c.R, hd = W / 2;
    const int64_t rowb = L + 4 * (L / 128) + 2 * R;   /* the layout, written out here on purpose */
    const int64_t brow = (int64_t)c.bs * rowb;
    if (rowb % 16) {
        std::printf("row i8 L%d R%d: row_bytes %lld is not a multiple of 16\n", L, R, (long long)rowb);
        return 1;
    }
    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<uint16_t> row0((size_t)c.M * c.pitch), gain((size_t)L);
    for (auto& x : row0) x = f2b(u(rng));
    for (auto& g : gain) g = f2b(u(rng) * 2.0f);
    /* Fillers that are not zero and not one: an untouched byte stays visible. */
    std::vector<uint8_t> kv0((size_t)c.blocks * brow), host0((size_t)c.blocks * brow),
        ring0((size_t)c.slots * brow);
    for (auto& x : kv0) x = (uint8_t)(0x40 + (rng() & 0x3f));
    for (auto& x : host0) x = (uint8_t)(0x40 + (rng() & 0x3f));
    for (auto& x : ring0) x = (uint8_t)(0x40 + (rng() & 0x3f));
    /* 16-byte aligned planes, as the ABI's cache is: std::vector's allocation is. */
    std::vector<int32_t> slot(c.slot.begin(), c.slot.end()), pos(c.pos.begin(), c.pos.end()),
        res(c.res.begin(), c.res.end());
    RadTensor t_sm = tens(slot.data(), RAD_I32, { c.M });
    RadTensor t_pos = tens(pos.data(), RAD_I32, { c.M });
    auto kv_t = [&](void* p) { return tens(p, RAD_I8, { c.blocks, 1, c.bs, rowb }); };

    /* (a) the fused op into the int8 cache. */
    std::vector<uint16_t> rowF = row0;
    std::vector<uint8_t> kvF = kv0;
    int rc_f;
    {
        std::vector<RadTensor> t{ tens(rowF.data(), RAD_BF16, { c.M, W }, { c.pitch, 1 }),
                                  tens(gain.data(), RAD_BF16, { L }), t_pos, t_sm, kv_t(kvF.data()) };
        std::vector<RadParam> p{ pi("M", c.M), pi("latent", L), pi("rope_dim", R),
                                 pi("block_size", c.bs), pf("theta", c.theta), pf("eps", c.eps),
                                 pf("scale", c.scale), ps("mode", "gptj"), ps("dtype", "bf16") };
        RadArgs a = mk(t, p);
        rc_f = r4d_rad_mla_row_norm_rope_store(&a, nullptr);
    }

    /* (b) the chain: the row in bf16, then kv_store's int8 form. */
    std::vector<uint16_t> rowC = row0;
    std::vector<uint8_t> kvC = kv0;
    int rc_c = RAD_OK;
    {
        std::vector<RadTensor> t{ tens(rowC.data(), RAD_BF16, { c.M, L }, { c.pitch, 1 }),
                                  tens(gain.data(), RAD_BF16, { L }),
                                  tens(rowC.data(), RAD_BF16, { c.M, L }, { c.pitch, 1 }) };
        std::vector<RadParam> p{ pi("M", c.M), pi("n", L), pf("eps", c.eps), ps("dtype", "bf16") };
        RadArgs a = mk(t, p);
        rc_c = r4d_rad_rmsnorm(&a, nullptr);
    }
    if (rc_c == RAD_OK) {
        std::vector<RadTensor> t{ tens((uint16_t*)rowC.data() + L, RAD_BF16, { c.M, R }, { c.pitch, 1 }),
                                  t_pos };
        std::vector<RadParam> p{ pi("M", c.M), pi("head_dim", R), pi("n_head", 1),
                                 pi("n_head_kv", 0), pf("theta", c.theta), pf("scale", c.scale),
                                 ps("mode", "gptj"), pi("rotary_dim", R) };
        RadArgs a = mk(t, p);
        rc_c = r4d_rad_rope(&a, nullptr);
    }
    if (rc_c == RAD_OK) {
        std::vector<RadTensor> t{ tens(rowC.data(), RAD_BF16, { c.M, hd }, { c.pitch, 1 }),
                                  tens((uint16_t*)rowC.data() + hd, RAD_BF16, { c.M, hd },
                                       { c.pitch, 1 }),
                                  t_sm, kv_t(kvC.data()) };
        std::vector<RadParam> p{ pi("M", c.M), pi("head_dim", hd), pi("n_head_kv", 1),
                                 pi("block_size", c.bs), ps("kv_dtype", "i8"),
                                 pi("rope_dim", R) };
        RadArgs a = mk(t, p);
        rc_c = r4d_rad_kv_store_i8(&a, nullptr);
    }

    /* (c) kv_store from the fused op's row: the store's own bytes, not the chain's. */
    std::vector<uint8_t> kvS = kv0;
    int rc_s = RAD_OK;
    {
        std::vector<RadTensor> t{ tens(rowF.data(), RAD_BF16, { c.M, hd }, { c.pitch, 1 }),
                                  tens((uint16_t*)rowF.data() + hd, RAD_BF16, { c.M, hd },
                                       { c.pitch, 1 }),
                                  t_sm, kv_t(kvS.data()) };
        std::vector<RadParam> p{ pi("M", c.M), pi("head_dim", hd), pi("n_head_kv", 1),
                                 pi("block_size", c.bs), ps("kv_dtype", "i8"),
                                 pi("rope_dim", R) };
        RadArgs a = mk(t, p);
        rc_s = r4d_rad_kv_store_i8(&a, nullptr);
    }

    /* (d) kv_store_streamed: the mirror always, the ring where the block is resident. */
    std::vector<uint8_t> hostD = host0, ringD = ring0;
    int rc_d = RAD_OK;
    {
        std::vector<RadTensor> t{ tens(rowF.data(), RAD_BF16, { c.M, hd }, { c.pitch, 1 }),
                                  tens((uint16_t*)rowF.data() + hd, RAD_BF16, { c.M, hd },
                                       { c.pitch, 1 }),
                                  t_sm, tens(ringD.data(), RAD_I8, { c.slots, 1, c.bs, rowb }),
                                  tens(res.data(), RAD_I32, { c.blocks }),
                                  tens(hostD.data(), RAD_U8, { c.blocks, brow }) };
        std::vector<RadParam> p{ pi("M", c.M), pi("head_dim", hd), pi("n_head_kv", 1),
                                 pi("block_size", c.bs), ps("kv_dtype", "i8"),
                                 pi("rope_dim", R) };
        RadArgs a = mk(t, p);
        rc_d = r4d_rad_kv_store_streamed(&a, nullptr);
    }

    int bad = 0;
    if (rc_f != RAD_OK || rc_c != RAD_OK || rc_s != RAD_OK || rc_d != RAD_OK) {
        std::printf("row i8 L%d R%d M%d: launch failed (fused %d, chain %d, store %d, streamed %d)\n",
                    L, R, c.M, rc_f, rc_c, rc_s, rc_d);
        return 1;
    }
    for (size_t i = 0; i < rowF.size(); ++i)
        if (rowF[i] != rowC[i]) {
            if (!bad) std::printf("row i8 L%d R%d: row column %zu is %04x, the chain's %04x\n", L, R,
                                  i, rowF[i], rowC[i]);
            ++bad;
        }
    for (size_t i = 0; i < kvF.size(); ++i) {
        if (kvF[i] != kvC[i]) {
            if (!bad) std::printf("row i8 L%d R%d: cache byte %zu is %02x, the chain's %02x\n", L, R,
                                  i, kvF[i], kvC[i]);
            ++bad;
        }
        if (kvS[i] != kvF[i]) {
            if (!bad) std::printf("row i8 L%d R%d: kv_store byte %zu is %02x, the fused %02x\n", L,
                                  R, i, kvS[i], kvF[i]);
            ++bad;
        }
    }
    /* The streamed form's two planes: the mirror holds every named slot's row at its block's
     * offset, the ring only a resident block's, and every other byte is the filler it started
     * with -- a slot no token names keeps it, and so does a block. */
    std::vector<int> used((size_t)c.blocks, 0);
    for (int s : slot)
        if (s >= 0 && s < c.blocks * c.bs) used[(size_t)(s / c.bs)] = 1;
    auto named = [&](int64_t k, int64_t o) {
        return used[(size_t)k] &&
               std::find(slot.begin(), slot.end(), (int32_t)(k * c.bs + o)) != slot.end();
    };
    for (int64_t k = 0; k < c.blocks; ++k)
        for (int64_t o = 0; o < c.bs; ++o)
            for (int64_t j = 0; j < rowb; ++j) {
                const int64_t i = ((k * c.bs + o) * rowb) + j;   /* the same offset in kvF */
                const uint8_t want = named(k, o) ? kvF[(size_t)i] : host0[(size_t)i];
                if (hostD[(size_t)i] != want) {
                    if (!bad) std::printf("row i8 L%d R%d: mirror byte %lld is %02x, want %02x\n",
                                          L, R, (long long)i, hostD[(size_t)i], want);
                    ++bad;
                }
            }
    for (int64_t r = 0; r < c.slots; ++r) {
        int64_t src = -1;
        for (int64_t k = 0; k < c.blocks; ++k)
            if (used[(size_t)k] && res[(size_t)k] == r) src = k;
        for (int64_t o = 0; o < c.bs; ++o)
            for (int64_t j = 0; j < rowb; ++j) {
                const int64_t i = ((r * c.bs + o) * rowb) + j;
                const uint8_t want = (src >= 0 && named(src, o))
                                         ? kvF[(size_t)(((src * c.bs + o) * rowb) + j)]
                                         : ring0[(size_t)i];
                if (ringD[(size_t)i] != want) {
                    if (!bad) std::printf("row i8 L%d R%d: ring byte %lld is %02x, want %02x\n", L,
                                          R, (long long)i, ringD[(size_t)i], want);
                    ++bad;
                }
            }
    }
    /* libref's own store row over the same tokens, slots and cache filler: the oracle the latent
     * columns are held to, in place of a form restated here. The two stores disagree on two slot
     * values and write neither of them: this kernel treats slot <= 0 as null (block 0 is the
     * padding block), and skips a slot past the id space as padding, while libref writes every
     * slot >= 0 and refuses one past the blocks. So the oracle is given those tokens as the
     * padding slot -1, which is the row for them both: the filler, untouched. */
    std::vector<int32_t> slotL = slot;
    for (int32_t& s : slotL)
        if (s <= 0 || s >= c.blocks * c.bs) s = -1;
    std::vector<uint8_t> kvL;
    {
        std::string why;
        if (!libref_store(rowF.data(), rowF.data() + hd, c.pitch, c.M, hd, R, c.bs, c.blocks, rowb,
                          slotL, kv0, kvL, why)) {
            std::printf("row i8 L%d R%d: %s\n", L, R, why.c_str());
            return 1;
        }
    }
    /* The row's columns: L codes, L/128 f32 scales, R rope columns. The rope columns are the row's
     * own bf16 (no rotation, no rounding), which is the one column group an oracle can name
     * without the quantiser. The latent columns are checked against the chain above, and then
     * against libref's store, which is what X-07b's reader has to make sense of. */
    for (int m = 0; m < c.M; ++m) {
        const int s = slot[(size_t)m];
        if (s <= 0 || s >= c.blocks * c.bs) continue;   /* the store writes no row for it */
        const int64_t base = ((int64_t)(s / c.bs) * c.bs + (s % c.bs)) * rowb;
        if (base % 16) {
            if (!bad) std::printf("row i8 L%d R%d: slot %d starts at %lld, not 16-byte aligned\n", L,
                                  R, s, (long long)base);
            ++bad;
        }
        const uint8_t* const row = kvF.data() + base;
        const uint16_t* const d = (const uint16_t*)(row + L + 4 * (L / 128));
        for (int i = 0; i < R; ++i) {
            const uint16_t want = rowF[(size_t)m * c.pitch + (size_t)(L + i)];
            if (d[i] != want) {
                if (!bad) std::printf("row i8 L%d R%d: token %d rope column %d is %04x, the row's %04x\n",
                                      L, R, m, i, d[i], want);
                ++bad;
            }
        }
        /* libref copies the same bf16 columns, so its rope columns are this row's, byte for byte. */
        if (std::memcmp(row + L + 4 * (L / 128), kvL.data() + base + L + 4 * (L / 128),
                        (size_t)(2 * R)) != 0) {
            if (!bad) std::printf("row i8 L%d R%d: token %d rope columns are not libref's\n", L, R, m);
            ++bad;
        }
        /* The store's own bytes against libref's: the same row, the same slots, the same cache
         * filler, through libref's int8 store. Three things are held per block. The scale, to 4e-6
         * relative: both divide the block's absmax by 127 and multiply by the normaliser, one in
         * f32 and one in f32 after a double rotation, so the two absmaxes differ by the kernel's
         * rotation error. And, column by column, the code (equal, or one apart where the ratio sits
         * on a rounding boundary) and the decoded value `scale * code`, within one code step
         * (absmax / 127) plus that rotation error -- each store is within half a step of the rotated
         * column, so two honest stores of the same block are within one step of each other, and this
         * is the tightest bound that does not assume the two rotations round alike. The half step
         * itself is checked exactly where the rotation is (i8_edge's one-hot block, and
         * r4d_selftest's constant row on the card): there both stores must write the same codes. A
         * block libref calls dead is the design note's scale 1 / codes 0, and the kernel's must be
         * that block exactly. */
        const int8_t* const codes = (const int8_t*)row;
        const float* const scales = (const float*)(row + L);
        const int8_t* const ref_codes = (const int8_t*)(kvL.data() + base);
        const float* const ref_scales = (const float*)(kvL.data() + base + L);
        for (int b = 0; b < L / 128; ++b) {
            const float sc = scales[b], rsc = ref_scales[b];
            if (rsc == 1.0f) {   /* libref's dead block: scale 1 and codes 0 */
                if (sc != rsc) {
                    if (!bad) std::printf("row i8 L%d R%d: token %d block %d scale %g, libref's dead block's %g\n",
                                          L, R, m, b, (double)sc, (double)rsc);
                    ++bad;
                }
                for (int i = 0; i < 128; ++i)
                    if (codes[b * 128 + i] != 0) {
                        if (!bad) std::printf("row i8 L%d R%d: token %d block %d is live (code %d), "
                                              "libref's block is dead\n", L, R, m, b,
                                              (int)codes[b * 128 + i]);
                        ++bad;
                        break;
                    }
                continue;
            }
            if (!(std::fabs(sc - rsc) <= rsc * 4e-6f)) {
                if (!bad) std::printf("row i8 L%d R%d: token %d block %d scale %g, libref's %g\n",
                                      L, R, m, b, (double)sc, (double)rsc);
                ++bad;
                continue;
            }
            /* libref stores (absmax / 127) * 1/sqrt(128), so the block's absmax is in its scale. */
            const double amax = (double)rsc * 127.0 / 0.08838834764831845;
            const double step = amax / 127.0 * 1.0001 + amax * 1e-6;
            for (int i = 0; i < 128; ++i) {
                const int ck = codes[b * 128 + i], cl = ref_codes[b * 128 + i];
                if (std::abs(ck - cl) > 1) {
                    if (!bad) std::printf("row i8 L%d R%d: token %d block %d column %d is code %d, "
                                          "libref's %d\n", L, R, m, b, i, ck, cl);
                    ++bad;
                }
                const double dec = (double)sc * (double)ck, rdec = (double)rsc * (double)cl;
                if (std::fabs(dec - rdec) > step) {
                    if (!bad) std::printf("row i8 L%d R%d: token %d block %d column %d decodes to %g, "
                                          "libref's %g (step %g)\n", L, R, m, b, i, dec, rdec, step);
                    ++bad;
                }
            }
        }
    }
    std::printf("row i8 L%-4d R%-3d M%d pitch%-4d bs%d blocks%d: %s\n", L, R, c.M, c.pitch, c.bs,
                c.blocks, bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}

/* X-07's edge blocks, through kv_store's int8 form at the served geometry (L 512, R 64, a 656-byte
 * row): a block with one live column, so the rotation is exactly +-1 and the stored scale is the
 * number `(1/127) * 1/sqrt(128)` and nothing like a tolerance; blocks that are all zero; and a
 * block holding an inf, whose rotated absmax is not finite. The design note has the store write
 * scale 1 / codes 0 for an absmax of zero or not finite -- an inf scale would write codes 0 that
 * decode to NaN -- and the power-of-two block is where the scale itself is compared exactly, as
 * libref's form computes it. The rope columns are the bf16 the store was handed. */
int i8_edge() {
    const int64_t L = 512, R = 64, hd = 288, rowb = 656;
    const int M = 2, bs = 4, blocks = 2;
    const float kRot = 0.08838834764831845f;   /* 1 / sqrt(128) */
    std::vector<uint16_t> k((size_t)M * hd, f2b(0.0f)), v((size_t)M * hd, f2b(0.0f));
    k[(size_t)0] = f2b(1.0f);                     /* token 0, block 0: one live column */
    k[(size_t)2 * 128] = f2b(INFINITY);           /* token 1, block 2: not finite */
    for (int m = 0; m < M; ++m)                   /* v's last R columns are the row's rope */
        for (int64_t i = 0; i < R; ++i)
            v[(size_t)m * hd + (size_t)(hd - R + i)] = f2b(0.25f * (float)(m * R + i + 1));
    std::vector<int32_t> slot{ 0, 5 };            /* token 0 at slot 0, token 1 at block 1's slot 1 */
    std::vector<uint8_t> kv((size_t)blocks * bs * rowb, 0x5a);
    {
        std::vector<RadTensor> t{ tens(k.data(), RAD_BF16, { M, hd }), tens(v.data(), RAD_BF16, { M, hd }),
                                  tens(slot.data(), RAD_I32, { M }),
                                  tens(kv.data(), RAD_I8, { blocks, 1, bs, rowb }) };
        std::vector<RadParam> p{ pi("M", M), pi("head_dim", hd), pi("n_head_kv", 1),
                                 pi("block_size", bs), ps("kv_dtype", "i8"), pi("rope_dim", R) };
        RadArgs a = mk(t, p);
        const int rc = r4d_rad_kv_store_i8(&a, nullptr);
        if (rc != RAD_OK) {
            std::printf("row i8 edge: launch failed (%d)\n", rc);
            return 1;
        }
    }
    int bad = 0;
    /* The same rows through libref's own store: at these blocks the rotation is exact, so the two
     * stores must write the same bytes, not the same bytes within a tolerance -- the one place the
     * row's quantiser is checked against libref with nothing between them. */
    {
        std::vector<uint8_t> kvL;
        std::string why;
        std::vector<int32_t> sm = slot;
        if (!libref_store(k.data(), v.data(), hd, M, hd, R, bs, blocks, rowb, sm, kv, kvL, why)) {
            std::printf("row i8 edge: %s\n", why.c_str());
            return 1;
        }
        if (std::memcmp(kvL.data(), kv.data(), kv.size()) != 0) {
            for (size_t i = 0; i < kv.size(); ++i)
                if (kvL[i] != kv[i]) {
                    std::printf("row i8 edge: byte %zu is %02x, libref's %02x\n", i, kv[i], kvL[i]);
                    break;
                }
            std::printf("row i8 edge: the store's bytes are not libref's\n");
            return 1;
        }
    }
    const float onehot = (1.0f / 127.0f) * kRot;   /* amax 1, so the scale is exact */
    for (int m = 0; m < M; ++m) {
        const int s = slot[(size_t)m];
        const uint8_t* const row = kv.data() + ((size_t)(s / bs) * bs + (size_t)(s % bs)) * rowb;
        const int8_t* const codes = (const int8_t*)row;
        const float* const scales = (const float*)(row + L);
        /* Token 0's block 0 is the one-hot one; token 1's block 2 holds the inf. Everything else
         * in both rows is zero, so it must be scale 1 / codes 0. */
        for (int b = 0; b < L / 128; ++b) {
            const bool live = m == 0 && b == 0;
            const float want = live ? onehot : 1.0f;
            if (scales[b] != want) {
                if (!bad)
                    std::printf("row i8 edge: token %d block %d scale %.9g, want %.9g\n", m, b,
                                (double)scales[b], (double)want);
                ++bad;
            }
            for (int i = 0; i < 128; ++i) {
                const int8_t wantc = live ? 127 : 0;   /* the rotation of e_0 is all ones */
                if (codes[b * 128 + i] != wantc) {
                    if (!bad)
                        std::printf("row i8 edge: token %d block %d code %d is %d, want %d\n", m, b, i,
                                    (int)codes[b * 128 + i], (int)wantc);
                    ++bad;
                }
            }
        }
        /* The live block's decoded columns are the rotated one-hot column, +1/sqrt(128). */
        for (int i = 0; i < 128 && m == 0; ++i) {
            const double dec = (double)scales[0] * (double)codes[i];
            if (std::fabs(dec - (double)kRot) > (double)kRot * 1e-6) {
                if (!bad)
                    std::printf("row i8 edge: block 0 column %d decodes to %g, want %g\n", i, dec,
                                (double)kRot);
                ++bad;
            }
        }
        const uint16_t* const d = (const uint16_t*)(row + L + 4 * (L / 128));
        for (int64_t i = 0; i < R; ++i) {
            const uint16_t want = v[(size_t)m * hd + (size_t)(hd - R + i)];
            if (d[i] != want) {
                if (!bad) std::printf("row i8 edge: token %d rope column %d is %04x, want %04x\n", m,
                                      (int)i, d[i], want);
                ++bad;
            }
        }
    }
    std::printf("row i8 edge (power of two, zero and inf blocks): %s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}

/* D-02 (#88): DFly2's chain reads rows k + 1 of every sequence's block of the final norm in round
 * k. The plugin now gathers every round's rows in one launch and normalises them in one launch,
 * the norm into the first half of a [rounds * n, 2E] plane, where it issued a gather and a norm of
 * n rows a round. Held here to those per-round launches byte for byte: at one sequence and at
 * sixteen the hoisted launch is on either side of the row count the norm's block width turns on
 * (r4d_act_threads), and a row's bytes must not depend on it. The plane's other half is the
 * embedding norm's, written a round later: neither form may touch it. */
int hoist(int n, int rounds, int E, uint32_t seed) {
    const int blk = rounds + 1, Mz = n * blk, RN = rounds * n, P = 2 * E;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    std::vector<uint16_t> z((size_t)Mz * E), gain((size_t)E), plane0((size_t)RN * P);
    for (auto& x : z) x = f2b(u(rng));
    for (auto& x : gain) x = f2b(u(rng));
    for (auto& x : plane0) x = f2b(u(rng));
    std::vector<int32_t> idx((size_t)RN);
    for (int k = 0; k < rounds; ++k)
        for (int s = 0; s < n; ++s) idx[(size_t)(k * n + s)] = s * blk + k + 1;   /* iota(k + 1, step blk) */
    auto gather = [&](int32_t* ix, uint16_t* y, int M) {
        std::vector<RadTensor> t{ tens(z.data(), RAD_BF16, { Mz, E }), tens(ix, RAD_I32, { M }),
                                  tens(y, RAD_BF16, { M, E }) };
        std::vector<RadParam> p{ pi("M", M), pi("n", E), ps("dtype", "bf16") };
        RadArgs a = mk(t, p);
        return r4d_rad_gather_rows(&a, nullptr);
    };
    auto norm = [&](uint16_t* x, uint16_t* y, int M) {
        std::vector<RadTensor> t{ tens(x, RAD_BF16, { M, E }), tens(gain.data(), RAD_BF16, { E }),
                                  tens(y, RAD_BF16, { M, E }, { P, 1 }) };
        std::vector<RadParam> p{ pi("M", M), pi("n", E), pf("eps", 1e-6), ps("dtype", "bf16") };
        RadArgs a = mk(t, p);
        return r4d_rad_rmsnorm(&a, nullptr);
    };
    /* a launch a round, each into the first n rows of its buffers (kept here a round) */
    std::vector<uint16_t> zkR((size_t)RN * E, 0), plR = plane0;
    int rc = RAD_OK;
    for (int k = 0; k < rounds && rc == RAD_OK; ++k) {
        std::vector<uint16_t> zk((size_t)n * E), cc(plane0.begin(), plane0.begin() + (size_t)n * P);
        rc = gather(idx.data() + (size_t)k * n, zk.data(), n);
        if (rc == RAD_OK) rc = norm(zk.data(), cc.data(), n);
        std::memcpy(zkR.data() + (size_t)k * n * E, zk.data(), zk.size() * 2);
        for (int r = 0; r < n; ++r)
            std::memcpy(plR.data() + ((size_t)k * n + r) * P, cc.data() + (size_t)r * P, (size_t)E * 2);
    }
    /* one launch each */
    std::vector<uint16_t> zkH((size_t)RN * E, 0), plH = plane0;
    int rc2 = gather(idx.data(), zkH.data(), RN);
    if (rc2 == RAD_OK) rc2 = norm(zkH.data(), plH.data(), RN);
    if (rc != RAD_OK || rc2 != RAD_OK) {
        std::printf("hoist n%d rounds%d E%d: launch failed (a round %d, hoisted %d)\n", n, rounds, E, rc, rc2);
        return 1;
    }
    int bad = 0, pad = 0;
    for (size_t i = 0; i < zkH.size(); ++i) bad += zkH[i] != zkR[i];
    for (int r = 0; r < RN; ++r)
        for (int j = 0; j < P; ++j) {
            const size_t o = (size_t)r * P + (size_t)j;
            if (j < E) bad += plH[o] != plR[o];
            else       pad += plH[o] != plane0[o];
        }
    /* the gathered rows are z's own */
    for (int i = 0; i < RN; ++i)
        bad += std::memcmp(zkH.data() + (size_t)i * E, z.data() + (size_t)idx[(size_t)i] * E, (size_t)E * 2) != 0;
    if (bad || pad)
        std::printf("hoist n%d rounds%d E%d: %d byte(s) differ from the per-round launches, %d of "
                    "the other half written\n", n, rounds, E, bad, pad);
    return (bad || pad) ? 1 : 0;
}

}  // namespace

int main() {
    int fails = 0;
    /* The served shape: latent 512, rope 64, a bf16 gain, gptj, one token a row. */
    fails += one(base());
    {   /* The row's pitch wider than its columns: the padding is nobody's to write. */
        Cfg c = base();
        c.pitch = 608;
        fails += one(c);
    }
    {   /* An f32 gain (the WF32 form) and one token. */
        Cfg c = base();
        c.wf32 = true;
        c.M = 1;
        c.slot = { 3 };
        c.pos = { 5 };
        fails += one(c);
    }
    {   /* Every slot padding: the row is written, the cache is not. */
        Cfg c = base();
        c.slot = { -1, c.blocks * c.bs, -1 };
        fails += one(c);
    }
    {   /* imrope: three position components, gptj's adjacent pairs, two blocks of four. */
        Cfg c = base();
        c.mode = "imrope_gptj";
        c.plane = true;
        c.M = 5;
        c.bs = 4;
        c.blocks = 3;
        c.slot = { 0, 3, 8, 11, 5 };
        c.pos = { 3, 7, 11, 5, 9, 1, 2, 4, 8, 16, 0, 1, 2, 3, 4 };
        fails += one(c);
    }
    {   /* A narrower latent, five tokens, a plain rope theta and a scaled position. */
        Cfg c = base();
        c.L = 288;
        c.R = 64;
        c.pitch = 352;
        c.M = 5;
        c.bs = 4;
        c.blocks = 3;
        c.slot = { 0, 3, 8, 11, 5 };
        c.pos = { 3, 7, 11, 5, 9 };
        c.theta = 10000.0;
        c.scale = 0.7071067811865476;
        fails += one(c);
    }
    {   /* A rotary wider than a block: rope_dim 1024 is 512 pairs, twice kThreads, so the rope
         * phase's loop has to stride over them; a pair the first pass leaves unwritten is a
         * column of the row and of the slot that keeps its pre-rotation value. */
        Cfg c = base();
        c.L = 512;
        c.R = 1024;
        c.pitch = 1536;
        c.M = 2;
        c.bs = 4;
        c.blocks = 3;
        c.slot = { 0, 8 };
        c.pos = { 3, 7 };
        fails += one(c);
    }
    {   /* X-07 (#136): the int8 latent row -- the served geometry, three tokens into two blocks,
         * one of them not resident in the ring, and a fourth that is padding both ways. */
        I8Cfg c;
        c.slot = { 3, 9, -1, c.blocks * c.bs };
        c.pos = { 3, 7, 11, 5 };
        c.M = 4;
        c.res = { -1, 0, -1 };
        fails += i8(c);
    }
    {   /* A wider rotary: 192 rope columns, so the row is 912 bytes, still a multiple of 16. */
        I8Cfg c;
        c.R = 192;
        c.pitch = 704;
        c.slot = { 5, 0, 16 };
        c.pos = { 1, 2, 3 };
        c.res = { 0, -1, 1 };
        c.slots = 3;
        fails += i8(c);
    }
    {   /* One token, one block, the whole ring resident. */
        I8Cfg c;
        c.M = 1;
        c.bs = 4;
        c.blocks = 1;
        c.slots = 1;
        c.slot = { 2 };
        c.pos = { 9 };
        c.res = { 0 };
        fails += i8(c);
    }
    fails += i8_edge();
    {   /* D-02: the chain's hoisted gather and norm, at the stream counts and depths served. */
        int h = 0, cases = 0;
        for (int E : { 256, 1000, 1536, 4096 })
            for (int n : { 1, 2, 8, 16 })
                for (int rounds = 1; rounds <= 5; ++rounds, ++cases)
                    h += hoist(n, rounds, E, (uint32_t)(E + 31 * n + rounds));
        std::printf("chain hoist (D-02): %d case(s), %s\n", cases, h ? "FAIL" : "ok");
        fails += h;
    }
    std::printf("row front: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
