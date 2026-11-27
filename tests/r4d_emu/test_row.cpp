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
 * a launch of each a round (hoist()). */
#include "ref_host.h"
#include "r4d_plugin.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
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
