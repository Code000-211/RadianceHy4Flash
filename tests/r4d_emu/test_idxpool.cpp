/* tests/r4d_emu: the Hy4-flash compressed indexer's key side -- the gated group pool
 * (r4d_idx_pool_gated_f32.hip: idx_pool_gated, idx_pool_gated_fp8, idx_pool_gated_i8) -- on the
 * host SIMT emulator, held to libref's row of the same op on the same bytes.
 *
 * Both sides are compared byte for byte: libref's pool_group is the kernel's arithmetic in the
 * kernel's order (the gated pool, the max-subtracted exp, the f64 mean square, the two roundings,
 * the rotary), so no tolerance is right here and none is given.
 *
 * The cases are chosen for what a value comparison alone does not show:
 *
 *  - a workgroup whose groups all have page < 0 and that has nothing to write beside the cache
 *    returns before it reads a weight (K-24's early-out). libref writes nothing to a dead page
 *    either, so the 0x7e an untouched buffer is filled with must come back as it went in -- and
 *    every byte of every buffer is compared, not only the rows a live page names;
 *  - the early-out must not fire when `out` (the tap) or `q` (the codes) is present: those are
 *    written for a dead page's group too, so a fixture with every page dead shows that they are;
 *  - the rotation's cos and sin are taken before the serial f64 norm sum (K-24's overlap), which
 *    must leave the rotated bytes where they were;
 *  - both width forms are reached: W < 16 runs one group a workgroup, W >= 16 four (kWideMinW).
 *
 * Needs libref, like test_hy4 and test_kda (ctest sets RADIANCE_HOME). */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <string>
#include <vector>

static const RadKernelInfo* libref_op(const char* op) {
    static void* h = nullptr;
    if (!h) {
        const char* home = std::getenv("RADIANCE_HOME");
        if (!home || !*home) return nullptr;
        h = dlopen((std::string(home) + "/kernels/libref.so").c_str(), RTLD_NOW | RTLD_LOCAL);
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

static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: return 4;
        case RAD_BF16: return 2;
        case RAD_I64: return 8;
        default: return 1;
    }
}

static RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam pf(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

struct Opd {
    uint32_t dt = RAD_BF16;
    std::vector<int64_t> shape;
    int64_t pitch = 0;
    std::vector<uint8_t> bytes;
    bool absent = false, out = false;   /* `out`: the op writes it, so libref's bytes are the oracle */
    int64_t elements() const {
        /* A pitched row still holds shape[1] elements: the last row ends the buffer. */
        if (shape.size() == 2 && pitch) return (shape[0] - 1) * pitch + shape[1];
        int64_t n = 1;
        for (int64_t e : shape) n *= e;
        return n;
    }
    RadTensor tensor(void* data) const {
        RadTensor t{};
        t.data = absent ? nullptr : data;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[(size_t)i];
            t.stride[i] = st;
            st *= shape[(size_t)i];
        }
        if (shape.size() == 2 && pitch) t.stride[0] = pitch;
        return t;
    }
};

static Opd opd(uint32_t dt, std::vector<int64_t> shape, int64_t pitch = 0, bool out = false) {
    Opd o;
    o.dt = dt;
    o.shape = std::move(shape);
    o.pitch = pitch;
    o.out = out;
    o.bytes.assign((size_t)o.elements() * esz(dt), 0x7e);
    return o;
}
static Opd absent() { Opd o; o.absent = true; return o; }

/* Every element offset of a rank 1-4 tensor: a cache page is 128 codes in four rows, and all of
 * them are the op's to write, so all of them are compared. */
static std::vector<int64_t> offsets(const RadTensor& t) {
    std::vector<int64_t> o;
    const int64_t rank = (int64_t)t.rank;
    std::vector<int64_t> ix((size_t)rank, 0);
    for (;;) {
        int64_t off = 0;
        for (int64_t i = 0; i < rank; ++i) off += ix[(size_t)i] * t.stride[i];
        o.push_back(off);
        int64_t d = rank - 1;
        for (; d >= 0; --d) {
            if (++ix[(size_t)d] < t.shape[d]) break;
            ix[(size_t)d] = 0;
        }
        if (d < 0) break;
    }
    return o;
}

static void put(Opd& o, int64_t off, double v) {
    switch (o.dt) {
        case RAD_BF16: ((uint16_t*)o.bytes.data())[off] = f2b((float)v); break;
        case RAD_F32: ((float*)o.bytes.data())[off] = (float)v; break;
        case RAD_I32: ((int32_t*)o.bytes.data())[off] = (int32_t)v; break;
        default: o.bytes[(size_t)off] = (uint8_t)(int)v; break;
    }
}

/* A value that moves with both indices: every dot product is then distinct, and the gated sum
 * lands where bf16 rounding is a real constraint rather than a formality. */
static double sample(int64_t r, int64_t c, int64_t salt) {
    const uint32_t h = (uint32_t)(r * 1315423911u + c * 2654435761u + (uint32_t)salt * 40503u);
    const uint32_t m = (h ^ (h >> 15)) * 2246822519u;
    return (double)((int32_t)(m >> 8) % 2000) / 1000.0 - 0.5;
}
static void fill_mat(Opd& o, int64_t salt) {
    const RadTensor t = o.tensor(o.bytes.data());
    for (int64_t r = 0; r < t.shape[0]; ++r)
        for (int64_t c = 0; c < t.shape[1]; ++c) put(o, r * t.stride[0] + c, sample(r, c, salt));
}
static void fill_vec(Opd& o, int64_t salt) {
    const RadTensor t = o.tensor(o.bytes.data());
    for (int64_t i = 0; i < t.shape[0]; ++i) put(o, i * t.stride[0], sample(i, 7, salt));
}
/* The bytes no launch is meant to write keep another value than 0x7e, so a launch that writes
 * them anyway is visible. */
static void fill_bytes(Opd& o) {
    for (size_t i = 0; i < o.bytes.size(); ++i) o.bytes[i] = (uint8_t)(0x31 + (uint8_t)(i * 11));
}

static int g_cases, g_bad, g_only = -1;   /* argv[1]: run one case, to bisect a fixture */

/* libref's op and the kernel's, on two copies of the same operands, compared byte for byte. */
static void check(const char* op, RadLaunchFn kernel, const char* what, std::vector<RadParam> p,
                  std::vector<Opd>& ops) {
    const int id = g_cases++;
    if (g_only >= 0 && g_only != id) return;
    const RadKernelInfo* ref = libref_op(op);
    if (!ref || !ref->launch) {
        std::printf("FAIL %s: libref's %s not loadable (set RADIANCE_HOME)\n", what, op);
        ++g_bad;
        return;
    }
    const size_t n = ops.size();
    std::vector<std::vector<uint8_t>> a(n), b(n);
    std::vector<RadTensor> ta(n), tb(n);
    for (size_t i = 0; i < n; ++i) {
        if (ops[i].absent) { ta[i] = ops[i].tensor(nullptr); tb[i] = ta[i]; continue; }
        a[i] = b[i] = ops[i].bytes;
        ta[i] = ops[i].tensor(a[i].data());
        tb[i] = ops[i].tensor(b[i].data());
    }
    RadArgs ar{}, ak{};
    ar.t = ta.data(); ar.n_t = (int)n; ar.p = p.data(); ar.n_p = (int)p.size(); ar.world_size = 1;
    ak.t = tb.data(); ak.n_t = (int)n; ak.p = p.data(); ak.n_p = (int)p.size(); ak.world_size = 1;
    const int rr = ref->launch(&ar, nullptr), rk = kernel(&ak, nullptr);
    if (rr != RAD_OK || rk != RAD_OK) {
        std::printf("FAIL %s: libref rc=%d kernel rc=%d\n", what, rr, rk);
        ++g_bad;
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        if (ops[i].absent) continue;
        const size_t e = esz(ops[i].dt);
        if (!ops[i].out) {
            /* An input comes back as it went in, padding and all. */
            if (std::memcmp(a[i].data(), ops[i].bytes.data(), ops[i].bytes.size()) ||
                std::memcmp(b[i].data(), ops[i].bytes.data(), ops[i].bytes.size())) {
                std::printf("FAIL %s: operand %zu changed\n", what, i);
                ++g_bad;
                return;
            }
            continue;
        }
        for (int64_t off : offsets(ta[i])) {
            if (std::memcmp(a[i].data() + (size_t)off * e, b[i].data() + (size_t)off * e, e)) {
                std::printf("FAIL %s: operand %zu element %lld differs\n", what, i, (long long)off);
                ++g_bad;
                return;
            }
        }
    }
    std::printf("ok   %-18s %s\n", op, what);
}

/* --- fixtures ------------------------------------------------------------------------------ */

/* A group closes when its page is a cache page: -1 (qsa_work: the block does not complete this
 * step) and a page past the cache (a dead slot) both mean nothing to write. */
enum PgMode { PG_LIVE, PG_DEAD, PG_MIXED };

static std::vector<int32_t> page_of(PgMode m, int64_t W, int64_t pages) {
    std::vector<int32_t> p((size_t)W, -1);
    int32_t next = 0;
    for (int64_t w = 0; w < W; ++w) {
        switch (m) {
            case PG_LIVE: p[(size_t)w] = (int32_t)(next++ % pages); break;
            case PG_DEAD: p[(size_t)w] = -1; break;
            case PG_MIXED:
                /* groups 4..7 -- one whole workgroup of the wide form -- dead, a group dead by
                 * value, a group dead by range, the rest with a page of their own */
                if (w >= 4 && w < 8) p[(size_t)w] = -1;
                else if (w % 9 == 8) p[(size_t)w] = (int32_t)(pages + 3);
                else if (w % 4 == 2) p[(size_t)w] = -1;
                else p[(size_t)w] = (int32_t)(next++ % pages);
                break;
        }
    }
    return p;
}

struct Bf {
    int64_t W, pages, rd, pad;
    PgMode pg;
    bool imrope, gain_bf16, taps, cache, chain;
};

static void bf16_case(const Bf& f) {
    const int64_t n = 128, ratio = 4;
    std::vector<Opd> ops;
    Opd k = opd(RAD_BF16, { f.W * ratio, n }, f.pad ? n + f.pad : 0); fill_mat(k, 1);
    ops.push_back(std::move(k));
    Opd wv = opd(RAD_F32, { n, n }); fill_mat(wv, 2); ops.push_back(std::move(wv));
    Opd wg = opd(RAD_F32, { n, n }); fill_mat(wg, 3); ops.push_back(std::move(wg));
    Opd ap = opd(RAD_F32, { ratio, n }); fill_mat(ap, 4); ops.push_back(std::move(ap));
    Opd gn = opd(f.gain_bf16 ? RAD_BF16 : RAD_F32, { n }); fill_vec(gn, 5); ops.push_back(std::move(gn));
    Opd po = f.imrope ? opd(RAD_I32, { f.W, 3 }) : opd(RAD_I32, { f.W });
    {
        const RadTensor t = po.tensor(po.bytes.data());
        for (int64_t w = 0; w < f.W; ++w)
            for (int64_t j = 0; j < (f.imrope ? 3 : 1); ++j) put(po, w * t.stride[0] + j, 17 + 4 * w + 3 * j);
    }
    ops.push_back(std::move(po));
    /* A page and a cache go together: a group closes only by writing its page. */
    const std::vector<int32_t> pages = page_of(f.pg, f.W, f.pages);
    if (f.cache) {
        Opd pg = opd(RAD_I32, { f.W });
        for (int64_t w = 0; w < f.W; ++w) put(pg, w, pages[(size_t)w]);
        ops.push_back(std::move(pg));
        Opd ca = opd(RAD_BF16, { f.pages, 1, 4, n / 4 }, 0, true); fill_bytes(ca);
        ops.push_back(std::move(ca));
    } else {
        ops.push_back(absent());
        ops.push_back(absent());
    }
    if (f.taps) {
        Opd ou = opd(RAD_BF16, { f.W, n }, f.pad ? n + f.pad : 0, true); fill_bytes(ou);
        ops.push_back(std::move(ou));
    } else {
        ops.push_back(absent());
    }

    std::vector<RadParam> params = { pi("n", n), pi("ratio", ratio), pi("rotary_dim", f.rd),
                                     pf("eps", 1e-6), pf("theta", 1e7),
                                     ps("mode", f.imrope ? "imrope_gptj" : "gptj") };
    if (f.imrope) params.push_back(ps("sections", "16 24 24"));
    params.push_back(pi("chain", f.chain ? 1 : 0));

    char what[192];
    std::snprintf(what, sizeof(what), "W %3lld rd %3lld %s%s%s%s%.7s%s", (long long)f.W, (long long)f.rd,
                  f.imrope ? "imrope " : "gptj   ", f.gain_bf16 ? "gain-bf16 " : "",
                  f.taps ? "out " : "no-out ", f.cache ? "cache " : "",
                  f.pg == PG_MIXED ? "mixed-pages" : f.pg == PG_DEAD ? "dead-pages " : "live-pages ",
                  f.chain ? " chain" : "");
    check("idx_pool_gated", r4d_rad_idx_pool_gated, what, params, ops);
}

struct Q {
    const char* op;         /* idx_pool_gated_fp8 or idx_pool_gated_i8 */
    int64_t W, pages;
    PgMode pg;
    bool codes;             /* `q` and its scale: written for a dead page's group too */
    bool cache;
    bool chain;
};

static void quant_case(const Q& f) {
    const int64_t n = 128, ratio = 4;
    const bool i8 = !std::strcmp(f.op, "idx_pool_gated_i8");
    std::vector<Opd> ops;
    Opd k = opd(RAD_BF16, { f.W * ratio, n }); fill_mat(k, 7); ops.push_back(std::move(k));
    Opd wv = opd(RAD_F32, { n, n }); fill_mat(wv, 8); ops.push_back(std::move(wv));
    Opd wg = opd(RAD_F32, { n, n }); fill_mat(wg, 9); ops.push_back(std::move(wg));
    Opd ap = opd(RAD_F32, { ratio, n }); fill_mat(ap, 10); ops.push_back(std::move(ap));
    Opd gn = opd(RAD_F32, { n }); fill_vec(gn, 11); ops.push_back(std::move(gn));
    Opd po = opd(RAD_I32, { f.W });
    for (int64_t w = 0; w < f.W; ++w) put(po, w, 31 + 4 * w);
    ops.push_back(std::move(po));
    const std::vector<int32_t> pages = page_of(f.pg, f.W, f.pages);
    if (f.cache) {
        Opd pg = opd(RAD_I32, { f.W });
        for (int64_t w = 0; w < f.W; ++w) put(pg, w, pages[(size_t)w]);
        ops.push_back(std::move(pg));
        Opd ca = opd(RAD_U8, { f.pages, 1, 4, 34 }, 0, true); fill_bytes(ca);
        ops.push_back(std::move(ca));
    } else {
        ops.push_back(absent());
        ops.push_back(absent());
    }
    if (f.codes) {
        Opd q8 = opd(i8 ? RAD_I8 : RAD_F8E4M3, { f.W, n }, 0, true); fill_bytes(q8);
        ops.push_back(std::move(q8));
        Opd qs = opd(RAD_F32, { f.W, 1 }, 0, true); fill_bytes(qs);
        ops.push_back(std::move(qs));
    } else {
        ops.push_back(absent());
        ops.push_back(absent());
    }

    std::vector<RadParam> params = { pi("n", n), pi("ratio", ratio), pi("rotary_dim", n),
                                     pf("eps", 1e-6), pf("theta", 1e7), ps("mode", "gptj"),
                                     pi("chain", f.chain ? 1 : 0) };

    char what[192];
    std::snprintf(what, sizeof(what), "W %3lld %s%s%.7s%s", (long long)f.W,
                  f.codes ? "codes " : "no-codes ", f.cache ? "cache " : "",
                  f.pg == PG_MIXED ? "mixed-pages" : f.pg == PG_DEAD ? "dead-pages " : "live-pages ",
                  f.chain ? " chain" : "");
    const RadLaunchFn fn = i8 ? r4d_rad_idx_pool_gated_i8 : r4d_rad_idx_pool_gated_fp8;
    check(f.op, fn, what, params, ops);
}

int main(int argc, char** argv) {
    if (argc > 1) g_only = std::atoi(argv[1]);
    /* Decode, the two groups of a lone sequence: the first closes, the second does not. */
    bf16_case({ 2, 4, 128, 0, PG_MIXED, false, false, true, true, true });
    /* Decode, no group closes and nothing is written beside the cache: the early-out's case, three
     * steps in four. Not one byte may move. */
    bf16_case({ 2, 4, 128, 0, PG_DEAD, false, false, false, true, true });
    /* The wide form (four groups a workgroup) with one whole workgroup dead, groups dead by value
     * and by range, live pages around them. */
    bf16_case({ 20, 12, 128, 0, PG_MIXED, false, false, false, true, true });
    /* Every page dead but the tap asked for: the early-out must not fire. */
    bf16_case({ 20, 12, 128, 0, PG_DEAD, false, false, true, false, true });
    bf16_case({ 16, 16, 128, 0, PG_LIVE, false, false, true, true, false });
    /* M-RoPE: the three position components of a group. */
    bf16_case({ 5, 6, 128, 0, PG_MIXED, true, false, true, true, false });
    /* A rotary narrower than the channel count, a bf16 gain, and rows pitched past n. */
    bf16_case({ 7, 6, 64, 8, PG_MIXED, false, true, true, true, false });
    /* The quantised forms: the cache page, the codes, or both. */
    quant_case({ "idx_pool_gated_fp8", 6, 8, PG_MIXED, true, true, false });
    quant_case({ "idx_pool_gated_fp8", 6, 8, PG_DEAD, false, true, true });
    quant_case({ "idx_pool_gated_fp8", 6, 8, PG_DEAD, true, false, true });
    quant_case({ "idx_pool_gated_i8", 4, 6, PG_MIXED, true, true, false });
    quant_case({ "idx_pool_gated_i8", 4, 6, PG_DEAD, true, false, true });

    std::printf("idxpool: %d cases, %d failed\n", g_cases, g_bad);
    return g_bad ? 1 : 0;
}
