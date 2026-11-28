/* tests/r4d_emu: Hy4-flash's kv_b projections on WMMA (r4d_mla_kv_bf16.hip, WP16.3) on the host SIMT
 * emulator.
 *
 *   - mla_project_kv, both modes, against libref's op within one bf16 ulp (a different sum order);
 *   - the law: a token's bytes are the same whatever M, `mb` (every value of the row's tunable),
 *     `stage` (K-50's kv_expand_staged, 16 waves over a token block staged in LDS, or kvp_body's
 *     expand; which of the two ran is asserted by name), the chained form or the operand layout
 *     (16-byte fragments and global_load_tr_b128, or the element-by-element path a pitched operand
 *     takes) -- every row of every run equal to the same token run alone, at M 1, through the
 *     plain aligned path;
 *   - mla_expand_gate byte-equal to mla_project_kv's expand followed by gate_mul (its fusion
 *     claim, r4d_fuse.cpp), on both paths, every `mb` and both `stage` arms;
 *   - the staged kernel on aligned operands whose pitches are wider than their widths;
 *   - mla_expand_gate_had_quant_i8g, the 1024-thread fused quantiser, at 32 heads: the grid
 *     the K-17 pad steps (`r4d_cliff_rows`, one head past 32 at every multiple-of-2048 wave
 *     count). The extra head's workgroup must return at once, so every head's bytes equal a
 *     run of the same rows on their own and sit next to libref's, and the padded grid is what
 *     the launcher launches wherever the rule says, and only there;
 *   - the same quantiser above 64 rows, where K-50b's prefill form runs (four token tiles a
 *     workgroup, which of the two forms ran asserted by name): the prefill heights' bytes equal
 *     both the row alone's -- the 16-row form's -- and the chain the 64-row cap used to force,
 *     mla_expand_gate's gated product through had_quant_act_i8g, bit for bit.
 *
 * The emulator's WMMA sums in its own order (emu.cpp), not the card's: what is checked is that the
 * kernels compute every output from the same fragments in the same chain, which is what the law
 * says; r4d_selftest holds the card's bytes to the same rules. Needs libref (RADIANCE_HOME).
 * The quantiser runs here through the real launcher: no engine record ever reaches an emulated
 * launch, so its chained form is its plain one (r4d_chain.h) and the card's wait/done traffic
 * stays the card's, with the 108-case kbench check pass as its byte evidence. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_common.h"  // r4d_cliff_rows: the grid math the quantiser's launcher calls
#include "ref_host.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

extern const char* emu_last_kernel;   /* the emulator's (emu/hip/hip_runtime.h): the last kernel launched */

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

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed * 2654435761u + 12345u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float u() { return (float)(next() >> 8) / 8388608.0f - 1.0f; }
    float n() { return (u() + u() + u() + u()) * 0.866f; }
};

static RadParam pi(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam ps(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

static int g_fails = 0, g_cases = 0;
static void ok(bool good, const std::string& what) {
    ++g_cases;
    if (!good) ++g_fails;
    std::printf("%s %s\n", good ? "ok  " : "FAIL", what.c_str());
}

constexpr int64_t kKd = 192, kVd = 256, kL = 512;

/* A [rows, cols] bf16 matrix with row pitch `pitch` (elements) and its first element `off`
 * elements into the allocation: off 1 or an odd pitch puts it off the 16-byte grid. */
struct Mat {
    int64_t rows = 0, cols = 0, pitch = 0, off = 0;
    std::vector<uint16_t> buf;
    Mat(int64_t r, int64_t c, int64_t p, int64_t o, uint16_t fillv = 0x7e7e)
        : rows(r), cols(c), pitch(p), off(o), buf((size_t)(o + r * p + 16), fillv) {}
    uint16_t* data() {
        /* the allocation's own alignment is the vector's: start the matrix on a 16-byte boundary */
        uintptr_t a = (uintptr_t)buf.data();
        const int64_t pad = (int64_t)(((16 - (a & 15)) & 15) / 2);
        return buf.data() + pad + off;
    }
    uint16_t& at(int64_t i, int64_t j) { return data()[i * pitch + j]; }
    RadTensor t() {
        RadTensor x{};
        x.data = data();
        x.dtype = RAD_BF16;
        x.rank = 2;
        x.shape[0] = rows; x.shape[1] = cols;
        x.stride[0] = pitch; x.stride[1] = 1;
        return x;
    }
};

static int run(RadLaunchFn fn, std::vector<RadTensor> t, std::vector<RadParam> p) {
    RadArgs a{};
    a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
    return fn(&a, nullptr);
}
static std::vector<RadParam> params(int64_t M, int64_t H, const char* mode, int mb, int chain, int stage = 0) {
    std::vector<RadParam> p = { pi("M", M), pi("heads", H), pi("k_dim", kKd), pi("v_dim", kVd),
                                pi("latent_dim", kL), ps("dtype", "bf16"), pi("mb", mb), pi("chain", chain),
                                pi("stage", stage) };
    if (mode) p.push_back(ps("mode", mode));
    return p;
}

/* The inputs of one geometry, drawn once: every run copies its rows out of these. */
struct Draw {
    int64_t H, M;
    std::vector<uint16_t> x, w, g;   /* x [M, H*xw], w [H*448, 512], g [M, H*256] */
};
static Draw draw(int64_t H, int64_t M, bool absorb, uint32_t seed) {
    Draw d{ H, M, {}, {}, {} };
    Rng rng(seed);
    const int64_t xw = absorb ? kKd : kL;
    d.x.resize((size_t)(M * H * xw));
    for (auto& v : d.x) v = f2b(rng.n());
    d.w.resize((size_t)(H * (kKd + kVd) * kL));
    for (auto& v : d.w) v = f2b(rng.n() * 0.05f);
    d.g.resize((size_t)(M * H * kVd));
    for (auto& v : d.g) v = f2b(rng.n() * 3.0f);
    return d;
}

/* One run over rows [r0, r0 + n) of the draw; returns y [n, H * yw] packed. `skew` 1 puts every
 * operand off the 16-byte grid (odd pitches, a one-element offset), the element-by-element path;
 * `wide` elements (a multiple of 8) widen every pitch and keep the operands on the grid. */
static std::vector<uint16_t> project(const Draw& d, bool absorb, bool gate, int64_t r0, int64_t n, int mb,
                                     int chain, bool skew, int* rc, int stage = 0, int64_t wide = 0) {
    const int64_t H = d.H, xw = absorb ? kKd : kL, yw = absorb ? kL : kVd;
    const int64_t o = skew ? 1 : 0, sp = skew ? 3 : wide;
    Mat x(n, H * xw, H * xw + sp, o), w(H * (kKd + kVd), kL, kL + sp, o), y(n, H * yw, H * yw + sp, o),
        g(n, H * kVd, H * kVd + sp, o);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < H * xw; ++j) x.at(i, j) = d.x[(size_t)((r0 + i) * H * xw + j)];
    for (int64_t i = 0; i < H * (kKd + kVd); ++i)
        for (int64_t j = 0; j < kL; ++j) w.at(i, j) = d.w[(size_t)(i * kL + j)];
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < H * kVd; ++j) g.at(i, j) = d.g[(size_t)((r0 + i) * H * kVd + j)];
    if (gate)
        *rc = run(r4d_rad_mla_expand_gate, { x.t(), w.t(), g.t(), y.t() }, params(n, H, nullptr, mb, chain, stage));
    else
        *rc = run(r4d_rad_mla_project_kv, { x.t(), w.t(), y.t() },
                  params(n, H, absorb ? "absorb" : "expand", mb, chain, stage));
    std::vector<uint16_t> out((size_t)(n * H * yw));
    bool pad_ok = true;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < H * yw; ++j) out[(size_t)(i * H * yw + j)] = y.at(i, j);
        for (int64_t j = H * yw; j < y.pitch; ++j) pad_ok &= y.at(i, j) == 0x7e7e;
    }
    if (!pad_ok) *rc = -999;
    return out;
}

/* gate_mul over a packed [n, H*256] expansion, through the emulated libr4d row (r4d_ihc_bf16.hip) */
static std::vector<uint16_t> gate_chain(const Draw& d, int64_t r0, int64_t n, std::vector<uint16_t> ao,
                                        int* rc) {
    const int64_t W = d.H * kVd;
    Mat a(n, W, W, 0), g(n, W, W, 0), y(n, W, W, 0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < W; ++j) {
            a.at(i, j) = ao[(size_t)(i * W + j)];
            g.at(i, j) = d.g[(size_t)((r0 + i) * W + j)];
        }
    *rc = run(r4d_rad_gate_mul, { a.t(), g.t(), y.t() }, { pi("M", n), pi("n", W), ps("dtype", "bf16") });
    std::vector<uint16_t> out((size_t)(n * W));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < W; ++j) out[(size_t)(i * W + j)] = y.at(i, j);
    return out;
}

/* libref's mla_project_kv on the whole draw, for the bound */
static bool vs_libref(const Draw& d, bool absorb, const std::vector<uint16_t>& got, std::string* why) {
    const RadKernelInfo* ref = libref_op("mla_project_kv");
    if (!ref) { *why = "libref's mla_project_kv not loadable (set RADIANCE_HOME)"; return false; }
    const int64_t H = d.H, M = d.M, xw = absorb ? kKd : kL, yw = absorb ? kL : kVd;
    Mat x(M, H * xw, H * xw, 0), w(H * (kKd + kVd), kL, kL, 0), y(M, H * yw, H * yw, 0);
    std::memcpy(x.data(), d.x.data(), d.x.size() * 2);
    std::memcpy(w.data(), d.w.data(), d.w.size() * 2);
    if (run(ref->launch, { x.t(), w.t(), y.t() }, params(M, H, absorb ? "absorb" : "expand", 0, 0)) != RAD_OK) {
        *why = "libref refused";
        return false;
    }
    int64_t bad = 0;
    for (int64_t i = 0; i < M * H * yw; ++i) {
        const double e = b2f(y.data()[i]), g = b2f(got[(size_t)i]);
        if (!std::isfinite(g) || std::fabs(g - e) > 2e-5 + 0.0079 * std::fabs(e)) {
            if (!bad) {
                char b[160];
                std::snprintf(b, sizeof b, "element %lld got %.9g want %.9g", (long long)i, g, e);
                *why = b;
            }
            ++bad;
        }
    }
    return bad == 0;
}

/* ---- mla_expand_gate_had_quant_i8g, the quantiser the K-17 pad steps ---- */

/* A [rows, cols] matrix with `elem`-byte elements (the i8 codes, the f32 scales) and a guard
 * tail per row (cols..pitch stays 0x7e): a store into it, from the padded head or otherwise,
 * fails the run. The vector's own allocation is 16-byte aligned, every alignment the launcher
 * asks of these operands. */
struct QMat {
    int64_t rows = 0, cols = 0, pitch = 0;
    size_t elem = 1;
    std::vector<uint8_t> buf;
    QMat() = default;
    QMat(int64_t r, int64_t c, int64_t p, size_t e)
        : rows(r), cols(c), pitch(p), elem(e), buf((size_t)(r * p + 16) * e, 0x7e) {}
    uint8_t* data() { return buf.data(); }
    uint8_t& at(int64_t i, int64_t j) { return buf[(size_t)(i * pitch + j) * elem]; }
    void load_rows(const void* src) {  /* packed [rows, cols] into the strided matrix */
        const uint8_t* s = (const uint8_t*)src;
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j)
                std::memcpy(&at(i, j), s + (size_t)(i * cols + j) * elem, elem);
    }
    std::vector<uint8_t> packed() const {
        std::vector<uint8_t> out((size_t)rows * cols * elem);
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j)
                std::memcpy(&out[(size_t)(i * cols + j) * elem],
                            &buf[(size_t)(i * pitch + j) * elem], elem);
        return out;
    }
    bool tail_untouched() const {
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = cols; j < pitch; ++j)
                for (size_t e = 0; e < elem; ++e)
                    if (buf[(size_t)(i * pitch + j) * elem + e] != 0x7e) return false;
        return true;
    }
    RadTensor t(uint32_t dt) {
        RadTensor x{};
        x.data = data(); x.dtype = dt; x.rank = 2;
        x.shape[0] = rows; x.shape[1] = cols; x.stride[0] = pitch; x.stride[1] = 1;
        return x;
    }
};

struct QuantOut {
    std::vector<int8_t> q;     /* [n, H * kVd] codes  */
    std::vector<float> sc;     /* [n, H * kVd / 128] scales */
    std::vector<uint16_t> y;   /* [n, H * kVd], empty without the optional operand */
    int rc = 0;                /* the launcher's code, or -999 when a guard tail was written */
};

/* One run of the real launcher over rows [r0, r0 + n) of the draw. The quantiser takes aligned,
 * unpitched operands only (it refuses the skewed layout), so this is its one path. */
static QuantOut quant(const Draw& d, int64_t r0, int64_t n, int chain, bool with_y) {
    const int64_t H = d.H, W = H * kVd, GS = W / 128;
    QuantOut o;
    Mat x(n, H * kL, H * kL, 0), w(H * (kKd + kVd), kL, kL, 0), g(n, H * kVd, H * kVd, 0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < H * kL; ++j) x.at(i, j) = d.x[(size_t)((r0 + i) * H * kL + j)];
    for (int64_t i = 0; i < H * (kKd + kVd); ++i)
        for (int64_t j = 0; j < kL; ++j) w.at(i, j) = d.w[(size_t)(i * kL + j)];
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < H * kVd; ++j) g.at(i, j) = d.g[(size_t)((r0 + i) * H * kVd + j)];
    QMat q(n, W, W + 8, 1), sc(n, GS, GS + 4, 4), y(with_y ? n : 1, W, W + 8, 2);
    std::vector<RadTensor> t = { x.t(), w.t(), g.t(), q.t(RAD_I8), sc.t(RAD_F32) };
    if (with_y) t.push_back(y.t(RAD_BF16));
    o.rc = run(r4d_rad_mla_expand_gate_had_quant_i8g, t,
               { pi("M", n), pi("heads", H), pi("k_dim", kKd), pi("v_dim", kVd), pi("latent_dim", kL),
                 pi("group", 128), ps("dtype", "bf16"), pi("chain", chain) });
    const auto qb = q.packed();
    o.q.assign(qb.begin(), qb.end());
    const auto sb = sc.packed();
    o.sc.resize((size_t)n * GS);
    std::memcpy(o.sc.data(), sb.data(), sb.size());
    if (with_y) {
        const auto yb = y.packed();
        o.y.resize((size_t)n * W);
        std::memcpy(o.y.data(), yb.data(), yb.size());
        if (!y.tail_untouched()) o.rc = -999;
    }
    if (!q.tail_untouched() || !sc.tail_untouched()) o.rc = -999;
    return o;
}

/* had_quant_act_i8g over a packed [n, H * kVd] bf16 product, through the emulated libr4d row: the
 * second half of the unfused chain -- mla_expand_gate's gated product, then the standalone
 * quantiser the fusion absorbs -- which the fused launch must equal bit for bit. */
static QuantOut quant_chain(const Draw& d, int64_t n, const std::vector<uint16_t>& prod) {
    const int64_t W = d.H * kVd, GS = W / 128;
    QuantOut o;
    QMat x(n, W, W, 2), q(n, W, W + 8, 1), sc(n, GS, GS + 4, 4);
    x.load_rows(prod.data());
    o.rc = run(r4d_rad_had_quant_act_i8g, { x.t(RAD_BF16), q.t(RAD_I8), sc.t(RAD_F32) },
               { pi("M", n), pi("n", W), pi("group", 128), ps("dtype", "bf16") });
    const auto qb = q.packed();
    o.q.assign(qb.begin(), qb.end());
    const auto sb = sc.packed();
    o.sc.resize((size_t)n * GS);
    std::memcpy(o.sc.data(), sb.data(), sb.size());
    if (!q.tail_untouched() || !sc.tail_untouched()) o.rc = -999;
    return o;
}

static bool quant_equal(const QuantOut& a, const QuantOut& b) {
    return a.q == b.q &&
           std::memcmp(a.sc.data(), b.sc.data(), a.sc.size() * sizeof(float)) == 0 &&
           (a.y.empty() || a.y == b.y);
}

/* libref's mla_expand_gate_had_quant_i8g on the whole draw: the codes next to the reference's
 * (its WMMA order differs, and a product sitting on a rounding boundary can code one step
 * apart), its scales and y inside the bound vs_libref holds. */
static bool quant_vs_libref(const Draw& d, const QuantOut& got, std::string* why) {
    const RadKernelInfo* ref = libref_op("mla_expand_gate_had_quant_i8g");
    if (!ref) { *why = "libref's mla_expand_gate_had_quant_i8g not loadable (set RADIANCE_HOME)"; return false; }
    const int64_t H = d.H, M = d.M, W = H * kVd, GS = W / 128;
    Mat x(M, H * kL, H * kL, 0), w(H * (kKd + kVd), kL, kL, 0), g(M, H * kVd, H * kVd, 0), y(M, W, W, 0);
    std::memcpy(x.data(), d.x.data(), d.x.size() * 2);
    std::memcpy(w.data(), d.w.data(), d.w.size() * 2);
    std::memcpy(g.data(), d.g.data(), d.g.size() * 2);
    QMat rq(M, W, W, 1), rs(M, GS, GS, 4);
    if (run(ref->launch, { x.t(), w.t(), g.t(), rq.t(RAD_I8), rs.t(RAD_F32), y.t() },
            { pi("M", M), pi("heads", H), pi("k_dim", kKd), pi("v_dim", kVd), pi("latent_dim", kL),
              pi("group", 128), ps("dtype", "bf16"), pi("chain", 0) }) != RAD_OK) {
        *why = "libref refused";
        return false;
    }
    const auto rc = rq.packed(), rsc = rs.packed();
    int64_t bad = 0;
    char b[200];
    for (int64_t i = 0; i < M * W; ++i) {
        const int dq = std::abs((int)got.q[(size_t)i] - (int)(int8_t)rc[(size_t)i]);
        if (dq > 1) {
            if (!bad) std::snprintf(b, sizeof b, "code %lld got %d want %d", (long long)i,
                                    (int)got.q[(size_t)i], (int)(int8_t)rc[(size_t)i]);
            ++bad;
        }
    }
    for (int64_t i = 0; i < M * GS; ++i) {
        float e = 0.f;
        std::memcpy(&e, &rsc[(size_t)i * 4], sizeof e);
        const float g = got.sc[(size_t)i];
        if (!std::isfinite(g) || std::fabs(g - e) > 2e-5f + 0.0079f * std::fabs(e)) {
            if (!bad) std::snprintf(b, sizeof b, "scale %lld got %.9g want %.9g", (long long)i, g, e);
            ++bad;
        }
    }
    for (int64_t i = 0; i < M * W; ++i) {
        const double e = b2f(y.data()[i]), g = b2f(got.y[(size_t)i]);
        if (!std::isfinite(g) || std::fabs(g - e) > 2e-5 + 0.0079 * std::fabs(e)) {
            if (!bad) std::snprintf(b, sizeof b, "y %lld got %.9g want %.9g", (long long)i, g, e);
            ++bad;
        }
    }
    if (bad) { *why = std::string(b) + " (+" + std::to_string(bad - 1) + " more)"; return false; }
    return true;
}

int main() {
    struct G { int64_t H, M; };
    for (const bool absorb : { true, false })
        for (const G& g : { G{ 1, 1 }, G{ 2, 17 }, G{ 2, 37 }, G{ 1, 70 }, G{ 2, 64 }, G{ 1, 133 } }) {
            const char* mode = absorb ? "absorb" : "expand";
            const Draw d = draw(g.H, g.M, absorb, (uint32_t)(g.H * 100 + g.M + (absorb ? 0 : 7)));
            const int64_t yw = (absorb ? kL : kVd) * g.H;
            int rc = 0;
            /* the reference bytes: every token alone, aligned, plain, the law's mb for one row */
            std::vector<uint16_t> alone((size_t)(g.M * yw));
            bool rc_ok = true;
            for (int64_t t = 0; t < g.M; ++t) {
                const auto r = project(d, absorb, false, t, 1, 0, 0, false, &rc);
                rc_ok &= rc == RAD_OK;
                std::memcpy(&alone[(size_t)(t * yw)], r.data(), r.size() * 2);
            }
            char w[200];
            std::snprintf(w, sizeof w, "%s heads %lld: every token alone launches", mode, (long long)g.H);
            ok(rc_ok, w);
            std::string why;
            std::snprintf(w, sizeof w, "%s heads %lld M %lld within one bf16 ulp of libref", mode,
                          (long long)g.H, (long long)g.M);
            ok(vs_libref(d, absorb, alone, &why), std::string(w) + (why.empty() ? "" : ": " + why));
            for (const bool skew : { false, true })
                for (const int chain : { 0, 1 })
                    for (const int mb : { 0, 1, 2, 4, 8 }) {
                        if (skew && (chain || mb > 1)) continue;   /* one form off the grid */
                        const auto all = project(d, absorb, false, 0, g.M, mb, chain, skew, &rc);
                        std::snprintf(w, sizeof w,
                                      "%s heads %lld M %lld mb %d%s%s: every row's bytes equal the row alone",
                                      mode, (long long)g.H, (long long)g.M, mb, chain ? " chained" : "",
                                      skew ? " off the 16-byte grid" : "");
                        ok(rc == RAD_OK && all == alone, w);
                        if (absorb) continue;
                        /* the fused gate against the chain it replaces, on these bytes */
                        const auto fused = project(d, false, true, 0, g.M, mb, chain, skew, &rc);
                        int rc2 = 0;
                        const auto chained = gate_chain(d, 0, g.M, alone, &rc2);
                        std::snprintf(w, sizeof w,
                                      "mla_expand_gate heads %lld M %lld mb %d%s%s == expand + gate_mul, bit for bit",
                                      (long long)g.H, (long long)g.M, mb, chain ? " chained" : "",
                                      skew ? " off the 16-byte grid" : "");
                        ok(rc == RAD_OK && rc2 == RAD_OK && fused == chained, w);
                    }
            /* K-50: the staged expansion (auto at four token tiles, aligned, unchained) and kvp_body's
             * (stage 1) at the prefill heights, expand with and without the gate: each row's bytes
             * are the row alone's, so the two arms are byte-equal to each other. Which kernel ran is
             * asserted, or a broken route would compare kvp_body with itself. */
            if (!absorb)
                for (const int mb : { 0, 4 })
                    for (const bool gate : { false, true }) {
                        const bool staged = mb == 4 || g.M > 16;   /* kvp_mb: four token tiles */
                        const auto on = project(d, false, gate, 0, g.M, mb, 0, false, &rc, 0);
                        const int rc_on = rc;
                        const bool on_staged = std::strstr(emu_last_kernel, "kv_expand_staged") != nullptr;
                        const auto off = project(d, false, gate, 0, g.M, mb, 0, false, &rc, 1);
                        const bool off_staged = std::strstr(emu_last_kernel, "kv_expand_staged") != nullptr;
                        std::snprintf(w, sizeof w,
                                      "expand%s heads %lld M %lld mb %d: stage 0 ran %s, stage 1 kvp_body's",
                                      gate ? " + gate" : "", (long long)g.H, (long long)g.M, mb,
                                      staged ? "kv_expand_staged" : "kvp_body's");
                        ok(on_staged == staged && !off_staged, w);
                        /* every pitch (x, w, g, y) 8 elements wider than its width, still on the grid */
                        int rc_w = 0;
                        const auto wide = project(d, false, gate, 0, g.M, mb, 0, false, &rc_w, 0, 8);
                        std::snprintf(w, sizeof w,
                                      "expand%s heads %lld M %lld mb %d: staged at pitches 8 past the widths "
                                      "== unstaged, bit for bit",
                                      gate ? " + gate" : "", (long long)g.H, (long long)g.M, mb);
                        ok(rc_w == RAD_OK && rc == RAD_OK && wide == off &&
                               (std::strstr(emu_last_kernel, "kv_expand_staged") != nullptr) == staged, w);
                        std::snprintf(w, sizeof w,
                                      "expand%s heads %lld M %lld mb %d: staged == unstaged, bit for bit",
                                      gate ? " + gate" : "", (long long)g.H, (long long)g.M, mb);
                        ok(rc_on == RAD_OK && rc == RAD_OK && on == off, w);
                        if (gate) continue;
                        std::snprintf(w, sizeof w, "expand heads %lld M %lld mb %d: staged == the row alone",
                                      (long long)g.H, (long long)g.M, mb);
                        ok(on == alone, w);
                    }
            /* a slice that starts mid-matrix: rows [5, M) as their own launch */
            if (g.M > 5) {
                const auto tail = project(d, absorb, false, 5, g.M - 5, 0, 0, false, &rc);
                std::snprintf(w, sizeof w, "%s heads %lld rows 5..%lld as a launch of their own: the same bytes",
                              mode, (long long)g.H, (long long)g.M);
                ok(rc == RAD_OK && std::equal(tail.begin(), tail.end(), alone.begin() + 5 * yw), w);
            }
        }
    /* ---- the quantiser the K-17 pad stepped: 32 heads, a grid that steps to 33 ---- */
    for (const int64_t M : { 1LL, 16LL, 32LL }) {
        const Draw d = draw(32, M, false, (uint32_t)(1300 + M));
        char wq[200];
        const QuantOut all = quant(d, 0, M, 0, true);
        std::snprintf(wq, sizeof wq, "quant heads 32 M %lld: the launch over all rows runs clean",
                      (long long)M);
        ok(all.rc == RAD_OK && !all.y.empty(), wq);
        /* every token's bytes are its own: alone, and inside the M-row launch, bit for bit (a
         * padded head that wrote anywhere, or a body that read its index, breaks this) */
        bool law = all.rc == RAD_OK;
        for (int64_t t = 0; law && t < std::min<int64_t>(M, 2); ++t) {
            const QuantOut one = quant(d, t, 1, 0, true);
            law &= one.rc == RAD_OK &&
                   std::equal(one.q.begin(), one.q.end(), all.q.begin() + t * 32 * kVd) &&
                   std::memcmp(one.sc.data(), all.sc.data() + t * 64, 64 * sizeof(float)) == 0 &&
                   std::equal(one.y.begin(), one.y.end(), all.y.begin() + t * 32 * kVd);
        }
        /* and a slice that starts mid-draw, as its own launch */
        if (M > 5) {
            const QuantOut tail = quant(d, 5, M - 5, 0, true);
            law &= tail.rc == RAD_OK &&
                   std::equal(tail.q.begin(), tail.q.end(), all.q.begin() + 5 * 32 * kVd) &&
                   std::memcmp(tail.sc.data(), all.sc.data() + 5 * 64, (size_t)(M - 5) * 64 * 4) == 0 &&
                   std::equal(tail.y.begin(), tail.y.end(), all.y.begin() + 5 * 32 * kVd);
        }
        std::snprintf(wq, sizeof wq,
                      "quant heads 32 M %lld: every row's bytes equal the row alone, bit for bit",
                      (long long)M);
        ok(law, wq);
        std::snprintf(wq, sizeof wq, "quant heads 32 M %lld: codes, scales and y next to libref",
                      (long long)M);
        std::string why;
        ok(all.rc == RAD_OK && quant_vs_libref(d, all, &why),
           std::string(wq) + (why.empty() ? "" : ": " + why));
        if (M == 1) {
            /* the chained wrapper, padded head included (the emulator carries no engine record,
             * so this drives the chained instantiation with the chain calls inert) */
            const QuantOut ch = quant(d, 0, M, 1, true);
            ok(ch.rc == RAD_OK && quant_equal(ch, all),
               "quant heads 32 M 1: the chained form's bytes equal the plain form's");
            /* and the optional output operand left out: the codes and scales are the same */
            const QuantOut ny = quant(d, 0, M, 0, false);
            ok(ny.rc == RAD_OK && ny.y.empty() && ny.q == all.q &&
               std::memcmp(ny.sc.data(), all.sc.data(), all.sc.size() * sizeof(float)) == 0,
               "quant heads 32 M 1: without the optional bf16 output the codes and scales stand");
        }
    }
    /* ---- K-50b: the prefill form of the fused quantiser, four token tiles a workgroup from 64
     * rows up. A prefill chunk above 64 rows used to fall back to mla_expand_gate followed by the
     * standalone quantiser, a 33.5 MB round trip a layer-step; the prefill form keeps it fused. Its
     * bytes are held to the row alone's -- the 16-row form, a row's bytes being the row's own --
     * and to that unfused chain, bit for bit. The draw adds 2 to the seed: the codes, scales and y
     * check's y bound is one bf16 step of the gated product, and at a row whose expansion sits on
     * libref's rounding boundary the product can land two steps off, which is the reference's
     * WMMA order and not the form (the bit-for-bit checks below do not depend on the draw). */
    for (const int64_t H : { 32LL, 2LL })
        for (const int64_t M : { 65LL, 70LL, 133LL }) {
            if (H == 32 && M != 65) continue;   /* 32 heads is the K-17 pad's geometry, and its time */
            const Draw d = draw(H, M, false, (uint32_t)(1500 + H * 1000 + M + 2));
            const int64_t W = H * kVd, GS = W / 128;
            char wq[240];
            const QuantOut all = quant(d, 0, M, 0, true);
            const bool prefill = std::strstr(emu_last_kernel, "kv_expand_gate_quant<true") != nullptr;
            const bool rows_only = std::strstr(emu_last_kernel, "kv_expand_gate_quant<false") != nullptr;
            std::snprintf(wq, sizeof wq,
                          "quant heads %lld M %lld: the prefill form (four token tiles a workgroup) ran",
                          (long long)H, (long long)M);
            ok(all.rc == RAD_OK && prefill && !rows_only, wq);
            /* across the two forms: a row alone (M 1, the 16-row form) is the row inside the
             * prefill launch, bit for bit -- the first tile's first, middle and last row, the
             * seam rows 16 and 64, and the last group's tail */
            bool law = all.rc == RAD_OK;
            for (const int64_t t : std::initializer_list<int64_t>{ 0, 7, 15, 16, 17, 47, 63, 64, M - 1 }) {
                if (t >= M) continue;
                const QuantOut one = quant(d, t, 1, 0, true);
                law &= one.rc == RAD_OK &&
                       std::equal(one.q.begin(), one.q.end(), all.q.begin() + t * W) &&
                       std::memcmp(one.sc.data(), all.sc.data() + t * GS, (size_t)GS * sizeof(float)) == 0 &&
                       std::equal(one.y.begin(), one.y.end(), all.y.begin() + t * W);
            }
            std::snprintf(wq, sizeof wq,
                          "quant heads %lld M %lld: every row's bytes equal the row alone's, both forms",
                          (long long)H, (long long)M);
            ok(law, wq);
            /* the item's verify: what the cap used to force, the gated product through the
             * standalone quantiser, is the fused launch's product, codes and scales bit for bit */
            int rc_e = 0;
            const auto prod = project(d, false, true, 0, M, 0, 0, false, &rc_e);
            const QuantOut ch = quant_chain(d, M, prod);
            std::snprintf(wq, sizeof wq,
                          "quant heads %lld M %lld == mla_expand_gate then had_quant_act_i8g, bit for bit",
                          (long long)H, (long long)M);
            ok(rc_e == RAD_OK && all.rc == RAD_OK && ch.rc == RAD_OK &&
                   all.y == prod && all.q == ch.q &&
                   std::memcmp(all.sc.data(), ch.sc.data(), ch.sc.size() * sizeof(float)) == 0,
               wq);
            std::snprintf(wq, sizeof wq, "quant heads %lld M %lld: codes, scales and y next to libref",
                          (long long)H, (long long)M);
            std::string why;
            ok(all.rc == RAD_OK && quant_vs_libref(d, all, &why),
               std::string(wq) + (why.empty() ? "" : ": " + why));
            /* a slice across the tile seams, rows 5..M as a launch of their own */
            const QuantOut tail = quant(d, 5, M - 5, 0, true);
            std::snprintf(wq, sizeof wq,
                          "quant heads %lld M %lld: rows 5..%lld as a launch of their own are the same bytes",
                          (long long)H, (long long)M, (long long)M);
            ok(tail.rc == RAD_OK &&
                   std::equal(tail.q.begin(), tail.q.end(), all.q.begin() + 5 * W) &&
                   std::memcmp(tail.sc.data(), all.sc.data() + 5 * GS,
                               (size_t)(M - 5) * GS * sizeof(float)) == 0 &&
                   std::equal(tail.y.begin(), tail.y.end(), all.y.begin() + 5 * W),
               wq);
        }
    /* The launcher's own grid math, called as the launcher calls it (per_row = (v_dim / group) *
     * ceil(M / 16) = 2 zt, waves = 1024 / 32, or ceil(M / 64) from 64 rows up, K-50b): the step at
     * 32 heads fires at every one of the multiples of 2048 that the token-tile count puts the grid
     * on, and never on the counts clear of them. The M 1 / 16 / 32 rows of the item's verify are
     * the zt 1 and 2 lines. */
    for (const int zt : { 1, 2, 3, 4 }) {
        char wg[160];
        std::snprintf(wg, sizeof wg,
                      "cliff rows: 32 heads x %d zt x 32 waves (a %d-wave multiple) steps to 33",
                      zt, 2048 * zt);
        ok(r4d_cliff_rows(32, 2 * zt, 32) == 33, wg);
    }
    ok(r4d_cliff_rows(8, 4, 32) == 8, "cliff rows: 8 heads x 2 zt x 32 waves (1024, off the window) stays");
    std::printf("%d case(s), %d failure(s)\n", g_cases, g_fails);
    return g_fails ? 1 : 0;
}
