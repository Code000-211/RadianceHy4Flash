/* row_topk and cast_row_topk (r4d_rowtopk_bf16.hip): the device kernels (emulated) against each
 * other and a host top-R, on rows built to break them: ties inside and across chunks, NaN of every
 * kind, +-inf, -0.0, values on and next to a bf16 rounding midpoint, and whole rows of one value.
 *
 *   - cast_row_topk over an f32 plane is held to row_topk over the plane `cast` f32 -> bf16 writes
 *     (rounded here with the cast kernel's own f32_to_bf16, r4d_common.h), byte for byte: idx, val
 *     and the pair plane, at the draft head's shapes (rows 1, 5, 8, 40; a 120832 vocabulary's slice
 *     at TP1, TP2 and TP4) and at chunk-boundary widths.
 *   - Both are held to a host top-R where a lane's list cannot overflow (R <= PT1, see the kernel's
 *     header); above that the kernel may keep fewer than R of a chunk, by design, and only the two
 *     kernels' agreement is required.
 *   - The launch form is checked: one launch (stage 2 in the last wave) at R <= 16, two above. A
 *     run of launches on one output at varying heights checks that the arrival counters are left
 *     zero.
 *   - The rounding itself over all 2^32 f32 patterns: f32_to_bf16 against the round-to-nearest-even
 *     formula the cast documents, and the count of patterns (NaN only) where libref's cast, which
 *     quiets a NaN, differs.
 *
 * R4D_EMU_TOPK_DUMP=<file> writes a hash of each case's outputs, one line a case; built with
 * -DTOPK_BASE the test runs row_topk alone, so a base build's dump can be compared line for line.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d_common.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern const char* emu_last_kernel;

namespace {

uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

/* The cast's rounding, as narrow_bf16_kernel does it (r4d_common.h). */
uint16_t cast_bf16(float f) { return f32_to_bf16(f); }

/* A value the generator draws: mostly ordinary, a share at a rounding midpoint or one ulp off it,
 * and the specials. */
float draw(std::mt19937& g, int mode) {
    const uint32_t r = g();
    switch (mode) {
        case 0: return std::ldexp((float)(int)(r % 4001) - 2000.f, -6);           /* many ties */
        case 1: {                                                                    /* midpoints */
            const uint32_t hi = (0x3f00u + (r % 512u)) << 16 | ((r >> 9) & 1u ? 0x80000000u : 0u);
            const uint32_t lo = (r >> 10) % 3 == 0 ? 0x8000u : (r >> 10) % 3 == 1 ? 0x7fffu : 0x8001u;
            return flt(hi | lo);
        }
        case 2: {                                                                    /* specials */
            static const uint32_t sp[] = { 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u,
                                           0x7f800001u, 0xff800001u, 0x7fffffffu, 0xffffffffu,
                                           0x7f80ffffu, 0x7f7fffffu, 0xff7fffffu, 0x00000000u,
                                           0x80000000u, 0x00000001u, 0x80000001u, 0x7f7f8000u };
            return flt(sp[r % 16]);
        }
        default: return flt(r);                                                      /* any bits */
    }
}

/* A row: a background of one kind and, sometimes, a cluster of large values packed into one lane's
 * slice of one chunk or spread at its stride, where a short list could lose one. */
void fill_row(std::mt19937& g, float* x, int N, int kind) {
    for (int j = 0; j < N; ++j) {
        const uint32_t r = g();
        int mode = kind;
        if (kind == 4) mode = r % 100 < 85 ? 3 : r % 100 < 95 ? 1 : 2;   /* mixed */
        x[j] = draw(g, mode);
    }
    if (kind == 5) for (int j = 0; j < N; ++j) x[j] = 1.0f;              /* one value: all ties */
    if (kind == 6) for (int j = 0; j < N; ++j) x[j] = j % 2 ? -0.0f : 0.0f;
    if (g() % 3 == 0 && N >= 512) {
        const int c0 = (int)(g() % (unsigned)(N / 8)) * 8, lane = (int)(g() % 32u);
        for (int q = 0; q < 24; ++q) {                                   /* one lane's slice */
            const int j = c0 + lane * 8 + (q / 8) * 256 + (q % 8);
            if (j < N) x[j] = 1000.0f + (float)(q % 3);
        }
    }
}

struct Host { std::vector<int32_t> idx; std::vector<float> val; std::vector<float> pairs; };

/* The exact top-R of the rounded row: descending key (the kernel's order-preserving image of the
 * bf16, so NaN and -0.0 rank as the kernel ranks them), ties to the lower column. */
void host_topk(const std::vector<uint16_t>& xb, int M, int N, int R, int off, Host* h) {
    h->idx.assign((size_t)M * R, 0); h->val.assign((size_t)M * R, 0.f);
    h->pairs.assign((size_t)M * R * 2, 0.f);
    std::vector<unsigned long long> keys((size_t)N);
    for (int m = 0; m < M; ++m) {
        for (int j = 0; j < N; ++j) {
            uint32_t u = (uint32_t)xb[(size_t)m * N + j] << 16;
            u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            keys[(size_t)j] = ((unsigned long long)u << 32) | (0xFFFFFFFFu - (unsigned)j);
        }
        std::partial_sort(keys.begin(), keys.begin() + std::min(R, N), keys.end(),
                          [](unsigned long long a, unsigned long long b) { return a > b; });
        for (int r = 0; r < R; ++r) {
            const unsigned long long w = r < N ? keys[(size_t)r] : 0ull;
            uint32_t u = (uint32_t)(w >> 32);
            u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
            const float v = w ? flt(u) : -INFINITY;
            const int id = (int)(0xFFFFFFFFu - (unsigned)(w & 0xFFFFFFFFull));
            h->val[(size_t)m * R + r] = v;
            h->idx[(size_t)m * R + r] = w ? id + off : 0;
            const int pid = w ? id + off : -1;
            std::memcpy(&h->pairs[((size_t)m * R + r) * 2], &pid, 4);
            h->pairs[((size_t)m * R + r) * 2 + 1] = v;
        }
    }
}

RadTensor ten(void* d, uint32_t dt, std::initializer_list<int64_t> sh) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = (int)sh.size();
    int i = 0;
    for (int64_t s : sh) t.shape[i++] = s;
    int64_t st = 1;
    for (int k = t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
    return t;
}

struct Out { std::vector<int32_t> idx; std::vector<float> val, pairs; std::string kern; int rc = 0; };

/* One launch of `fn` over x (bf16 or f32), with the pair plane when `off` >= 0 (the sharded form). */
int run(int (*fn)(const RadArgs*, RadStream), void* x, uint32_t xdt, int M, int N, int R, int pt1,
        int off, Out* o, std::vector<unsigned char>& scratch, int32_t* idx_buf = nullptr,
        float* val_buf = nullptr) {
    o->idx.assign((size_t)M * R, 12345); o->val.assign((size_t)M * R, 777.f);
    o->pairs.assign((size_t)M * R * 2, 555.f);
    int32_t* ib = idx_buf ? idx_buf : o->idx.data();
    float* vb = val_buf ? val_buf : o->val.data();
    RadTensor t[4] = { ten(x, xdt, { M, N }), ten(ib, RAD_I32, { M, R }), ten(vb, RAD_F32, { M, R }),
                       ten(o->pairs.data(), RAD_F32, { M, R, 2 }) };
    RadParam p[7]{};
    int np = 0;
    p[np++] = { "M", RAD_P_INT, M };
    p[np++] = { "N", RAD_P_INT, N };
    p[np++] = { "R", RAD_P_INT, R };
    p[np].key = "dtype"; p[np].kind = RAD_P_STR; p[np++].sval = xdt == RAD_F32 ? "f32" : "bf16";
    p[np++] = { "pt1", RAD_P_INT, pt1 };
    if (off >= 0) p[np++] = { "vocab_off", RAD_P_INT, off };
    RadArgs a{};
    a.t = t; a.n_t = off >= 0 ? 4 : 3; a.p = p; a.n_p = np;
    const int64_t need = r4d_rad_rowtopk_scratch(&a);
    if (need <= 0) return o->rc = -100;
    scratch.assign((size_t)need, 0xA5);
    a.scratch = scratch.data(); a.scratch_bytes = need;
    o->rc = fn(&a, nullptr);
    o->kern = emu_last_kernel ? emu_last_kernel : "";
    if (idx_buf) std::memcpy(o->idx.data(), idx_buf, (size_t)M * R * 4);
    if (val_buf) std::memcpy(o->val.data(), val_buf, (size_t)M * R * 4);
    return o->rc;
}

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
uint64_t hash(const Out& o, bool pairs) {
    uint64_t h = fnv(o.idx.data(), o.idx.size() * 4);
    h = fnv(o.val.data(), o.val.size() * 4, h);
    return pairs ? fnv(o.pairs.data(), o.pairs.size() * 4, h) : h;
}
bool same(const Out& a, const Out& b, bool pairs) {
    return a.rc == 0 && b.rc == 0 && a.idx == b.idx &&
           std::memcmp(a.val.data(), b.val.data(), a.val.size() * 4) == 0 &&
           (!pairs || std::memcmp(a.pairs.data(), b.pairs.data(), a.pairs.size() * 4) == 0);
}
bool same_host(const Out& a, const Host& h, bool pairs) {
    return a.rc == 0 && a.idx == h.idx &&
           std::memcmp(a.val.data(), h.val.data(), a.val.size() * 4) == 0 &&
           (!pairs || std::memcmp(a.pairs.data(), h.pairs.data(), a.pairs.size() * 4) == 0);
}

}  // namespace

int main(int argc, char** argv) {
    const char* dump_path = std::getenv("R4D_EMU_TOPK_DUMP");
    FILE* dump = dump_path ? std::fopen(dump_path, "w") : nullptr;
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    int cases = 0, fails = 0, host_skip = 0;
    std::vector<unsigned char> scratch;

    /* ---- the rounding over every f32 pattern */
    {
        uint64_t bad = 0, nan_lib = 0, other_lib = 0;
        for (uint64_t i = 0; i <= 0xFFFFFFFFull; ++i) {
            const uint32_t u = (uint32_t)i;
            const uint16_t r = cast_bf16(flt(u));
            if (r != (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16)) ++bad;
            if (r != rad_f32_to_bf16(flt(u))) {
                if ((u & 0x7fffffffu) > 0x7f800000u) ++nan_lib; else ++other_lib;
            }
        }
        ++cases;
        if (bad || other_lib) ++fails;
        std::printf("rounding: 2^32 patterns, %llu off the cast's formula; libref's cast differs on "
                    "%llu NaN and %llu other patterns\n", (unsigned long long)bad,
                    (unsigned long long)nan_lib, (unsigned long long)other_lib);
    }

    /* ---- the grid */
    std::mt19937 g(83);
    const int V = 120832;
    std::vector<int> widths = { V, V / 2, V / 4, 8, 16, 2040, 2048, 2056, 4104, 6144 };
    std::vector<int> rows_set = { 1, 5, 8, 40 };
    if (quick) { widths = { V / 4, 2056 }; rows_set = { 1, 5 }; }
    for (int N : widths) for (int M : rows_set) for (int R : { 8, 16, 32 }) {
        if (R > N) continue;
        const bool big = N >= V / 4;
        /* The draft path's R (8) at every row count; 16 and 32 at the tall rows of the full slices
         * only, to bound the emulator's time. */
        if (big && R != 8 && M != 5) continue;
        for (int kind : { 4, 0, 1, 2, 3, 5, 6 }) {
            if (big && kind != 4 && kind != 1 && !(kind == 2 && M <= 5)) continue;
            for (int pt1 : { 8, 2, 4, 16 }) {
                if (pt1 != 8 && (big || kind != 4)) continue;
                if (R > pt1 * 32) continue;
                for (int sharded = 0; sharded < 2; ++sharded) {
                    if (big && sharded != (N == V ? 0 : 1)) continue;
                    const int off = sharded ? (N == V / 4 ? 3 * (V / 4) : N) : -1;
                    std::vector<float> x((size_t)M * N);
                    for (int m = 0; m < M; ++m) fill_row(g, x.data() + (size_t)m * N, N, kind);
                    std::vector<uint16_t> xb(x.size());
                    for (size_t i = 0; i < x.size(); ++i) xb[i] = cast_bf16(x[i]);

                    Out a, b;
                    run(r4d_rad_rowtopk, xb.data(), RAD_BF16, M, N, R, pt1, off, &a, scratch);
                    const bool want_fused = R <= 16;
                    bool ok = a.rc == 0;
#ifndef TOPK_BASE
                    ok = ok && (a.kern.find("fused") != std::string::npos) == want_fused;
                    run(r4d_rad_cast_rowtopk, x.data(), RAD_F32, M, N, R, pt1, off, &b, scratch);
                    ok = ok && b.rc == 0 && (b.kern.find("fused") != std::string::npos) == want_fused;
                    ok = ok && same(a, b, sharded);
#endif
                    Host h;
                    host_topk(xb, M, N, R, sharded ? off : 0, &h);
                    if (R <= pt1) ok = ok && same_host(a, h, sharded);
                    else if (!same_host(a, h, sharded)) ++host_skip;
                    ++cases;
                    if (!ok) {
                        ++fails;
                        std::printf("FAIL N %d M %d R %d kind %d pt1 %d sharded %d (rc %d/%d, %s / %s)\n",
                                    N, M, R, kind, pt1, sharded, a.rc, b.rc, a.kern.c_str(),
                                    b.kern.c_str());
                    }
                    if (dump) {
                        std::fprintf(dump, "N %d M %d R %d kind %d pt1 %d sharded %d row_topk %016llx",
                                     N, M, R, kind, pt1, sharded,
                                     (unsigned long long)hash(a, sharded));
#ifndef TOPK_BASE
                        std::fprintf(dump, " cast_row_topk %016llx",
                                     (unsigned long long)hash(b, sharded));
#endif
                        std::fprintf(dump, "\n");
                    }
                }
            }
        }
    }

#ifndef TOPK_BASE
    /* ---- one output, many launches at varying heights: the counters must be left zero, or a
     * later launch's last wave would be the wrong one (a stale row, or none at all). */
    {
        const int N = 2056, R = 8, Mmax = 40;
        std::vector<int32_t> ib((size_t)Mmax * R);
        std::vector<float> vb((size_t)Mmax * R);
        int bad = 0;
        for (int it = 0; it < 12; ++it) {
            const int M = 1 + (it * 7) % Mmax;
            std::vector<float> x((size_t)M * N);
            for (int m = 0; m < M; ++m) fill_row(g, x.data() + (size_t)m * N, N, 4);
            std::vector<uint16_t> xb(x.size());
            for (size_t i = 0; i < x.size(); ++i) xb[i] = cast_bf16(x[i]);
            Out a, b;
            run(r4d_rad_cast_rowtopk, x.data(), RAD_F32, M, N, R, 8, -1, &b, scratch, ib.data(),
                vb.data());
            Host h;
            host_topk(xb, M, N, R, 0, &h);
            if (!same_host(b, h, false) || b.kern.find("fused") == std::string::npos) ++bad;
            run(r4d_rad_rowtopk, xb.data(), RAD_BF16, M, N, R, 8, -1, &a, scratch, ib.data(),
                vb.data());
            if (!same(a, b, false)) ++bad;
        }
        ++cases;
        if (bad) { ++fails; std::printf("FAIL repeated launches on one output: %d bad\n", bad); }
    }
    /* ---- the refusals: an f32 op handed a bf16 plane */
    {
        std::vector<uint16_t> xb(64 * 8, 0);
        Out o;
        run(r4d_rad_cast_rowtopk, xb.data(), RAD_BF16, 8, 64, 8, 8, -1, &o, scratch);
        ++cases;
        if (o.rc == 0) { ++fails; std::printf("FAIL cast_row_topk took a bf16 plane\n"); }
    }
#endif

    if (dump) std::fclose(dump);
    std::printf("topk: %d case(s), %d failure(s); %d case(s) above PT1 differ from the exact host "
                "top-R (allowed, both kernels agree)\n", cases, fails, host_skip);
    return fails != 0;
}
