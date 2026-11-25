/* tests/r4d_emu: gemm_nt_f32 (r4d_glm.hip) on the host SIMT emulator. The form the launcher picks
 * at each shape (a tile of rows x columns a wave with its steps of k in flight, by M and N; the
 * shapes reach every one, with K past and short of whole groups of steps) must match the
 * arithmetic written out here bit for bit: lane l folds k = l, l+32, ... with fmaf, then the xor
 * butterfly over 16, 8, 4, 2, 1. Tail rows and columns and the pitch's padding stay untouched.
 * From 768 rows at N 128-255, 384 at N 256-511 and 256 at N 512 or more the staged form (x widened
 * into LDS, 8 x 8 a wave) takes over; its
 * shapes cover a K short of a whole stage and of a whole step, tail rows and columns, waves with
 * no columns, and x's pitch with and without 16-byte vectors.
 *
 * R4D_EMU_F32_DUMP=<file> runs a grid instead (M 1..4096, N 128/256/512, K 3072, f32 and bf16
 * weights; R4D_EMU_F32_DUMP_M=<m> one M of it) and appends a line a launch with an FNV-1a hash of
 * its output, so the kernel before and after a change can be compared byte for byte.
 *
 * router_sigmoid_topk from the same unit: the wave-per-row select must match a serial scan written
 * out here bit for bit, on random and on tied ranks, with the outputs' pitch padding untouched. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern const char* emu_last_kernel;

static float gemm_model(const uint16_t* x, const float* w, int k) {
    float lane[32];
    for (int l = 0; l < 32; ++l) {
        float s = 0;
        for (int j = l; j < k; j += 32) s = std::fmaf(b2f(x[j]), w[j], s);
        lane[l] = s;
    }
    for (int mask = 16; mask; mask >>= 1) {
        float next[32];
        for (int l = 0; l < 32; ++l) next[l] = lane[l] + lane[l ^ mask];
        std::memcpy(lane, next, sizeof lane);
    }
    return lane[0];
}

static void router_model(const float* lg, const float* bias, int bp, int ne, int k, int norm,
                         float factor, int* ids, float* w) {
    float prob[512], rank[512], chosen[32], total = 0;
    int index[32];
    for (int e = 0; e < ne; ++e) {
        prob[e] = 1.0f / (1.0f + expf(-lg[e]));
        rank[e] = prob[e] + bias[(size_t)e * bp];
    }
    for (int j = 0; j < k; ++j) {
        int best = 0;
        for (int e = 1; e < ne; ++e)
            if (rank[e] > rank[best]) best = e;
        index[j] = best;
        chosen[j] = prob[best];
        total += chosen[j];
        rank[best] = -INFINITY;
    }
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    for (int j = 0; j < k; ++j) {
        ids[j] = index[j];
        w[j] = chosen[j] * mult;
    }
}

static void router_cases(std::mt19937& rng, int& cases, int& fails) {
    struct Shape {
        int M, ne, k, norm;
        bool ties;
    };
    const Shape shapes[] = { { 1, 288, 8, 1, false }, { 6, 288, 8, 1, false },
                             { 20, 288, 8, 1, true }, { 48, 288, 8, 1, false },
                             { 6, 288, 8, 1, true },  { 5, 8, 8, 0, true },
                             { 5, 32, 32, 0, true },  { 7, 300, 8, 1, false },
                             { 4, 512, 16, 1, false }, { 5, 512, 16, 1, true },
                             { 3, 512, 32, 1, true } };
    std::uniform_int_distribution<int> ud(-1000, 1000);
    for (const Shape& sh : shapes) {
        const int M = sh.M, ne = sh.ne, k = sh.k, lp = ne + 5, bp = 2, ip = k + 3, wp = k + 1;
        const float factor = 2.5f;
        std::vector<float> lg((size_t)M * lp), bias((size_t)ne * bp);
        for (auto& v : lg) v = sh.ties ? (float)(ud(rng) % 8) * 0.5f : (float)ud(rng) / 200;
        for (auto& v : bias) v = sh.ties ? (float)(ud(rng) % 5) * 0.125f : (float)ud(rng) / 2000;
        RadTensor t[4]{};
        t[0].data = lg.data();
        t[0].dtype = RAD_F32;
        t[0].rank = 2;
        t[0].shape[0] = M;
        t[0].shape[1] = ne;
        t[0].stride[0] = lp;
        t[0].stride[1] = 1;
        t[1].data = bias.data();
        t[1].dtype = RAD_F32;
        t[1].rank = 1;
        t[1].shape[0] = ne;
        t[1].stride[0] = bp;
        t[2].dtype = RAD_I32;
        t[2].rank = 2;
        t[2].shape[0] = M;
        t[2].shape[1] = k;
        t[2].stride[0] = ip;
        t[2].stride[1] = 1;
        t[3].dtype = RAD_F32;
        t[3].rank = 2;
        t[3].shape[0] = M;
        t[3].shape[1] = k;
        t[3].stride[0] = wp;
        t[3].stride[1] = 1;
        RadParam p[6]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "n_expert", RAD_P_INT, ne };
        p[2] = { "top_k", RAD_P_INT, k };
        p[3] = { "norm", RAD_P_INT, sh.norm };
        p[4].key = "route_scale";
        p[4].kind = RAD_P_F64;
        p[4].dval = factor;
        p[5].key = "dtype";
        p[5].kind = RAD_P_STR;
        p[5].sval = "f32";
        RadArgs a{};
        a.t = t;
        a.n_t = 4;
        a.p = p;
        a.n_p = 6;
        std::vector<int> ids((size_t)M * ip, -77);
        std::vector<float> w((size_t)M * wp, -7.25f);
        t[2].data = ids.data();
        t[3].data = w.data();
        ++cases;
        const int rc = r4d_rad_router_sigmoid_topk(&a, nullptr);
        int bad = 0;
        for (int r = 0; r < M && rc == RAD_OK; ++r) {
            int mi[32];
            float mw[32];
            router_model(&lg[(size_t)r * lp], bias.data(), bp, ne, k, sh.norm, factor, mi, mw);
            bad += std::memcmp(mi, &ids[(size_t)r * ip], (size_t)k * 4) != 0;
            bad += std::memcmp(mw, &w[(size_t)r * wp], (size_t)k * 4) != 0;
            for (int c = k; c < ip; ++c) bad += ids[(size_t)r * ip + c] != -77;
            for (int c = k; c < wp; ++c) bad += w[(size_t)r * wp + c] != -7.25f;
        }
        if (rc != RAD_OK || bad) {
            std::printf("FAIL router wave select against the scan at M=%d ne=%d k=%d%s (rc %d): "
                        "%d\n",
                        M, ne, k, sh.ties ? " tied" : "", rc, bad);
            ++fails;
        }
    }
}

struct GemmCase {
    int M, N, K, xp;
};

/* One shape against the model, f32 weight then the same weight stored bf16. With `dump` the
 * outputs' hashes go there instead of a FAIL line only. Returns the words that differ. */
static int gemm_case(std::mt19937& rng, const GemmCase& sh, FILE* dump) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const int M = sh.M, N = sh.N, K = sh.K, xp = sh.xp, yp = N + 3;
    std::vector<uint16_t> x((size_t)M * xp);
    std::vector<float> w((size_t)N * K);
    for (auto& v : x) v = f2b(nd(rng));
    // The weight's values are bf16 values, so the stored-bf16 run below reads the same weight.
    std::vector<uint16_t> packed(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
        packed[i] = f2b(nd(rng) * 0.03f);
        w[i] = b2f(packed[i]);
    }
    RadTensor t[3]{};
    t[0].data = x.data();
    t[0].dtype = RAD_BF16;
    t[0].rank = 2;
    t[0].shape[0] = M;
    t[0].shape[1] = K;
    t[0].stride[0] = xp;
    t[0].stride[1] = 1;
    t[1].data = w.data();
    t[1].dtype = RAD_F32;
    t[1].rank = 2;
    t[1].shape[0] = N;
    t[1].shape[1] = K;
    t[1].stride[0] = K;
    t[1].stride[1] = 1;
    t[2].dtype = RAD_F32;
    t[2].rank = 2;
    t[2].shape[0] = M;
    t[2].shape[1] = N;
    t[2].stride[0] = yp;
    t[2].stride[1] = 1;
    RadParam p[5]{};
    p[0] = { "M", RAD_P_INT, M };
    p[1] = { "N", RAD_P_INT, N };
    p[2] = { "K", RAD_P_INT, K };
    p[3].key = "dtype";
    p[3].kind = RAD_P_STR;
    p[3].sval = "bf16";
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p;
    a.n_p = 4;
    std::vector<float> y((size_t)M * yp, -7.25f);
    t[2].data = y.data();
    const int rc = r4d_rad_gemm_nt_f32(&a, nullptr);
    const char* form = emu_last_kernel;
    int bad = 0;
    for (int r = 0; r < M && rc == RAD_OK; ++r) {
        for (int c = 0; c < N; ++c) {
            const float e = gemm_model(&x[(size_t)r * xp], &w[(size_t)c * K], K);
            bad += std::memcmp(&e, &y[(size_t)r * yp + c], 4) != 0;
        }
        for (int c = N; c < yp; ++c) bad += y[(size_t)r * yp + c] != -7.25f;
    }
    // The same weight stored as bf16 (`weight_bf16`): widened in the kernel, the output is the
    // f32 weight's byte for byte, the pitch's fill included.
    std::vector<float> packed_y(y.size(), -7.25f);
    t[1].dtype = RAD_BF16;
    t[1].data = packed.data();
    p[4] = { "weight_bf16", RAD_P_INT, 1 };
    a.n_p = 5;
    t[2].data = packed_y.data();
    const int packed_rc = r4d_rad_gemm_nt_f32(&a, nullptr);
    const char* packed_form = emu_last_kernel;
    const int packed_bad =
        packed_rc != RAD_OK || std::memcmp(y.data(), packed_y.data(), y.size() * sizeof(float));
    bad += packed_bad;
    if (dump) {
        const std::vector<float>* outs[2] = { &y, &packed_y };
        for (int o = 0; o < 2; ++o) {
            uint64_t h = 1469598103934665603ull;
            const unsigned char* b = (const unsigned char*)outs[o]->data();
            for (size_t i = 0; i < outs[o]->size() * sizeof(float); ++i)
                h = (h ^ b[i]) * 1099511628211ull;
            std::fprintf(dump, "M %d N %d K %d xp %d w %s rc %d %016llx model_bad %d form %s\n", M,
                         N, K, xp, o ? "bf16" : "f32", o ? packed_rc : rc, (unsigned long long)h,
                         o ? packed_bad : bad - packed_bad, o ? packed_form : form);
            std::fflush(dump);
        }
    }
    if (rc != RAD_OK || bad)
        std::printf("FAIL %s at M=%d N=%d K=%d xp=%d against its model (rc %d): %d words\n", form,
                    M, N, K, xp, rc, bad);
    return rc != RAD_OK || bad;
}

int main() {
    std::mt19937 rng(17);
    int fails = 0, cases = 0;
    if (const char* dump = std::getenv("R4D_EMU_F32_DUMP")) {
        FILE* f = std::fopen(dump, "a");
        if (!f) return 2;
        const char* only = std::getenv("R4D_EMU_F32_DUMP_M");
        for (int M : { 1, 4, 16, 40, 300, 1024, 2048, 4096 }) {
            if (only && std::atoi(only) != M) continue;
            for (int N : { 128, 256, 512 }) {
                std::mt19937 grid_rng(1000003u * (unsigned)M + (unsigned)N);
                ++cases;
                fails += gemm_case(grid_rng, { M, N, 3072, 3072 }, f);
            }
        }
        std::fclose(f);
        std::printf("%d case(s), %d failure(s)\n", cases, fails);
        return fails ? 1 : 0;
    }
    // One wave an output (M 1), the 2x1, 4x1 (M under 16 or N up to 8), 4x4 and 8x4 tiles; from
    // 768 rows at N 128-255, 384 at N 256-511 and 256 at N 512+ the staged 8x8 form, each side of
    // each bound: x's pitch odd (staged a value at a time) and
    // a multiple of 8 (16 bytes at a time), K past and short of a whole stage of 256 and of a whole
    // step of 32, M no multiple of 8, and N no multiple of 32 (the last workgroup's last waves
    // have no columns) or of 8.
    const GemmCase shapes[] = {
        { 1, 288, 512, 517 },    { 2, 288, 129, 134 },    { 3, 36, 129, 134 },
        { 8, 288, 129, 134 },    { 33, 32, 512, 517 },    { 70, 36, 129, 134 },
        { 70, 4, 512, 517 },     { 7, 31, 129, 134 },     { 65, 13, 130, 135 },
        { 16, 288, 1056, 1061 }, { 24, 288, 1040, 1045 }, { 40, 36, 1024, 1029 },
        { 1, 512, 3072, 3077 },  { 767, 128, 300, 305 },  { 768, 127, 300, 305 },
        { 769, 128, 300, 305 },  { 775, 130, 600, 605 },  { 383, 256, 520, 520 },
        { 400, 256, 520, 520 },  { 800, 203, 264, 264 },  { 776, 136, 1000, 1000 },
        { 255, 512, 96, 101 },   { 256, 512, 96, 101 },   { 768, 128, 24, 24 },
    };
    for (const GemmCase& sh : shapes) {
        ++cases;
        fails += gemm_case(rng, sh, nullptr);
    }
    router_cases(rng, cases, fails);
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
