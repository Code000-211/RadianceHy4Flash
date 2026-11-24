/* tests/r4d_emu: the device sampler (r4d_sample_chain_f32.hip, r4d_sample_stages_f32.hip) on the
 * host SIMT emulator, through the published entry points.
 *
 * sample_chain: the same logits and uniforms must give the same three outputs whatever order the
 * block's threads run in. Each case runs in thread order, again, and in reverse order
 * (emu_reverse_order), and all three must match a serial host oracle bit for bit: the oracle
 * takes the top_k and nucleus thresholds from a sort and the nucleus mass in q62, so it shares
 * neither the radix passes nor their accumulation order with the kernel. The rows are adversarial
 * for the nucleus: top_p sits exactly on a boundary between two probabilities, where one ulp of
 * the summed mass moves the cut.
 *
 * The negative control is `old_topp_kernel` below, the nucleus passes as they were with an f32 LDS
 * atomicAdd: over the same rows its threshold must differ between the two orders on some row, or
 * the comparison above could not have caught the defect.
 *
 * sample_topk and sample_logprobs: the per-group histograms of the radix select against a brute
 * force (descending logit, ties to the lower id), in both orders: vector and scalar loads, tie
 * groups that need the id pass, -inf entries, top-1, greedy rows, the pair plane.
 *
 * tests/halo_emu/test_sample.cpp includes this file with SAMPLE_HALO defined: the sample_chain
 * cases on libhalo's copy of the kernel (its stage kernels already count per wave group). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_sample.h"
#ifdef SAMPLE_HALO
#include "halo.h"
#include "halo_plugin.h"
typedef HaloSpecDraw R4DSpecDraw;
#define r4d_rad_sample_chain halo_rad_sample_chain
#else
#include "r4d.h"
#include "r4d_plugin.h"
#endif
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int fails = 0, cases = 0;
static void check(bool ok, const char* what, int a = 0, int b = 0, int c = 0) {
    ++cases;
    if (!ok) { ++fails; std::printf("FAIL %s (%d, %d, %d)\n", what, a, b, c); }
}

static uint32_t mono(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static uint64_t q62(float p) {
    return p > 0.0f ? (uint64_t)(std::fmin(p, 2.0f) * 4611686018427387904.0f) : 0ull;
}
/* The block reduction's order: thread t's value, then the pairwise fold over 256 threads. */
static float fold256(float* red) {
    for (int k = 128; k; k >>= 1) for (int t = 0; t < k; ++t) red[t] += red[t + k];
    return red[0];
}

/* ------------------------------------------------------------------ sample_chain: the oracle */

struct Chain { float temp, top_p, min_p; int top_k; };

/* The row's probabilities as the kernel forms them: the same expf, the same sum order for Z. */
static std::vector<float> probs(const float* lg, int V, float temp, float* inv_z_out) {
    float m = -INFINITY;
    for (int v = 0; v < V; ++v) m = std::fmax(m, lg[v]);
    const float inv_t = 1.0f / temp;
    float red[256] = {};
    for (int v = 0; v < V; ++v) red[v % 256] += std::exp((lg[v] - m) * inv_t);
    const float z = fold256(red);
    const float inv_z = z > 0.0f ? 1.0f / z : 0.0f;
    std::vector<float> p((size_t)V);
    for (int v = 0; v < V; ++v) p[v] = std::exp((lg[v] - m) * inv_t) * inv_z;
    *inv_z_out = inv_z;
    return p;
}

/* The nucleus threshold from a sort: the largest key whose tokens, with every larger one, hold at
 * least top_p of the mass. 0 when the candidates never reach it (no cut). */
static uint32_t nucleus_key(const std::vector<float>& p, uint32_t k_key, float top_p) {
    std::vector<uint32_t> keys;
    for (float x : p) if (!k_key || mono(x) >= k_key) keys.push_back(mono(x));
    std::sort(keys.begin(), keys.end(), std::greater<uint32_t>());
    const uint64_t target = q62(top_p);
    uint64_t cum = 0;
    for (uint32_t key : keys) {
        float x;
        const uint32_t u = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
        std::memcpy(&x, &u, 4);
        cum += q62(x);
        if (cum >= target) return key;
    }
    return 0u;
}

static R4DSpecDraw chain_oracle(const float* lg, int V, const Chain& c, uint32_t q, const float* u) {
    R4DSpecDraw d{};
    if (!(c.temp > 0.0f)) {
        int best = 0;
        for (int v = 1; v < V; ++v) if (lg[v] > lg[best]) best = v;
        d.tok = d.tok_excl = (uint32_t)best;
        d.p_query = q == (uint32_t)best ? 1.0f : 0.0f;
        return d;
    }
    float inv_z = 0.0f;
    const std::vector<float> p = probs(lg, V, c.temp, &inv_z);
    uint32_t k_key = 0u;
    if (c.top_k > 0 && c.top_k < V) {
        std::vector<uint32_t> keys;
        for (float x : p) keys.push_back(mono(x));
        std::nth_element(keys.begin(), keys.begin() + (c.top_k - 1), keys.end(), std::greater<uint32_t>());
        k_key = keys[c.top_k - 1];
    }
    const uint32_t p_key = c.top_p < 1.0f ? nucleus_key(p, k_key, c.top_p) : 0u;
    const float floor_p = c.min_p > 0.0f ? c.min_p * inv_z : 0.0f;
    auto keep = [&](float x) {
        const uint32_t key = mono(x);
        if (k_key && key < k_key) return false;
        if (p_key && key < p_key) return false;
        return !(x < floor_p);
    };
    float kt[256] = {}, ke[256] = {}, pq[256] = {};
    for (int v = 0; v < V; ++v) {
        if (!keep(p[v])) continue;
        kt[v % 256] += p[v];
        if ((uint32_t)v == q) pq[v % 256] += p[v]; else ke[v % 256] += p[v];
    }
    const float keep_total = fold256(kt), excl_total = fold256(ke), p_query = fold256(pq);
    /* The inverse CDF over contiguous slices, as the kernel walks it. */
    const int chunk = (V + 255) / 256;
    auto draw = [&](float uu, float total, uint32_t excl) -> uint32_t {
        if (!(total > 0.0f)) return 0u;
        const float want = uu * total;
        float acc = 0.0f, last_base = 0.0f;
        int who = 256, last_who = 0;
        float mine[256] = {};
        for (int t = 0; t < 256; ++t)
            for (int v = t * chunk; v < std::min(V, (t + 1) * chunk); ++v)
                if (keep(p[v]) && (uint32_t)v != excl) mine[t] += p[v];
        for (int t = 0; t < 256; ++t) {
            if (!(mine[t] > 0.0f)) continue;
            last_who = t;
            last_base = acc;
            if (acc + mine[t] >= want) { who = t; break; }
            acc += mine[t];
        }
        if (who == 256) { who = last_who; acc = last_base; }
        uint32_t pick = 0xFFFFFFFFu, last = 0xFFFFFFFFu;
        for (int v = who * chunk; v < std::min(V, (who + 1) * chunk); ++v) {
            if (!keep(p[v]) || (uint32_t)v == excl || !(p[v] > 0.0f)) continue;
            last = (uint32_t)v;
            acc += p[v];
            if (acc >= want) { pick = (uint32_t)v; break; }
        }
        if (pick == 0xFFFFFFFFu) pick = last;
        return pick < (uint32_t)V ? pick : 0u;
    };
    d.tok = draw(u[0], keep_total, 0xFFFFFFFFu);
    d.tok_excl = draw(u[1], excl_total, q);
    d.p_query = keep_total > 0.0f ? p_query / keep_total : 0.0f;
    return d;
}

/* M positions through r4d_rad_sample_chain, `reverse` choosing the thread order. */
static std::vector<R4DSpecDraw> run_chain(std::vector<float>& lg, int M, int V, const Chain& c,
                                          std::vector<uint32_t>& query, std::vector<float>& u,
                                          int reverse) {
    std::vector<R4DSpecDraw> out((size_t)M, R4DSpecDraw{0xdeadu, 0xdeadu, -7.0f});
    RadTensor t[4]{};
    t[0].data = lg.data(); t[0].dtype = RAD_F32; t[0].rank = 2;
    t[0].shape[0] = M; t[0].shape[1] = V; t[0].stride[0] = V; t[0].stride[1] = 1;
    t[1].data = query.data(); t[1].dtype = RAD_U32; t[1].rank = 1; t[1].shape[0] = M; t[1].stride[0] = 1;
    t[2].data = u.data(); t[2].dtype = RAD_F32; t[2].rank = 1; t[2].shape[0] = 2 * M; t[2].stride[0] = 1;
    t[3].data = out.data(); t[3].dtype = RAD_U32; t[3].rank = 2;
    t[3].shape[0] = M; t[3].shape[1] = 3; t[3].stride[0] = 3; t[3].stride[1] = 1;
    RadParam p[6]{};
    p[0] = {"M", RAD_P_INT, M}; p[1] = {"n_vocab", RAD_P_INT, V};
    p[2].key = "temp"; p[2].kind = RAD_P_F64; p[2].dval = c.temp;
    p[3].key = "p"; p[3].kind = RAD_P_F64; p[3].dval = c.top_p;
    p[4].key = "min_p"; p[4].kind = RAD_P_F64; p[4].dval = c.min_p;
    p[5] = {"k", RAD_P_INT, c.top_k};
    RadArgs a{}; a.t = t; a.n_t = 4; a.p = p; a.n_p = 6;
    emu_reverse_order = reverse;
    const int rc = r4d_rad_sample_chain(&a, nullptr);
    emu_reverse_order = 0;
    if (rc != 0) out.assign((size_t)M, R4DSpecDraw{0xbadu, 0xbadu, -9.0f});
    return out;
}

/* The nucleus passes before the fix, kept as the negative control: bucket mass summed in f32
 * through an LDS atomic, so the sum takes the order the threads arrive in. */
static void old_topp_kernel(const float* p, uint32_t vocab, float top_p, uint32_t* p_key_out) {
    __shared__ float mass[256];
    __shared__ uint32_t prefix;
    const uint32_t tid = threadIdx.x;
    if (tid == 0) prefix = 0u;
    __syncthreads();
    float target = top_p;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) mass[tid] = 0.0f;
        __syncthreads();
        const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint32_t v = tid; v < vocab; v += 256) {
            const uint32_t key = mono(p[v]);
            if ((key & mask) == (prefix & mask)) atomicAdd(&mass[(key >> shift) & 0xFFu], p[v]);
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            uint32_t b = 0u;
            for (int i = 255; i >= 0; --i) {
                if (acc + mass[i] >= target) { b = (uint32_t)i; break; }
                acc += mass[i];
            }
            target -= acc;
            prefix |= b << shift;
        }
        __syncthreads();
    }
    if (tid == 0) *p_key_out = prefix;
}
static uint32_t old_topp(const std::vector<float>& p, float top_p, int reverse) {
    uint32_t key = 0;
    emu_reverse_order = reverse;
    hipLaunchKernelGGL(old_topp_kernel, dim3(1), dim3(256), 0, nullptr, p.data(),
                       (uint32_t)p.size(), top_p, &key);
    emu_reverse_order = 0;
    return key;
}

static bool same(const std::vector<R4DSpecDraw>& a, const std::vector<R4DSpecDraw>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(R4DSpecDraw));
}

static void test_chain() {
    std::mt19937 rng(47);
    std::normal_distribution<float> gauss(0.0f, 2.5f);
    std::uniform_real_distribution<float> uni(0.0f, 0.999f);
    int old_flips = 0, rows = 0;
    for (int V : {1000, 4099, 16384}) for (int round = 0; round < 6; ++round) {
        const int M = 4;
        const float temp = round % 3 == 0 ? 1.0f : round % 3 == 1 ? 0.7f : 1.3f;
        std::vector<float> lg((size_t)M * V), u((size_t)2 * M);
        std::vector<uint32_t> query((size_t)M);
        for (float& x : lg) x = gauss(rng);
        for (float& x : u) x = uni(rng);
        for (int m = 0; m < M; ++m) {
            /* the draft's proposal: the mode, which every filter keeps, or none on the last row */
            const float* row = &lg[(size_t)m * V];
            query[m] = m == M - 1 ? 0xFFFFFFFFu : (uint32_t)(std::max_element(row, row + V) - row);
        }
        /* top_p on a boundary of row 0's sorted mass, between the 20th and the 60th percentile
         * of the vocabulary's tokens by rank among the likely ones: any float sum of the mass
         * above the cut lands within an ulp of it, on either side. */
        float inv_z = 0.0f;
        const std::vector<float> p0 = probs(lg.data(), V, temp, &inv_z);
        std::vector<float> sorted = p0;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        const int cut = 20 + (int)(rng() % 200);
        uint64_t cum = 0;
        for (int i = 0; i < cut; ++i) cum += q62(sorted[i]);
        const float top_p = (float)((double)cum / 4611686018427387904.0);

        for (int form = 0; form < 4; ++form) {
            Chain c{temp, top_p, 0.0f, 0};
            if (form == 1) c.top_k = 400;            /* the nucleus over a top_k set */
            if (form == 2) c.min_p = 0.02f;
            if (form == 3) { c.top_k = 12; c.top_p = 0.999f; }   /* a top_k set short of top_p: no cut */
            const auto fwd = run_chain(lg, M, V, c, query, u, 0);
            const auto again = run_chain(lg, M, V, c, query, u, 0);
            const auto rev = run_chain(lg, M, V, c, query, u, 1);
            check(same(fwd, again), "sample_chain: a rerun differs", V, round, form);
            check(same(fwd, rev), "sample_chain: the thread order changes the draw", V, round, form);
            std::vector<R4DSpecDraw> ref;
            for (int m = 0; m < M; ++m)
                ref.push_back(chain_oracle(&lg[(size_t)m * V], V, c, query[m], &u[(size_t)2 * m]));
            check(same(fwd, ref), "sample_chain: not the host oracle's draw", V, round, form);
        }
        /* Temperature 0 is greedy: the argmax, ties to the lower id, in either order. */
        {
            lg[7] = lg[V - 3] = 50.0f;   /* a tie at the top of row 0 */
            const Chain g{0.0f, 0.9f, 0.0f, 40};
            const auto fwd = run_chain(lg, M, V, g, query, u, 0), rev = run_chain(lg, M, V, g, query, u, 1);
            std::vector<R4DSpecDraw> ref;
            for (int m = 0; m < M; ++m)
                ref.push_back(chain_oracle(&lg[(size_t)m * V], V, g, query[m], &u[(size_t)2 * m]));
            check(same(fwd, rev) && same(fwd, ref) && fwd[0].tok == 7u, "sample_chain: greedy", V, round);
        }
        /* The negative control, on the row the boundary was taken from. */
        ++rows;
        old_flips += old_topp(p0, top_p, 0) != old_topp(p0, top_p, 1);
    }
    std::printf("sample_chain: f32-atomic nucleus (negative control) changed its threshold with the "
                "thread order on %d of %d row(s)\n", old_flips, rows);
    check(old_flips > 0, "negative control: the f32 atomic sum never showed its order dependence");
}

/* ------------------------------------------------------------------ sample_topk, sample_logprobs */
#ifndef SAMPLE_HALO

struct TopkOut { std::vector<int32_t> ci; std::vector<float> cv; std::vector<uint32_t> pr; int rc; };

static TopkOut run_topk(float* lg, int M, int V, int ld, int n_cand, std::vector<RadSampleParams>& sp,
                        bool pairs, int voff, int reverse) {
    TopkOut o;
    o.ci.assign((size_t)M * n_cand, 12345);
    o.cv.assign((size_t)M * n_cand, 777.0f);
    o.pr.assign((size_t)M * n_cand * 2, 0x5a5a5a5au);
    RadTensor t[5]{};
    t[0].data = lg; t[0].dtype = RAD_F32; t[0].rank = 2;
    t[0].shape[0] = M; t[0].shape[1] = V; t[0].stride[0] = ld; t[0].stride[1] = 1;
    t[1].data = sp.data(); t[1].dtype = RAD_U32; t[1].rank = 1;
    t[1].shape[0] = (int64_t)M * (int64_t)(sizeof(RadSampleParams) / 4); t[1].stride[0] = 1;
    t[2].data = o.ci.data(); t[2].dtype = RAD_I32; t[2].rank = 2;
    t[2].shape[0] = M; t[2].shape[1] = n_cand; t[2].stride[0] = n_cand; t[2].stride[1] = 1;
    t[3] = t[2]; t[3].data = o.cv.data(); t[3].dtype = RAD_F32;
    t[4].data = pairs ? o.pr.data() : nullptr; t[4].dtype = RAD_U32; t[4].rank = 2;
    t[4].shape[0] = M; t[4].shape[1] = 2 * n_cand; t[4].stride[0] = 2 * n_cand; t[4].stride[1] = 1;
    RadParam p[3]{};
    p[0] = {"M", RAD_P_INT, M}; p[1] = {"n_vocab", RAD_P_INT, V}; p[2] = {"vocab_off", RAD_P_INT, voff};
    RadArgs a{}; a.t = t; a.n_t = 5; a.p = p; a.n_p = 3;
    emu_reverse_order = reverse;
    o.rc = r4d_rad_sample_topk(&a, nullptr);
    emu_reverse_order = 0;
    return o;
}

/* Token ids of the row in the select's order: descending logit, ties to the lower id. */
static std::vector<int32_t> ranked(const float* row, int V) {
    std::vector<int32_t> id((size_t)V);
    for (int v = 0; v < V; ++v) id[v] = v;
    std::sort(id.begin(), id.end(), [&](int32_t a, int32_t b) {
        const uint32_t ka = mono(row[a]), kb = mono(row[b]);
        return ka != kb ? ka > kb : a < b;
    });
    return id;
}

static void test_topk() {
    std::mt19937 rng(30);
    const uint32_t ninf_bits = 0xff800000u;
    /* V 4096 and 12416 take the float4 loads on an aligned row; 1001 and a misaligned base the
     * scalar ones. */
    for (int V : {1001, 4096, 12416}) for (int n_cand : {64, 100, 1024}) for (int shape = 0; shape < 4; ++shape) {
        const int M = 5, ld = V + (shape == 1 ? 8 : 0);
        const bool misaligned = shape == 2;
        std::vector<float> store((size_t)M * ld + 8, -1234.0f);
        const size_t off = (((uintptr_t)store.data() & 15u) ? (16 - ((uintptr_t)store.data() & 15u)) / 4 : 0)
                           + (misaligned ? 1 : 0);
        float* lg = store.data() + off;
        for (int m = 0; m < M; ++m) for (int v = 0; v < V; ++v) {
            float x;
            switch ((m + shape) % 4) {
                case 0:  x = std::ldexp((float)(int)(rng() % 4001) - 2000.0f, -6); break;   /* many ties */
                case 1:  x = (float)(rng() % 7) - 3.0f; break;                               /* seven values */
                case 2:  x = rng() % 5 == 0 ? -INFINITY : std::ldexp((float)(int)(rng() % 60000), -9); break;
                default: x = std::ldexp((float)(int)(rng() & 0xffffff) - 8388608.0f, -18); break;
            }
            lg[(size_t)m * ld + v] = x;
        }
        std::vector<RadSampleParams> sp((size_t)M);
        const int ks[5] = {0, 1, 20, n_cand, 3 * n_cand};
        for (int m = 0; m < M; ++m) {
            sp[m] = RadSampleParams{};
            sp[m].top_k = ks[(m + shape) % 5];
            sp[m].temp = 1.0f;
        }
        if (shape == 3) sp[2].flags = RAD_SP_GREEDY;
        const bool pairs = shape != 0;
        const int voff = shape == 3 ? 100000 : 0;
        const TopkOut fwd = run_topk(lg, M, V, ld, n_cand, sp, pairs, voff, 0);
        const TopkOut rev = run_topk(lg, M, V, ld, n_cand, sp, pairs, voff, 1);
        bool ok = fwd.rc == 0 && rev.rc == 0;
        check(ok && fwd.ci == rev.ci && !std::memcmp(fwd.cv.data(), rev.cv.data(), fwd.cv.size() * 4) &&
                  fwd.pr == rev.pr,
              "sample_topk: the thread order changes the set", V, n_cand, shape);
        for (int m = 0; m < M && ok; ++m) {
            const float* row = lg + (size_t)m * ld;
            const bool greedy = (sp[m].flags & RAD_SP_GREEDY) != 0;
            int k = sp[m].top_k > 0 ? sp[m].top_k : n_cand;
            k = std::min(k, std::min(n_cand, V));
            if (greedy) k = 0;
            const std::vector<int32_t> id = ranked(row, V);
            for (int r = 0; r < n_cand && ok; ++r) {
                const int32_t want_i = r < k ? id[r] : -1;
                const float want_v = r < k ? row[id[r]] : -INFINITY;
                const size_t at = (size_t)m * n_cand + r;
                ok = fwd.ci[at] == want_i && !std::memcmp(&fwd.cv[at], &want_v, 4);
                if (ok && pairs) {
                    uint32_t vb;
                    std::memcpy(&vb, &want_v, 4);
                    ok = fwd.pr[2 * at] == (r < k ? (uint32_t)(want_i + voff) : 0xffffffffu) &&
                         fwd.pr[2 * at + 1] == (r < k ? vb : ninf_bits);
                }
            }
        }
        check(ok, "sample_topk: not the brute-force set", V, n_cand, shape);
        /* a row's padding is not the kernel's to write */
        if (shape == 1) check(lg[V] == -1234.0f && lg[(size_t)ld + V + 7] == -1234.0f, "sample_topk: wrote the pitch", V);

        /* sample_logprobs takes the same select at the op's k. */
        if (n_cand == 64) {
            const int k = 21, st_ld = 2 + 2 * k;
            for (int reverse = 0; reverse < 2; ++reverse) {
                std::vector<float> st((size_t)M * st_ld, 99.0f);
                RadTensor t[3]{};
                t[0].data = lg; t[0].dtype = RAD_F32; t[0].rank = 2;
                t[0].shape[0] = M; t[0].shape[1] = V; t[0].stride[0] = ld; t[0].stride[1] = 1;
                t[2].data = st.data(); t[2].dtype = RAD_F32; t[2].rank = 2;
                t[2].shape[0] = M; t[2].shape[1] = st_ld; t[2].stride[0] = st_ld; t[2].stride[1] = 1;
                RadParam p[4]{};
                p[0] = {"M", RAD_P_INT, M}; p[1] = {"n_vocab", RAD_P_INT, V};
                p[2] = {"k", RAD_P_INT, k}; p[3] = {"vocab_off", RAD_P_INT, voff};
                RadArgs a{}; a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
                emu_reverse_order = reverse;
                bool lok = r4d_rad_sample_logprobs(&a, nullptr) == 0;
                emu_reverse_order = 0;
                for (int m = 0; m < M && lok; ++m) {
                    const float* row = lg + (size_t)m * ld;
                    const std::vector<int32_t> id = ranked(row, V);
                    const float* o = &st[(size_t)m * st_ld];
                    lok = o[0] == row[id[0]];
                    for (int r = 0; r < k && lok; ++r)
                        lok = o[2 + 2 * r] == (float)(id[r] + voff) && !std::memcmp(&o[3 + 2 * r], &row[id[r]], 4);
                }
                check(lok, "sample_logprobs: not the brute-force top k", V, shape, reverse);
            }
        }
    }
}

#endif

int main() {
    test_chain();
#ifndef SAMPLE_HALO
    test_topk();
#endif
    std::printf("sample: %d check(s), %d failure(s)\n", cases, fails);
    return fails != 0;
}
