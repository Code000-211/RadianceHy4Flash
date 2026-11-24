// r4d_ihc.h -- the independent hyper-connection's device pieces, shared by ihc_pre
// (r4d_ihc_bf16.hip), the fused junction rmsnorm_ihc_junction (same unit) and its all-reduce twin
// ar_rmsnorm_ihc_junction (r4d_ar_ihc_junction.hip). One summation order for all of them, so the
// fused rows are ihc_pre's bytes.
//
// ihc_pre's order (WP16.2; the scalar kernel it replaced strided single columns by 256 threads):
// a workgroup of kIhcThreads = 512 a token row. The row's W = hc * n values are pieces of eight
// consecutive values (the last one shorter when W % 8 != 0); thread t owns pieces t, t + 512, ...
// and keeps rows + 1 f32 sums: for each projection j, fn[j][c] * f[c], and f[c]^2, each product
// rounded to f32 and added (no FMA), the pieces ascending and the eight values of a piece
// ascending. The 32 lanes of a wave then add by r4d_xor_sum<32> (offsets 16, 8, 4, 2, 1), and one
// thread adds the 16 waves' sums from 0.0f in wave order. A row's bytes depend on its own values
// only (not on M, the pitch or the alignment: the scalar-load form runs the same order).
#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include "r4d_common.h"
#include "r4d_engram_gate.h"   // the Engram form's gate stage (P-07)
#include "r4d_q8.h"   // q8_row_had_i8g: the next norm's rotated int8 codes (G-3b)

namespace r4d_ihc {

enum { kThreads = 512, kWaves = kThreads / 32 };

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Eight bf16 of a 16-byte word as floats.
__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        f[2 * k] = __uint_as_float(w[k] << 16);
        f[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u);
    }
}
__device__ __forceinline__ uint4 pack8(const uint16_t (&b)[8]) {
    return make_uint4((uint32_t)b[0] | ((uint32_t)b[1] << 16), (uint32_t)b[2] | ((uint32_t)b[3] << 16),
                      (uint32_t)b[4] | ((uint32_t)b[5] << 16), (uint32_t)b[6] | ((uint32_t)b[7] << 16));
}

// One piece's contribution: `f` the piece's `cnt` values (cnt <= 8) at flattened column `c0`.
// VF: fn's rows are 16-byte aligned with a pitch of whole float4 and the piece is whole.
template <int R, bool VF>
__device__ __forceinline__ void acc_piece(float (&acc)[R + 1], const float (&f)[8], uint32_t cnt,
                                          const float* __restrict__ fn, uint64_t fp, uint32_t c0) {
#pragma unroll
    for (int j = 0; j < R; ++j) {
        const float* __restrict__ fr = fn + (uint64_t)j * fp + c0;
        float w[8];
        if (VF) {
            const float4 a = *(const float4*)fr, b = *(const float4*)(fr + 4);
            w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w;
            w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[j] += w[e] * f[e];
        } else {
            for (uint32_t e = 0; e < cnt; ++e) acc[j] += fr[e] * f[e];
        }
    }
    if (VF) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[R] += f[e] * f[e];
    } else {
        for (uint32_t e = 0; e < cnt; ++e) acc[R] += f[e] * f[e];
    }
}

// The lanes, then the waves through `part`; one thread forms r, the gates (into `gate`) and the
// write weights (into `post_row` when POST). Every thread calls; a barrier on entry to `part`'s
// reads and one behind the gates, so `gate` is readable on return.
template <int HC, bool POST>
__device__ __forceinline__ void finish(float (&acc)[(POST ? 2 * HC : HC) + 1],
                                       float (*part)[(POST ? 2 * HC : HC) + 1], float* gate,
                                       const float* __restrict__ scale, const float* __restrict__ base,
                                       float* __restrict__ post_row, uint32_t width, float eps,
                                       float heps, float pmag) {
    constexpr int R = POST ? 2 * HC : HC;
    const uint32_t t = threadIdx.x, lane = t & 31u, wave = t >> 5;
#pragma unroll
    for (int k = 0; k <= R; ++k) {
        const float s = r4d_xor_sum<32>(acc[k]);
        if (lane == 0) part[wave][k] = s;
    }
    __syncthreads();
    if (t == 0) {
        float s[R + 1];
#pragma unroll
        for (int k = 0; k <= R; ++k) {
            float v = 0.0f;
            for (int w = 0; w < kWaves; ++w) v += part[w][k];
            s[k] = v;
        }
        const float inv = 1.0f / sqrtf(s[R] / (float)width + eps);
#pragma unroll
        for (int j = 0; j < HC; ++j) gate[j] = sigmoid(s[j] * inv * scale[0] + base[j]) + heps;
        if constexpr (POST) {
#pragma unroll
            for (int j = 0; j < HC; ++j)
                post_row[j] = pmag * sigmoid(s[HC + j] * inv * scale[1] + base[HC + j]) + heps;
        }
    }
    __syncthreads();
}

// x[i] = bf16(sum_j gate[j] * f32(h[j][i])), j ascending, for the 8-column groups q = t, t + 512,
// ... of the row (n % 8 == 0, rows 16-byte aligned). `store(q, word)` takes each group's word.
template <int HC, class S>
__device__ __forceinline__ void collapse8(const uint16_t* __restrict__ hr, const float* gate,
                                          uint32_t n, S store) {
    for (uint32_t q = threadIdx.x; q < n / 8u; q += kThreads) {
        float v[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = 0.0f;
#pragma unroll
        for (int j = 0; j < HC; ++j) {
            float f[8];
            unpack8(*(const uint4*)(hr + (uint32_t)j * n + 8u * q), f);
            const float g = gate[j];
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += g * f[e];
        }
        uint16_t b[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) b[e] = f32_to_bf16(v[e]);
        store(q, pack8(b));
    }
}

// ------------------------------------------------------------------ the fused junction's row
// `[add] -> [all_reduce] -> rmsnorm -> ihc_post -> [ihc_pre -> rmsnorm]` over one token row, at
// kThreads threads, the unfused rows' bytes:
//
//   * both norms are rmsnorm_bf16's (r4d_model_bf16.hip): the sum of squares in kR4dSumLanes'
//     partition (r4d_lanes_sumsq), the wave butterfly, the waves from 0.0f in order (norm_sum, a
//     copy of its block_sum: the waves past the eighth add 0.0f there and nothing here), rsqrtf,
//     then `x * s * (g + wadd) * mult` per element (wadd 0 and mult 1 passed at run time, as the
//     standalone norm gets them), rounded once. The norm reads its input as stored: the raw row
//     here is held in LDS as the bf16 the chain stores.
//   * ihc_post's `bf16(post[j] * y + h)` with the product rounded first (__fmul_rn / __fadd_rn);
//   * ihc_pre's order above, over the h just written (from the same registers);
//   * the Engram form (EG): engram_gate's own row (r4d_engram_gate.h) between the write and those
//     sums, over the h as stored, the first 256 threads' waves taking the gate's partition;
//   * the collapse x rounded to bf16 into LDS and normed from there.
//
// The row's input (`a`, or `a + b`, or the reduced sum) is in `ybuf` (n bf16, 16-byte aligned) on
// entry, each thread having written its own 8-column groups; `ybuf` ends holding x (TAIL) or y.
template <int HC, bool POST, bool EG = false>
struct JuncLds {
    float red1[kWaves], red2[kWaves];
    float part[kWaves][(POST ? 2 * HC : HC) + 1];
    float gate[HC], pv[HC];
    /* EG: engram_gate's two wave-sum arrays, in its order (wave, value) */
    float eg1[EG ? r4d_eg::kWaves : 1][EG ? 1 + 2 * HC : 1];
    float eg2[EG ? r4d_eg::kWaves : 1][EG ? HC : 1];
};

// rmsnorm_bf16's block_sum: the wave butterfly, then the waves from 0.0f in order.
__device__ __forceinline__ float norm_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < (uint32_t)kWaves; ++w) s += lds[w];
    return s;
}

// The norm of the bf16 row in `buf` with the f32 gain `g`, word by word to `put(q, word)`.
template <class P>
__device__ __forceinline__ void norm_row(const uint16_t* buf, const float* __restrict__ g,
                                         uint32_t n, float eps, float wadd, float mult, float* red,
                                         P put) {
    float ss = r4d_lanes_sumsq(buf, n, [](uint16_t v) { return bf16_to_f32(v); });
    ss = norm_sum(ss, red);
    const float s = rsqrtf(ss / (float)n + eps);
    for (uint32_t q = threadIdx.x; q < n / 8u; q += kThreads) {
        float f[8];
        unpack8(*(const uint4*)(buf + 8u * q), f);
        const float4 g0 = *(const float4*)(g + 8u * q), g1 = *(const float4*)(g + 8u * q + 4u);
        const float gv[8] = { g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w };
        uint16_t b[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) b[e] = f32_to_bf16(f[e] * s * (gv[e] + wadd) * mult);
        put(q, pack8(b));
    }
}

// Passed by value as a kernel argument: no padding anywhere (the pointers and 64-bit fields first,
// the 32-bit fields filling a whole 8-byte multiple), so the argument bytes are the fields' and a
// recorded pass's audit (core/runtime/ctx.cpp tapes_agree) compares the same bytes every step.
struct JuncRow {
    const float* w_post;
    float* post_row;          // [hc] f32: read (this connection), written (the next one's, POST && TAIL)
    uint16_t* hr;             // [hc * n] the streams, in place
    const float* fn;          // [rows][hc * n] at pitch fp
    uint64_t fp;
    const float *scale, *base, *w_next;
    uint16_t* xnr;            // [n] the next block's normed input (TAIL)
    uint16_t* yr;             // [n] the post norm's output, when asked
    uint16_t* xr;             // [n] the collapse, when asked
    signed char* cq;          // [n] the next block's input norm's rotated int8 codes (TAIL), when asked
    float* cs;                // [n / 128] their scales
    uint64_t cqp, csp;        // the row pitches of cq and cs
    uint32_t n;
    float eps, ieps, heps, pmag, wadd, mult;
    /* the Engram form (EG): its key [hc * n] and val [n] bf16 rows at the pitches egkp / egvp, its
     * mask [M], its four WT weights (wk, wq, wb [hc * n], wv [n]) and engram_gate's two floats */
    const void *eg_key, *eg_val, *eg_wk, *eg_wq, *eg_wv, *eg_wb;
    const int32_t* eg_mask;
    uint64_t egkp, egvp;
    float eg_eps, eg_inv_sqrt_n;
    uint32_t eg_w_bf16 = 0;
    uint32_t pad_ = 0;
};
static_assert(sizeof(JuncRow) == 24 * 8 + 12 * 4, "JuncRow carries no padding");

template <int HC, bool POST, bool TAIL, bool EG = false>
__device__ __forceinline__ void junction_row(JuncLds<HC, POST, EG>& L, uint16_t* ybuf,
                                            const JuncRow& r) {
    constexpr int R = POST ? 2 * HC : HC;
    const uint32_t t = threadIdx.x, n = r.n;
    if (t < (uint32_t)HC) L.pv[t] = r.post_row[t];
    __syncthreads();   // the input row in ybuf and pv
    // 1. y = rmsnorm(input), back into ybuf: every thread has read what the sum needs before
    //    norm_sum's barrier, and each writes only its own words.
    norm_row(ybuf, r.w_post, n, r.eps, r.wadd, r.mult, L.red1, [&](uint32_t q, uint4 w) {
        *(uint4*)(ybuf + 8u * q) = w;
        if (r.yr) *(uint4*)(r.yr + 8u * q) = w;
    });
    __syncthreads();
    // 2. h += post * y, and the next connection's sums from the same registers.
    float acc[R + 1];
#pragma unroll
    for (int k = 0; k <= R; ++k) acc[k] = 0.0f;
    const uint32_t pieces = (uint32_t)HC * n / 8u;
    for (uint32_t p = t; p < pieces; p += kThreads) {
        const uint32_t c0 = 8u * p, j = c0 / n, i0 = c0 - j * n;
        float hf[8], yf[8], f[8];
        unpack8(*(const uint4*)(r.hr + c0), hf);
        unpack8(*(const uint4*)(ybuf + i0), yf);
        const float pj = L.pv[j];
        uint16_t b[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            b[e] = f32_to_bf16(__fadd_rn(__fmul_rn(pj, yf[e]), hf[e]));
            f[e] = bf16_to_f32(b[e]);
        }
        *(uint4*)(r.hr + c0) = pack8(b);
        if constexpr (TAIL && !EG) acc_piece<R, true>(acc, f, 8u, r.fn, r.fp, c0);
    }
    if constexpr (EG) {
        // 2b. The Engram's gate over the h just written (engram_gate's row: its first 256 threads'
        //     waves own the columns), then the next connection's sums over the gated h as stored.
        //     `eg_mask` is this block's rows (the caller has added the row it holds).
        __syncthreads();
        const bool on = r.eg_mask[0] != 0 && t < (uint32_t)r4d_eg::kThreads;
        if (r.eg_w_bf16)
            r4d_eg::row<HC, uint16_t>((const uint16_t*)r.eg_key, (const uint16_t*)r.eg_val,
                                      (const uint16_t*)r.eg_wk, (const uint16_t*)r.eg_wq,
                                      (const uint16_t*)r.eg_wv, (const uint16_t*)r.eg_wb, r.hr, n,
                                      r.eg_eps, r.eg_inv_sqrt_n, L.eg1, L.eg2, on);
        else
            r4d_eg::row<HC, float>((const uint16_t*)r.eg_key, (const uint16_t*)r.eg_val,
                                   (const float*)r.eg_wk, (const float*)r.eg_wq,
                                   (const float*)r.eg_wv, (const float*)r.eg_wb, r.hr, n, r.eg_eps,
                                   r.eg_inv_sqrt_n, L.eg1, L.eg2, on);
        __syncthreads();
        for (uint32_t p = t; p < pieces; p += kThreads) {
            float f[8];
            unpack8(*(const uint4*)(r.hr + 8u * p), f);
            acc_piece<R, true>(acc, f, 8u, r.fn, r.fp, 8u * p);
        }
    }
    if constexpr (TAIL) {
        // 3. gates and post (finish's barriers also publish h to the block), the collapse into
        //    ybuf, its norm.
        finish<HC, POST>(acc, L.part, L.gate, r.scale, r.base, r.post_row, (uint32_t)HC * n,
                         r.ieps, r.heps, r.pmag);
        collapse8<HC>(r.hr, L.gate, n, [&](uint32_t q, uint4 w) {
            *(uint4*)(ybuf + 8u * q) = w;
            if (r.xr) *(uint4*)(r.xr + 8u * q) = w;
        });
        __syncthreads();
        norm_row(ybuf, r.w_next, n, r.eps, r.wadd, r.mult, L.red2, [&](uint32_t q, uint4 w) {
            *(uint4*)(r.xnr + 8u * q) = w;
            if (r.cq) *(uint4*)(ybuf + 8u * q) = w;   // each thread its own words, read before
        });
        // G-3b: had_quant_act_i8g's codes over the bf16 row just stored (q8_row_had_i8g, a wave a
        // 128-column group), so the consumers of xn read them and no quantiser follows.
        if (r.cq) {
            __syncthreads();
            q8_row_had_i8g<kThreads>(ybuf, r.cq, r.cs, n);
        }
    }
}

// A junction launch's operands, checked (r4d_ihc_junction_args, r4d_ihc_bf16.hip): the plain row
// and the all-reduce twin read the same list.
struct R4dIhcJunction {
    uint16_t* a = nullptr;
    const uint16_t* b = nullptr;
    JuncRow r{};
    long long M = 0, ap = 0, bp = 0, pp = 0, hp = 0, xnp = 0, yp = 0, xp = 0;
    int hc = 0;
    bool post = false, tail = false;
    bool eg = false;        // the Engram form: the gate between the write and the sums
    bool eg_w_bf16 = false; // its four weights are bf16 (else f32)
};

}  // namespace r4d_ihc

struct RadArgs;
/* RAD_OK or the RAD_E_* the operands earn; `o` filled on RAD_OK. */
int r4d_ihc_junction_args(const RadArgs* a, r4d_ihc::R4dIhcJunction* o);
