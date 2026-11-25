// The compressed indexer group key's arithmetic (Hy4-flash, docs/OPS.md "idx_pool_gated",
// "qsa_work_pool_tail"): one source for the two kernels that produce the key, so both write the
// same bytes.
//
//   val_s[c] = sum_j wv[c][j] * x_s[j]              gat_s[c] = sum_j wg[c][j] * x_s[j] + ape[s][c]
//   p_s[c]   = exp(gat_s[c] - max_s gat) / sum_s exp(..)         s ascending
//   y[c]     = sum_s val_s[c] * p_s[c]
//   z[c]     = bf16( gain[c] * (y[c] * rsqrt(sum_c y^2 / n + eps)) )      the sum in f64
//   then the last `rotary_dim` channels of z, adjacent pairs, rotated at the group's first position
//   and rounded again.
//
// The caller owns the raw keys: the pool kernel (r4d_idx_pool_gated_f32.hip) reads its `k` plane of
// gathered rows, qsa_work_pool_tail (r4d_qsa_work_pool_tail_bf16.hip) has the group's gather do it
// in place. Either way the keys are the bf16 raw keys widened to f32 in LDS -- the same values, so
// the two write the same key -- and the group's page and first position arrive the same way.
//
// One thread a channel, 128 threads; a workgroup pools GPW groups so one read of a weight row serves
// GPW groups (the 128 KiB of weights are the cost, not the arithmetic). Each thread's dot products
// run j ascending with a separate multiply and add (the unit is built -ffp-contract=off), the exp,
// the trig and the sum of squares are f64 and the norm's sum runs in channel order on one thread per
// group, so the result is byte-equal to libref: the order of every sum is libref's.

#ifndef R4D_IDX_POOL_BODY_H
#define R4D_IDX_POOL_BODY_H

#include "r4d_common.h"
#include "r4d_fwht.h"
#include "r4d_q8.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kN       = 128;
constexpr int kRatio   = 4;
constexpr int kThreads = kN;

/* Rotary component of pair j in imrope_gptj with sections (t, h, w): the h position at j % 3 == 1
 * below 3 * h, the w position at j % 3 == 2 below 3 * w, else t (libref's imrope_component). */
__device__ __forceinline__ int imrope_comp(int j, int sec_h, int sec_w) {
    if (j % 3 == 1 && j < 3 * sec_h) return 1;
    if (j % 3 == 2 && j < 3 * sec_w) return 2;
    return 0;
}

#define IDX_POOL_MATH_PARAMS                                                                    \
    const float (&xs)[GPW][kRatio][kN], const int32_t (&ps)[GPW][3], const int32_t (&pw)[GPW],   \
    const float* __restrict__ wv, const float* __restrict__ wg, const float* __restrict__ ape,  \
    const void* __restrict__ gain, uint16_t* __restrict__ cache, uint16_t* __restrict__ out,    \
    int c, int g0, int W, int out_ld, long cache_pitch, int pages, int rd, float eps,           \
    double theta, int sec_h, int sec_w, int gain_bf16, uint8_t* __restrict__ q8,                \
    float* __restrict__ qs, long q_ld, long qs_ld, float norm, int i8
/* FP8 (idx_pool_gated_fp8): nothing is rounded to bf16; the f32 key goes through the 128-wide
 * Hadamard (r4d_fwht_wave, a wave a group, four consecutive channels a lane), times `norm`, and into
 * E4M3 codes at a power-of-two scale: dsa_pool's packed page (`cache` as bytes, the scale at byte
 * 128) and/or q8/qs.
 *
 * `xs` is the caller's LDS block of raw keys (thread c holds channel c of every row, a group past W
 * holds zeros), `ps` the group's position -- its one component, or imrope's three -- and `pw` the
 * group's page, -1 for a group with nothing to write. */
template <int GPW, bool IMROPE, bool FP8>
__device__ __forceinline__ void idx_pool_math(IDX_POOL_MATH_PARAMS) {

    __shared__ float ys[GPW][kN];
    __shared__ float rs[GPW];

    float val[GPW][kRatio], gat[GPW][kRatio];
#pragma unroll
    for (int g = 0; g < GPW; ++g)
#pragma unroll
        for (int s = 0; s < kRatio; ++s) { val[g][s] = 0.0f; gat[g][s] = 0.0f; }

    /* Row c of both weights, j ascending; every x is an LDS broadcast. */
    const float4* wvr = (const float4*)(wv + (size_t)c * kN);
    const float4* wgr = (const float4*)(wg + (size_t)c * kN);
    for (int j4 = 0; j4 < kN / 4; ++j4) {
        const float4 a = wvr[j4], b = wgr[j4];
#pragma unroll
        for (int g = 0; g < GPW; ++g)
#pragma unroll
            for (int s = 0; s < kRatio; ++s) {
                const float4 x = *(const float4*)&xs[g][s][4 * j4];
                float v = val[g][s], t = gat[g][s];
                v = v + a.x * x.x;  t = t + b.x * x.x;
                v = v + a.y * x.y;  t = t + b.y * x.y;
                v = v + a.z * x.z;  t = t + b.z * x.z;
                v = v + a.w * x.w;  t = t + b.w * x.w;
                val[g][s] = v; gat[g][s] = t;
            }
    }

    /* The softmax over the slots and the pooled value, per group. */
#pragma unroll
    for (int g = 0; g < GPW; ++g) {
        float gs[kRatio];
#pragma unroll
        for (int s = 0; s < kRatio; ++s) gs[s] = gat[g][s] + ape[s * kN + c];
        float mx = -INFINITY;
#pragma unroll
        for (int s = 0; s < kRatio; ++s) mx = mx < gs[s] ? gs[s] : mx;
        float e[kRatio], den = 0.0f;
#pragma unroll
        for (int s = 0; s < kRatio; ++s) {
            e[s] = (float)exp((double)(gs[s] - mx));
            den = den + e[s];
        }
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < kRatio; ++s) {
            const float p = __fdiv_rn(e[s], den);
            acc = acc + val[g][s] * p;
        }
        ys[g][c] = acc;
    }
    __syncthreads();

    /* The rotation's factors depend on the group's position and this channel alone, so they are
     * taken here: the f64 sum below runs on GPW threads and no others, so the other threads' pow,
     * cos and sin go in its latency instead of standing behind it, and each thread keeps its
     * groups' cosine and sine for the store. */
    const int rc = c - (kN - rd);
    float rcs[GPW], rsn[GPW];
#pragma unroll
    for (int g = 0; g < GPW; ++g) {
        rcs[g] = 1.0f;
        rsn[g] = 0.0f;
    }
    if (rc >= 0) {
        const int i = rc >> 1;
        const int comp = IMROPE ? imrope_comp(i, sec_h, sec_w) : 0;
        const float inv_freq = (float)(1.0 / pow(theta, (double)(2 * i) / (double)rd));
#pragma unroll
        for (int g = 0; g < GPW; ++g) {
            const int w = g0 + g;
            if (w < W) {
                const int pv = IMROPE ? ps[g][comp] : ps[g][0];
                const float ang = (float)pv * inv_freq;
                rcs[g] = (float)cos((double)ang);
                rsn[g] = (float)sin((double)ang);
            }
        }
    }

    /* The mean square, in channel order and in f64, one thread a group: libref's sum exactly. */
    if (c < GPW) {
        double sq = 0.0;
        for (int i = 0; i < kN; ++i) sq += (double)ys[c][i] * (double)ys[c][i];
        rs[c] = __fdiv_rn(1.0f, __fsqrt_rn((float)(sq / (double)kN) + eps));
    }
    __syncthreads();

    float z[GPW];
    const float gc = gain_bf16 ? bf16_to_f32(((const uint16_t*)gain)[c]) : ((const float*)gain)[c];
#pragma unroll
    for (int g = 0; g < GPW; ++g)
        z[g] = FP8 ? gc * (ys[g][c] * rs[g]) : bf16_to_f32(f32_to_bf16(gc * (ys[g][c] * rs[g])));
#pragma unroll
    for (int g = 0; g < GPW; ++g) ys[g][c] = z[g];
    __syncthreads();

    /* The rotation: channel c of the last rd takes pair i = (c - (n - rd)) / 2 and computes its own
     * half of the pair from the two rounded values, with the cosine and sine taken above. */
#pragma unroll
    for (int g = 0; g < GPW; ++g) {
        const int w = g0 + g;
        float r = z[g];
        if (rc >= 0 && w < W) {
            const float cs = rcs[g], sn = rsn[g];
            const float x0 = ys[g][c & ~1], x1 = ys[g][(c & ~1) | 1];
            if (FP8) r = (rc & 1) ? x1 * cs + x0 * sn : x0 * cs - x1 * sn;
            else
                r = (rc & 1) ? bf16_to_f32(f32_to_bf16(x1 * cs + x0 * sn))
                             : bf16_to_f32(f32_to_bf16(x0 * cs - x1 * sn));
        }
        if (FP8) {
            z[g] = r;
            continue;
        }
        if (w < W) {
            const uint16_t h = f32_to_bf16(r);
            if (out) out[(size_t)w * out_ld + c] = h;
            if (cache) {
                const int pg = pw[g];
                if (pg >= 0 && pg < pages) cache[(size_t)pg * cache_pitch + c] = h;
            }
        }
    }
    if (!FP8) return;
    /* Every pair has read both of its rotated inputs: the f32 keys replace them in LDS. */
    __syncthreads();
#pragma unroll
    for (int g = 0; g < GPW; ++g) ys[g][c] = z[g];
    __syncthreads();
    const int lane = c & 31, wave = c >> 5;
    for (int g = wave; g < GPW; g += kThreads / 32) {
        const int w = g0 + g;
        if (w >= W) continue;
        float v[4];
#pragma unroll
        for (int d = 0; d < 4; ++d) v[d] = ys[g][lane * 4 + d];
        r4d_fwht_wave<kN>(v, lane);
        float am = 1e-4f;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            v[d] = v[d] * norm;
            am = fmaxf(am, fabsf(v[d]));
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) am = fmaxf(am, __shfl_xor(am, mask, 32));
        /* i8 (idx_pool_gated_i8): symmetric int8 codes at an f32 scale amax / 127, else E4M3 at a
         * power-of-two scale. */
        float sc;
        uint32_t packed;
        if (i8) {
            sc = am * (1.0f / 127.0f);
            packed = q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / sc);
        } else {
            const uint32_t bits = __float_as_uint(am * (1.0f / 448.0f));
            sc = __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0));
            packed = q8_pack4<false>(v[0], v[1], v[2], v[3], 1.0f / sc);
        }
        if (q8) {
            uint8_t* dst = q8 + (size_t)w * q_ld + lane * 4;
            if (((uintptr_t)dst & 3) == 0)
                *(uint32_t*)dst = packed;
            else
#pragma unroll
                for (int d = 0; d < 4; ++d) dst[d] = (uint8_t)(packed >> (8 * d));
            if (!lane) qs[(size_t)w * qs_ld] = sc;
        }
        if (cache) {
            const int pg = pw[g];
            if (pg >= 0 && pg < pages) {
                uint8_t* dst = (uint8_t*)cache + (size_t)pg * cache_pitch;
                *(uint32_t*)(dst + lane * 4) = packed;
                if (!lane) *(float*)(dst + 128) = sc;
            }
        }
    }
}
}  // namespace

#endif  // R4D_IDX_POOL_BODY_H
