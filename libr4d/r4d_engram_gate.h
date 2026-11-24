// r4d_engram_gate.h -- the Engram gate's token row, shared by engram_gate (r4d_engram_bf16.hip)
// and the junction form that runs it between ihc_post and ihc_pre (rmsnorm_ihc_junction's `engram`
// param; r4d_ihc.h's junction_row). Both run one partition: the block's first kThreads threads own
// the columns t, t + kThreads, ... and the totals are the first kWaves waves' in wave order. A
// block larger than kThreads (the junction's 512) puts its spare waves in the barriers and adds
// nothing, so a fused row's h is engram_gate's bytes whatever the workgroup's size.
//
//   main = w_val * val * rsqrt(mean(val^2) + eps); per stream c:
//   kn = w_key*key*rsqrt(..), qn = w_query*h_c*rsqrt(..), d = (sum kn*qn) n^-0.5,
//   g = sigmoid(sign(d) sqrt(max(|d|, 1e-6))), h_c += bf16(g*main + w_branch*main)
#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

#include "r4d_common.h"

namespace r4d_eg {

enum { kThreads = 256, kWaves = kThreads / 32 };

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float w(const uint16_t* p, uint32_t i) { return bf16_to_f32(p[i]); }
__device__ __forceinline__ float w(const float* p, uint32_t i) { return p[i]; }

// A workgroup-wide sum of K values: lanes through r4d_xor_sum, waves through LDS in wave order.
// Every thread returns the totals. `on` is wave-uniform: a masked row's waves are all off, a fused
// block's spare waves are off, and both still reach the barriers. The trailing barrier lets the
// array be reused.
template <int K>
__device__ __forceinline__ void block_sum(float (&v)[K], float (*part)[K], bool on) {
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    if (on) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float s = r4d_xor_sum<32>(v[k]);
            if (lane == 0) part[wave][k] = s;
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) {
        float s = 0.0f;
        for (int wv = 0; wv < kWaves; ++wv) s += part[wv][k];
        v[k] = s;
    }
    __syncthreads();
}

// One token row: `kr` the row's hc streams of key, `vr` its val, `hr` its hc streams of h, updated
// in place; `wk`/`wq`/`wb` are hc * n wide, `wv` n wide, all WT (bf16 or f32). `part1` is
// kWaves x (1 + 2 hc) and `part2` kWaves x hc of LDS. Every thread of the block calls it.
template <int HC, typename WT>
__device__ __forceinline__ void row(const uint16_t* __restrict__ kr, const uint16_t* __restrict__ vr,
                                    const WT* __restrict__ wk, const WT* __restrict__ wq,
                                    const WT* __restrict__ wv, const WT* __restrict__ wb,
                                    uint16_t* __restrict__ hr, uint32_t n, float eps,
                                    float inv_sqrt_n, float (*part1)[1 + 2 * HC],
                                    float (*part2)[HC], bool on) {
    constexpr int K1 = 1 + 2 * HC;
    const uint32_t t = threadIdx.x;

    // The squares: val, then key and h of each stream.
    float s1[K1];
#pragma unroll
    for (int k = 0; k < K1; ++k) s1[k] = 0.0f;
    if (on) {
        for (uint32_t i = t; i < n; i += kThreads) {
            const float v = bf16_to_f32(vr[i]);
            s1[0] += v * v;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const float kv = bf16_to_f32(kr[(uint32_t)c * n + i]);
                const float hv = bf16_to_f32(hr[(uint32_t)c * n + i]);
                s1[1 + 2 * c] += kv * kv;
                s1[2 + 2 * c] += hv * hv;
            }
        }
    }
    block_sum<K1>(s1, part1, on);
    const float inv_val = 1.0f / sqrtf(s1[0] / (float)n + eps);
    float ik[HC], iq[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        ik[c] = 1.0f / sqrtf(s1[1 + 2 * c] / (float)n + eps);
        iq[c] = 1.0f / sqrtf(s1[2 + 2 * c] / (float)n + eps);
    }

    // The dots of the normed key and query.
    float dot[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) dot[c] = 0.0f;
    if (on) {
        for (uint32_t i = t; i < n; i += kThreads) {
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const uint32_t j = (uint32_t)c * n + i;
                const float kn = w(wk, j) * bf16_to_f32(kr[j]) * ik[c];
                const float qn = w(wq, j) * bf16_to_f32(hr[j]) * iq[c];
                dot[c] += kn * qn;
            }
        }
    }
    block_sum<HC>(dot, part2, on);
    float gate[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float d = dot[c] * inv_sqrt_n;
        const float sgn = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
        gate[c] = sigmoid(sgn * sqrtf(fmaxf(fabsf(d), 1e-6f)));
    }

    // The update, rounded once to bf16 and added to the stream, which rounds again.
    if (on) {
        for (uint32_t i = t; i < n; i += kThreads) {
            const float mn = w(wv, i) * bf16_to_f32(vr[i]) * inv_val;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                const uint32_t j = (uint32_t)c * n + i;
                const float upd = bf16_to_f32(f32_to_bf16(
                    __fadd_rn(__fmul_rn(gate[c], mn), __fmul_rn(w(wb, j), mn))));
                hr[j] = f32_to_bf16(__fadd_rn(bf16_to_f32(hr[j]), upd));
            }
        }
    }
}

}  // namespace r4d_eg
