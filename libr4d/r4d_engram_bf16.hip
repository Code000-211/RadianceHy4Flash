// r4d_engram_bf16.hip -- Hy4-flash's Engram: engram_ids, engram_norm, engram_gate.
//
//   engram_ids   c_k = map[t_k] (the pad id before the document start, and for an id outside the
//                map), m_k = c_k * mult[k] in wrapping int64, mix_g = m_0 ^ .. ^ m_{g-1}, head i of
//                order g: floor_mod(mix_g, vocab_sizes[i]) + offsets[i]; mask = tok not in
//                {pad, image, video}. `state` is ngram_ids' window of committed ids. With
//                `ahead_ids`, `tok` runs on past the step's rows into the last sequence's following
//                prompt, whose ids (no mask) go there: the read-ahead hint of a host gather. The
//                caller's cu ends at the step's last row, the last sequence's (the host libraries
//                check it; here it is the contract).
//   engram_norm  y[i] = bf16(gain[i] * x[i] * rsqrt(mean_head(x^2) + eps)), one rms a head
//   engram_gate  main = w_val * val * rsqrt(mean(val^2) + eps); per stream c:
//                kn = w_key*key*rsqrt(..), qn = w_query*h_c*rsqrt(..), d = (sum kn*qn) n^-0.5,
//                g = sigmoid(sign(d) sqrt(max(|d|, 1e-6))), h_c += bf16(g*main + w_branch*main);
//                rows with mask 0 are untouched
//
// libref/ref_hy4.cpp is the oracle, kernels/engram.md the design; written for this repository.
//
// engram_ids is two launches. The first, one thread a (row, head), reads the window `state` as it
// was before the step and writes ids and mask. The second, one thread a sequence, rewrites the
// window by ple_conv's rule (a decode step, `num_accepted` present: the old ids shift down one and
// every id of the step is appended; otherwise the last ngram-1 ids land at offset zero). Integer
// results are exact: ids and masks are byte-identical to libref's.
//
// engram_gate: one workgroup of 256 threads a token row, every reduction a workgroup sum; three
// phases (the squares of val, key and h per stream; the dots; the update), each reading its
// operands again from the cache. The sums are f32 (libref's are double), so the gate agrees to
// rounding: d to 1e-5 relative, h within one bf16 ulp of the update's size. LDS under 1 KiB.
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include <cmath>
#include <cstdint>

namespace {

enum { kEgThreads = 256, kEgWaves = kEgThreads / 32 };
/* The gate's rows a workgroup at prefill: its four weight vectors are the same for every
 * row of a chunk, so the block holds them over its rows (decode keeps a row a block). */
enum { kEgRows = 4 };

__device__ __forceinline__ float eg_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ int64_t eg_floor_mod(int64_t a, int64_t b) {
    const int64_t r = a % b;
    return r < 0 ? r + b : r;
}

// The sequence a row belongs to: the last s with cu[s] <= m. -1 past the last row.
__device__ __forceinline__ int32_t eg_seq_of(const int32_t* __restrict__ cu, int32_t n_seq,
                                             int32_t m) {
    if (m < cu[0] || m >= cu[n_seq]) return -1;
    int32_t lo = 0, hi = n_seq;  // cu[lo] <= m < cu[hi]
    while (hi - lo > 1) {
        const int32_t mid = (lo + hi) >> 1;
        if (cu[mid] <= m) lo = mid;
        else hi = mid;
    }
    return lo;
}

struct EngramIds {
    const int32_t* tok;
    const int32_t* state;
    const int32_t* cu;
    const int32_t* pos;
    const int32_t* map;
    const int64_t* mult;
    const int64_t* vsz;
    const int64_t* off;
    const int32_t* nacc;   // may be null
    const int32_t* sidx;   // may be null
    int32_t* ahead;        // may be null: the following prompt's ids, A rows past the step's M
    int32_t M, A, heads, ngram, pad, msize, image, video, n_seq, W;
    uint64_t sp, ip, ap;   // the window's row pitch, ids' and ahead's row pitch
};

template <bool I64>
__device__ __forceinline__ void engram_ids_body(EngramIds p, void* __restrict__ ids_,
                                                int32_t* __restrict__ mask) {
    const uint64_t g = (uint64_t)blockIdx.x * 256u + threadIdx.x;
    if (g >= (uint64_t)(p.M + p.A) * (uint64_t)p.heads) return;
    const int32_t m = (int32_t)(g / (uint32_t)p.heads), head = (int32_t)(g % (uint32_t)p.heads);
    // A row past the step's is the last sequence's following prompt (`ahead`).
    const bool ah = m >= p.M;
    const int32_t s = ah ? p.n_seq - 1 : eg_seq_of(p.cu, p.n_seq, m);
    if (s < 0) return;
    const int32_t slot = p.sidx ? p.sidx[s] : s;
    if (slot < 0) return;
    const int32_t ctx = p.ngram - 1, per = p.heads / ctx;
    const int32_t o = p.nacc ? p.nacc[s] - 1 : 0;
    if (o < 0 || o + ctx > p.W) return;
    const int32_t bos = p.cu[s], t = m - bos;
    const int32_t* __restrict__ st = p.state + (uint64_t)slot * p.sp + o;
    const int32_t order = head / per + 1;   // the head's mix XORs shifts 0 .. order
    const int32_t pm = ah ? p.pos[p.M - 1] + 1 + (m - p.M) : p.pos[m];
    const int64_t map_pad = p.map[p.pad];
    int64_t mix = 0;
    for (int32_t k = 0; k <= order; ++k) {
        int64_t c = map_pad;
        if (pm >= k) {
            const int32_t q = ctx + t - k;
            const int32_t tk = q < ctx ? st[q] : p.tok[bos + q - ctx];
            if (tk >= 0 && tk < p.msize) c = p.map[tk];
        }
        const int64_t x = (int64_t)((uint64_t)c * (uint64_t)p.mult[k]);
        mix = k == 0 ? x : (mix ^ x);
    }
    const int64_t vs = p.vsz[head];
    const int64_t id = (vs > 0 ? eg_floor_mod(mix, vs) : 0) + p.off[head];
    if (ah) {
        p.ahead[(uint64_t)(m - p.M) * p.ap + head] = (int32_t)id;
        return;
    }
    if (I64) ((int64_t*)ids_)[(uint64_t)m * p.ip + head] = id;
    else     ((int32_t*)ids_)[(uint64_t)m * p.ip + head] = (int32_t)id;
    if (head == 0) {
        const int32_t t0 = p.tok[m];
        mask[m] = t0 != p.pad && t0 != p.image && t0 != p.video ? 1 : 0;
    }
}

// CH `r4dchain`: the chained form (r4d_chain.h). Each kernel of this file keeps its body (early
// returns and all) in a device function and the wrapper brackets it, so every block counts once.
template <bool I64, class CH>
__global__ __launch_bounds__(256) void engram_ids_kernel(EngramIds p, void* __restrict__ ids_,
                                                         int32_t* __restrict__ mask) {
    if (CH::on) r4d_chain_wait();
    engram_ids_body<I64>(p, ids_, mask);
    if (CH::on) r4d_chain_done();
}

// The window after the step; one thread a sequence, the old ids copied out before any is written.
__device__ __forceinline__ void engram_state_body(EngramIds p) {
    const int32_t s = (int32_t)(blockIdx.x * 64u + threadIdx.x);
    if (s >= p.n_seq) return;
    const int32_t bos = p.cu[s], T = p.cu[s + 1] - bos;
    if (T <= 0) return;
    const int32_t slot = p.sidx ? p.sidx[s] : s;
    if (slot < 0) return;
    const int32_t ctx = p.ngram - 1;
    const int32_t o = p.nacc ? p.nacc[s] - 1 : 0;
    if (o < 0 || o + ctx > p.W || (p.nacc && ctx - 1 + T > p.W)) return;
    int32_t* __restrict__ sb = const_cast<int32_t*>(p.state) + (uint64_t)slot * p.sp;
    /* The new window's first ctx entries, every old id read before any is written. Constant
     * indices only, so they stay in registers (a dynamic index would spill them to scratch). */
    int32_t nv[7];
#pragma unroll
    for (int32_t i = 0; i < 7; ++i) {
        nv[i] = 0;
        if (i < ctx) {
            if (p.nacc) {   // the old ids shifted down one; the last entry is the step's first id
                nv[i] = i + 1 < ctx ? sb[o + i + 1] : p.tok[bos];
            } else {        // the last ctx ids of (old window, step)
                const int32_t src = T - ctx + i;
                nv[i] = src >= 0 ? p.tok[bos + src] : sb[o + T + i];
            }
        }
    }
#pragma unroll
    for (int32_t i = 0; i < 7; ++i)
        if (i < ctx) sb[i] = nv[i];
    if (p.nacc)
        for (int32_t t = 1; t < T; ++t) sb[ctx - 1 + t] = p.tok[bos + t];
}

template <class CH>
__global__ __launch_bounds__(64) void engram_state_kernel(EngramIds p) {
    if (CH::on) r4d_chain_wait();
    engram_state_body(p);
    if (CH::on) r4d_chain_done();
}

// A gain or gate weight as f32: bf16 (the checkpoint's) or f32 (the plugin's declared form).
__device__ __forceinline__ float eg_w(const uint16_t* p, uint32_t i) { return bf16_to_f32(p[i]); }
__device__ __forceinline__ float eg_w(const float* p, uint32_t i) { return p[i]; }

/* Elements a lane of the norm's wave keeps in registers: a head up to 256 wide, one read. */
enum { kEgNormKeep = 8 };

// One wave a (row, head), eight to a workgroup.
template <typename WT>
__device__ __forceinline__ void engram_norm_body(
    const uint16_t* __restrict__ e, const WT* __restrict__ gain, uint16_t* __restrict__ y,
    uint32_t M, uint32_t heads, uint32_t n, float eps, uint64_t ep, uint64_t yp) {
    const uint64_t job = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5);
    const uint32_t lane = threadIdx.x & 31;
    if (job >= (uint64_t)M * heads) return;
    const uint64_t m = job / heads, hd = job % heads;
    const uint16_t* __restrict__ er = e + m * ep + hd * n;
    /* The head is 96 wide in this model, three elements a lane, so the row stays in registers
     * between the sum and the scale instead of being read off the card twice (and, when the rows
     * are in the host arena, over the link twice). A head wider than kEgNormKeep lanes' worth reads
     * its row again, as before. */
    const bool keep_row = n <= (uint32_t)(32 * kEgNormKeep);
    float keep[kEgNormKeep];
    float sq = 0.0f;
    if (keep_row) {
#pragma unroll
        for (int k = 0; k < kEgNormKeep; ++k) {
            keep[k] = 0.0f;
            const uint32_t i = lane + 32u * (uint32_t)k;
            if (i < n) {
                keep[k] = bf16_to_f32(er[i]);
                sq += keep[k] * keep[k];
            }
        }
    } else {
        for (uint32_t i = lane; i < n; i += 32) {
            const float v = bf16_to_f32(er[i]);
            sq += v * v;
        }
    }
    sq = r4d_xor_sum<32>(sq);
    const float inv = 1.0f / sqrtf(sq / (float)n + eps);
    uint16_t* __restrict__ yr = y + m * yp + hd * n;
    const WT* __restrict__ gr = gain + hd * n;
    if (keep_row) {
#pragma unroll
        for (int k = 0; k < kEgNormKeep; ++k) {
            const uint32_t i = lane + 32u * (uint32_t)k;
            if (i < n) yr[i] = f32_to_bf16(eg_w(gr, i) * keep[k] * inv);
        }
    } else {
        for (uint32_t i = lane; i < n; i += 32)
            yr[i] = f32_to_bf16(eg_w(gr, i) * bf16_to_f32(er[i]) * inv);
    }
}

template <typename WT, class CH>
__global__ __launch_bounds__(256) void engram_norm_kernel(
    const uint16_t* __restrict__ e, const WT* __restrict__ gain, uint16_t* __restrict__ y,
    uint32_t M, uint32_t heads, uint32_t n, float eps, uint64_t ep, uint64_t yp) {
    if (CH::on) r4d_chain_wait();
    engram_norm_body<WT>(e, gain, y, M, heads, n, eps, ep, yp);
    if (CH::on) r4d_chain_done();
}

// A workgroup-wide sum of K values: lanes through r4d_xor_sum, waves through LDS in wave order.
// Every thread returns the totals. The trailing barrier lets the array be reused.
template <int K>
__device__ __forceinline__ void eg_block_sum(float (&v)[K], float (*part)[K]) {
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const float s = r4d_xor_sum<32>(v[k]);
        if (lane == 0) part[wave][k] = s;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) {
        float s = 0.0f;
        for (int w = 0; w < kEgWaves; ++w) s += part[w][k];
        v[k] = s;
    }
    __syncthreads();
}

// One workgroup R rows: one at a decode height, where a row a block is what keeps every block busy,
// and kEgRows at a prefill chunk's, where the four weight vectors are the same for every row of the
// block, so holding them in registers over its rows reads them once instead of once a row (half of
// the op's traffic). R is a constant so the rows' sums stay in registers; the per-row sums keep
// their order (i ascending a thread, then the lanes' tree, then the waves), so the results are the
// one-row kernel's, bit for bit.
template <int HC, typename WT, int R>
__device__ __forceinline__ void engram_gate_body(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ val,
    const int32_t* __restrict__ mask, const WT* __restrict__ wk,
    const WT* __restrict__ wq, const WT* __restrict__ wv,
    const WT* __restrict__ wb, uint16_t* __restrict__ h, uint32_t n, float eps,
    float inv_sqrt_n, uint64_t kp, uint64_t vp, uint64_t hp, uint32_t M) {
    constexpr int K1 = 1 + 2 * HC;
    constexpr int KD = HC * R;
    __shared__ float part1[kEgWaves][K1];
    __shared__ float part2[kEgWaves][KD];
    __shared__ float invs[R][K1];   /* a row's inverses: val, then key and h a stream */
    const uint64_t row0 = (uint64_t)blockIdx.x * (uint64_t)R;
    const uint32_t t = threadIdx.x;
    /* The block's rows, and the mask's verdict on each: a row past M is not one of the batch's.
     * `ri` keeps the mask read on a row that exists, whichever way the check falls. */
    bool act[R], live = false;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const uint64_t rw = row0 + (uint64_t)r, ri = rw < (uint64_t)M ? rw : 0;
        act[r] = rw < (uint64_t)M && mask[ri] != 0;
        live = live || act[r];
    }
    if (!live) return;

    // The squares: val, then key and h of each stream, a row at a time.
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s1[K1];
#pragma unroll
        for (int k = 0; k < K1; ++k) s1[k] = 0.0f;
        if (act[r]) {
            const uint16_t* __restrict__ kr = key + (row0 + (uint64_t)r) * kp;
            const uint16_t* __restrict__ vr = val + (row0 + (uint64_t)r) * vp;
            const uint16_t* __restrict__ hr = h + (row0 + (uint64_t)r) * hp;
            for (uint32_t i = t; i < n; i += kEgThreads) {
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
        eg_block_sum<K1>(s1, part1);
        if (t == 0) {
#pragma unroll
            for (int k = 0; k < K1; ++k) invs[r][k] = 1.0f / sqrtf(s1[k] / (float)n + eps);
        }
    }
    __syncthreads();   /* the sums are a wave's lane 0's; every thread reads invs below */

    // The dots of the normed key and query: a stream's two weight vectors are read once for the
    // block's rows.
    float dot[KD];
#pragma unroll
    for (int k = 0; k < KD; ++k) dot[k] = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        float ik[R], iq[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            ik[r] = invs[r][1 + 2 * c];
            iq[r] = invs[r][2 + 2 * c];
        }
        for (uint32_t i = t; i < n; i += kEgThreads) {
            const uint32_t j = (uint32_t)c * n + i;
            const float wkk = eg_w(wk, j), wqq = eg_w(wq, j);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (!act[r]) continue;
                const float kn = wkk * bf16_to_f32(key[(row0 + (uint64_t)r) * kp + j]) * ik[r];
                const float qn = wqq * bf16_to_f32(h[(row0 + (uint64_t)r) * hp + j]) * iq[r];
                dot[r * HC + c] += kn * qn;
            }
        }
    }
    eg_block_sum<KD>(dot, part2);
    float gate[KD];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const float d = dot[r * HC + c] * inv_sqrt_n;
            const float sgn = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
            gate[r * HC + c] = eg_sigmoid(sgn * sqrtf(fmaxf(fabsf(d), 1e-6f)));
        }
    }

    // The update, rounded once to bf16 and added to the stream, which rounds again. val's weight
    // and the branch weights are read once for the block's rows, as above.
    for (uint32_t i = t; i < n; i += kEgThreads) {
        const float wvi = eg_w(wv, i);
        float mn[R];
#pragma unroll
        for (int r = 0; r < R; ++r)
            mn[r] = act[r] ? wvi * bf16_to_f32(val[(row0 + (uint64_t)r) * vp + i]) * invs[r][0]
                           : 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const uint32_t j = (uint32_t)c * n + i;
            const float wbj = eg_w(wb, j);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (!act[r]) continue;
                const float upd = bf16_to_f32(f32_to_bf16(
                    __fadd_rn(__fmul_rn(gate[r * HC + c], mn[r]), __fmul_rn(wbj, mn[r]))));
                uint16_t* __restrict__ hr = h + (row0 + (uint64_t)r) * hp;
                hr[j] = f32_to_bf16(__fadd_rn(bf16_to_f32(hr[j]), upd));
            }
        }
    }
}

template <int HC, typename WT, class CH, int R>
__global__ __launch_bounds__(kEgThreads) void engram_gate_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ val,
    const int32_t* __restrict__ mask, const WT* __restrict__ wk,
    const WT* __restrict__ wq, const WT* __restrict__ wv,
    const WT* __restrict__ wb, uint16_t* __restrict__ h, uint32_t n, float eps,
    float inv_sqrt_n, uint64_t kp, uint64_t vp, uint64_t hp, uint32_t M) {
    if (CH::on) r4d_chain_wait();
    engram_gate_body<HC, WT, R>(key, val, mask, wk, wq, wv, wb, h, n, eps, inv_sqrt_n, kp, vp, hp,
                                M);
    if (CH::on) r4d_chain_done();
}

bool eg_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
bool eg_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
bool eg_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }
bool eg_i64(const RadTensor* t) { return t && t->dtype == RAD_I64; }

}  // namespace

extern "C" int r4d_rad_engram_ids(const RadArgs* a, RadStream s) {
    enum { TOK = 0, STATE, CU, POS, MAP, MULT, VSZ, OFF, NACC, SIDX, IDS, MASK, AHEAD };
    const RadTensor *tok = r4d_opt(a, TOK), *st = r4d_opt(a, STATE), *cu = r4d_opt(a, CU),
                    *pos = r4d_opt(a, POS), *map = r4d_opt(a, MAP), *mult = r4d_opt(a, MULT),
                    *vsz = r4d_opt(a, VSZ), *off = r4d_opt(a, OFF), *nacc = r4d_opt(a, NACC),
                    *sidx = r4d_opt(a, SIDX), *ids = r4d_opt(a, IDS), *mask = r4d_opt(a, MASK),
                    *ahead = a->n_t > AHEAD ? r4d_opt(a, AHEAD) : nullptr;
    if (!tok || !st || !cu || !pos || !map || !mult || !vsz || !off || !ids || !mask)
        return R4D_NO(RAD_E_INVAL);
    if (!eg_i32(tok) || !eg_i32(st) || !eg_i32(cu) || !eg_i32(pos) || !eg_i32(map) ||
        !eg_i32(mask) || (nacc && !eg_i32(nacc)) || (sidx && !eg_i32(sidx)) || !eg_i64(mult) ||
        !eg_i64(vsz) || !eg_i64(off) || (ids->dtype != RAD_I32 && ids->dtype != RAD_I64))
        return R4D_NO(RAD_E_DTYPE);
    const long long heads = r4d_geti_or(a, "heads", 0), ngram = r4d_geti_or(a, "ngram", 0),
                    pad = r4d_geti_or(a, "pad", -1), msize = r4d_geti_or(a, "map_size", 0),
                    image = r4d_geti_or(a, "image", -1), video = r4d_geti_or(a, "video", -1);
    if (heads < 1 || ngram < 2 || ngram > 8 || heads % (ngram - 1) || msize < 1 ||
        msize > INT32_MAX)
        return R4D_NO(RAD_E_SHAPE);
    if (pad < 0 || pad >= msize) return R4D_NO(RAD_E_INVAL);
    const long long ctx = ngram - 1;
    /* `tok` holds the step's rows, then the A tokens whose ids go to `ahead` (i32 [A, heads]). */
    const long long A = ahead ? r4d_numel(ahead) / heads : 0;
    if (ahead && (!eg_i32(ahead) || ahead->rank != 2 || ahead->shape[1] != heads ||
                  ahead->stride[1] != 1))
        return R4D_NO(RAD_E_SHAPE);
    const long long M = r4d_numel(tok) - A;
    if (tok->rank < 1 || M < 1 || M > INT32_MAX / 64 || !r4d_contig(tok) || !r4d_contig(cu) ||
        cu->rank != 1 || cu->shape[0] < 2 || r4d_numel(pos) < M || r4d_numel(map) < msize ||
        r4d_numel(mult) < ngram || r4d_numel(vsz) < heads || r4d_numel(off) < heads ||
        r4d_numel(mask) < M || st->rank < 2 || st->rank > 3 || (st->rank == 3 && st->shape[1] != 1) ||
        st->shape[st->rank - 1] < ctx || st->stride[st->rank - 1] != 1 ||
        ids->rank != 2 || ids->shape[0] < M || ids->shape[1] != heads || ids->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(pos) || !r4d_contig(map) || !r4d_contig(mult) || !r4d_contig(vsz) ||
        !r4d_contig(off) || !r4d_contig(mask) || (nacc && !r4d_contig(nacc)) ||
        (sidx && !r4d_contig(sidx)))
        return R4D_NO(RAD_E_STRIDE);
    EngramIds p{};
    p.tok = (const int32_t*)tok->data;
    p.state = (const int32_t*)st->data;
    p.cu = (const int32_t*)cu->data;
    p.pos = (const int32_t*)pos->data;
    p.map = (const int32_t*)map->data;
    p.mult = (const int64_t*)mult->data;
    p.vsz = (const int64_t*)vsz->data;
    p.off = (const int64_t*)off->data;
    p.nacc = nacc ? (const int32_t*)nacc->data : nullptr;
    p.sidx = sidx ? (const int32_t*)sidx->data : nullptr;
    p.ahead = A > 0 ? (int32_t*)ahead->data : nullptr;
    p.A = A > 0 ? (int32_t)A : 0;
    p.ap = A > 0 ? (uint64_t)ahead->stride[0] : 0;
    p.M = (int32_t)M;
    p.heads = (int32_t)heads;
    p.ngram = (int32_t)ngram;
    p.pad = (int32_t)pad;
    p.msize = (int32_t)msize;
    p.image = (int32_t)image;
    p.video = (int32_t)video;
    p.n_seq = (int32_t)(cu->shape[0] - 1);
    /* [slots, W], or [slots, 1, W] as the engine hands a CONV group's state. */
    p.W = (int32_t)st->shape[st->rank - 1];
    p.sp = (uint64_t)st->stride[0];
    p.ip = (uint64_t)ids->stride[0];
    if (A > 0 && M + A > INT32_MAX / 64) return R4D_NO(RAD_E_SHAPE);
    const uint32_t blocks = (uint32_t)(((M + A) * heads + 255) / 256);
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;   /* decode heights */
#define R4D_EGI_K(I_, C_) R4D_EMIT(s, (engram_ids_kernel<I_, C_>), dim3(blocks), dim3(256), 0, p, \
                                   ids->data, (int32_t*)mask->data)
    int rc = ch ? (ids->dtype == RAD_I64 ? R4D_EGI_K(true, r4dchain) : R4D_EGI_K(false, r4dchain))
                : (ids->dtype == RAD_I64 ? R4D_EGI_K(true, R4dUnchained)
                                         : R4D_EGI_K(false, R4dUnchained));
#undef R4D_EGI_K
    if (rc != RAD_OK) return rc;
    const dim3 sg((uint32_t)((p.n_seq + 63) / 64));
    return ch ? R4D_EMIT(s, (engram_state_kernel<r4dchain>), sg, dim3(64), 0, p)
              : R4D_EMIT(s, (engram_state_kernel<R4dUnchained>), sg, dim3(64), 0, p);
}

extern "C" int r4d_rad_engram_norm(const RadArgs* a, RadStream s) {
    const RadTensor *e = r4d_opt(a, 0), *gain = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!e || !gain || !y) return R4D_NO(RAD_E_INVAL);
    if (!eg_bf16(e) || (!eg_bf16(gain) && !eg_f32(gain)) || !eg_bf16(y)) return R4D_NO(RAD_E_DTYPE);
    const long long H = r4d_geti_or(a, "heads", 0), n = r4d_geti_or(a, "n", 0);
    const float eps = (float)r4d_getf(a, "eps", -1.0);
    if (H < 1 || n < 1 || !std::isfinite(eps) || eps < 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long ep = r4d_rows_pitch(e, H * n, &M), yp = r4d_rows_pitch(y, H * n, &My);
    if (ep < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || My != M || (M * H + 7) / 8 > INT32_MAX || r4d_numel(gain) != H * n)
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(gain)) return R4D_NO(RAD_E_STRIDE);
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
#define R4D_EGN_K(W_, C_)                                                                        \
    R4D_EMIT(s, (engram_norm_kernel<W_, C_>), dim3((uint32_t)((M * H + 7) / 8)), dim3(256), 0,   \
             (const uint16_t*)e->data, (const W_*)gain->data, (uint16_t*)y->data, (uint32_t)M,   \
             (uint32_t)H, (uint32_t)n, eps, (uint64_t)ep, (uint64_t)yp)
    if (eg_f32(gain)) return ch ? R4D_EGN_K(float, r4dchain) : R4D_EGN_K(float, R4dUnchained);
    return ch ? R4D_EGN_K(uint16_t, r4dchain) : R4D_EGN_K(uint16_t, R4dUnchained);
#undef R4D_EGN_K
}

extern "C" int r4d_rad_engram_gate(const RadArgs* a, RadStream s) {
    enum { KEY = 0, VAL, MASK, WK, WQ, WV, WB, H };
    const RadTensor *key = r4d_opt(a, KEY), *val = r4d_opt(a, VAL), *mask = r4d_opt(a, MASK),
                    *wk = r4d_opt(a, WK), *wq = r4d_opt(a, WQ), *wv = r4d_opt(a, WV),
                    *wb = r4d_opt(a, WB), *h = r4d_opt(a, H);
    if (!key || !val || !mask || !wk || !wq || !wv || !wb || !h) return R4D_NO(RAD_E_INVAL);
    /* The four weights bf16 (the checkpoint's) or all four f32 (the plugin's declared form). */
    const bool wf32 = eg_f32(wk);
    auto wok = [&](const RadTensor* t) { return wf32 ? eg_f32(t) : eg_bf16(t); };
    if (!eg_bf16(key) || !eg_bf16(val) || !eg_bf16(h) || !wok(wk) || !wok(wq) || !wok(wv) ||
        !wok(wb) || !eg_i32(mask))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0);
    const float eps = (float)r4d_getf(a, "eps", -1.0);
    if (n < 1 || hc < 1 || hc > 8 || !std::isfinite(eps) || eps < 0) return R4D_NO(RAD_E_SHAPE);
    const long long W = hc * n;
    long long M = 0, Mk = 0, Mv = 0;
    const long long hp = r4d_rows_pitch(h, W, &M), kp = r4d_rows_pitch(key, W, &Mk),
                    vp = r4d_rows_pitch(val, n, &Mv);
    if (hp < 0 || kp < 0 || vp < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 1 || Mk != M || Mv != M || r4d_numel(mask) < M || r4d_numel(wk) != W ||
        r4d_numel(wq) != W || r4d_numel(wv) != n || r4d_numel(wb) != W)
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(mask) || !r4d_contig(wk) || !r4d_contig(wq) || !r4d_contig(wv) ||
        !r4d_contig(wb))
        return R4D_NO(RAD_E_STRIDE);
    const float inv_sqrt_n = (float)std::pow((double)n, -0.5);
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    /* A prefill chunk's rows share the four weight vectors, so a block holds them over kEgRows
     * rows: a quarter of the weight traffic, at the cost of the rows a block running in series.
     * With kEgRows rows a block a chunk of M rows fills M/kEgRows blocks, so the split starts at
     * 256 rows, where that fills a R9700's 64 CUs; a decode step's handful of rows keeps a row a
     * block, where every block is busy and a quarter of the vectors is all there is to read. */
    const bool rpg = M >= 256;   /* the block's rows: kEgRows at a prefill height, else one */
#define R4D_EGT_K(HC_, WT_, C_, R_)                                                              \
    R4D_EMIT(s, (engram_gate_kernel<HC_, WT_, C_, R_>),                                          \
                    dim3((uint32_t)((M + (R_) - 1) / (R_))), dim3(kEgThreads), 0,                \
                    (const uint16_t*)key->data, (const uint16_t*)val->data,                      \
                    (const int32_t*)mask->data, (const WT_*)wk->data,                            \
                    (const WT_*)wq->data, (const WT_*)wv->data,                                  \
                    (const WT_*)wb->data, (uint16_t*)h->data, (uint32_t)n, eps, inv_sqrt_n,      \
                    (uint64_t)kp, (uint64_t)vp, (uint64_t)hp, (uint32_t)M)
#define R4D_EG_ROWS(HC_, WT_, C_)                                                                \
    (rpg ? R4D_EGT_K(HC_, WT_, C_, kEgRows) : R4D_EGT_K(HC_, WT_, C_, 1))
#define R4D_EGT(HC_, WT_)                                                                        \
    return ch ? R4D_EG_ROWS(HC_, WT_, r4dchain) : R4D_EG_ROWS(HC_, WT_, R4dUnchained)
#define R4D_EG(HC_) do { if (wf32) R4D_EGT(HC_, float); R4D_EGT(HC_, uint16_t); } while (0)
    switch (hc) {
        case 1: R4D_EG(1);
        case 2: R4D_EG(2);
        case 3: R4D_EG(3);
        case 4: R4D_EG(4);
        case 5: R4D_EG(5);
        case 6: R4D_EG(6);
        case 7: R4D_EG(7);
        default: R4D_EG(8);
    }
#undef R4D_EG
#undef R4D_EGT
#undef R4D_EGT_K
}
