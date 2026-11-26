// The QSA indexer's key side in one op (K-24): this step's closing groups' raw keys gathered out of
// the ring (qsa_work), pooled into a group key (idx_pool_gated) and the tail's rows stored
// (qsa_tail_store) -- three ops over the same step, the same token keys and the same ring.
//
// Why one: the three are a chain of launches over 128 keys a workgroup, and at decode a sequence
// closes a group one step in `ratio`, so most of those launches find nothing to do and the launches
// themselves are the cost (`idx_pool_gated` 12 x 9.4 us + `qsa_work` 12 x 3.0 + `qsa_tail_store`
// 12 x 3.0 a pass, notes/backlog/audit/A-kernels-decode.md T-10). Fused, a workgroup resolves its
// items once and gathers them straight into the LDS block the pool's arithmetic reads, so a decode
// step is one launch where it was three.
//
// What is fused is the exact (bf16) key path only: RADIANCE_HY4_INDEXER=fp8|int8 pool through the
// Hadamard and keep the key f32 (idx_pool_gated_fp8/_i8), and a tap reads the staged raw keys
// (`stage`), so both keep the three ops. The plugin declares one path or the other, never both.
//
// The tail is the part that cannot always run here. A token's key lands at row pos % ring, and
// position p + ring shares that row, so a chunk's own store can collide with a gather this same
// kernel does. A group completing at p1 gathers positions [p0 - ratio + 1, p1] and a store writes
// this chunk's positions, whose span is at most the chunk's row count, so the two cannot share a
// row when the chunk's rows <= ring - ratio + 1: under that bound the store is done here, per
// sequence, and the plugin issues no qsa_tail_store; over it this kernel skips the sequence's rows
// and qsa_tail_store runs after it as before. A short sequence beside a long one of the same step
// is then stored twice with the same bytes, which is harmless -- the row guard is the same one on
// both sides.
//
// Not chainable: the indexer's key side is not in the chained set (docs/OPS.md), and it already
// serialises against the key norm that precedes it.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_idx_pool_body.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kMaxSeq = 256;   /* qsa_work's bound on the scan a workgroup does */

#define QSA_FRONT_PARAMS                                                                        \
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,                            \
    const int32_t* __restrict__ cu, uint16_t* __restrict__ state,                               \
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ bt,                           \
    const float* __restrict__ wv, const float* __restrict__ wg, const float* __restrict__ ape,  \
    const void* __restrict__ gain, const int32_t* __restrict__ rope, int32_t* __restrict__ page, \
    int32_t* __restrict__ bpos, int32_t* __restrict__ nc, uint16_t* __restrict__ cache,         \
    int T, int n, int ratio, int ring, int seqs, int slots, int W, int rd, float eps,           \
    double theta, int sec_h, int sec_w, int k_ld, int s_slot, int s_row, int sidx_ld, int bt_ld, \
    int bstep, int rplane, long cache_pitch, int pages, int gain_bf16

/* One workgroup -- `blockIdx.x` of the pool's grid over the work list -- holds GPW of the work
 * list's items: it stores the tail's rows it may store, resolves the items' pages, positions and
 * key sources in one scan (qsa_work's), gathers each item's `ratio` raw keys into LDS and pools
 * them in place. */
template <int GPW, bool IMROPE>
__device__ __forceinline__ void qsa_work_pool_tail_body(QSA_FRONT_PARAMS) {

    __shared__ float xs[GPW][kRatio][kN];
    __shared__ int32_t ps[GPW][3];   /* the group's first position, three components under M-RoPE */
    __shared__ int32_t pw[GPW];      /* the group's page, -1 for none */
    __shared__ int32_t gs[GPW];      /* the group's tail slot, -1 when it gathers nothing */
    __shared__ int32_t gp[GPW];      /* keys below this index come from the tail */
    __shared__ int32_t gt[GPW];      /* the token row of key 0 (valid from key gp on) */
    __shared__ int32_t gq[GPW];      /* the tail row of key 0 */
    __shared__ int closed;

    const int c  = (int)threadIdx.x;
    const int g0 = (int)blockIdx.x * GPW;

    /* ---- the tail: the rows of every chunk whose gather cannot reach them (see the header) ------
     * Before the scan below, which returns early and is followed by barriers. Strided over the
     * grid, the body qsa_tail_store runs for a token. */
    for (int t = (int)blockIdx.x; t < T; t += (int)gridDim.x) {
        int s = -1;
        for (int i = 0; i < seqs && i < kMaxSeq; ++i)
            if (t >= cu[i] && t < cu[i + 1]) { s = i; break; }
        if (s < 0) continue;
        const int t0 = cu[s], t1 = cu[s + 1];
        if (t1 - t0 > ring - ratio + 1) continue;
        if (t < t1 - ring) continue;
        const int sl = sidx[(size_t)s * sidx_ld];
        if (sl < 0 || sl >= slots) continue;
        const size_t src = (size_t)t * k_ld;
        const size_t dst = (size_t)sl * s_slot + (size_t)(pos[t] % ring) * s_row;
        for (int d = c; d < n; d += kThreads) state[dst + d] = k[src + d];
    }

    /* ---- the work list: qsa_work's items, resolved once ---------------------------------- */
    if (!c) {
        int run = 0;
        /* qsa_work publishes the work list: an item past the real count is page -1 with no
         * position, and the first `seqs` items carry the per-sequence count. Both are read by the
         * scorer and the selection, so they are written here as they are there. */
        const int wend = W < g0 + GPW ? W : g0 + GPW;
        for (int w = g0; w < wend; ++w) {
            page[w] = -1;
            for (int j = 0; j < (IMROPE ? 3 : 1); ++j) bpos[(size_t)w * bstep + j] = 0;
        }
        for (int i = g0; i < seqs && i < kMaxSeq && i < wend; ++i)
            nc[i] = cu[i + 1] > cu[i] ? (pos[cu[i + 1] - 1] + 1) / ratio : 0;
#pragma unroll
        for (int g = 0; g < GPW; ++g) {
            pw[g] = -1;
            gs[g] = -1;
            gp[g] = 0;
            gt[g] = 0;
            gq[g] = 0;
            ps[g][0] = 0;
            ps[g][1] = 0;
            ps[g][2] = 0;
        }
        int any = 0;
        for (int i = 0; i < seqs && i < kMaxSeq; ++i) {
            const int t0 = cu[i], t1 = cu[i + 1];
            if (t1 <= t0) continue;
            const int p0 = pos[t0], p1 = pos[t1 - 1];
            const int lo = p0 / ratio;               /* ceil((p0 - ratio + 1) / ratio) */
            const int hi = (p1 + 1) / ratio - 1;     /* floor((p1 - ratio + 1) / ratio) */
            for (int b = lo; b <= hi; ++b) {
                const int w = run + (b - lo);
                if (w < g0) continue;
                if (w >= g0 + GPW) break;
                const int g = w - g0;
                const int first = b * ratio;
                const int sl = sidx[(size_t)i * sidx_ld];
                pw[g] = sl < 0 ? -1 : bt[(size_t)i * bt_ld + b];
                any |= pw[g] >= 0 && pw[g] < pages;
                gs[g] = sl;
                /* The key sources: a key below p0 is in the tail at row q % ring, the rest is this
                 * step's row t0 + (q - p0). */
                const int from = p0 - first;
                gp[g] = from > 0 ? (from < ratio ? from : ratio) : 0;
                gt[g] = t0 + first - p0;
                gq[g] = first % ring;
                /* M-RoPE: the group turns at its first token's three components, from this step's
                 * plane when that token is in it (`qsa_work`'s rule). */
                if (IMROPE) {
                    for (int j = 0; j < 3; ++j)
                        ps[g][j] = first >= p0 ? rope[(size_t)j * rplane + t0 + (first - p0)]
                                               : first + (rope[t0] - p0);
                } else {
                    ps[g][0] = first;
                }
                /* Published as qsa_work publishes them. */
                page[w] = pw[g];
                for (int j = 0; j < (IMROPE ? 3 : 1); ++j)
                    bpos[(size_t)w * bstep + j] = ps[g][j];
            }
            run += hi >= lo ? hi - lo + 1 : 0;
        }
        closed = any;
    }
    __syncthreads();
    if (!closed) return;

    /* ---- the gather, straight into the pool's LDS block --------------------------------- */
#pragma unroll
    for (int g = 0; g < GPW; ++g)
#pragma unroll
        for (int i = 0; i < kRatio; ++i) {
            const uint16_t* src = 0;
            if (gs[g] >= 0)
                src = i >= gp[g] ? k + (size_t)(gt[g] + i) * k_ld
                                 : state + (size_t)gs[g] * s_slot +
                                       (size_t)((gq[g] + i) % ring) * s_row;
            xs[g][i][c] = src ? bf16_to_f32(src[c]) : 0.0f;
        }
    __syncthreads();

    idx_pool_math<GPW, IMROPE, false>(xs, ps, pw, wv, wg, ape, gain, cache, (uint16_t*)0, c, g0, W,
                                      0, cache_pitch, pages, rd, eps, theta, sec_h, sec_w,
                                      gain_bf16, (uint8_t*)0, (float*)0, 0, 0, 0.0f, 0);
}

template <int GPW, bool IMROPE>
__global__ __launch_bounds__(kThreads)
void qsa_work_pool_tail_kernel(QSA_FRONT_PARAMS) {
    qsa_work_pool_tail_body<GPW, IMROPE>(k, pos, cu, state, sidx, bt, wv, wg, ape, gain, rope,
                                         page, bpos, nc, cache, T, n, ratio, ring, seqs, slots, W,
                                         rd, eps, theta, sec_h, sec_w, k_ld, s_slot, s_row,
                                         sidx_ld, bt_ld, bstep, rplane, cache_pitch, pages,
                                         gain_bf16);
}
#undef QSA_FRONT_PARAMS

constexpr int kWideGroups = 4;       /* groups a workgroup on the wide form */
constexpr int kWideMinW   = 16;      /* the wide form from this many groups */

}  // namespace

// ============================================================== the plugin ABI entry point

//   k      [T][n]                  bf16, this step's raw index keys, k_ld elements a row
//   pos    [T]                     i32, each token's position in its sequence
//   cu     [seqs + 1]              i32, cu_seqlens
//   state  [slots][1][ring][n]     bf16, the tail group's cache for this layer: read by the gather
//                                  and written with the rows the gather cannot reach
//   sidx   [seqs][sidx_ld]         i32, the tail group's state row; column 0 is the slot
//   bt     [seqs][bt_ld]           i32, the block-key group's block table
//   wv,wg  [n][n]                  f32, the value and gate projections
//   ape    [ratio][n]              f32, the additive group position encoding, a row a key
//   gain   [n]                     f32 or bf16, the key's per-channel gain
//   rope   [3][rplane]            i32, optional: this step's rotary components (RadBatch::rope_pos)
//   page   [W]                     i32, each item's block page, -1 past the real count
//   bpos   [W][bstep]              i32, an item's first-token position at [w * bstep]
//   nc     [seqs]                  i32, complete blocks per sequence
//   cache  [pages][1][ratio][n/ratio]  bf16 in-out, optional: the indexer's key pages
extern "C" int r4d_rad_qsa_work_pool_tail(const RadArgs* a, RadStream stream) {
    enum { QF_K = 0, QF_POS, QF_CU, QF_STATE, QF_SIDX, QF_BT, QF_WV, QF_WG, QF_APE, QF_GAIN,
           QF_ROPE, QF_PAGE, QF_BPOS, QF_NC, QF_CACHE };
    const RadTensor* k  = r4d_opt(a, QF_K);
    const RadTensor* po = r4d_opt(a, QF_POS);
    const RadTensor* cu = r4d_opt(a, QF_CU);
    const RadTensor* tl = r4d_opt(a, QF_STATE);
    const RadTensor* sx = r4d_opt(a, QF_SIDX);
    const RadTensor* bt = r4d_opt(a, QF_BT);
    const RadTensor* wv = r4d_opt(a, QF_WV);
    const RadTensor* wg = r4d_opt(a, QF_WG);
    const RadTensor* ap = r4d_opt(a, QF_APE);
    const RadTensor* gn = r4d_opt(a, QF_GAIN);
    const RadTensor* rp = r4d_opt(a, QF_ROPE);       /* optional: multi-component positions */
    const RadTensor* pg = r4d_opt(a, QF_PAGE);
    const RadTensor* bp = r4d_opt(a, QF_BPOS);
    const RadTensor* nc = r4d_opt(a, QF_NC);
    const RadTensor* ca = r4d_opt(a, QF_CACHE);      /* optional: only the key pages are written */
    if (!k || !po || !cu || !tl || !sx || !bt || !wv || !wg || !ap || !gn || !pg || !bp || !nc)
        return R4D_NO(RAD_E_INVAL);
    if (k->rank != 2 || bt->rank != 2) return R4D_NO(RAD_E_SHAPE);
    /* The tail state arrives as its group declared it: [slots, heads, ring, n], heads 1 (see
     * r4d_qsa_tail_store_bf16.hip). */
    if (tl->rank != 4 || tl->shape[1] != 1) return R4D_NO(RAD_E_SHAPE);
    if (tl->stride[3] != 1) return R4D_NO(RAD_E_STRIDE);
    const long long n     = r4d_geti_or(a, "n", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long ring  = r4d_geti_or(a, "ring", 0);
    const long long rd    = r4d_geti_or(a, "rotary_dim", 0);
    /* The arithmetic is compiled for the key width and group size (r4d_idx_pool_body.h). */
    if (n != kN || ratio != kRatio || ring < ratio || rd < 2 || rd > n || (rd & 1))
        return R4D_NO(RAD_E_SHAPE);
    const double eps = r4d_getf(a, "eps", 0.0), theta = r4d_getf(a, "theta", 0.0);
    const float eps_f = (float)eps, theta_f = (float)theta;
    if (!(eps_f > 0.0f) || !std::isfinite(eps_f) || !(theta_f > 1.0f)) return R4D_NO(RAD_E_INVAL);
    const char* mode = r4d_gets(a, "mode");
    const bool imrope = mode && !std::strcmp(mode, "imrope_gptj");
    if (!mode || (!imrope && std::strcmp(mode, "gptj"))) return R4D_NO(RAD_E_INVAL);
    long long sec_h = 0, sec_w = 0;
    if (imrope) {
        const char* s = r4d_gets(a, "sections");
        long t = 0, h = 0, w = 0;
        if (!s || std::sscanf(s, "%ld %ld %ld", &t, &h, &w) != 3 || t < 0 || h < 0 || w < 0 ||
            t + h + w != rd / 2)
            return R4D_NO(RAD_E_SHAPE);
        sec_h = h;
        sec_w = w;
    }
    if (k->dtype != RAD_BF16 || tl->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (wv->dtype != RAD_F32 || wg->dtype != RAD_F32 || ap->dtype != RAD_F32 ||
        (gn->dtype != RAD_F32 && gn->dtype != RAD_BF16) || po->dtype != RAD_I32 ||
        bt->dtype != RAD_I32 || pg->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    const long long T    = k->shape[0];
    const long long seqs = r4d_numel(cu) - 1;
    if (seqs < 1 || seqs > kMaxSeq || T < 1 || r4d_numel(po) < T) return R4D_NO(RAD_E_SHAPE);
    if (k->shape[1] != n || tl->shape[2] < ring || tl->shape[3] != n) return R4D_NO(RAD_E_SHAPE);
    if (bt->shape[0] < seqs) return R4D_NO(RAD_E_SHAPE);
    const long long sxp = sx->rank >= 2 ? r4d_row_pitch(sx) : 1;
    if (sxp < 1 || r4d_dim(sx, 0) < seqs) return R4D_NO(RAD_E_SHAPE);
    /* The work count comes from `page`, as qsa_work's does. */
    const long long W = r4d_numel(pg);
    if (W < 1 || W > (1ll << 30) || pg->rank != 1 || pg->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    /* nc is written by the first seqs items' workgroups, so the launch must be that wide. */
    if (W < seqs) return R4D_NO(RAD_E_SHAPE);
    long long bstep = 1;
    if (bp->rank == 2) {
        if (bp->shape[0] < W || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1] ||
            (imrope && bp->shape[1] < 3))
            return R4D_NO(RAD_E_SHAPE);
        bstep = bp->stride[0];
    } else if (imrope || r4d_numel(bp) < W) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (bp->dtype != RAD_I32 || nc->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(nc) < seqs) return R4D_NO(RAD_E_SHAPE);
    if (rp && (rp->rank != 2 || rp->shape[0] < 3 || rp->shape[1] < T || !r4d_contig(rp) ||
               rp->dtype != RAD_I32))
        return R4D_NO(RAD_E_SHAPE);
    if (!rp && imrope) return R4D_NO(RAD_E_INVAL);
    auto dense = [&](const RadTensor* t, long long rows) {
        return t->rank == 2 && t->shape[0] == rows && t->shape[1] == n && t->stride[1] == 1 &&
               t->stride[0] == n;
    };
    if (!dense(wv, n) || !dense(wg, n) || !dense(ap, ratio)) return R4D_NO(RAD_E_SHAPE);
    if (gn->rank != 1 || gn->shape[0] != n || gn->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    long long cache_pitch = 0, pages = 0;
    if (ca) {
        if (ca->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
        if (ca->rank != 4 || ca->shape[0] < 1 || ca->shape[1] != 1 ||
            ca->shape[2] * ca->shape[3] != n || ca->stride[3] != 1 ||
            ca->stride[2] != ca->shape[3] || ca->stride[0] < n)
            return R4D_NO(RAD_E_SHAPE);
        cache_pitch = ca->stride[0];
        pages = ca->shape[0];
    }
    const long long kp = r4d_row_pitch(k), btp = r4d_row_pitch(bt);
    if (kp < 0 || btp < 0) return R4D_NO(RAD_E_STRIDE);

    const int gain_bf16 = gn->dtype == RAD_BF16;
    const bool wide = W >= kWideMinW;
    const int gpw = wide ? kWideGroups : 1;
    const unsigned grid = (unsigned)((W + gpw - 1) / gpw);
#define R4D_QFRONT(G_, IM_)                                                                      \
    R4D_EMIT(stream, (qsa_work_pool_tail_kernel<G_, IM_>), dim3(grid), dim3(kThreads), 0,        \
             (const uint16_t*)r4d_p(k), (const int32_t*)r4d_p(po), (const int32_t*)r4d_p(cu),    \
             (uint16_t*)r4d_wp(tl), (const int32_t*)r4d_p(sx), (const int32_t*)r4d_p(bt),        \
             (const float*)r4d_p(wv), (const float*)r4d_p(wg), (const float*)r4d_p(ap), r4d_p(gn), \
             (const int32_t*)r4d_p(rp), (int32_t*)r4d_wp(pg), (int32_t*)r4d_wp(bp),              \
             (int32_t*)r4d_wp(nc), (uint16_t*)r4d_wp(ca), (int)T, (int)n, (int)ratio, (int)ring, \
             (int)seqs, (int)tl->shape[0], (int)W, (int)rd, eps_f, (double)theta_f, (int)sec_h,  \
             (int)sec_w, (int)kp, (int)tl->stride[0], (int)tl->stride[2], (int)sxp, (int)btp,    \
             (int)bstep, rp ? (int)rp->shape[1] : 0, (long)cache_pitch, (int)pages, gain_bf16)
    if (wide) return imrope ? R4D_QFRONT(kWideGroups, true) : R4D_QFRONT(kWideGroups, false);
    return imrope ? R4D_QFRONT(1, true) : R4D_QFRONT(1, false);
#undef R4D_QFRONT
}
