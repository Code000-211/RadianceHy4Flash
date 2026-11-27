// r4d_topk.h - the key, the register list and the row emit that row_topk's stages share.
//
// r4d_rowtopk_bf16.hip's two stages rank on one 64-bit key, and so does a kernel that produces the
// candidates itself instead of reading them off a plane (logits_topk, in
// r4d_gemm_fp8a16_nt_m1.hip: the head's matvec keeps its blocks' top-RC and hands stage 2 the
// candidates). They must agree to the byte -- one order, one tie rule, one pad -- so the key, the
// lane's sorted list, the round and the emit live here, and both units include them.
//
// The key is one 64-bit integer, so a round is a plain max. The high 32 bits are the float's
// order-preserving image (invert all bits if negative, set the sign bit if not), the low 32 are
// ~index, so a larger key is a larger value and, on a tie, the lower column (a dense top-k's tie
// rule). The index also makes every key distinct, so a round identifies its winner by equality.
//
// PT bounds how many relevant candidates one lane's slice can hold:
//
//   * Stage 1 gives a lane every 32nd eight-column group, and PT must cover a whole vector load:
//     eight bf16 arrive at once, so PT=4 can lose a candidate to a single load (and does disagree
//     with torch.topk on smooth data). PT=8 places one load in full; beyond that a chunk would need
//     nine of its own top-RC in one of 32 slices.
//   * Stage 2's slice is not random: stage 1 writes chunk-major, so at stride 32 with RC=32 lane 0
//     receives every chunk's maximum, exactly where the row's top-R concentrates. Stage 2 therefore
//     uses PT = 32 >= R: one lane can hold the entire answer, so it is exact for any distribution.
//
// RC is the only possible approximation, and the caller's choice. A chunk hands on its top-RC, so
// the row's top-R survives unless more than RC of it falls in one chunk. RC=R is exact for any
// data; a smaller RC shortens stage 1's chain at a risk that falls fast with the chunk count (at
// R=32, RC=8 and 64 chunks it takes nine of the 32 in one sixty-fourth of the row). A caller that
// rescores exactly then loses a candidate, not correctness.
#pragma once
#include "r4d_common.h"
#include <cstdint>

#define R4D_TOPK_WAVE  32
#define R4D_TOPK_PT2   32      // stage 2: >= R, so a lane can hold the whole answer
#define R4D_TOPK_WPB   8      // independent waves per block: one chunk each, no barrier between
#define R4D_TOPK_UNROLL 4    // vector loads issued before any of them is consumed

__device__ __forceinline__ unsigned long long tk_key(unsigned short bf, unsigned idx) {
  unsigned u = (unsigned)bf << 16;
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - idx);
}

__device__ __forceinline__ float tk_val(unsigned long long k) {
  unsigned u = (unsigned)(k >> 32);
  u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
  return __uint_as_float(u);
}

__device__ __forceinline__ int tk_idx(unsigned long long k) {
  return (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
}

// The larger of this lane's key and its partner's at lane ^ N, read through DPP or permlanex16
// (see r4d_xor_lane).
template <int N>
__device__ __forceinline__ unsigned long long tk_xor_max(unsigned long long m) {
  const unsigned long long p = r4d_xor_lane_u64<N>(m);
  return p > m ? p : m;
}

// The insert's only guard is its own key compare. A float guard against the tail's value costs
// ~18% at M=64 in stage 1 (a branch breaks the unrolled straight-line body), and it is wrong with
// NaN: the key orders a positive NaN above +inf and a negative one below -inf, while a float
// compare is false, so once a NaN is the tail every later candidate would be dropped. Every stage
// and the merge rank on the key alone, so they agree on one order.
template <int PT>
__device__ __forceinline__ void tk_insert(unsigned long long* t, unsigned long long k) {
  if (k > t[PT - 1]) {
    unsigned long long c = k;
#pragma unroll
    for (int q = 0; q < PT; q++) {
      const unsigned long long hi = t[q] > c ? t[q] : c;
      c = t[q] > c ? c : t[q];
      t[q] = hi;
    }
  }
}

// One reduction round over the wave: every lane gets the winner and its owner drops its head. No
// LDS and no barrier.
template <int PT>
__device__ __forceinline__ unsigned long long tk_round(unsigned long long* t) {
  unsigned long long m = t[0];
  m = tk_xor_max<16>(m);
  m = tk_xor_max<8>(m);
  m = tk_xor_max<4>(m);
  m = tk_xor_max<2>(m);
  m = tk_xor_max<1>(m);
  if (m && t[0] == m) {
#pragma unroll
    for (int q = 0; q < PT - 1; q++) t[q] = t[q + 1];
    t[PT - 1] = 0ull;
  }
  return m;
}

// One element of the row as stage 1 ranks it. A bf16 plane is read as is. An f32 plane (the
// cast_row_topk op: a head's f32 logits) is rounded here exactly as `cast` f32 -> bf16 rounds it
// (r4d_encoder_bf16.hip's narrow_bf16_kernel calls this same f32_to_bf16, NaN included), so the
// keys, and so every selection and every reported value, are those of the bf16 plane the cast
// would have written. The cast's launch and its plane are what this saves.
__device__ __forceinline__ unsigned short tk_bf(unsigned short v) { return v; }
__device__ __forceinline__ unsigned short tk_bf(float v) { return f32_to_bf16(v); }

// The R winners of row m over its n stage-1 survivors at P, written by one wave (`lane` 0..31).
// Stage 2's kernel runs it with a wave per row; the fused stage 1 runs it in the row's last wave
// to finish (see r4d_rowtopk_stage1_body), and logits_topk runs it over the winners of a block's
// eight waves. PT is a lane's list, at least R (see the header): 32 serves every R, and a launch
// for R <= 8 or R <= 16 takes the shorter list, whose insert and whose round's shift are that much
// shorter. The output is the same for any PT >= R.
//
// `off` is the first global column this rank's plane stands for, and `OP` the optional (id, value)
// pair plane an all-gather moves: eight bytes an entry, an int32 id in the first float slot, the
// layout core/sample/sampler.h holds as buf_pairs_. Both are inert at off = 0 with OP null, which
// is the single-rank launch bit for bit.
//
// An empty slot is not token `off`. A chunk that ran out of candidates leaves a zero key; OI gets
// column 0 for it, without the offset, so a pad cannot look like this rank's first real token. The
// pair plane writes -1, the sentinel the merge drops (not -INFINITY, a valid masked logit).
template <int PT>
__device__ __forceinline__ void r4d_rowtopk_stage2_row(
    const unsigned long long* __restrict__ P, float* __restrict__ OV, int* __restrict__ OI,
    float* __restrict__ OP, int off, int R, int n, int m, int lane) {
  unsigned long long t[PT];
#pragma unroll
  for (int q = 0; q < PT; q++) t[q] = 0ull;

  // The key decides, as in stage 1 (see tk_insert). A NaN a chunk handed on displaced a real
  // candidate, so dropping it here can leave the row short of R; the pad filling the gap is what
  // the merge reports as id -1. An empty slot's key is 0 and never places.
  // UNROLL keys are loaded before the first is placed, as in stage 1: one load a pass would put a
  // memory latency between consecutive inserts. A lane places its keys in the same order.
  const unsigned long long* p = P + (size_t)m * n;
  int i = lane;
  for (; i + (R4D_TOPK_UNROLL - 1) * R4D_TOPK_WAVE < n; i += R4D_TOPK_UNROLL * R4D_TOPK_WAVE) {
    unsigned long long k[R4D_TOPK_UNROLL];
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++) k[u] = p[i + u * R4D_TOPK_WAVE];
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++) tk_insert<PT>(t, k[u]);
  }
  for (; i < n; i += R4D_TOPK_WAVE) tk_insert<PT>(t, p[i]);

  for (int r = 0; r < R; r++) {
    const unsigned long long w = tk_round<PT>(t);
    if (lane == 0) {
      // A zero key is an empty slot, not a real column.
      const float v = w ? tk_val(w) : -INFINITY;
      OV[(size_t)m * R + r] = v;
      OI[(size_t)m * R + r] = w ? (int)tk_idx(w) + off : 0;
      if (OP) {
        float* const q = OP + ((size_t)m * R + r) * 2;
        const int id = w ? (int)tk_idx(w) + off : -1;
        __builtin_memcpy(q, &id, sizeof(int));
        q[1] = v;
      }
    }
  }
}

// Where the fused form's stage 2 writes: the outputs and the row's arrival counters.
struct R4dTopkOut {
  float* OV;
  int* OI;
  float* OP;
  unsigned* ctr;     // one a row, zero at entry and left zero (rowtopk_ctr)
  int off, R;
};
