/* r4d_row_i8.h -- the rotated int8 latent row of Hy4-flash's KV cache (design note X-07).
 *
 * One token's row of one group is
 *
 *   [0, L)                  L int8 codes: the row's L latent columns, each block of 128 rotated by
 *                           the tree's Walsh-Hadamard (r4d_fwht.h, one wave a block, so the four
 *                           contiguous columns a lane holds are r4d_fwht_wave<128>'s) and quantised
 *                           against the block's scale
 *   [L, L + 4 * L/128)      one f32 scale a block, in block order
 *   [L + 4 * L/128, + 2R)   the row's R rotary columns as they are: the bf16 values the writer's
 *                           rope produced (bf16_to_f32, rope, f32_to_bf16), not quantised -- a
 *                           rope column carries its own angle and must not be mixed into a block
 *
 * at L + R = 2 * head_dim (the group declares the honest head_dim, (L + R) / 2, because its
 * attention op is the ordinary one). A row is L + 4 * (L / 128) + 2 * R bytes and a multiple of
 * 16 for every L the latent group uses (512 + 16 + 128 = 656), so the readers' uint4 loads land
 * whole. Hy4-flash's latent cache is one KV head: the row is the group.
 *
 * A block has two scales (design note X-07 §2). The quantiser's is its absmax / 127, the
 * activation path's (r4d_q8.h:18's q8_top), and the codes are the rotated block over it; it is
 * not the stored one. The stored scale also carries the rotation's 1 / sqrt(128), so a decoded
 * column is `scale * code ~= WHT(x_b) / sqrt(128)`, the rotated value itself. Folding the
 * normaliser into the divisor as well would scale every code by sqrt(128) and clip most of them.
 * A block whose absmax is zero or not finite stores codes 0 and scale 1, as an activation block
 * does, so an inf out of the rotation cannot make a NaN block on decode.
 *
 * libref does not borrow this header (its form builds the rotation as an explicit matrix, and its
 * block is normalised before it is quantised); the kernels of the three store rows and the
 * readers of X-07b share it. The geometry it and the host sites agree on is r4d_row_i8_geom.h. */
#ifndef R4D_ROW_I8_H
#define R4D_ROW_I8_H

#include <hip/hip_runtime.h>
#include <cstdint>

#include "r4d_q8.h"       // q8_pack4, q8_top, and the r4d_fwht.h wave it rotates with
#include "r4d_row_i8_geom.h"   // kBlock, kRotScale, the row's layout

namespace r4d_row_i8 {

/* Whether a block has any code worth writing: a finite, positive absmax. An all-zero block and a
 * block whose absmax is inf or NaN are stored as scale 1 / codes 0 (design note X-07 §2), and the
 * absmax is a wave maximum, so the answer is the same in every lane of the wave. */
__device__ __forceinline__ bool block_live(float amax) {
    return amax > 0.0f && amax < 3.402823466e38f;   /* FLT_MAX: false for an inf too */
}

/* A block's quantiser scale: its absmax / 127, the activation path's (q8_row_i8g). The codes are
 * the rotated block over this one. */
__device__ __forceinline__ float block_qscale(float amax) {
    return amax * (1.0f / q8_top<true>());
}

/* A block's stored scale: the quantiser scale with the rotation's 1 / sqrt(kBlock) folded in, so
 * the reader's `scale * code` is the rotated column and no separate normalisation is needed. */
__device__ __forceinline__ float block_scale(float amax) {
    return block_qscale(amax) * kRotScale;
}

/* Quantise one block of the row: a wave's 128 columns, the four contiguous ones in `v` rotated in
 * place, the wave's absmax, the scale written by the wave's lane 0 and the lane's four codes
 * packed behind the row's other blocks. THR is the workgroup size; waves take blocks by stride,
 * so the bytes do not depend on it. */
template <uint32_t THR>
__device__ __forceinline__ void quant_block(float* v, uint8_t* __restrict__ codes,
                                            float* __restrict__ scales, uint32_t lane,
                                            uint32_t block) {
    r4d_fwht_wave<kBlock>(v, lane);
    float m = fabsf(v[0]);
    m = fmaxf(m, fabsf(v[1]));
    m = fmaxf(m, fabsf(v[2]));
    m = fmaxf(m, fabsf(v[3]));
    m = r4d_wave_max(m);
    const bool live = block_live(m);
    if (lane == 0) scales[block] = live ? block_scale(m) : 1.0f;
    *(uint32_t*)(codes + block * kBlock + lane * 4u) =
        live ? q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / block_qscale(m)) : 0u;
}

/* Quantise a whole row of `L` latent columns held as bf16 in `row` (L a multiple of kBlock).
 * Every thread of the workgroup takes part, so the wave shuffles are complete. */
template <uint32_t THR>
__device__ __forceinline__ void quant_row(const uint16_t* row, uint8_t* __restrict__ codes,
                                          float* __restrict__ scales, uint32_t L) {
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t b = threadIdx.x >> 5; b < L / kBlock; b += THR / 32u) {
        const uint32_t c0 = b * kBlock + lane * 4u;
        float v[4] = {bf16_to_f32(row[c0]), bf16_to_f32(row[c0 + 1]),
                      bf16_to_f32(row[c0 + 2]), bf16_to_f32(row[c0 + 3])};
        quant_block<THR>(v, codes, scales, lane, b);
    }
}

/* A stored code as the row means it: the block's scale times the code (X-07b's readers). */
__device__ __forceinline__ float dequant(int8_t code, float scale) {
    return scale * (float)code;
}

}  // namespace r4d_row_i8

#endif  // R4D_ROW_I8_H
