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
 * The scale is the activation path's (r4d_q8.h:18's q8_top, 1 / sqrt(128) folded in), so a block
 * of an all-zero row stores codes 0 and scale 1 as an activation block does. The rotation is
 * orthogonal and 1 / sqrt(128) normalises it, so a decoded column is its stored value to the
 * quantiser's step -- the row's own bound (design note X-07 §2).
 *
 * libref does not borrow this header (its form builds the rotation as an explicit matrix); the
 * kernels of the three store rows and the readers of X-07b share it. */
#ifndef R4D_ROW_I8_H
#define R4D_ROW_I8_H

#include <hip/hip_runtime.h>
#include <cstdint>

#include "r4d_q8.h"       // q8_pack4, q8_top, and the r4d_fwht.h wave it rotates with

namespace r4d_row_i8 {

/* The columns of one quantised block: a wave holds a block, four columns a lane. */
constexpr uint32_t kBlock = 128;

/* 1 / sqrt(kBlock): the rotation's normaliser, folded into the block's scale so the reader's
 * dequantisation is one multiply (design note X-07 §2). */
constexpr float kRotScale = 0.08838834764831845f;

/* The row's byte count at L latent columns and R rotary ones. */
__host__ __device__ __forceinline__ constexpr int row_bytes(int L, int R) {
    return L + 4 * (L / (int)kBlock) + 2 * R;
}

/* Where the row's three parts start, in bytes. */
__host__ __device__ __forceinline__ constexpr int codes_offset() { return 0; }
__host__ __device__ __forceinline__ constexpr int scales_offset(int L) { return L; }
__host__ __device__ __forceinline__ constexpr int rope_offset(int L) { return L + 4 * (L / (int)kBlock); }

/* A block's scale: its absmax / (127 * sqrt(kBlock)), and 1 for an all-zero block as the
 * activation path stores it (q8_row_i8g). */
__device__ __forceinline__ float block_scale(float amax) {
    return amax > 0.0f ? amax * (1.0f / q8_top<true>()) * kRotScale : 1.0f;
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
    const float sc = block_scale(m);
    if (lane == 0) scales[block] = sc;
    *(uint32_t*)(codes + block * kBlock + lane * 4u) =
        q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / sc);
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
