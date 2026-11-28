/* r4d_row_i8_geom.h -- the int8 latent row's geometry (design note X-07), for host and device.
 *
 * The row is [L int8 codes][L / 128 f32 scales][R rope columns as bf16] at L + R = 2 * head_dim,
 * L a multiple of the block, so L + 4 * (L / 128) + 2 * R bytes (656 at Hy4-flash's 512 latent and
 * 64 rotary columns, a multiple of 16 at every geometry the model uses).
 *
 * Everything here is arithmetic on integers and one f32 constant, so the sites that only size the
 * row -- the shape hook that publishes both stores' operands (r4d_shapes.cpp), the row's own
 * selftest -- take it without the device code, and no site restates a formula. r4d_row_i8.h holds
 * the quantiser that writes the row and includes this.
 */
#pragma once

#include <cstdint>

namespace r4d_row_i8 {

/* The columns of one quantised block: a wave holds a block, four columns a lane. */
constexpr int kBlock = 128;

/* 1 / sqrt(kBlock): the rotation's normaliser, folded into the block's scale so the reader's
 * dequantisation is one multiply (design note X-07 §2). */
constexpr float kRotScale = 0.08838834764831845f;

/* The row's byte count at L latent columns and R rotary ones. */
constexpr int64_t row_bytes(int64_t L, int64_t R) {
    return L + 4 * (L / kBlock) + 2 * R;
}

/* Where the row's three parts start, in bytes. */
constexpr int64_t codes_offset() { return 0; }
constexpr int64_t scales_offset(int64_t L) { return L; }
constexpr int64_t rope_offset(int64_t L) { return L + 4 * (L / kBlock); }

/* The row of one head of an int8 cache, in elements of its dtype -- which for int8 is its bytes,
 * and the extent both stores publish as the cache operand's last one. Without a `rope_dim` (R <= 0)
 * it is the generic K then V row, 2 * head_dim + 4; with one it is the latent row above at
 * L = 2 * head_dim - R, and 0 when the geometry has no such row (R odd, or L not positive and not
 * a whole number of blocks). */
constexpr int64_t row_elems(int64_t head_dim, int64_t R) {
    if (R <= 0) return 2 * head_dim + 4;
    const int64_t L = 2 * head_dim - R;
    if ((R & 1) || L <= 0 || L % kBlock) return 0;
    return row_bytes(L, R);
}

}  // namespace r4d_row_i8
