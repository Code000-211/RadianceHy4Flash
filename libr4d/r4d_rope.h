// The rotary's angle, for the kernels that turn a pair: `rope` and `rope_rows` (r4d_model_bf16.hip)
// launch the rotation, mla_row_norm_rope_store turns the row it writes, and mla_attn_paged_sink /
// mla_attn_sel_sink turn the query fragment they read (their `rope_pos` operand), so the launch that
// rotated q_rope is not needed on those rows. All of them call rope_turn and rope_comp rather than
// restating them, so there is one expression, no fast-math, and no accumulation in a narrower type:
// the per-component position (mrope / imrope / axial), the base ladder's exponent over `den`, yarn's
// frequency scaling, the proportional cut at `nfreq`, and the accurate sincosf. Do not "simplify" it
// here -- a kernel that turns by a different angle is not byte-identical to the launch it replaces.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#include "rad_yarn.h"
#include "r4d_common.h"

__device__ __forceinline__ int rope_comp(int mode, uint32_t i, const int* sec, int n_sec,
                                         uint32_t* j) {
    *j = i;
    if (mode == 3) {
        if (i % 3 == 1 && (int)i < 3 * sec[1]) return 1;
        if (i % 3 == 2 && (int)i < 3 * sec[2]) return 2;
        return 0;
    }
    int c = 0, start = 0;
    while (c + 1 < n_sec && (int)i >= start + sec[c]) { start += sec[c]; ++c; }
    *j = i - (uint32_t)start;
    return c;
}
/* The angle for pair i from a position source: `at(c)` is component c's position over rscale
 * (pos[c * plane + m] / div), asked for component 0 and, with a component pitch, for the one
 * rope_comp picks. rope_turn reads it from `pos`; rope_turn_at from a RopePos read earlier (K-29). */
template <class At>
__device__ __forceinline__ void rope_turn_by(At at, uint32_t i, int den, uint32_t nfreq,
                                             float theta, int mode, const int* sec, int n_sec,
                                             uint32_t plane, RadYarn yarn, RadLlama3 l3, float* c,
                                             float* s) {
    const float p0 = at(0);
    float inv = 1.0f / powf(theta, (float)(2 * i) / (float)den);
    float p = p0;
    if (plane) {
        uint32_t j = i;
        const int comp = rope_comp(mode, i, sec, n_sec, &j);
        p = at(comp);
        if (mode == 4)
            inv = 1.0f / powf(theta, (float)(2 * j) / (float)(2 * sec[comp]));
    }
    inv = yarn.frequency(inv, (int)i);
    inv = l3.frequency(inv);
    /* The proportional ladder rotates the whole head with every frequency past nfreq zero (HF's
     * _compute_proportional_rope_parameters): pair i is (i, i + head_dim/2), and pairs past nfreq
     * turn by 0 (cos 1, sin 0), returning the value exactly. */
    if (i >= nfreq) inv = 0.0f;
    const float ang = p * inv;
    sincosf(ang, s, c);
    if (yarn.factor != 0.0f) { *c *= yarn.magnitude; *s *= yarn.magnitude; }
}
__device__ __forceinline__ void rope_turn(const int32_t* __restrict__ pos, uint32_t m, uint32_t i,
                                          int den, uint32_t nfreq, float theta, float rscale,
                                          int mode, const int* sec, int n_sec, uint32_t plane,
                                          RadYarn yarn, RadLlama3 l3, float* c, float* s) {
    const float div = rscale != 0.0f ? rscale : 1.0f;
    rope_turn_by([&](int comp) { return (float)pos[(size_t)comp * plane + m] / div; }, i, den,
                 nfreq, theta, mode, sec, n_sec, plane, yarn, l3, c, s);
}

/* The step's rotary position for one token, read once (K-29): p[0] is pos[m] and, with a component
 * pitch, p[c] is pos[c * plane + m] for the mode's components (rope_comp's, at most `n_comp`).
 * rope_turn_at takes it instead of reading `pos` itself, so that a chained block can read it in
 * front of its wait (r4d_chain.h) and the wait covers the round trip. Both values are rope_turn's
 * own expression, so the bytes are the same whichever side of the wait the read happens. */
struct RopePos { float p[3]; };
__device__ __forceinline__ RopePos rope_pos(const int32_t* __restrict__ pos, uint32_t m,
                                            uint32_t plane, float rscale, int n_comp) {
    const float div = rscale != 0.0f ? rscale : 1.0f;
    RopePos r;
    r.p[0] = (float)pos[m] / div;
    r.p[1] = r.p[0];
    r.p[2] = r.p[0];
    /* Only the components the mode indexes: a table has as many rows as the mode has sections
     * (r4d_rad_rope), so a two-section axial table has no third row to read. */
    if (plane && n_comp > 1) r.p[1] = (float)pos[(size_t)plane + m] / div;
    if (plane && n_comp > 2) r.p[2] = (float)pos[(size_t)2 * (size_t)plane + m] / div;
    return r;
}
__device__ __forceinline__ void rope_turn_at(const RopePos& rp, uint32_t i, int den,
                                             uint32_t nfreq, float theta, int mode, const int* sec,
                                             int n_sec, uint32_t plane, RadYarn yarn, RadLlama3 l3,
                                             float* c, float* s) {
    rope_turn_by([&](int comp) { return rp.p[comp]; }, i, den, nfreq, theta, mode, sec, n_sec,
                 plane, yarn, l3, c, s);
}
