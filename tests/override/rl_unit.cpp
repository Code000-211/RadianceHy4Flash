/* rl_unit.cpp -- the unit of tests/override's argmax library (rl_rows.cpp) that its overrides
 * rebuild. RL_TOKEN is the token every row is given, so a reply's tokens say which build ran;
 * RL_SCRATCH the scratch the launch asks for; RL_INIT defines the row's init here as well, which
 * is a hook declare used. */
#include "rad_abi.h"

#ifndef RL_TOKEN
#define RL_TOKEN 3
#endif
#ifndef RL_SCRATCH
#define RL_SCRATCH 64
#endif

extern "C" {

/* Operands as libref's argmax takes them: logits, the rows' parameters, one token per row. */
int rl_argmax(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 3 || !a->t[2].data || a->t[2].rank < 1) return RAD_E_INVAL;
    const RadTensor& tok = a->t[2];
    const int64_t stride = tok.stride[0] ? tok.stride[0] : 1;
    for (int64_t m = 0; m < tok.shape[0]; ++m) {
        if (tok.dtype == RAD_I32) static_cast<int32_t*>(tok.data)[m * stride] = RL_TOKEN;
        else if (tok.dtype == RAD_I64) static_cast<int64_t*>(tok.data)[m * stride] = RL_TOKEN;
        else return RAD_E_DTYPE;
    }
    return RAD_OK;
}

int64_t rl_argmax_scratch(const RadArgs*) { return RL_SCRATCH; }

#ifdef RL_INIT
int rl_argmax_init(const RadParam*, int, int, int, void** out) {
    *out = nullptr;
    return RAD_OK;
}
#endif

}  /* extern "C" */
