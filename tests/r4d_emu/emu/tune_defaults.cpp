/* The row table's tuned axes, for the emu units that read them.
 *
 * An emu unit is a libr4d source compiled on the host. r4d_moe.hip publishes GLM's router and
 * scatter rows as well as its gather, and a row's args read the tuned configuration through
 * r4d_cfg_gemm (libr4d/r4d_rows.cpp). The RadTunable table is not part of the emulator, and the
 * cases here launch that file's gather only, so no axis is ever read: this answers with the
 * untuned defaults the table declares (tGemmSkinny: wv 1, sk 0, mb 0, npw 1, nt 1), which is what
 * an install with no tuned key runs. A GEMM case added to the emu would need the real table. */
#include "r4d_args.h"

#include "rad_plugin.h"

#include <cstring>

extern "C" void r4d_cfg_gemm(const RadArgs*, R4dGemmCfg* out) {
    std::memset(out, 0, sizeof(*out));
    out->wv = 1;
    out->npw = 1;
    out->nt = 1;
}

/* row_topk's axes (tTopk), for r4d_rowtopk_bf16.hip: a parameter of the call wins, as in
 * r4d_rows.cpp's tune_i, so a case can pick its stage-1 depth; otherwise the table's defaults
 * (nch 0, pt1 8). */
extern "C" void r4d_cfg_topk(const RadArgs* a, R4dTopkCfg* out) {
    long long v = 0;
    out->rc  = 0;
    out->nch = rad_args_geti(a, "nch", &v) ? (int)v : 0;
    out->pt1 = rad_args_geti(a, "pt1", &v) ? (int)v : 8;
}
