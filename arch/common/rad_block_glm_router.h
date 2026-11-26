/* GLM noaux_tc router. Correction bias changes selection only; mixture weights
 * come from the uncorrected sigmoid probabilities. All routing arithmetic is F32. With `groups`
 * above 1 the selection is group-limited (router_sigmoid_group_topk, Ling-3.0's 8 groups, 4 kept). */
#ifndef RAD_BLOCK_GLM_ROUTER_H
#define RAD_BLOCK_GLM_ROUTER_H
#include "rad_glm_tp.h"
#include <cmath>

namespace rad {
namespace arch {
struct GlmRouter {
    struct Config {
        int64_t experts = 288, top_k = 8, rows = 32, groups = 1, topk_groups = 1;
        int norm = 1;
        /* Declare the projection weight as stored bf16 (gemm_nt_f32's `weight_bf16`) when its
         * stored encoding is bf16 or unknown; otherwise, or when false, it is declared f32. */
        bool weight_bf16 = false;
        /* Run the prefill projection on the bf16 WMMA tile (gemm_nt_f32_wmma): the column split's
         * part where that split is set up, and otherwise the replicated projection's own prefill
         * rows. LOSSY: the same bf16 products and f32 accumulation, summed in the tile's order, so
         * a near-tie routing pick can flip. Off by default; the rows below GlmTp::kSplitRows
         * (decode, and a verify step that short) keep gemm_nt_f32 whatever it says
         * (notes/backlog/lossy/P-13.md). */
        bool wmma = false;
        double route_scale = 2.5;
    };
    struct Src {
        const char *weight = nullptr, *correction = nullptr;
    };
    struct Wire {
        rad_buf input = 0, logits = 0, ids = 0, weights = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    rad_weight projection = 0, correction = 0;
    rad_op op_project = 0, op_project_tile = 0, op_select = 0;
    SplitF32Proj split{};

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int layer, const char* name,
                const Config& cfg, const Wire& wire, const Src& src) {
        g = geom;
        c = cfg;
        w = wire;
        if (!b || !name || !src.weight || !src.correction || g.n_embd < 1 || c.experts < 1 ||
            c.experts > 512 || c.top_k < 1 || c.top_k > c.experts || c.rows < 1 ||
            c.rows > g.max_tok || !std::isfinite(c.route_scale) || c.route_scale <= 0 ||
            c.groups < 1 || c.experts % c.groups || c.topk_groups < 1 || c.topk_groups > c.groups ||
            !w.input || !w.logits || !w.ids || !w.weights || w.logits == w.weights)
            return RAD_E_INVAL;
        const char *pn = nm.f("%s.weight", name), *cn = nm.f("%s.correction", name);
        RAD_ARCH_TRY(map_copy(b, pn, src.weight));
        RAD_ARCH_TRY(map_copy(b, cn, src.correction));
        const bool packed = c.weight_bf16 && weight_codes_dtype(b, pn, RAD_BF16) == RAD_BF16;
        projection = decl_w(b, pn, packed ? RAD_BF16 : RAD_F32, { c.experts, g.n_embd },
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(layer));
        correction = decl_w(b, cn, RAD_F32, { c.experts }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(layer));
        if (!projection || !correction) return RAD_E_INVAL;
        /* P-13 / F-19: with the switch on, the prefill band of the replicated projection runs on
         * the bf16 tile too, which is the only prefill path at one rank and the one a world the
         * expert count does not divide evenly keeps. Where the column split takes those rows, the
         * band never runs, so it is not declared and the graph stays the split's own (the lane
         * re-checks the split against this band, at one rank, with the switch on at both).
         * gemm_nt_f32 keeps every row below GlmTp::kSplitRows in the op declared here, so decode
         * and verify steps shorter than that do not move. */
        const bool tile = c.wmma && c.rows >= GlmTp::kSplitRows &&
                          !SplitF32Proj::columns_split(g.world, c.experts) &&
                          glm_projection_tile_serves(g.n_embd, c.experts, packed);
        op_project = declare_glm_projection(b, tile ? GlmTp::kSplitRows - 1 : c.rows, c.experts,
                                            g.n_embd, projection, packed, w.input, w.logits);
        if (tile)
            op_project_tile = declare_glm_projection(b, c.rows, c.experts, g.n_embd, projection,
                                                     packed, w.input, w.logits, true);
        RAD_ARCH_TRY(split.declare(b, nm, g, name, c.rows, c.experts, g.n_embd, projection, w.input,
                                   w.logits, packed, c.wmma));
        if (c.groups > 1)
            op_select =
                rw(b,
                   RAD_OP(b, "router_sigmoid_group_topk",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("n_group", c.groups),
                                     RAD_INT("topk_group", c.topk_groups), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction)),
                   { w.logits }, { w.ids, w.weights });
        else
            op_select =
                rw(b,
                   RAD_OP(b, "router_sigmoid_topk",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction)),
                   { w.logits }, { w.ids, w.weights });
        return op_project && op_select ? RAD_OK : RAD_E_INVAL;
    }
    void project(RadCtx* ctx, int64_t first, int64_t rows, int64_t output_first = 0) const {
        const RadOperand in = brow_slice(w.input, first, rows, g.n_embd);
        const RadOperand out = brow_slice(w.logits, output_first, rows, c.experts);
        if (split.takes(rows)) split.step(ctx, rows, in, projection, out);
        else if (op_project_tile && rows >= GlmTp::kSplitRows)
            RAD_ISSUE_N(ctx, op_project_tile, rows, in, RAD_W(projection), out);
        else RAD_ISSUE_N(ctx, op_project, rows, in, RAD_W(projection), out);
    }
    void step(RadCtx* ctx, int64_t first, int64_t rows, int64_t output_first = 0) const {
        project(ctx, first, rows, output_first);
        RAD_ISSUE_N(ctx, op_select, rows, brow_slice(w.logits, output_first, rows, c.experts),
                    RAD_W(correction), brow_slice(w.ids, output_first, rows, c.top_k),
                    brow_slice(w.weights, output_first, rows, c.top_k));
    }
};
}
} // namespace rad::arch
#endif
