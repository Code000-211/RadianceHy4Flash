/* r4d_fuse_chain_test.cpp -- the chain libr4d's `replaces` hook publishes for
 * rmsnorm_ihc_junction and its all-reduce twin (libr4d/r4d_fuse.cpp), step by step.
 *
 * The hook is what `rad-check` and the graph dump read: they run the steps the hook names out of
 * the same library and compare the bytes against the fused launch, so a chain that drops a step,
 * repeats one, or wires a temporary nothing writes is a wrong answer no kernel-side test can see.
 * This file holds the hook to the chain it claims, without a card.
 *
 * The order (arch/hyv4flash: the junction runs ihc_post, then the Engram's gate, then ihc_pre, so
 * the input norm reads the gated residual):
 *
 *   [add] -> [all_reduce] -> rmsnorm -> ihc_post -> [engram_gate] -> ihc_pre -> rmsnorm -> [codes]
 *
 * `tail` 0 stops behind ihc_post: the Engram then writes h between two launches and this chain has
 * no ihc_pre at all.
 */
#include "rad_test.h"

#include "rad_plugin.h"

#include <cstdint>
#include <string>
#include <vector>

extern "C" int r4d_fuse_rmsnorm_ihc_junction(const RadParam* p, int n_p, uint64_t present,
                                             int step, RadFuseStep* out);
extern "C" int r4d_fuse_ar_rmsnorm_ihc_junction(const RadParam* p, int n_p, uint64_t present,
                                                int step, RadFuseStep* out);

namespace {

/* The operands of the junction kernel, in the order its schema has them. */
enum {
    OPD_A = 0, OPD_B = 1, OPD_W_POST = 2, OPD_POST = 3, OPD_H = 4, OPD_FN = 5, OPD_SCALE = 6,
    OPD_BASE = 7, OPD_W_NEXT = 8, OPD_XN = 9, OPD_Y = 10, OPD_X = 11, OPD_XQ = 12, OPD_XS = 13,
    OPD_EG_KEY = 14,
};

struct ChainStep {
    std::string      op;
    std::vector<int> from;
};

using FuseFn = int (*)(const RadParam*, int, uint64_t, int, RadFuseStep*);

/* The whole chain, walked the way a caller walks it: steps until RAD_E_NOTFOUND. */
std::vector<ChainStep> chain_of(FuseFn fn, const RadParam* p, int n_p, uint64_t present) {
    std::vector<ChainStep> out;
    for (int k = 0; k < 16; ++k) {
        RadFuseStep s{};
        const int rc = fn(p, n_p, present, k, &s);
        if (rc == RAD_E_NOTFOUND) break;
        CHECK_EQ(rc, RAD_OK);
        if (rc != RAD_OK) break;
        out.push_back({ s.op ? std::string(s.op) : std::string("<null>"),
                        std::vector<int>(s.from, s.from + s.n_from) });
    }
    return out;
}

std::string ops_of(const std::vector<ChainStep>& c) {
    std::string s;
    for (const auto& x : c) s += (s.empty() ? "" : " ") + x.op;
    return s;
}

int count_of(const std::vector<ChainStep>& c, const char* op) {
    int n = 0;
    for (const auto& s : c) n += s.op == op;
    return n;
}

struct Geom {
    std::vector<RadParam> p;
    Geom(long long M, long long n, long long hc, long long rows, long long tail, long long engram,
         long long world) {
        p = { RAD_INT("M", M), RAD_INT("n", n), RAD_INT("hc", hc), RAD_INT("rows", rows),
              RAD_INT("tail", tail), RAD_INT("engram", engram), RAD_INT("world_size", world),
              RAD_INT("dtype", 3), RAD_F64("eps", 1e-6), RAD_F64("ihc_eps", 1e-6),
              RAD_F64("hc_eps", 1e-6), RAD_F64("eg_eps", 1e-6), RAD_F64("post_mag", 2.0) };
    }
    const RadParam* data() const { return p.data(); }
    int size() const { return (int)p.size(); }
};

constexpr uint64_t no_b = ~(1ull << OPD_B);
constexpr uint64_t no_bxy = no_b & ~((1ull << OPD_Y) | (1ull << OPD_X));

/* The Engram form: the gate sits between ihc_post and ihc_pre, exactly once, and ihc_pre still
 * runs -- before P-07's step shift it did not, and the gate ran twice instead. */
TEST(the_junction_chain_puts_the_engram_gate_in_front_of_ihc_pre) {
    const Geom g(3, 4096, 3072, 6144, 1, 1, 1);
    const auto c = chain_of(r4d_fuse_rmsnorm_ihc_junction, g.data(), g.size(), no_bxy);
    CHECK_EQ(ops_of(c), std::string("rmsnorm ihc_post engram_gate ihc_pre rmsnorm "
                                    "had_quant_act_i8g"));
    CHECK_EQ(count_of(c, "engram_gate"), 1);
    CHECK_EQ(count_of(c, "ihc_pre"), 1);

    /* the gate reads the h ihc_post wrote (operand 4), and ihc_pre reads it after the gate */
    CHECK_EQ(c[1].from[2], (int)OPD_H);           /* ihc_post: y, post, h */
    CHECK_EQ(c[2].from.back(), (int)OPD_H);       /* engram_gate writes h in place */
    CHECK_EQ(c[3].from[0], (int)OPD_H);           /* ihc_pre: h, fn, scale, base, x, post */
    /* ihc_pre's x is the temp the norm behind it reads: the wiring a caller resolves */
    CHECK_EQ(c[3].from[4], c[4].from[0]);
    CHECK_EQ(c[3].from[4], (int)RAD_FUSE_TMP(1));
}

/* With `y` and `x` among the operands the caller passed, the chain names those: ihc_pre writes
 * `x`, the norm behind it reads `x` and writes `xn`, the codes are read off `xn`. */
TEST(the_junction_chain_wires_ihc_pre_into_the_next_norm) {
    const Geom g(2, 1024, 512, 1024, 1, 1, 1);
    const auto c = chain_of(r4d_fuse_rmsnorm_ihc_junction, g.data(), g.size(), no_b);
    CHECK_EQ((int)c.size(), 6);
    CHECK_EQ(c[1].from[0], (int)OPD_Y);           /* the norm's y is the caller's */
    CHECK_EQ(c[3].from[4], (int)OPD_X);           /* ihc_pre: x */
    CHECK_EQ(c[4].from[0], (int)OPD_X);           /* the norm reads it */
    CHECK_EQ(c[4].from[2], (int)OPD_XN);          /* and writes xn */
    CHECK_EQ(c[5].from[0], (int)OPD_XN);          /* the codes are read off xn */
}

/* Without `engram` the chain is the one it always was: five steps, no gate. */
TEST(the_plain_junction_chain_has_no_gate) {
    const Geom g(1, 1024, 512, 1024, 1, 0, 1);
    const auto c = chain_of(r4d_fuse_rmsnorm_ihc_junction, g.data(), g.size(), no_b);
    CHECK_EQ(ops_of(c), std::string("rmsnorm ihc_post ihc_pre rmsnorm had_quant_act_i8g"));
}

/* `tail` 0: the junction stops behind ihc_post, so the Engram writes h between two launches. */
TEST(a_tail_zero_junction_stops_behind_ihc_post) {
    const Geom g(1, 1024, 512, 512, 0, 1, 1);
    const auto c = chain_of(r4d_fuse_rmsnorm_ihc_junction, g.data(), g.size(), no_b);
    CHECK_EQ(ops_of(c), std::string("rmsnorm ihc_post"));
}

/* The gate's operands absent: the chain has no gate to publish, and saying so is what stops the
 * plugin declaring the form. */
TEST(the_engram_form_is_not_published_without_its_operands) {
    const Geom g(1, 1024, 512, 1024, 1, 1, 1);
    const uint64_t present = no_b & ~(1ull << OPD_EG_KEY);
    RadFuseStep s{};
    CHECK_EQ(r4d_fuse_rmsnorm_ihc_junction(g.data(), g.size(), present, 2, &s), RAD_E_NOTFOUND);
}

/* The head form (rows == hc) has no post to write and no such gate to run. */
TEST(the_engram_form_is_refused_for_the_head_rows) {
    const Geom g(1, 1024, 512, 512, 1, 1, 1);
    RadFuseStep s{};
    CHECK_EQ(r4d_fuse_rmsnorm_ihc_junction(g.data(), g.size(), no_b, 2, &s), RAD_E_UNSUPPORTED);
}

/* The all-reduce twin publishes the same chain behind the exact reduction over `a`. */
TEST(the_all_reduce_twin_publishes_the_reduction_in_front) {
    const Geom g(3, 4096, 3072, 6144, 1, 1, 2);
    const auto c = chain_of(r4d_fuse_ar_rmsnorm_ihc_junction, g.data(), g.size(), no_b);
    CHECK_EQ(ops_of(c), std::string("all_reduce rmsnorm ihc_post engram_gate ihc_pre rmsnorm "
                                    "had_quant_act_i8g"));
    CHECK_EQ(c[0].from[0], (int)OPD_A);           /* in place over a */
    CHECK_EQ(c[0].from[1], (int)RAD_FUSE_NONE);
}

/* `b` present adds the MoE join's add in front of everything. */
TEST(the_add_step_comes_first_when_b_is_present) {
    const Geom g(1, 1024, 512, 512, 1, 0, 1);
    const auto c = chain_of(r4d_fuse_rmsnorm_ihc_junction, g.data(), g.size(),
                            RAD_FUSE_PRESENT_ALL);
    CHECK_EQ(c[0].op, std::string("add"));
    CHECK_EQ((int)c[0].from.size(), 3);
    CHECK_EQ(c[0].from[0], (int)OPD_A);
    CHECK_EQ(c[0].from[1], (int)OPD_B);
    CHECK_EQ(c[0].from[2], (int)OPD_A);
}

}  /* namespace */

RAD_TEST_MAIN()
