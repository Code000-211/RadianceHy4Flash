/* r4d_fuse_test.cpp -- the chains libr4d's fused rows declare (libr4d/r4d_fuse.cpp).
 *
 * A fused row's `replaces` hook is what a caller runs the kernel against: r4d_selftest and
 * rad-kbench's fusion oracle declare the step ops the hook names, run them on the same inputs as
 * the fused kernel and memcmp the results. The claim therefore has to be exact in the vocabulary --
 * the right ops, in the right order, with every operand wired -- and this test holds the K-24 row
 * (qsa_work_pool_tail) to it without a card: the hook is host code over the ABI, like the row
 * table, and the chain it publishes is checked here.
 *
 * The three steps and the fused op's operands (libr4d/r4d_rows.cpp, oQsaWorkPoolTail):
 *
 *   qsa_work        k pos cu state sidx bt rope stage=tmp page bpos nc
 *   idx_pool_gated  stage=tmp wv wg ape gain pos=bpos page cache out=none
 *   qsa_tail_store  k pos cu state sidx
 *
 * The pool reads the same `page` qsa_work wrote and the same `bpos` as its positions, and writes
 * the cache with no `out` -- which is how the plugin issues the unfused pair. The tail store is
 * exact only where the kernel stores it: a chunk of more than `ring - ratio + 1` tokens is left to
 * the caller, and the hook must decline that geometry rather than claim a chain the kernel did not
 * do (rad_abi.h, RadFuseFn: a chain must be exact).
 */
#include "rad_test.h"

#include "rad_abi.h"
#include "r4d_plugin.h"

#include <cstring>
#include <string>

namespace {

/* The fused op's operands, by index, as libr4d/r4d_rows.cpp lists them. */
enum { kK = 0, kPos, kCu, kState, kSidx, kBt, kWv, kWg, kApe, kGain, kRope, kPage, kBpos, kNc,
       kCache, kN_OPD };

/* Every operand but `rope`, which a gptj geometry may omit. */
constexpr uint64_t kAllPresent = (1ull << kN_OPD) - 1;

RadParam P(const char* k, long long v) { return RAD_INT(k, v); }
/* RAD_RANGE, as the plugin declares M: the row's `M` is a range and the step's copy keeps its kind
 * (r4d_fuse.cpp, Step::mirror). */
RadParam Pd(const char* k, long long hi) { return RAD_RANGE(k, 1, hi); }
RadParam Pf(const char* k, double v) { return RAD_F64(k, v); }
RadParam Ps(const char* k, const char* v) { return RAD_STR(k, v); }

/* A decode step of one sequence: three tokens, a group of four, a tail ring of seven -- nothing
 * closes, which is the step this op exists for, and three is under the kernel's store bound of
 * ring - ratio + 1 = 4, so the chain covers the whole of the fused op's work. */
struct Geo {
    RadParam p[16];
    int n = 0;
    Geo& add(const RadParam& x) { p[n++] = x; return *this; }
    Geo& decode() {
        add(Pd("M", 3)).add(P("n", 128)).add(P("ratio", 4)).add(P("ring", 7))
            .add(P("seqs", 1)).add(P("rotary_dim", 64)).add(Pf("eps", 1e-6))
            .add(Pf("theta", 5e5)).add(Ps("mode", "gptj")).add(P("block_size", 256))
            .add(Ps("dtype", "bf16")).add(P("work", 1));
        return *this;
    }
    const RadParam* operator()() const { return p; }
};

const RadParam* find(const RadFuseStep& s, const char* key) {
    for (int i = 0; i < s.n_p; ++i)
        if (s.p[i].key && !strcmp(s.p[i].key, key)) return &s.p[i];
    return nullptr;
}

int64_t ival(const RadFuseStep& s, const char* key) {
    const RadParam* q = find(s, key);
    return q ? (q->kind == RAD_P_RANGE ? q->ihi : q->ival) : -1;
}

std::string sval(const RadFuseStep& s, const char* key) {
    const RadParam* q = find(s, key);
    return q && q->sval ? q->sval : "";
}

/* Runs the whole chain and returns how many steps it declared. */
int chain(const Geo& g, uint64_t present, RadFuseStep* out, int cap) {
    int n = 0;
    for (int i = 0;; ++i) {
        RadFuseStep s = {};
        const int rc = r4d_fuse_qsa_work_pool_tail(g(), g.n, present, i, &s);
        if (rc != RAD_OK) return n;   /* the end of the chain, or a declined geometry */
        if (n >= cap) return -1;
        out[n++] = s;
    }
}

}  // namespace

/* The three steps, in order, with every operand wired: what the fusion oracle declares and runs. */
TEST(qsa_work_pool_tail_chain_is_the_three_ops) {
    const Geo g = Geo().decode();
    RadFuseStep s[4];
    CHECK_EQ(chain(g, kAllPresent, s, 4), 3);

    CHECK_EQ(std::string(s[0].op), std::string("qsa_work"));
    /* k pos cu state sidx bt rope stage page bpos nc, with the stage a temporary. */
    const int16_t w0[] = { kK, kPos, kCu, kState, kSidx, kBt, kRope, RAD_FUSE_TMP(0),
                           kPage, kBpos, kNc };
    CHECK_EQ(s[0].n_from, (int)(sizeof w0 / sizeof w0[0]));
    for (int i = 0; i < s[0].n_from && i < (int)(sizeof w0 / sizeof w0[0]); ++i)
        CHECK_EQ((int)s[0].from[i], (int)w0[i]);
    CHECK_EQ(ival(s[0], "M"), 3);
    CHECK_EQ(ival(s[0], "ratio"), 4);
    CHECK_EQ(ival(s[0], "ring"), 7);
    CHECK_EQ(ival(s[0], "seqs"), 1);
    CHECK_EQ(sval(s[0], "dtype"), std::string("bf16"));

    /* The pool is one row a block: `work` is the page's width, not the step's token count. */
    CHECK_EQ(std::string(s[1].op), std::string("idx_pool_gated"));
    CHECK_EQ(ival(s[1], "M"), 1 /* work */);
    CHECK_EQ(ival(s[1], "ratio"), 4);
    CHECK_EQ(ival(s[1], "rotary_dim"), 64);
    CHECK_EQ(sval(s[1], "mode"), std::string("gptj"));
    const int16_t w1[] = { RAD_FUSE_TMP(0), kWv, kWg, kApe, kGain, kBpos, kPage, kCache,
                           RAD_FUSE_NONE };
    CHECK_EQ(s[1].n_from, (int)(sizeof w1 / sizeof w1[0]));
    for (int i = 0; i < s[1].n_from && i < (int)(sizeof w1 / sizeof w1[0]); ++i)
        CHECK_EQ((int)s[1].from[i], (int)w1[i]);

    CHECK_EQ(std::string(s[2].op), std::string("qsa_tail_store"));
    const int16_t w2[] = { kK, kPos, kCu, kState, kSidx };
    CHECK_EQ(s[2].n_from, (int)(sizeof w2 / sizeof w2[0]));
    for (int i = 0; i < s[2].n_from && i < (int)(sizeof w2 / sizeof w2[0]); ++i)
        CHECK_EQ((int)s[2].from[i], (int)w2[i]);
    CHECK_EQ(ival(s[2], "ring"), 7);

    /* The chain ends, and it is the end the caller learns the length from. */
    RadFuseStep past = {};
    CHECK_EQ(r4d_fuse_qsa_work_pool_tail(g(), g.n, kAllPresent, 3, &past), RAD_E_NOTFOUND);
}

TEST(qsa_work_pool_tail_rope_only_where_passed) {
    const Geo g = Geo().decode();
    RadFuseStep s[4];
    CHECK_EQ(chain(g, kAllPresent & ~(1ull << kRope), s, 4), 3);
    CHECK_EQ((int)s[0].from[6], (int)RAD_FUSE_NONE);   /* qsa_work's own optional rope */
    CHECK_EQ((int)s[1].from[0], (int)RAD_FUSE_TMP(0));
}

TEST(qsa_work_pool_tail_over_the_tail_bound_declines) {
    /* The kernel stores a sequence's tail while its chunk holds at most ring - ratio + 1 = 4
     * tokens (r4d_qsa_work_pool_tail_bf16.hip); a longer chunk it leaves to qsa_tail_store, so the
     * three-op chain is not the fused op's own work there and the hook declines. */
    Geo g = Geo().decode();
    for (int i = 0; i < g.n; ++i)
        if (!strcmp(g.p[i].key, "M")) g.p[i].ihi = 5;
    RadFuseStep s[4];
    CHECK_EQ(chain(g, kAllPresent, s, 4), 0);          /* no chain at all, not two of three */
    RadFuseStep t = {};
    CHECK_EQ(r4d_fuse_qsa_work_pool_tail(g(), g.n, kAllPresent, 0, &t), RAD_E_UNSUPPORTED);
    /* M == ring - ratio + 1 is still the kernel's own store, so the chain stands there. */
    for (int i = 0; i < g.n; ++i)
        if (!strcmp(g.p[i].key, "M")) g.p[i].ihi = 4;
    CHECK_EQ(chain(g, kAllPresent, s, 4), 3);
}

TEST(qsa_work_pool_tail_without_cache_has_no_chain) {
    /* The fused op has no `out`: a geometry without a cache has nowhere for the pooled key to go,
     * which is also why the plugin never issues it that way. */
    const Geo g = Geo().decode();
    RadFuseStep s = {};
    CHECK_EQ(r4d_fuse_qsa_work_pool_tail(g(), g.n, kAllPresent & ~(1ull << kCache), 0, &s),
             RAD_E_UNSUPPORTED);
}

TEST(qsa_work_pool_tail_imrope_sections_reach_the_pool) {
    Geo g = Geo().decode();
    for (int i = 0; i < g.n; ++i)
        if (!strcmp(g.p[i].key, "mode")) g.p[i].sval = "imrope_gptj";
    g.add(Ps("sections", "[24,20,20]"));
    RadFuseStep s[4];
    CHECK_EQ(chain(g, kAllPresent, s, 4), 3);
    CHECK_EQ(sval(s[1], "mode"), std::string("imrope_gptj"));
    CHECK_EQ(sval(s[1], "sections"), std::string("[24,20,20]"));
}

/* ---------------------------------------------------------------- the row's operand descriptions
 *
 * rad-kbench draws a row's inputs from its `opd_shape` hook and hands them to the kernel
 * (tools/opshapes.cpp, and `--op` in replay mode), and the fusion oracle compares the fused launch
 * with its chain over those buffers (core/runtime/oracle.cpp), so a description the kernel refuses
 * is a row no tool can run at that geometry, whatever the plugin wires by hand. These are the
 * descriptions r4d_qsa_work_pool_tail_bf16.hip checks before it launches: a `mode` that changes an
 * operand's rank or presence has to show up here too, which is what the M-RoPE half pins.
 */
TEST(qsa_work_pool_tail_describes_the_operands_its_kernel_accepts) {
    for (int im = 0; im < 2; ++im) {
        Geo g = Geo().decode();
        for (int i = 0; i < g.n; ++i)
            if (!strcmp(g.p[i].key, "mode")) g.p[i].sval = im ? "imrope_gptj" : "gptj";
        RadOpdDesc d[kN_OPD];
        for (int i = 0; i < kN_OPD; ++i)
            REQUIRE_EQ(r4d_shape_qsa_work_pool_tail(g(), g.n, i, &d[i]), RAD_OK);

        /* k: [M, n], the width the kernel's block copy is compiled for. */
        CHECK_EQ(d[kK].rank, 2u);
        CHECK_EQ(d[kK].shape[1], 128);
        /* positions and cu_seqlens: drawn as runs, not uniform draws. The work list's bounds come
         * off the positions (`lo`, `hi`, and the `k` row a block reads), and a token at a position
         * the step does not hold sends the gather past `k`'s rows. */
        CHECK((d[kPos].flags & RAD_OPD_F_IDX_CU) != 0);
        CHECK((d[kCu].flags & RAD_OPD_F_IDX_CU) != 0);
        /* state: [slots, 1, ring, n] */
        CHECK_EQ(d[kState].rank, 4u);
        CHECK_EQ(d[kState].shape[1], 1);
        CHECK_EQ(d[kState].shape[2], 7);
        CHECK_EQ(d[kState].shape[3], 128);
        /* gain: one dense row, not an [n, 1] column. */
        CHECK_EQ(d[kGain].rank, 1u);
        CHECK_EQ(d[kGain].shape[0], 128);
        /* page: one i32 a work item. */
        CHECK_EQ(d[kPage].rank, 1u);
        CHECK_EQ(d[kPage].dtype, (uint32_t)RAD_I32);
        /* bpos: one position a work item, or the group's three components under M-RoPE. */
        CHECK_EQ(d[kBpos].rank, im ? 2u : 1u);
        CHECK_EQ(d[kBpos].shape[0], 1 /* work */);
        if (im) CHECK(d[kBpos].shape[1] >= 3);
        /* rope: absent under gptj, the step's three planes under M-RoPE (the kernel refuses a null
         * plane there). */
        CHECK_EQ((d[kRope].flags & RAD_OPD_F_ABSENT) != 0, im == 0);
        if (im) {
            CHECK_EQ(d[kRope].rank, 2u);
            CHECK(d[kRope].shape[0] >= 3);
        }
        /* cache: the pool's page, [work, 1, ratio, n / ratio]. */
        CHECK_EQ(d[kCache].rank, 4u);
        CHECK_EQ(d[kCache].shape[1], 1);
        CHECK_EQ(d[kCache].shape[2] * d[kCache].shape[3], 128);
    }
}

RAD_TEST_MAIN()
