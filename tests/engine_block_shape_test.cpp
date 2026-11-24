/* D-07: the block and the rounds a step's chain is staged at (Engine::block_shape, from the depth
 * rule's last decision), and the block a drafter's chain may be cut to (Engine::any_block_of, the
 * rule declare() resolves from rad_drafter_any_block).
 *
 * A block drafter proposes its whole chain in one pass, so the rounds the depth rule will not take
 * are a head pass and a correction the verify never reads: a drafter that serves any block from the
 * anchor and one row up is staged at the last decision's rounds plus one (Scheduler::chain_rounds).
 * A token is the same either way, so the check is on the staged shape alone.
 */
#include "rad_test.h"

#include "engine.h"

namespace rad {
struct EngineBlockShapeTestAccess {
    /* A block drafter of five rows: four drafts, as DFly2 declares. `decided` is the depths the
     * rule's last step took (Engine::depth_last_), `load` the drafting rows of the step just run
     * and `counted` the load-aware steps exploration has counted so far. */
    static void stage(Engine& e, int64_t any_block, const DraftDepthCfg& cfg, int decided,
                      int load = 0, uint64_t counted = 0) {
        e.drafter_ = RadDrafterDecl{};
        e.drafter_.kind = RAD_DRAFT_BLOCK;
        e.drafter_.depth = 4;
        e.drafter_.block = 5;
        e.drafter_any_block_ = any_block;
        e.depth_block_ = cfg.on;
        e.depth_cfg_ = cfg;
        e.depth_last_ = decided;
        e.depth_note_rows_ = load;
        e.depth_explore_n_ = counted;
        e.depth_explore_step_ = false;
    }
    static void shape(Engine& e, int* block, int* steps) { e.block_shape(block, steps); }
    static bool explored(const Engine& e) { return e.depth_explore_step_; }
    static uint64_t counted(const Engine& e) { return e.depth_explore_n_; }
    static int64_t any_block_of(int kind, bool tree, int64_t declared) {
        return Engine::any_block_of(kind, tree, declared);
    }
};
}

namespace {
rad::DraftDepthCfg depth_rule_on() {
    rad::DraftDepthCfg c;
    c.on = true;
    c.min = 1;
    c.max = 4;
    return c;
}
/* The block and the rounds of a step's chain, after staging a block drafter of five rows. */
struct Shape { int block = 0, steps = 0; };
Shape shape(rad::Engine& e) {
    Shape s;
    rad::EngineBlockShapeTestAccess::shape(e, &s.block, &s.steps);
    return s;
}
}

TEST(a_step_is_staged_at_the_rounds_the_depth_rule_took_last_step) {
    const rad::DraftDepthCfg cfg = depth_rule_on();
    rad::Config config;
    rad::Engine engine(config);
    /* Two rounds last step: this one may take three, never four: the chain is staged at four rows
     * (the anchor and three), not at the declared block of five. */
    rad::EngineBlockShapeTestAccess::stage(engine, 2, cfg, /*decided=*/2);
    Shape s = shape(engine);
    CHECK_EQ(s.steps, 3);
    CHECK_EQ(s.block, 4);
    /* The first step has no last decision: it runs the whole block, and the rule cuts nothing. */
    rad::EngineBlockShapeTestAccess::stage(engine, 2, cfg, /*decided=*/0);
    s = shape(engine);
    CHECK_EQ(s.steps, 4);
    CHECK_EQ(s.block, 5);
    /* The floor: the rule always takes draft-depth-min rounds, and the drafter needs the anchor
     * and one row. One round past a decision of one is below the floor of three. */
    rad::DraftDepthCfg floor = cfg;
    floor.min = 3;
    rad::EngineBlockShapeTestAccess::stage(engine, 2, floor, /*decided=*/1);
    s = shape(engine);
    CHECK_EQ(s.steps, 3);
    CHECK_EQ(s.block, 4);
}

TEST(a_block_is_staged_whole_without_the_depth_rule_or_without_an_any_block_drafter) {
    const rad::DraftDepthCfg cfg = depth_rule_on();
    rad::Config config;
    rad::Engine engine(config);
    /* A drafter that proposes its block only whole (rad_drafter_any_block 0): nothing to cut. */
    rad::EngineBlockShapeTestAccess::stage(engine, /*any_block=*/0, cfg, /*decided=*/2);
    Shape s = shape(engine);
    CHECK_EQ(s.steps, 4);
    CHECK_EQ(s.block, 5);
    /* No depth rule: the chain is the declared block. */
    rad::DraftDepthCfg off = cfg;
    off.on = false;
    rad::EngineBlockShapeTestAccess::stage(engine, /*any_block=*/2, off, /*decided=*/2);
    s = shape(engine);
    CHECK_EQ(s.steps, 4);
    CHECK_EQ(s.block, 5);
    /* And no exploring step is counted where the rule is off. */
    CHECK_EQ(rad::EngineBlockShapeTestAccess::counted(engine), (uint64_t)0);
}

TEST(an_exploring_step_keeps_the_whole_block_it_verifies) {
    rad::DraftDepthCfg cfg = depth_rule_on();
    cfg.load = true;
    cfg.load_from = 1;
    cfg.explore_every = 1;                     /* every load-aware step is an exploring one */
    rad::Config config;
    rad::Engine engine(config);
    /* One drafting row, the load of the step just run: the rule explores here, so the block stays
     * whole even though the last decision was two rounds (which would stage four rows). */
    rad::EngineBlockShapeTestAccess::stage(engine, 2, cfg, /*decided=*/2, /*load=*/1);
    Shape s = shape(engine);
    CHECK(rad::EngineBlockShapeTestAccess::explored(engine));
    CHECK_EQ(rad::EngineBlockShapeTestAccess::counted(engine), (uint64_t)1);
    CHECK_EQ(s.steps, 4);
    CHECK_EQ(s.block, 5);
    /* Five drafting rows is not a step exploration visits (kExploreBelow): it is staged at the
     * rule's rounds. */
    rad::EngineBlockShapeTestAccess::stage(engine, 2, cfg, /*decided=*/2, /*load=*/5);
    s = shape(engine);
    CHECK(!rad::EngineBlockShapeTestAccess::explored(engine));
    CHECK_EQ(s.steps, 3);
    CHECK_EQ(s.block, 4);
    /* Default cadence: exploration is the 20th load-aware step, not the 19th ... */
    cfg.explore_every = 20;
    rad::EngineBlockShapeTestAccess::stage(engine, 2, cfg, /*decided=*/2, /*load=*/1,
                                           /*counted=*/18);
    s = shape(engine);
    CHECK(!rad::EngineBlockShapeTestAccess::explored(engine));
    CHECK_EQ(s.steps, 3);
    /* ... and the 20th, which is staged whole and counted. */
    rad::EngineBlockShapeTestAccess::stage(engine, 2, cfg, /*decided=*/2, /*load=*/1,
                                           /*counted=*/19);
    s = shape(engine);
    CHECK(rad::EngineBlockShapeTestAccess::explored(engine));
    CHECK_EQ(rad::EngineBlockShapeTestAccess::counted(engine), (uint64_t)20);
    CHECK_EQ(s.steps, 4);
    CHECK_EQ(s.block, 5);
}

TEST(only_a_block_drafters_chain_is_staged_at_any_block) {
    using A = rad::EngineBlockShapeTestAccess;
    CHECK_EQ(A::any_block_of(RAD_DRAFT_BLOCK, /*tree=*/false, 2), (int64_t)2);
    CHECK_EQ(A::any_block_of(RAD_DRAFT_BLOCK, /*tree=*/false, 5), (int64_t)5);
    CHECK_EQ(A::any_block_of(RAD_DRAFT_BLOCK, /*tree=*/true, 2), (int64_t)0);
    CHECK_EQ(A::any_block_of(RAD_DRAFT_SERIAL, /*tree=*/false, 2), (int64_t)0);
    CHECK_EQ(A::any_block_of(RAD_DRAFT_NONE, /*tree=*/false, 2), (int64_t)0);
    CHECK_EQ(A::any_block_of(RAD_DRAFT_BLOCK, /*tree=*/false, 0), (int64_t)0);
}

RAD_TEST_MAIN()
