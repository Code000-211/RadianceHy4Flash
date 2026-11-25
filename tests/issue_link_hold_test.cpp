/* issue_link_hold_test -- which collectives take the stager's link hold (Ctx::issue_body).
 *
 * In a staged pass a collective holds the card's link (rad_link_hold on, then off), so the
 * stager's copy lists yield meanwhile. Not on lane 1: the gate is one word a card and its parity
 * is the hold, so two holds that overlap in time lose it. A pipelined prefill's lane-1 reduce ran
 * beside a lane-0 collective (the router's all_gather), both ending their hold in the kernel, and
 * the gate was left odd on every card with nothing holding: a yielding list stopped for good
 * (issue #2, Ctx::issue_body). This runs the pass on
 * the linked backend with a Program built by hand around a recording kernel, as issue_div_test
 * does, and counts the calls to rad_link_hold, which the link wraps (tests/CMakeLists.txt). It
 * checks the issue path only, not the hang, which needs four cards.
 */
#include "rad_test.h"

#include "device/device.h"
#include "runtime/ctx.h"
#include "runtime/stager.h"

#include <functional>
#include <vector>

using rad::Ctx;
using rad::CtxDesc;
using rad::Program;

extern "C" int __real_rad_link_hold(RadStream s, int on);
static std::vector<int> g_holds;   /* the `on` of each rad_link_hold call, in order */
extern "C" int __wrap_rad_link_hold(RadStream s, int on) {
    g_holds.push_back(on);
    return __real_rad_link_hold(s, on);
}

namespace {

int g_launches = 0;
int k_record(const RadArgs*, RadStream) { ++g_launches; return RAD_OK; }

const RadParamSpec   g_params[] = { { "M", RAD_P_INT, RAD_REQUIRED, 0 } };
const RadOperandSpec g_opds[]   = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
const RadOpSchema    g_schema   = { "all_reduce", g_params, 1, g_opds, 2, "any M" };

std::function<void(Ctx&)> g_body;
void trampoline(RadCtx* c, const RadBatch*) { if (g_body) g_body(*static_cast<Ctx*>(c)); }

/* Arms for every pass and stages nothing. */
struct NullStager final : rad::OpStager {
    bool begin_pass(const RadBatch*) override { return true; }
    int  before_op(rad_op, RadStream, bool*) override { return RAD_OK; }
    int  end_pass(RadStream) override { return RAD_OK; }
};

/* One all_reduce, served by a device row. */
struct Fixture {
    Program       P;
    RadKernelInfo info{};
    rad::KernelRow row;
    Ctx           ctx;
    NullStager    stager;
    RadBatch      batch{};
    rad_op        op = 1;

    explicit Fixture(bool staged) {
        P.meta.n_layers = 1;
        P.arena_bytes = 8192;
        P.scratch_bytes = 0;
        P.weights.resize(1);
        P.buffers.resize(1);
        P.ops.resize(1);
        for (const char* n : { "x", "y" }) {
            rad::BufferInfo b;
            b.name = n;
            b.decl.dtype = RAD_BF16;
            b.decl.rank = 2;
            b.decl.shape[0] = 64;
            b.decl.shape[1] = 4;
            b.decl.domain = RAD_DOMAIN_DEVICE;
            b.bytes = rad_dtype_bytes(RAD_BF16, 256);
            b.arena_offset = (int64_t)(P.buffers.size() - 1) * 1024;
            P.buffers.push_back(b);
        }
        info.name = "t_all_reduce";
        info.op = "all_reduce";
        info.family = "test";
        info.domain = RAD_DOMAIN_DEVICE;
        info.launch = k_record;
        row.info = &info;
        row.plugin = "fake";
        rad::OpInfo o;
        o.op = "all_reduce";
        o.index = 1;
        o.ranged_key = "M";
        o.range_lo = 1;
        o.range_hi = 64;
        o.schema = &g_schema;
        rad::Band b;
        b.hi = 64;
        b.miss[RAD_DOMAIN_HOST] = "no host kernel";
        rad::Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
        r.row = &row;
        r.geom.set_i("M", 64);
        o.bands.push_back(b);
        P.ops.push_back(o);
        if (staged) ctx.set_stager(&stager);
    }
    int start() {
        CtxDesc d;
        d.program = &P;
        d.rank = 0;
        d.world_size = 1;
        d.arch_step = trampoline;
        return ctx.init(d);
    }
    /* Issues the all_reduce on `lane` and returns the step's status. */
    int issue_on(int lane) {
        batch.phase = RAD_PHASE_PREFILL;
        batch.n_tok = 1;
        batch.n_seq = 1;
        g_holds.clear();
        g_launches = 0;
        g_body = [&](Ctx& c) {
            CHECK_OK(c.set_lane(lane));
            RadOperand opd[2] = { RAD_B(1), RAD_B(2) };
            CHECK_OK(c.issue(op, opd, 2, 64));
        };
        const int s = ctx.run_step(&batch);
        g_body = nullptr;
        return s;
    }
};

}  /* namespace */

TEST(a_staged_collective_on_lane_0_holds_the_link_and_releases_it) {
    Fixture f(true);
    REQUIRE_EQ(f.start(), RAD_OK);
    CHECK_OK(f.issue_on(0));
    CHECK_EQ(g_launches, 1);
    REQUIRE_EQ(g_holds.size(), 2u);
    CHECK_EQ(g_holds[0], 1);
    CHECK_EQ(g_holds[1], 0);
}

TEST(a_staged_collective_on_lane_1_takes_no_link_hold) {
    Fixture f(true);
    REQUIRE_EQ(f.start(), RAD_OK);
    CHECK_OK(f.issue_on(1));
    CHECK_EQ(g_launches, 1);
    CHECK(g_holds.empty());
}

TEST(an_unstaged_collective_takes_no_link_hold_on_either_lane) {
    Fixture f(false);
    REQUIRE_EQ(f.start(), RAD_OK);
    for (int lane : { 0, 1 }) {
        CHECK_OK(f.issue_on(lane));
        CHECK_EQ(g_launches, 1);
        CHECK(g_holds.empty());
    }
}

RAD_TEST_MAIN()
