/* w2_test.cpp -- "Weg 2" (abi/rad_w2.h, core/runtime/w2.h; issue #309), the parts that need no card:
 * the copy budget from the measured whole-unit list times, the window by distinct routed experts,
 * the switch and its knobs, the budget table a layer's control block carries, the post's sum, the
 * tail grid's bound, the decode band the form keeps inside (issue #324), and the switch-off
 * contract of the runtime entry (null: the block issues the layer as it always did). The plan in
 * the router's launch is tests/r4d_emu/test_moefuse.cpp's, the tail split's issue order
 * tests/hyv4flash_arch_test.cpp's, the split launches' bytes r4d_selftest's (r4d_selftest_w2.h). */
#include "rad_test.h"

#include "rad_runtime.h"
#include "rad_w2.h"
#include "runtime/w2.h"

#include <cstring>
#include <map>
#include <string>

using namespace rad;

namespace {

std::map<std::string, std::string> g_env;
const char* fake_env(const char* k) {
    const auto it = g_env.find(k);
    return it == g_env.end() ? nullptr : it->second.c_str();
}

}  // namespace

/* The owner's table (measurements/M-20.md, whole units, both cards at once): 90.6 us at one unit,
 * about 83 us a unit from two, and about 10 us of host hand-over. Not a single down copy's 57 us. */
TEST(the_budget_is_the_measured_whole_unit_list_time_plus_the_handover) {
    CHECK_EQ(rad_w2_list_us(0), 0.0);
    CHECK(std::abs(rad_w2_list_us(1) - 100.6) < 1e-9);
    CHECK(std::abs(rad_w2_list_us(2) - 176.0) < 1e-9);       /* measured 172.4 + 10 */
    CHECK(std::abs(rad_w2_list_us(4) - 342.0) < 1e-9);       /* measured 339.4 + 10 */
    CHECK(std::abs(rad_w2_list_us(8) - 674.0) < 1e-9);       /* measured 665.4 + 10 */
    /* a list of n whole units is never budgeted at n single down copies (57 us) */
    for (int n = 1; n <= RAD_W2_MAX_COPY; ++n) CHECK(rad_w2_list_us(n) > 57.0 * n);
}

TEST(the_window_holds_six_to_seven_units_at_dfly2_c8_and_one_at_c1) {
    /* DFly2 c8's window of about 620 us, c1's of about 154 us (issue #309). */
    CHECK_EQ(rad_w2_kmax(620.0, RAD_W2_MAX_COPY), 7);
    CHECK_EQ(rad_w2_kmax(600.0, RAD_W2_MAX_COPY), 7);
    CHECK_EQ(rad_w2_kmax(590.0, RAD_W2_MAX_COPY), 6);
    CHECK_EQ(rad_w2_kmax(154.0, RAD_W2_MAX_COPY), 1);
    /* no-spec c1 (66 us) holds none, no-spec c8 (281 us) three */
    CHECK_EQ(rad_w2_kmax(66.0, RAD_W2_MAX_COPY), 0);
    CHECK_EQ(rad_w2_kmax(281.0, RAD_W2_MAX_COPY), 3);
    /* the edges: exactly one list, a hair under it, the cap, and the ring's size */
    CHECK_EQ(rad_w2_kmax(100.6, RAD_W2_MAX_COPY), 1);
    CHECK_EQ(rad_w2_kmax(100.5, RAD_W2_MAX_COPY), 0);
    CHECK_EQ(rad_w2_kmax(620.0, 3), 3);
    CHECK_EQ(rad_w2_kmax(620.0, 0), 0);
    CHECK_EQ(rad_w2_kmax(1e9, 100), RAD_W2_MAX_COPY);
    CHECK_EQ(rad_w2_kmax(-5.0, RAD_W2_MAX_COPY), 0);
}

TEST(the_window_by_distinct_experts_goes_through_the_lanes_four_shapes) {
    /* design/M-20.md §2, §3: distinct routed experts a layer and the window measured there. */
    struct Shape { double distinct, window_us; int units; };
    const Shape shapes[] = { { 16.9, 66, 0 }, { 51.9, 154, 1 }, { 99.2, 281, 3 }, { 203.2, 620, 7 } };
    for (const Shape& s : shapes) {
        const double w = rad_w2_window_us((int)(s.distinct + 0.5));
        CHECK(std::abs(w - s.window_us) < 0.12 * s.window_us);
        CHECK_EQ(rad_w2_kmax(w, RAD_W2_MAX_COPY), s.units);
    }
}

TEST(the_switch_is_on_by_default_off_at_zero_and_its_knobs_are_clamped) {
    g_env.clear();
    W2Config c = w2_config(fake_env);
    CHECK(c.on);                                             /* on unless RADIANCE_W2=0 */
    CHECK_EQ(c.min_rows, 16);
    CHECK_EQ(c.kmax, RAD_W2_MAX_COPY);
    CHECK(c.ring_cached);                                    /* ordinary device memory + an acquire */
    g_env["RADIANCE_W2_RING"] = "uncached";
    CHECK(!w2_config(fake_env).ring_cached);
    g_env["RADIANCE_W2_RING"] = "vram";
    CHECK(w2_config(fake_env).ring_cached);
    g_env["RADIANCE_W2"] = "0";
    CHECK(!w2_config(fake_env).on);
    g_env["RADIANCE_W2"] = "";
    CHECK(w2_config(fake_env).on);
    g_env["RADIANCE_W2"] = "1";
    CHECK(w2_config(fake_env).on);
    g_env["RADIANCE_W2"] = "0";                              /* the knobs below do not turn it on */
    g_env["RADIANCE_W2_MIN_ROWS"] = "6";
    g_env["RADIANCE_W2_KMAX"] = "3";
    g_env["RADIANCE_W2_WIN_US0"] = "20.5";
    g_env["RADIANCE_W2_WIN_US_EXPERT"] = "2.25";
    c = w2_config(fake_env);
    CHECK(!c.on);
    CHECK_EQ(c.min_rows, 6);
    CHECK_EQ(c.kmax, 3);
    CHECK_EQ(c.win_us0, 20.5);
    CHECK_EQ(c.win_us_expert, 2.25);
    /* out of range, and not a number */
    g_env["RADIANCE_W2_MIN_ROWS"] = "0";
    g_env["RADIANCE_W2_KMAX"] = "99";
    g_env["RADIANCE_W2_WIN_US0"] = "abc";
    g_env["RADIANCE_W2_WIN_US_EXPERT"] = "-4";
    c = w2_config(fake_env);
    CHECK_EQ(c.min_rows, 1);
    CHECK_EQ(c.kmax, RAD_W2_MAX_COPY);
    CHECK_EQ(c.win_us0, RAD_W2_WIN_US0);
    CHECK_EQ(c.win_us_expert, 0.0);
    g_env.clear();
}

TEST(the_budget_table_rises_with_the_routed_experts_and_stops_at_the_cap) {
    W2Config c;
    uint8_t km[RAD_W2_MAX_EXPERT + 1];
    w2_fill_kmax(c, km);
    CHECK_EQ((int)km[0], 0);
    CHECK_EQ((int)km[17], 0);
    CHECK_EQ((int)km[52], 1);
    CHECK_EQ((int)km[99], 3);
    CHECK_EQ((int)km[203], 7);
    CHECK_EQ((int)km[RAD_W2_MAX_EXPERT], RAD_W2_MAX_COPY);
    for (int d = 1; d <= RAD_W2_MAX_EXPERT; ++d) {
        CHECK(km[d] >= km[d - 1]);
        CHECK_EQ((int)km[d], rad_w2_kmax(rad_w2_window_us(d), RAD_W2_MAX_COPY));
    }
    c.kmax = 2;
    w2_fill_kmax(c, km);
    for (int d = 0; d <= RAD_W2_MAX_EXPERT; ++d) CHECK(km[d] <= 2);
    CHECK_EQ((int)km[203], 2);
    c.kmax = 0;                                              /* copies off: every miss in place */
    w2_fill_kmax(c, km);
    for (int d = 0; d <= RAD_W2_MAX_EXPERT; ++d) CHECK_EQ((int)km[d], 0);
}

TEST(the_ring_and_the_control_blocks_are_what_the_vram_plan_reserves) {
    constexpr int64_t kGrain = (int64_t)2 << 20;
    const int64_t unit = 1499904;                            /* Hy4-flash's whole unit */
    const int64_t b = w2_device_bytes(unit, 41);
    CHECK_EQ(b % kGrain, (int64_t)0);
    /* the ring's three groups of eight slots and their done words, the layers' blocks, two words */
    CHECK(b >= 3 * 8 * unit + 4096 + 41 * (int64_t)sizeof(RadW2Layer));
    CHECK(b <= 3 * 8 * unit + 41 * (int64_t)sizeof(RadW2Layer) + 5 * kGrain);
    CHECK(w2_device_bytes(unit, 82) >= b);
    /* a layer's block: the mask, the entries and the budget table, a few KiB */
    CHECK(sizeof(RadW2Layer) < 16384u);
    CHECK_EQ(offsetof(RadW2Layer, mask) % 4, (size_t)0);
}

TEST(a_posts_sum_changes_with_every_word_of_it) {
    RadW2Job jobs[RAD_W2_MAX_COPY] = {};
    for (uint32_t j = 0; j < RAD_W2_MAX_COPY; ++j) jobs[j] = { 0x100000000ull + j * 0x180000ull, 0x7000000000ull + j * 0x200000ull, 0x16E000, 0 };
    const uint64_t s = rad_w2_box_sum(9, 4, 2, jobs);
    CHECK(s != rad_w2_box_sum(10, 4, 2, jobs));              /* another post's number */
    CHECK(s != rad_w2_box_sum(9, 3, 2, jobs));               /* a job fewer */
    CHECK(s != rad_w2_box_sum(9, 4, 1, jobs));               /* another group */
    for (uint32_t j = 0; j < 4; ++j) {
        RadW2Job m[RAD_W2_MAX_COPY];
        std::memcpy(m, jobs, sizeof m);
        m[j].src += 8;
        CHECK(s != rad_w2_box_sum(9, 4, 2, m));
        std::memcpy(m, jobs, sizeof m);
        m[j].dst ^= 0x200000;
        CHECK(s != rad_w2_box_sum(9, 4, 2, m));
        std::memcpy(m, jobs, sizeof m);
        m[j].bytes -= 4096;
        CHECK(s != rad_w2_box_sum(9, 4, 2, m));
    }
    /* two jobs swapped are not the same post */
    RadW2Job sw[RAD_W2_MAX_COPY];
    std::memcpy(sw, jobs, sizeof sw);
    std::swap(sw[0], sw[1]);
    CHECK(s != rad_w2_box_sum(9, 4, 2, sw));
    /* the sum is the header's term and the jobs' terms, which is how the plan's threads build it */
    uint64_t parts = rad_w2_box_base(9, 4, 2);
    for (uint32_t j = 0; j < 4; ++j) parts ^= rad_w2_job_term(j, jobs[j].src, jobs[j].dst, jobs[j].bytes);
    CHECK_EQ(parts, s);
}

TEST(the_tails_grid_holds_every_tile_its_experts_can_make) {
    /* RAD_W2_TAIL_EXPERTS experts' runs over T sorted rows: a run of r rows is ceil(r / 16) tiles. */
    for (uint32_t T : { 16u, 96u, 384u, 768u, 1024u }) {
        const uint32_t cap = rad_w2_tail_tiles(T);
        /* worst cases: every expert one row over a tile boundary, or one expert with every row */
        const uint32_t e = RAD_W2_TAIL_EXPERTS;
        uint32_t worst = 0;
        for (uint32_t big = 0; big <= e; ++big) {
            /* `big` experts share the rows left after the others took one row each */
            if (e - big > T) continue;
            const uint32_t left = T - (e - big);
            uint32_t tiles = e - big;
            if (big) {
                const uint32_t each = left / big, extra = left % big;
                for (uint32_t i = 0; i < big; ++i) tiles += ((each + (i < extra ? 1 : 0)) + 15) / 16;
            }
            worst = std::max(worst, tiles);
        }
        CHECK(worst <= cap);
        CHECK(cap <= e + T / 16);
    }
}

/* Issue #324: the form only inside the routed GEMMs' decode band (2 T <= 3 n_expert, libr4d's
 * r4d_moe_decode_band), whose grid alone reads a W2 operand. Hy4-flash routes 512 experts at top-16:
 * a verify pass of 48 rows (DFly2 c8 at depth 5, or c16 at depth 2) is the band's last, and c16 at
 * depth 3-5 (64-96 rows) is past it, where both launches of each GEMM ran every expert. */
TEST(the_form_stays_inside_the_routed_gemms_decode_band) {
    CHECK(rad_w2_in_band(768, 512));
    CHECK(!rad_w2_in_band(769, 512));
    CHECK(rad_w2_in_band(48 * 16, 512));                    /* c8 x (1 + 5), c16 x (1 + 2) */
    CHECK(!rad_w2_in_band(49 * 16, 512));
    for (int64_t rows : { 64, 80, 96 }) CHECK(!rad_w2_in_band(rows * 16, 512));   /* c16, depth 3-5 */
    CHECK(rad_w2_in_band(16 * 16, 512));                    /* the lowest height that takes the form */
    for (int64_t n : { 4, 8, 256, 512 })
        for (int64_t T = 1; T <= 2 * n; ++T) CHECK_EQ(rad_w2_in_band(T, n) != 0, 2 * T <= 3 * n);
}

TEST(with_no_context_or_the_switch_off_the_runtime_hands_out_no_control_block) {
    /* The block then issues the layer as it always did (hyv4flash_arch_test holds that side). */
    const rad_weight first[4] = { 1, 2, 3, 4 };
    CHECK(rad_expert_w2(nullptr, 0, first, 4, 48) == nullptr);
}
RAD_TEST_MAIN()
