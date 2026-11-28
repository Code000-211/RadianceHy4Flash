/* aql_inflight_test.cpp -- which dispatches a stopped queue may have faulted in (#337).
 *
 * Issue #337's dump printed the packet at the queue's read index and called it "the kernel that
 * ran". A chained kernel loses its barrier bit, so the read index moves on when the packet is read,
 * not when it completes, and the launch that faulted can be many packets behind it. That is what
 * made the dump read as a paradox: every parameter of the packet it printed was in range, and the
 * faulting address was in none of them.
 *
 * These cases are the ring arithmetic of core/device/aql_inflight.h on its own: no queue, no card. */
#include "rad_test.h"

#include "device/aql_inflight.h"  /* core/ is on the include path */

#include <cstdint>
#include <string>
#include <vector>

using namespace rad::aql;

namespace {

/* A ring of `size` packets whose slot `i & (size - 1)` waits for `need[i & mask]`. */
struct Ring {
    uint64_t size = 64;
    std::vector<uint64_t> need = std::vector<uint64_t>(size, 0);

    uint64_t at(uint64_t i) const { return need[i & (size - 1)]; }

    std::vector<uint64_t> run(uint64_t rd, uint64_t wr, uint64_t counter, bool open,
                              size_t cap = 32) const {
        return still_running(rd, wr, size, counter, open, [this](uint64_t i) { return at(i); },
                             cap);
    }
};

/* The report's own words for a list of packet indices: "5,6", and "" for none. */
std::string list(const std::vector<uint64_t>& v) {
    std::string s;
    for (uint64_t i : v) { if (!s.empty()) s += ','; s += std::to_string(i); }
    return s;
}

std::string range(uint64_t a, uint64_t b) {
    std::string s;
    for (uint64_t i = a; i < b; ++i) { if (!s.empty()) s += ','; s += std::to_string(i); }
    return s;
}

}  /* namespace */

/* The trap #337 fell into: the read index is 20 packets past a chained dispatch whose blocks have
 * not all counted. The chained one is the candidate; the packet the read index points at has not
 * run and is never one. */
TEST(read_index_past_a_chained_dispatch) {
    Ring r;
    r.need[5] = 900;             /* packet 5 is chained and waits for 900 */
    const uint64_t rd = 26, wr = 40, counter = 880;
    CHECK_EQ(list(r.run(rd, wr, counter, false)), range(5, 6));
    /* Once the counter reaches what it waits for, nothing may still be running. */
    CHECK(r.run(rd, wr, 900, false).empty());
}

/* A plain dispatch keeps its barrier bit: the read index passing it means it completed. Nothing is
 * reported, and the caller falls back to printing the packet at the read index itself. */
TEST(a_plain_dispatch_behind_the_read_index_is_done) {
    Ring r;
    const uint64_t rd = 30, wr = 33;
    CHECK(r.run(rd, wr, 0, false).empty());
}

/* An unbarriered packet went out (frontier_open, or a plain or collective packet launched since the
 * last barrier): every packet the ring still holds and the processor has read is a candidate --
 * when the cap is large enough to hold them. */
TEST(an_open_frontier_makes_every_read_packet_a_candidate) {
    Ring r;                      /* size 64: the ring holds [wr - 64, wr) */
    const uint64_t rd = 70, wr = 72;
    CHECK_EQ(list(r.run(rd, wr, 0, true, 100)), range(8, 70));
}

/* A chained dispatch whose blocks have all counted is done, and the card's counter says so whatever
 * the packet processor did: it is dropped as a candidate even with the frontier open. Dropping it
 * matters, because the report has room for six and a stale one in front of the right one is what
 * sent #337's readers after a launch that had not faulted. */
TEST(a_counted_chained_dispatch_is_done_with_the_frontier_open) {
    Ring r;
    for (uint64_t i = 0; i < 20; ++i) r.need[i] = 500;   /* a chained run, counter at 500 */
    CHECK(r.run(20, 20, 500, true).empty());
    /* One of them has not counted: exactly that one comes back, not its counted elders. */
    r.need[18] = 600;
    CHECK_EQ(list(r.run(20, 20, 500, true)), "18");
    /* A plain packet beside it is a candidate while the frontier is open, and only then. */
    r.need[19] = 0;
    CHECK_EQ(list(r.run(20, 20, 500, true)), "18,19");
    CHECK_EQ(list(r.run(20, 20, 500, false)), "18");
}

/* Only what the ring still holds: a slot older than wr - size has been rewritten, so the packet
 * index it used to carry says nothing about what its arguments are now. */
TEST(the_window_is_the_last_size_packets) {
    Ring r;                      /* size 64 */
    const uint64_t rd = 200, wr = 210;
    CHECK_EQ(list(r.run(rd, wr, 0, true, 200)), range(wr - r.size, rd));   /* 146 .. 199 */
    /* Nothing written yet at all: no window, no candidates. */
    CHECK(r.run(0, 0, 0, true).empty());
    r.size = 0;
    CHECK(still_running(4, 4, 0, 0, true, [](uint64_t) { return (uint64_t)0; }, 8).empty());
}

/* The cap keeps the newest packets written, nearest the read index, and they come back oldest
 * first: the queue stops at the first launch that faults. With an open frontier the ring holds a
 * lap of packets that are long done; printing the six oldest of them, thousands of packets behind
 * the read index, names launches that cannot have faulted. */
TEST(the_cap_keeps_the_newest_near_the_read_index) {
    Ring r;                      /* every slot plain: an open frontier makes them all candidates */
    const uint64_t rd = 4000, wr = 4000;
    CHECK_EQ(list(r.run(rd, wr, 0, true, 3)), "3997,3998,3999");
    CHECK(r.run(rd, wr, 0, true, 0).empty());
    /* Mixed: the newest three written are 47 (chained, not counted), 48 and 49 (plain). */
    r.need[47] = 900;
    CHECK_EQ(list(r.run(50, 50, 800, true, 3)), "47,48,49");
    /* A counted chained packet at the read index's edge is not a candidate, and the cap reaches
     * past it to the next ones. */
    r.need[49] = 700;
    CHECK_EQ(list(r.run(50, 50, 800, true, 3)), "46,47,48");
}

/* The oldest first, at most `cap` of them, none when the cap is 0. */
TEST(oldest_first_of_the_candidates) {
    Ring r;
    for (uint64_t i = 10; i < 20; ++i) r.need[i] = 500;   /* chained, counter at 400 */
    CHECK_EQ(list(r.run(30, 50, 400, false, 3)), "17,18,19");
    CHECK(r.run(30, 50, 400, false, 0).empty());
}

/* The counter as the host reads it can lag the card, so a chained kernel that finished may still be
 * a candidate. That way round is the safe one: a wrong candidate is printed beside the right one,
 * where a dropped one is not printed at all. */
TEST(a_lagging_counter_only_adds_candidates) {
    Ring r;
    r.need[3] = 700;
    r.need[4] = 700;
    CHECK_EQ(list(r.run(9, 12, 690, false)), range(3, 5));   /* both look unfinished */
    CHECK(r.run(9, 12, 700, false).empty());
}

/* A fault address only names a range it can have come from (fault_ran_off, aql.cpp's
 * fault_range_at). Run 2 of #337 faulted at 0xebbcff295000, which is above every 0x7c... range the
 * process held by 108 TiB: the range below it is not its, and an operand pointing into that range
 * must not be printed as the launch the fault address is past the end of. */
TEST(a_fault_address_names_only_a_range_it_can_come_from) {
    const uintptr_t base = 0x7ca0fb000000;
    const int64_t bytes = 502 * 1024;             /* #298 backend's 502.69 KiB */
    const uintptr_t end = base + (uintptr_t)bytes;
    CHECK(fault_ran_off(base, bytes, base));               /* the first byte */
    CHECK(fault_ran_off(base, bytes, end - 1));            /* the last */
    CHECK(fault_ran_off(base, bytes, end));                /* the end itself is the smallest run-off */
    CHECK(fault_ran_off(base, bytes, end + 4096));         /* one page past: a stride that ran off */
    CHECK(fault_ran_off(base, bytes, end + (uintptr_t)bytes));   /* a whole range's length past: the bound */
    CHECK(!fault_ran_off(base, bytes, end + (uintptr_t)bytes + 1));
    CHECK(!fault_ran_off(base, bytes, (uintptr_t)0xebbcff295000ull));   /* #337's run 2 */
    CHECK(!fault_ran_off(base, 0, base));                  /* an empty range explains nothing */
    CHECK(!fault_ran_off(base, bytes, base - 1));          /* below it: the caller's lookup picks it */
    /* A 24.37 GiB weight range: a run-off a few MiB past it is still its own. */
    const int64_t w = (int64_t)24 * 1024 * 1024 * 1024;
    CHECK(fault_ran_off(0x7cc1d6c36000ull, w, 0x7cc1d6c36000ull + (uintptr_t)w + 4096));
    CHECK(!fault_ran_off(0x7cc1d6c36000ull, w, 0x7cc1d6c36000ull + 2 * (uintptr_t)w + 4096));
}

RAD_TEST_MAIN()
