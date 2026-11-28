/* hostmem_test.cpp -- the memory cgroup limits bring-up reads before locking a table into host
 * memory (core/engine_hostmem.h), against cgroup files written into a temporary tree.
 *
 * A misread limit either refuses a copy that fits or locks one that does not, and under Docker's
 * --memory the second ends in the OOM killer partway through loading.
 */
#include "rad_test.h"

#include "engine_hostmem.h"
#include "rad_internal.h"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace rad;

namespace {

struct TmpTree {
    std::string root;
    TmpTree() {
        std::string t = radtest::tmp_path("tree.XXXXXX");
        root = mkdtemp(t.data()) ? t : "";
    }
    void put(const std::string& rel, const std::string& text) const {
        std::filesystem::create_directories(std::filesystem::path(root + rel).parent_path());
        std::ofstream(root + rel) << text;
    }
};

constexpr int64_t GiB = (int64_t)1 << 30;

}  /* namespace */

/* v2: the limit less the memory reclaim cannot give back, which is the usage less its file pages.
 * "max" is no limit, and so is a missing file. */
TEST(a_v2_cgroups_room_is_its_limit_less_unreclaimable_usage) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/a/memory.max", "8589934592\n");
    t.put("/a/memory.current", "3221225472\n");
    t.put("/a/memory.stat", "anon 1073741824\nfile 1073741824\nkernel 4096\n");
    int64_t room = -1;
    CHECK(cgroup_room(t.root + "/a", true, &room));
    CHECK_EQ(room, 8 * GiB - (3 * GiB - 1 * GiB));

    t.put("/b/memory.max", "max\n");
    t.put("/b/memory.current", "1024\n");
    CHECK(!cgroup_room(t.root + "/b", true, &room));
    CHECK(!cgroup_room(t.root + "/missing", true, &room));

    /* Usage past the limit leaves no room rather than a negative one. */
    t.put("/c/memory.max", "1073741824\n");
    t.put("/c/memory.current", "2147483648\n");
    CHECK(cgroup_room(t.root + "/c", true, &room));
    CHECK_EQ(room, (int64_t)0);
}

/* v1: the same from usage_in_bytes less total_cache (cache when there is no total_cache), and
 * the near-INT64_MAX value v1 reports for no limit is no limit. */
TEST(a_v1_cgroups_room_reads_its_cache_and_its_no_limit_sentinel) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/a/memory.limit_in_bytes", "4294967296\n");
    t.put("/a/memory.usage_in_bytes", "2147483648\n");
    t.put("/a/memory.stat", "cache 536870912\nrss 1073741824\ntotal_cache 1073741824\n");
    int64_t room = -1;
    CHECK(cgroup_room(t.root + "/a", false, &room));
    CHECK_EQ(room, 4 * GiB - (2 * GiB - 1 * GiB));

    t.put("/b/memory.limit_in_bytes", "4294967296\n");
    t.put("/b/memory.usage_in_bytes", "2147483648\n");
    t.put("/b/memory.stat", "cache 536870912\nrss 1073741824\n");
    CHECK(cgroup_room(t.root + "/b", false, &room));
    CHECK_EQ(room, 4 * GiB - (2 * GiB - GiB / 2));

    t.put("/c/memory.limit_in_bytes", "9223372036854771712\n");
    t.put("/c/memory.usage_in_bytes", "1024\n");
    CHECK(!cgroup_room(t.root + "/c", false, &room));
}

/* Every level from the process's own cgroup up to the root is a limit of its own: each one with
 * a limit is visited, leaf first, a level without one is passed over, and the walk stops when
 * asked. */
TEST(the_cgroup_walk_visits_every_limited_level_up_to_the_root) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/memory.max", "max\n");                       /* the root: no limit */
    t.put("/memory.current", "1\n");
    t.put("/docker/memory.max", "4294967296\n");
    t.put("/docker/memory.current", "1073741824\n");
    t.put("/docker/memory.stat", "file 0\n");
    t.put("/docker/ctr/memory.max", "max\n");            /* the container's own: none */
    t.put("/docker/ctr/memory.current", "1\n");
    t.put("/docker/ctr/app/memory.max", "17179869184\n");
    t.put("/docker/ctr/app/memory.current", "0\n");
    std::vector<std::pair<std::string, int64_t>> seen;
    cgroup_levels(t.root, "/docker/ctr/app", true, [&](const std::string& dir, int64_t room) {
        seen.emplace_back(dir, room);
        return true;
    });
    REQUIRE_EQ(seen.size(), (size_t)2);
    CHECK_EQ(seen[0].first, t.root + "/docker/ctr/app");
    CHECK_EQ(seen[0].second, 16 * GiB);
    CHECK_EQ(seen[1].first, t.root + "/docker");
    CHECK_EQ(seen[1].second, 3 * GiB);

    int visits = 0;
    cgroup_levels(t.root, "/docker/ctr/app", true, [&](const std::string&, int64_t) {
        ++visits;
        return false;
    });
    CHECK_EQ(visits, 1);

    /* The root itself, named "/". */
    seen.clear();
    t.put("/memory.max", "2147483648\n");
    cgroup_levels(t.root, "/", true, [&](const std::string& dir, int64_t room) {
        seen.emplace_back(dir, room);
        return true;
    });
    REQUIRE_EQ(seen.size(), (size_t)1);
    CHECK_EQ(seen[0].first, t.root);
}

/* ---- #273: a startup a little short of host memory waits for it -------------------------- */

namespace {

/* A host whose available memory is `avail` until the clock reaches `frees_at_ms`, then `later`:
 * the old process's memory coming back partway through the wait. The probe is lockable_room's
 * MemAvailable rule on that figure; the nap advances the clock. */
struct FakeHost {
    int64_t avail = 0, later = 0, frees_at_ms = INT64_MAX, clock_ms = 0;
    int probes = 0, naps = 0;
    int64_t need = 0, still = 0, margin = 0;
    std::function<bool(std::string*, int64_t*)> probe() {
        return [this](std::string* why, int64_t* short_by) {
            ++probes;
            const int64_t a = clock_ms >= frees_at_ms ? later : avail;
            if (a - still - margin >= need) return true;
            *short_by = need - (a - still - margin);
            *why = "only " + std::to_string(a >> 30) + " GiB is available";
            return false;
        };
    }
    std::function<void(int64_t)> nap() {
        return [this](int64_t ms) { ++naps; clock_ms += ms; };
    }
    int64_t want() const { return need + still + margin; }
};

}  /* namespace */

/* The bound: 2 GiB, or an eighth of what the check wants in all, whichever is more. The issue's
 * restart wanted 223.88 GiB and was 0.6 to 2.4 GiB short; a config asking for twice the host is
 * not a wait. A refusal no wait cures (RLIMIT_MEMLOCK) reports a negative shortfall. */
TEST(a_small_shortfall_is_two_gib_or_an_eighth_of_the_want) {
    CHECK(room_short_is_small(1, 4 * GiB));
    CHECK(room_short_is_small(2 * GiB, 4 * GiB));
    CHECK(!room_short_is_small(2 * GiB + 1, 4 * GiB));
    CHECK(room_short_is_small(28 * GiB, 224 * GiB));
    CHECK(!room_short_is_small(28 * GiB + 1, 224 * GiB));
    CHECK(!room_short_is_small(112 * GiB, 224 * GiB));
    CHECK(!room_short_is_small(-1, 224 * GiB));
}

/* Short by a little, and the memory comes back 12 s into the wait: startup goes on after the
 * re-check that sees it, saying it waited and that it re-checks. */
TEST(a_small_shortfall_freed_within_the_wait_starts) {
    FakeHost h;
    h.need = 60 * GiB; h.still = 160 * GiB; h.margin = 4 * GiB;
    h.avail = 222 * GiB;                       /* 2 GiB short */
    h.later = 230 * GiB;
    h.frees_at_ms = 12000;
    std::string why;
    bool ok = false;
    const Log was = log_level();
    log_set_level(Log::Info);
    const std::string log = stderr_of(
        [&] { ok = wait_for_room(h.probe(), h.want(), 90000, 5000, &why, h.nap()); });
    log_set_level(was);
    CHECK(ok);
    CHECK_EQ(h.naps, 3);                       /* 5, 10, 15 s: the third sees it */
    CHECK_EQ(h.probes, 4);
    CHECK(log.find("may still be releasing its memory") != std::string::npos);
    CHECK(log.find("still 2.00 GiB short after 5 s") != std::string::npos);
    CHECK(log.find("the room is there after 15 s") != std::string::npos);
}

/* Short by a little and never freed: it fails once the wait runs out, the reason the check gave
 * kept and the wait and its likely cause appended, after one re-check every step. */
TEST(a_small_shortfall_never_freed_fails_after_the_wait_with_the_hint) {
    FakeHost h;
    h.need = 60 * GiB; h.still = 160 * GiB; h.margin = 4 * GiB;
    h.avail = 223 * GiB;                       /* 1 GiB short */
    std::string why;
    bool ok = true;
    (void)stderr_of(
        [&] { ok = wait_for_room(h.probe(), h.want(), 90000, 5000, &why, h.nap()); });
    CHECK(!ok);
    CHECK_EQ(h.naps, 18);
    CHECK_EQ(h.clock_ms, (int64_t)90000);
    CHECK_EQ(h.probes, 19);
    CHECK(why.rfind("only 223 GiB is available", 0) == 0);
    CHECK(why.find("startup waited 90 s for it (--host-memory-wait-s)") != std::string::npos);
    CHECK(why.find("may still be releasing its memory") != std::string::npos);

    /* A wait that is not a whole number of steps stops at the wait, not past it. */
    FakeHost g = h;
    g.probes = g.naps = 0; g.clock_ms = 0;
    (void)stderr_of(
        [&] { ok = wait_for_room(g.probe(), g.want(), 12000, 5000, &why, g.nap()); });
    CHECK(!ok);
    CHECK_EQ(g.clock_ms, (int64_t)12000);
    CHECK_EQ(g.naps, 3);
}

/* Short by a lot, a misconfiguration: it fails at once, with no wait and the reason unchanged. A
 * refusal no wait cures fails at once too, and so does every refusal with the wait set to 0. */
TEST(a_large_shortfall_fails_at_once) {
    FakeHost h;
    h.need = 60 * GiB; h.still = 160 * GiB; h.margin = 4 * GiB;
    h.avail = 120 * GiB;                       /* 104 GiB short, past 28 */
    h.later = 250 * GiB;
    h.frees_at_ms = 5000;                      /* it would come back, were it waited for */
    std::string why;
    CHECK(!wait_for_room(h.probe(), h.want(), 90000, 5000, &why, h.nap()));
    CHECK_EQ(h.naps, 0);
    CHECK_EQ(h.probes, 1);
    CHECK_EQ(why, std::string("only 120 GiB is available"));

    int naps = 0;
    auto memlock = [](std::string* w, int64_t* s) { *w = "RLIMIT_MEMLOCK"; *s = -1; return false; };
    CHECK(!wait_for_room(memlock, 224 * GiB, 90000, 5000, &why, [&](int64_t) { ++naps; }));
    CHECK_EQ(naps, 0);

    FakeHost z;
    z.need = 60 * GiB; z.still = 160 * GiB; z.margin = 4 * GiB;
    z.avail = 223 * GiB; z.later = 250 * GiB; z.frees_at_ms = 1;
    CHECK(!wait_for_room(z.probe(), z.want(), 0, 5000, &why, z.nap()));
    CHECK_EQ(z.naps, 0);
}

/* A shortfall that grows past the bound partway through the wait (another process taking the
 * memory) stops the wait at that re-check. */
TEST(a_shortfall_that_grows_large_stops_the_wait) {
    FakeHost h;
    h.need = 60 * GiB; h.still = 160 * GiB; h.margin = 4 * GiB;
    h.avail = 223 * GiB;
    h.later = 100 * GiB;
    h.frees_at_ms = 10000;
    std::string why;
    bool ok = true;
    (void)stderr_of(
        [&] { ok = wait_for_room(h.probe(), h.want(), 90000, 5000, &why, h.nap()); });
    CHECK(!ok);
    CHECK_EQ(h.naps, 2);
    CHECK_EQ(why, std::string("only 100 GiB is available"));
}

/* lockable_room reports how far short it is, against this host's own MemAvailable: a want a TiB
 * past it is short by about a TiB, and nothing at all fits. */
TEST(lockable_room_reports_its_shortfall) {
    int64_t avail = -1;
    {
        FILE* f = std::fopen("/proc/meminfo", "r");
        REQUIRE(f != nullptr);
        char key[64];
        long long kb = 0;
        while (std::fscanf(f, "%63s %lld kB", key, &kb) == 2)
            if (std::string(key) == "MemAvailable:") avail = (int64_t)kb * 1024;
        std::fclose(f);
    }
    REQUIRE(avail > 0);
    const int64_t TiB = 1024 * GiB;
    std::string why;
    int64_t short_by = 0;
    CHECK(!lockable_room(avail + TiB, 0, 0, false, &why, &short_by));
    CHECK(short_by > TiB - 64 * GiB && short_by < TiB + 64 * GiB);
    CHECK(why.find("of memory is available") != std::string::npos);
    short_by = 7;
    CHECK(lockable_room(0, 0, 0, false, &why, &short_by));
    CHECK(lockable_room(0, 0, 0, false, &why));
}

RAD_TEST_MAIN()
