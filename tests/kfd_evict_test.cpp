/* kfd_evict_test.cpp -- issue #272 without a card: the pinned range's lock against swap and its
 * fallback (core/device/kfd_evict.h, a fake lock call), the driver's evicted_ms read from a fake
 * /sys/class/kfd tree, the stall dump's line, and the startup report's finding.
 *
 * The lock decides whether reclaim may swap out host memory the cards map, and every such
 * swap-out evicts all of the process's queues on every card; a refusal must never fail the start
 * and must be said once. The evicted_ms line is what tells an evicted queue from a hung kernel. */
#include "rad_test.h"

#include "device/kfd_evict.h"
#include "diag/diag.h"
#include "rad_internal.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace rad;

namespace {

struct TmpTree {
    std::string root;
    TmpTree() {
        std::string t = radtest::tmp_path("kfd.XXXXXX");
        root = mkdtemp(t.data()) ? t : "";
    }
    void put(const std::string& rel, const std::string& text) const {
        std::filesystem::create_directories(std::filesystem::path(root + rel).parent_path());
        std::ofstream(root + rel) << text;
    }
};

int g_calls = 0;
size_t g_last_n = 0;
int lock_ok(void*, size_t n) {
    ++g_calls;
    g_last_n = n;
    return 0;
}
int lock_refused(void*, size_t n) {
    ++g_calls;
    g_last_n = n;
    errno = ENOMEM;
    return -1;
}

constexpr int64_t MiB = (int64_t)1 << 20, GiB = (int64_t)1 << 30;

bool has(const std::vector<diag::Finding>& v, const char* id) {
    for (const diag::Finding& f : v)
        if (f.id == id) return true;
    return false;
}

}  /* namespace */

/* A lock that succeeds counts the range as locked, over its whole length, and says nothing. */
TEST(a_granted_lock_counts_the_range_locked_and_says_nothing) {
    PinLockTally t;
    std::string say = "stale";
    g_calls = 0;
    char buf[64];
    CHECK(lock_against_swap(buf, 3 * GiB, lock_ok, 8 * MiB, &t, &say));
    CHECK_EQ(g_calls, 1);
    CHECK_EQ((int64_t)g_last_n, 3 * GiB);
    CHECK_EQ(t.locked.load(), 3 * GiB);
    CHECK_EQ(t.unlocked.load(), (int64_t)0);
    CHECK(say.empty());
    pin_lock_forget(3 * GiB, true, &t);
    CHECK_EQ(t.locked.load(), (int64_t)0);
}

/* A refusal returns false (the caller keeps the range: the start never fails on it), counts the
 * bytes unlocked, and is said exactly once with the error, the limit and the fix. */
TEST(a_refused_lock_falls_back_counts_unlocked_and_says_it_once) {
    PinLockTally t;
    std::string say;
    char buf[64];
    CHECK(!lock_against_swap(buf, 2 * GiB, lock_refused, 8 * MiB, &t, &say));
    CHECK_EQ(t.unlocked.load(), 2 * GiB);
    CHECK_EQ(t.locked.load(), (int64_t)0);
    CHECK(say.find("not locked against swap") != std::string::npos);
    CHECK(say.find("RLIMIT_MEMLOCK " + humanb(8 * MiB)) != std::string::npos);
    CHECK(say.find(std::strerror(ENOMEM)) != std::string::npos);
    CHECK(say.find("memlock=-1") != std::string::npos);
    CHECK(say.find("LimitMEMLOCK=infinity") != std::string::npos);

    CHECK(!lock_against_swap(buf, GiB, lock_refused, 8 * MiB, &t, &say));
    CHECK(say.empty());
    CHECK_EQ(t.unlocked.load(), 3 * GiB);
    /* A later range the limit allows is still locked. */
    CHECK(lock_against_swap(buf, MiB, lock_ok, 8 * MiB, &t, &say));
    CHECK_EQ(t.locked.load(), MiB);
    pin_lock_forget(2 * GiB, false, &t);
    CHECK_EQ(t.unlocked.load(), GiB);

    PinLockTally u;
    CHECK(!lock_against_swap(buf, GiB, lock_refused, -2, &u, &say));
    CHECK(say.find("RLIMIT_MEMLOCK unlimited") != std::string::npos);
}

/* evicted_ms per gpu_id from the driver's tree; a missing card or a junk file is -1. */
TEST(evicted_ms_is_read_per_gpu_id_and_missing_is_unknown) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/proc/4242/stats_17663/evicted_ms", "18203\n");
    t.put("/proc/4242/stats_4402/evicted_ms", "0\n");
    t.put("/proc/4242/stats_9/evicted_ms", "junk\n");
    CHECK_EQ(kfd_evicted_ms(t.root + "/proc", 4242, 17663), (int64_t)18203);
    CHECK_EQ(kfd_evicted_ms(t.root + "/proc", 4242, 4402), (int64_t)0);
    CHECK_EQ(kfd_evicted_ms(t.root + "/proc", 4242, 9), (int64_t)-1);
    CHECK_EQ(kfd_evicted_ms(t.root + "/proc", 4242, 1), (int64_t)-1);
    CHECK_EQ(kfd_evicted_ms(t.root + "/proc", 4243, 17663), (int64_t)-1);
}

/* The dump's line: every card's count, its growth since the previous dump, VmSwap and VmLck;
 * the issue's recovered stop read as such (four cards each +16.7 s). */
TEST(the_dump_line_gives_each_cards_eviction_its_growth_and_swap) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    const uint32_t ids[4] = {17663, 4402, 51001, 23};
    const int64_t before[4] = {18709, 18744, 18733, 18696};
    const int64_t after[4] = {35429, 35465, 35454, 35416};
    for (int i = 0; i < 4; ++i)
        t.put(fmt("/proc/77/stats_%u/evicted_ms", ids[i]), fmt("%lld\n", (long long)before[i]));
    t.put("/status", "Name:\tradiance\nVmPin:\t       0 kB\nVmLck:\t       0 kB\n"
                     "VmSwap:\t  406900 kB\n");
    std::vector<std::pair<int, uint32_t>> cards;
    for (int i = 0; i < 4; ++i) cards.emplace_back(i, ids[i]);
    std::map<uint32_t, int64_t> last;
    std::string a = evict_line(t.root + "/proc", t.root + "/status", 77, cards, &last);
    CHECK(a.find(" card 0 18709, card 1 18744") != std::string::npos);
    CHECK(a.find("(+") == std::string::npos);   /* nothing to compare against yet */
    CHECK(a.find("card 3 18696") != std::string::npos);
    CHECK(a.find("VmSwap " + humanb(406900 * 1024LL)) != std::string::npos);
    CHECK(a.find("VmLck " + humanb(0)) != std::string::npos);

    for (int i = 0; i < 4; ++i)
        t.put(fmt("/proc/77/stats_%u/evicted_ms", ids[i]), fmt("%lld\n", (long long)after[i]));
    std::string b = evict_line(t.root + "/proc", t.root + "/status", 77, cards, &last);
    for (int i = 0; i < 4; ++i)
        CHECK(b.find(fmt("card %d %lld (+%lld)", i, (long long)after[i],
                         (long long)(after[i] - before[i]))) != std::string::npos);

    /* A card the driver does not list is "?", not a number, and has no growth. */
    cards.emplace_back(4, 999);
    std::string c = evict_line(t.root + "/proc", t.root + "/missing", 77, cards, &last);
    CHECK(c.find("card 4 ?") != std::string::npos);
    CHECK(c.find("card 0 35429 (+0)") != std::string::npos);
    CHECK(c.find("VmSwap ?, VmLck ?") != std::string::npos);
}

TEST(status_bytes_matches_the_whole_key) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/s", "VmLckX:\t 9 kB\nVmLck:\t  2048 kB\nVmSwap:\t 1 kB\n");
    CHECK_EQ(status_bytes(t.root + "/s", "VmLck"), (int64_t)2048 * 1024);
    CHECK_EQ(status_bytes(t.root + "/s", "VmSwap"), (int64_t)1024);
    CHECK_EQ(status_bytes(t.root + "/s", "VmPin"), (int64_t)-1);
}

/* The startup report's finding: pinned memory left unlocked is a warning naming the limit and
 * the fix; all of it locked is no finding. */
TEST(the_report_warns_when_pinned_memory_is_not_locked) {
    diag::Facts f;
    f.host.memlock_soft = 8 * MiB;
    diag::RunFacts run;
    run.pinned_bytes = 40 * GiB;
    run.pinned_unlocked_bytes = 0;
    CHECK(!has(diag::check(f, run), "pinned-unlocked"));
    run.pinned_unlocked_bytes = 40 * GiB;
    const auto found = diag::check(f, run);
    bool warn = false;
    for (const diag::Finding& x : found)
        if (x.id == "pinned-unlocked") {
            warn = x.level == diag::Finding::Warn;
            CHECK(x.what.find(humanb(40 * GiB)) != std::string::npos);
            CHECK(x.what.find(humanb(8 * MiB)) != std::string::npos);
            CHECK(x.fix.find("memlock=-1") != std::string::npos);
        }
    CHECK(warn);
}

/* Raising the soft limit to the hard one never lowers it. */
TEST(the_soft_memlock_limit_is_raised_to_the_hard_one) {
    struct rlimit rl {};
    REQUIRE(::getrlimit(RLIMIT_MEMLOCK, &rl) == 0);
    const int64_t soft = memlock_soft_to_hard();
    struct rlimit now {};
    REQUIRE(::getrlimit(RLIMIT_MEMLOCK, &now) == 0);
    CHECK(now.rlim_cur == rl.rlim_max);
    CHECK(soft == (now.rlim_cur == RLIM_INFINITY ? -2 : (int64_t)now.rlim_cur));
}

RAD_TEST_MAIN()
