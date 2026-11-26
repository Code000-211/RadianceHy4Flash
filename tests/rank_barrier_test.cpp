/* rank_barrier_test.cpp -- C-03: the rank barrier spins briefly and then parks in the futex.
 *
 * The barrier is the release point for the tensor-parallel rank threads (core/runtime/
 * rank_barrier.h). Its whole claim is a *bounded* spin: a peer that is microseconds behind is
 * covered by the spin, and a peer that is late is parked on std::atomic::wait instead of being
 * spun after. A revert to spinning the whole wait keeps the barrier correct and costs nothing in
 * a parity check -- it only burns the waiting thread's CPU for the length of the wait -- so the
 * cases here read the two things that distinguish the two forms:
 *
 *   - parks(): the number of arrivals that ran out of spin and went to the futex. A barrier that
 *     only spins never moves it; a barrier that parks after a late arrival moves it by one.
 *   - the waiting thread's own CPU time across a wait that lasts far longer than the spin. A
 *     bounded spin spends a few tens of microseconds; spinning the whole wait spends the wait.
 *
 * The old condvar barrier fails this file at compile time (it has neither park path nor spin
 * budget); a spin-the-whole-wait variant fails the CPU-time check with parks() == 0. Both were
 * checked by reverting core/runtime/rank_barrier.{h,cpp} and running the binary (C-03.md).
 *
 * No card, no device and no fixture: two threads and a clock, so this is a unit entry that runs
 * everywhere (the harness hides the cards for it like every other unit entry).
 */
#include "rad_test.h"
#include "runtime/rank_barrier.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <thread>

namespace {

int64_t thread_cpu_ns() {
    timespec t{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (int64_t)t.tv_sec * 1000000000ll + t.tv_nsec;
}
int64_t wall_ns() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000ll + t.tv_nsec;
}

}  /* namespace */

/* One late peer, one arrival that must wait for it. The late peer sleeps past any sane spin, so
 * the other arrival has to park; the wait then lasts the sleep, and the waiting thread's CPU time
 * is what the spin cost, not what the wait cost. */
TEST(a_late_rank_is_parked_in_the_futex_after_a_bounded_spin) {
    const int64_t lag_ms = 25;
    rad::RankBarrier b(2);
    std::atomic<int64_t> waiter_cpu_ns{-1}, waiter_wall_ns{-1};

    std::thread late([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(lag_ms));
        b.arrive_and_wait();
    });
    std::thread waiter([&] {
        const int64_t c0 = thread_cpu_ns(), w0 = wall_ns();
        b.arrive_and_wait();
        waiter_wall_ns.store(wall_ns() - w0);
        waiter_cpu_ns.store(thread_cpu_ns() - c0);
    });
    late.join();
    waiter.join();

    /* The futex path was taken: at least one park, and not a spin that re-parked in a loop. */
    CHECK(b.parks() >= 1);
    CHECK(b.parks() <= 8);
    /* It really waited for the late peer... */
    CHECK(waiter_wall_ns.load() >= (lag_ms - 5) * 1000000ll);
    /* ...and spent the wait parked: a spin held to its budget costs well under a millisecond of
     * PAUSE, so 8 ms is a generous ceiling that a spin-the-whole-wait barrier (25 ms of CPU, no
     * park) cannot pass. */
    CHECK(waiter_cpu_ns.load() < 8 * 1000000ll);
    CHECK(waiter_cpu_ns.load() < waiter_wall_ns.load());
}

/* The bound itself: a handful of pauses, not a number that makes the spin the wait. This is the
 * constant a revert would raise (a full wait cannot be spun in 2000 pauses). */
TEST(the_spin_before_parking_is_bounded) {
    CHECK(rad::RankBarrier::spin_budget() > 0);
    CHECK(rad::RankBarrier::spin_budget() <= 100000);
}

/* An arrival that completes the barrier never parks, and the counter is not moved by the common
 * case: n = 1 completes every round on the arriving thread itself. */
TEST(an_arrival_that_completes_the_barrier_does_not_park) {
    rad::RankBarrier b(1);
    for (int i = 0; i < 1000; ++i) b.arrive_and_wait();
    CHECK_EQ(b.parks(), (uint64_t)0);
}

/* abort() has to release a rank that is parked in the futex, not only one that is spinning: the
 * step loop exits through abort() on a device fault and the engine must be able to join its
 * ranks (spec §17). */
TEST(an_aborted_barrier_releases_a_parked_rank) {
    rad::RankBarrier b(3);
    std::atomic<bool> left{false};
    std::thread parked([&] { b.arrive_and_wait(); left.store(true); });

    /* Wait for the arriver to run out its spin and park before aborting, so the wake is the
     * futex's and not the spin's. */
    for (int i = 0; i < 2000 && b.parks() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(b.parks() >= 1);
    b.abort();
    parked.join();
    CHECK(left.load());
    for (int i = 0; i < 8; ++i) b.arrive_and_wait();   /* every later arrival returns at once */
    CHECK(true);
}

RAD_TEST_MAIN()