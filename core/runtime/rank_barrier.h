/* rank_barrier.h -- the release point for the tensor-parallel rank threads.
 *
 * Radiance is one process with one thread per rank, each owning one device (spec §1), so the step
 * batch is shared by pointer rather than serialised to other processes. The barrier only
 * establishes happens-before between the scheduler's release and each rank's first rad_issue, and
 * between the ranks' last issue and the scheduler's next batch build.
 *
 * Generation-based (sense reversal), so a fast rank re-entering for the next step cannot slip
 * through a barrier its slower peers have not left yet.
 *
 * C-03: every participant is already runnable when it arrives (it was just issuing), so a bounded
 * spin covers the common case in a few microseconds and only a genuinely lagging rank parks in the
 * futex behind std::atomic::wait. A condvar round trip costs ~1-2 us in the best case and 10-100 us
 * whenever the waiter actually deschedules (notes/backlog/optimisation-backlog.md C-03), and a
 * DFly2 TP2 step crosses eight rounds.
 *
 * `parks()` counts the arrivals that ran out of spin and went to the futex, and `spin_budget()` is
 * the bound the spin is held to; tests/rank_barrier_test.cpp reads both to hold the barrier to
 * "spin briefly, then park" rather than "spin for the whole wait". The counter costs one relaxed
 * add on the park path, which a step in a steady state does not take.
 */
#pragma once
#include "../rad_core.h"

#include <atomic>
#include <cstdint>
#include <thread>

namespace rad {

class RankBarrier {
public:
    explicit RankBarrier(int n_ranks) : n_(n_ranks < 1 ? 1 : n_ranks) {}

    RankBarrier(const RankBarrier&) = delete;
    RankBarrier& operator=(const RankBarrier&) = delete;

    void arrive_and_wait();

    /* Release everyone regardless of how many have arrived, and make every later arrival return
     * at once. Called on shutdown and on a device fault, so ranks are not left parked on a barrier
     * an exited rank can no longer complete; the engine must be able to exit (spec §17). */
    void abort();

    /* The futex path: arrivals that ran out the spin and parked, and the spin bound in pauses.
     * Read by the test that holds the barrier to a bounded spin rather than a whole wait. */
    uint64_t parks() const { return parks_.load(std::memory_order_relaxed); }
    static constexpr int spin_budget() { return kSpin; }

private:
    /* Pauses a step thread should spin before parking. Every arrival here follows a device wait,
     * so the last rank is normally microseconds behind; ~2000 pauses is a few tens of microseconds
     * and leaves a descheduled rank to the futex. */
    static constexpr int kSpin = 2000;

    const int               n_;
    std::atomic<int>        waiting_{0};
    std::atomic<uint32_t>   gen_{0};
    std::atomic<bool>       aborted_{false};
    std::atomic<uint64_t>   parks_{0};
};

}  /* namespace rad */
