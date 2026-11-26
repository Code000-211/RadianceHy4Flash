/* rank_barrier.cpp -- see rank_barrier.h for why this is a generation counter and not an IPC
 * handshake, and for why it spins before it parks (C-03). */
#include "rank_barrier.h"

namespace rad {

void RankBarrier::arrive_and_wait() {
    if (aborted_.load(std::memory_order_acquire)) return;

    const uint32_t my_gen = gen_.load(std::memory_order_acquire);
    if (waiting_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
        /* Last one in resets the counter and flips the generation, so a rank that re-enters at
         * once waits on the new generation instead of racing through. */
        waiting_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        gen_.notify_all();
        return;
    }

    /* Spin first: the peers were issuing on their cards and are microseconds behind, so this is
     * the whole wake for all but a lagging rank. */
    for (int i = 0; i < kSpin; ++i) {
        if (aborted_.load(std::memory_order_acquire)) return;
        if (gen_.load(std::memory_order_acquire) != my_gen) return;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }

    /* Park on the generation. wait() returns only when it changes from the value passed in, then
     * the acquire load orders everything the releasing rank wrote before it flipped. */
    uint32_t g = gen_.load(std::memory_order_acquire);
    while (g == my_gen && !aborted_.load(std::memory_order_acquire)) {
        parks_.fetch_add(1, std::memory_order_relaxed);
        gen_.wait(g, std::memory_order_acquire);
        g = gen_.load(std::memory_order_acquire);
    }
}

void RankBarrier::abort() {
    aborted_.store(true, std::memory_order_release);
    waiting_.store(0, std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
    gen_.notify_all();
}

}  /* namespace rad */
