/* aql_inflight.h -- which dispatches a stopped queue may have faulted in (issue #337).
 *
 * A queue's read index is not a completion mark. The packet processor moves it on when it has read
 * a packet, and a chained kernel loses its barrier bit (aql.cpp's chain plan: a chained launch waits
 * in its own prologue, so it needs no barrier behind it). The launch that runs can therefore be many
 * packets behind the read index: aql.cpp's reserve() says as much ("a kernel can still be running
 * when the read index is far past it"), and Stream::frontier_open records when a packet went out
 * without its bit. Issue #337's dump named the packet at the read index and called it "the kernel
 * that ran"; every parameter of it was a pointer into a live range, which is what any later launch
 * of the same kernel looks like, and the faulting address was in none of them. Three readers chased
 * a launch that had not faulted.
 *
 * What the engine does know, per ring slot, is the completion counter value the chained kernel in
 * that slot waits for (Stream::slot_need, 0 for a packet that is not chained) and the counter's
 * value (Stream::chain_ctr, which the caller reads live). A chained dispatch is done only once that
 * value is reached, so one whose value is not reached may still be running, and one whose value is
 * reached is done -- also when the frontier is open, because the count comes from the card, not
 * from the packet processor. A plain dispatch keeps its barrier bit, so the read index passing it
 * does mean it completed -- unless an unbarriered packet went out since (`open`), and then a plain
 * packet the ring still holds may be running.
 *
 * The queue stops at the first launch that faults, so the candidates come back oldest first. The
 * `cap` is applied to the newest of them, nearest the read index: the ring is long (kQueuePackets),
 * so with an open frontier it holds thousands of packets that were written a lap ago and are long
 * done, while the launch that faulted is one of the last few to go out. Byte- and state-only, no
 * queue and no card, so tests/aql_inflight_test.cpp covers it. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rad {
namespace aql {

/* The packet indices a queue may still be running, oldest first.
 *
 * `rd` and `wr` are the queue's read and write indices, `size` its packet count (the ring), and
 * `counter` the completion counter at the time of the call (Stream::chain_ctr, read live; a
 * lagging read only ever adds candidates, never drops one). `need_at(i)` is what the chained
 * dispatch at index `i` waits for, 0 when it is not chained; the caller answers from the slot
 * `i & (size - 1)`.
 *
 * The window is the last `size` packets written, [wr - size, wr), which is what the ring still
 * holds -- a slot older than that has been rewritten and says nothing about the packet that used to
 * be in it. Packets at or past `rd` are written and not read: they have not run.
 *
 * A chained packet whose counter value is reached is done and is dropped even when `open`, which is
 * the one thing the ring can still say with an open frontier; a packet that is not chained (`need`
 * 0) is done whenever the read index is past it and no unbarriered packet went out since.
 *
 * At most `cap` indices come back -- the newest ones, nearest `rd`, in oldest-first order; `cap` 0
 * reports none, for a caller that does not want the walk. */
template <class NeedAt>
std::vector<uint64_t> still_running(uint64_t rd, uint64_t wr, uint64_t size, uint64_t counter,
                                    bool open, NeedAt need_at, size_t cap) {
    std::vector<uint64_t> v;
    if (size == 0 || wr == 0 || cap == 0) return v;
    const uint64_t first = wr > size ? wr - size : 0;
    const uint64_t end = rd < wr ? rd : wr;
    for (uint64_t i = end; i > first && v.size() < cap;) {
        --i;
        const uint64_t need = need_at(i);
        if (need != 0) {
            if (need <= counter) continue;    /* chained, and its blocks have all counted */
        } else if (!open) {
            continue;                        /* kept its barrier bit: the read index means done */
        }
        v.push_back(i);
    }
    /* Collected newest first, so that the cap keeps the packets nearest the read index. */
    return std::vector<uint64_t>(v.rbegin(), v.rend());
}

/* Whether the live range `bytes` long at `base` explains the fault address `a`: it holds `a`, or `a`
 * has run off its end and has not gone further past that end than the range is long. The bound is
 * the point. A read that runs off one buffer keeps within one allocation's length of it, so an
 * address just above a range is that range's run-off; an address 108 TiB above every range the
 * process holds is not evidence about any of them, and naming the range below it names a launch the
 * fault address has nothing to do with (issue #337's run 2 faulted 108 TiB above every live range,
 * and the report then said an operand was "108.63 TiB past the end of this same range"). */
inline bool fault_ran_off(uintptr_t base, int64_t bytes, uintptr_t a) {
    if (bytes <= 0 || a < base) return false;
    const uintptr_t end = base + (uintptr_t)bytes;
    if (a < end) return true;                                 /* inside it */
    return (uint64_t)(a - end) <= (uint64_t)bytes;
}

/* One line for a candidate: whether it is chained and what it waits for, in the words the report
 * uses. `need` is that slot's chain_need, `counter` the completion counter as the card has it at
 * the call. */
inline const char* running_state(uint64_t need, uint64_t counter) {
    return need > counter ? "may still run (chained: its blocks have not all counted)"
                          : "may still run (went out without its barrier bit)";
}

}  /* namespace aql */
}  /* namespace rad */
