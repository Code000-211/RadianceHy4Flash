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
 * that slot waits for (Stream::slot_need, 0 for a packet that is not chained) and the largest
 * counter value it has read (Stream::chain_seen). A chained dispatch is done only once that value
 * is reached, so one whose value is not reached may still be running. A plain dispatch keeps its
 * barrier bit, so the read index passing it does mean it completed -- unless an unbarriered packet
 * went out since (`open`), and then every packet the ring still holds is a candidate.
 *
 * The queue stops at the first launch that faults, so the oldest candidate is the one to read
 * first. Byte- and state-only, no queue and no card, so tests/aql_inflight_test.cpp covers it. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rad {
namespace aql {

/* The packet indices a queue may still be running, oldest first.
 *
 * `rd` and `wr` are the queue's read and write indices, `size` its packet count (the ring), and
 * `counter` the completion counter as the host last read it (a lagging read only ever adds
 * candidates, never drops one). `need_at(i)` is what the chained dispatch at index `i` waits for,
 * 0 when it is not chained; the caller answers from the slot `i & (size - 1)`.
 *
 * The window is the last `size` packets written, [wr - size, wr), which is what the ring still
 * holds -- a slot older than that has been rewritten and says nothing about the packet that used to
 * be in it. Packets at or past `rd` are written and not read: they have not run. At most `cap`
 * indices come back; `cap` 0 reports none, for a caller that does not want the walk. */
template <class NeedAt>
std::vector<uint64_t> still_running(uint64_t rd, uint64_t wr, uint64_t size, uint64_t counter,
                                    bool open, NeedAt need_at, size_t cap) {
    std::vector<uint64_t> v;
    if (size == 0 || wr == 0 || cap == 0) return v;
    const uint64_t first = wr > size ? wr - size : 0;
    for (uint64_t i = first; i < wr && i < rd; ++i) {
        if (!open && need_at(i) <= counter) continue;
        v.push_back(i);
        if (v.size() >= cap) break;
    }
    return v;
}

/* One line for a candidate: whether it is chained and what it waits for, in the words the report
 * uses. `need` is that slot's chain_need, `counter` the value the host last read. */
inline const char* running_state(uint64_t need, uint64_t counter) {
    return need > counter ? "may still run (chained: its blocks have not all counted)"
                          : "may still run (went out without its barrier bit)";
}

}  /* namespace aql */
}  /* namespace rad */
