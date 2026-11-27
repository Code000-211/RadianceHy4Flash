/* kfd_evict.h -- host memory the cards map, held against swap, and the driver's eviction count
 * (issue #272).
 *
 * A pinned range (aql.cpp pin_pieces) is locked for the cards through ROCr, which on a driver with
 * the SVM API mirrors the range (HMM) rather than pinning its pages: /proc/<pid>/status shows
 * VmPin 0 however much is "pinned". The pages stay on the kernel's LRU, so reclaim can swap them
 * out. Each such move fires the driver's MMU notifier, which evicts every queue of the process on
 * every card until the pages are read back in; under steady pressure that stops all cards at once
 * for seconds, and a stop that outlasts scheduler.stuck-exit ends the process.
 *
 * Two parts, both without a card:
 *   - lock_against_swap: mlock2(MLOCK_ONFAULT) of a pinned range, which takes its pages off the
 *     LRU (compaction may still move them while vm.compact_unevictable_allowed is 1). A refusal --
 *     RLIMIT_MEMLOCK, most often -- never fails the allocation; it is counted, and said once with
 *     the limit and how to raise it.
 *   - evict_line: what the stall dump prints per card, the driver's evicted_ms
 *     (/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms) and its growth since the last
 *     dump, beside the process's VmSwap and VmLck. A count that grows while the queues stand is the
 *     driver holding them; a count that stays put while they stand is the card. */
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace rad {

/* What the pinned ranges hold now: bytes locked against swap, and bytes a refused lock left
 * swappable. A freed range is taken back out (pin_lock_forget). */
struct PinLockTally {
    std::atomic<int64_t> locked{0};
    std::atomic<int64_t> unlocked{0};
    std::atomic<bool>    told{false};      /* the refusal has been said */
};

/* The process's tally. */
PinLockTally& pin_lock_tally();

/* The lock call: 0, or -1 with errno set (mlock2(p, n, MLOCK_ONFAULT) in the engine). */
using PinLockFn = int (*)(void* p, size_t n);

/* Locks [p, p + len) with `lock` and counts the result in `t`. True when locked. On the first
 * refusal `t` sees, `*say` is the warning to log (the error, `memlock_soft` -- bytes, -2 for
 * unlimited, -1 unknown -- and the fix); otherwise `*say` is left empty. */
bool lock_against_swap(void* p, int64_t len, PinLockFn lock, int64_t memlock_soft, PinLockTally* t,
                       std::string* say);

/* A range counted by lock_against_swap is being freed. */
void pin_lock_forget(int64_t len, bool locked, PinLockTally* t);

/* Raises the soft RLIMIT_MEMLOCK to the hard one, which needs no privilege. Returns the soft
 * limit afterwards in bytes, -2 for unlimited, -1 when unreadable. */
int64_t memlock_soft_to_hard();

/* evicted_ms of KFD gpu_id `gpu_id` for process `pid` under `proc_root`
 * ("/sys/class/kfd/kfd/proc"), -1 when it cannot be read. */
int64_t kfd_evicted_ms(const std::string& proc_root, long pid, uint32_t gpu_id);

/* Field `key` ("VmSwap") of a /proc/<pid>/status file, in bytes; -1 when absent. */
int64_t status_bytes(const std::string& status_path, const char* key);

/* The dump's line for cards (ordinal, gpu_id). `last` holds each gpu_id's count at the previous
 * call and is updated, so the line gives the growth since then. */
std::string evict_line(const std::string& proc_root, const std::string& status_path, long pid,
                       const std::vector<std::pair<int, uint32_t>>& cards,
                       std::map<uint32_t, int64_t>* last);

}  /* namespace rad */
