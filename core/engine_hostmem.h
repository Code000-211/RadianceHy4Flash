/* engine_hostmem.h -- the memory cgroup limits bring-up checks before it locks a table into host
 * memory (engine_bringup.cpp, lockable_room). */
#pragma once
#include <cstdint>
#include <functional>
#include <string>

namespace rad {

/* A cgroup's limit less the usage reclaim cannot give back: memory.current less file pages (v2)
 * or usage_in_bytes less cache (v1), since page cache yields to a locked allocation and anonymous
 * memory does not. "max" and v1's near-INT64_MAX sentinel mean no limit. True, with `room` set,
 * when `dir` has a limit. */
bool cgroup_room(const std::string& dir, bool v2, int64_t* room);

/* Every level of the memory cgroup at `path` (as /proc/self/cgroup names it), from that level up
 * to the hierarchy's root mounted at `root`, leaf first: `visit(dir, room)` for each level with a
 * limit, until it returns false. Each level is a limit of its own. */
void cgroup_levels(const std::string& root, std::string path, bool v2,
                   const std::function<bool(const std::string& dir, int64_t room)>& visit);

/* Every level of every memory cgroup the process is in, leaf first: `visit(dir, room)` until it
 * returns false (cgroup_levels). /proc/self/cgroup reads "0::/path" on v2 and "N:memory:/path"
 * (or a comma list naming memory) on v1; each level up to the visible root is a limit of its
 * own. */
void memory_cgroups(const std::function<bool(const std::string& dir, int64_t room)>& visit);

/* The least room any of those levels leaves (cgroup_room), or INT64_MAX when none sets a limit:
 * a container's --memory, which the OOM killer enforces whatever MemAvailable says. */
int64_t cgroup_least_room();

/* Whether `need` more bytes fit in host memory after `still` more this process will allocate
 * first and `margin` (--host-headroom-mib) stays free, or why not. Checks MemAvailable; every
 * memory cgroup from the process's own up to the visible root (Docker's --memory, enforced by the
 * OOM killer); and, for an mlock()ed copy (`mlocks`), RLIMIT_MEMLOCK unless the process has
 * CAP_IPC_LOCK. Pinned memory a card reads is also mlock()ed where the limit allows (#272), and
 * VmLck counts it, but a refusal there never fails the start, so it is not checked here. An
 * unreadable limit is ignored, never guessed. */
bool lockable_room(int64_t need, int64_t still, int64_t margin, bool mlocks, std::string* why,
                   int64_t* short_by = nullptr);

/* #273: how far short a refusal may be and still be waited out -- 2 GiB, or an eighth of what the
 * check wanted in all (need + still + margin), whichever is more. After the stuck-device exit the
 * restarted server is short by the part of the old process's memory the kernel has not yet given
 * back (0.6-2.4 GiB of 224 GiB on the issue's host); a config asking far more than the host has is
 * a misconfiguration, and it still fails at once. */
bool room_short_is_small(int64_t short_by, int64_t want);

/* A fatal host-memory check that waits out a small shortfall. `probe` is the check: true when it
 * fits, else false with the reason in `why` and in `short_by` the bytes it is short (negative for
 * a refusal no wait can cure, such as RLIMIT_MEMLOCK). When the first probe is short by a small
 * amount (room_short_is_small of `want`) and `wait_ms` > 0, it re-probes every `step_ms` for up
 * to `wait_ms`, saying so once and then a line a re-check; it stops at once when the shortfall
 * grows past the bound. When the wait runs out, `why` is the last reason with the wait and its
 * likely cause appended. `nap` sleeps (tests pass a fake); empty sleeps the thread. */
bool wait_for_room(const std::function<bool(std::string* why, int64_t* short_by)>& probe,
                   int64_t want, int64_t wait_ms, int64_t step_ms, std::string* why,
                   const std::function<void(int64_t ms)>& nap = {});

}  /* namespace rad */
