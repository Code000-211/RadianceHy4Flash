/* kfd_evict.cpp -- see kfd_evict.h. */
#include "device/kfd_evict.h"

#include "rad_internal.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/resource.h>

namespace rad {

PinLockTally& pin_lock_tally() {
    static PinLockTally t;
    return t;
}

bool lock_against_swap(void* p, int64_t len, PinLockFn lock, int64_t memlock_soft, PinLockTally* t,
                       std::string* say) {
    say->clear();
    if (len <= 0) return true;
    if (lock(p, (size_t)len) == 0) {
        t->locked.fetch_add(len, std::memory_order_relaxed);
        return true;
    }
    const int err = errno;
    t->unlocked.fetch_add(len, std::memory_order_relaxed);
    if (t->told.exchange(true)) return false;
    const std::string limit = memlock_soft == -2 ? std::string("unlimited")
                              : memlock_soft < 0 ? std::string("unknown")
                                                 : humanb(memlock_soft);
    *say = fmt("device: pinned host memory is not locked against swap: mlock2 of %s at %p was "
               "refused (%s) with RLIMIT_MEMLOCK %s. The cards map it without pinning its pages, "
               "so reclaim can swap them out, and each swap-out evicts every queue of this process "
               "on every card until they are read back in (all cards stop together; the stall "
               "dump's evicted_ms shows it). Further refusals are counted, not logged. Raise the "
               "limit: Docker --ulimit memlock=-1, systemd LimitMEMLOCK=infinity, or memlock in "
               "/etc/security/limits.conf",
               humanb(len).c_str(), p, std::strerror(err), limit.c_str());
    return false;
}

void pin_lock_forget(int64_t len, bool locked, PinLockTally* t) {
    if (len <= 0) return;
    (locked ? t->locked : t->unlocked).fetch_sub(len, std::memory_order_relaxed);
}

int64_t memlock_soft_to_hard() {
    struct rlimit rl {};
    if (::getrlimit(RLIMIT_MEMLOCK, &rl) != 0) return -1;
    if (rl.rlim_cur != rl.rlim_max) {
        struct rlimit up = rl;
        up.rlim_cur = rl.rlim_max;
        if (::setrlimit(RLIMIT_MEMLOCK, &up) == 0) rl = up;
    }
    return rl.rlim_cur == RLIM_INFINITY ? -2 : (int64_t)rl.rlim_cur;
}

int64_t kfd_evicted_ms(const std::string& proc_root, long pid, uint32_t gpu_id) {
    std::ifstream f(fmt("%s/%ld/stats_%u/evicted_ms", proc_root.c_str(), pid, gpu_id));
    long long v = -1;
    if (!(f >> v) || v < 0) return -1;
    return (int64_t)v;
}

int64_t status_bytes(const std::string& status_path, const char* key) {
    std::ifstream f(status_path);
    std::string line;
    const size_t k = std::strlen(key);
    while (std::getline(f, line)) {
        if (line.compare(0, k, key) != 0 || line.size() <= k || line[k] != ':') continue;
        long long kb = -1;
        if (std::sscanf(line.c_str() + k + 1, "%lld", &kb) != 1 || kb < 0) return -1;
        return (int64_t)kb * 1024;
    }
    return -1;
}

std::string evict_line(const std::string& proc_root, const std::string& status_path, long pid,
                       const std::vector<std::pair<int, uint32_t>>& cards,
                       std::map<uint32_t, int64_t>* last) {
    std::string o = "driver queue eviction (KFD evicted_ms, in all / since the last dump):";
    bool first = true;
    for (const auto& [ord, id] : cards) {
        const int64_t ms = kfd_evicted_ms(proc_root, pid, id);
        o += fmt("%s card %d ", first ? "" : ",", ord);
        first = false;
        if (ms < 0) {
            o += "?";
            continue;
        }
        o += fmt("%" PRId64, ms);
        auto it = last->find(id);
        if (it != last->end() && it->second >= 0 && ms >= it->second)
            o += fmt(" (+%" PRId64 ")", ms - it->second);
        (*last)[id] = ms;
    }
    const int64_t swap = status_bytes(status_path, "VmSwap");
    const int64_t lck = status_bytes(status_path, "VmLck");
    o += fmt("; VmSwap %s, VmLck %s", swap < 0 ? "?" : humanb(swap).c_str(),
             lck < 0 ? "?" : humanb(lck).c_str());
    return o;
}

}  /* namespace rad */
