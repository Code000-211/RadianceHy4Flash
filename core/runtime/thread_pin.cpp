/* thread_pin.cpp -- see thread_pin.h. */
#include "thread_pin.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#endif

namespace rad {

int pin_thread_to_ccd(int slot)
{
#if defined(__linux__)
    const char* env = std::getenv("RADIANCE_PIN_CCD");
    if (!env || !*env || std::strcmp(env, "0") == 0) return 1;   /* off by default */

    const int cpus = get_nprocs();
    if (cpus < 4) return 1;

    /* Two cores a CCD on the small parts, eight on a desktop Ryzen: derive the CCD width from the
     * largest L3 sharing group the kernel reports, and fall back to half the machine. */
    int width = 0;
#if defined(__linux__)
    {
        char buf[64];
        for (int i = 0; i < cpus; ++i) {
            std::snprintf(buf, sizeof buf, "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", i);
            FILE* f = std::fopen(buf, "r");
            if (!f) continue;
            int lo = -1, hi = -1;
            if (std::fscanf(f, "%d-%d", &lo, &hi) == 2 && hi >= lo) { width = hi - lo + 1; }
            else if (std::fscanf(f, "%d", &lo) == 1) { width = 1; }
            std::fclose(f);
            break;
        }
    }
#endif
    if (width <= 0) width = cpus / 2;

    cpu_set_t set;
    CPU_ZERO(&set);
    const int base = 0;                       /* the first CCD; slot only orders the ranks in it */
    const int core = base + (slot % (width > 0 ? width : 1));
    CPU_SET(core, &set);
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    return rc == 0 ? 0 : -rc;
#else
    (void)slot;
    return 1;
#endif
}

}  /* namespace rad */
