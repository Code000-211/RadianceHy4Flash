/* ovt_unit_a.cpp -- one unit of tests/override's kernel library, and the one its override
 * rebuilds: a row's launch and scratch hooks, and a function another unit calls. */
#include "ovt.h"

extern "C" {

/* Counts in the library's own state, which a rebuilt copy of this unit must reach as well. */
int ovt_launch_a(const RadArgs*, RadStream) {
    ++ovt_launches;
    return OVT_ANSWER(1);
}

int64_t ovt_scratch_a(const RadArgs*) { return OVT_ANSWER(2); }

int ovt_helper_a(void) { return OVT_ANSWER(7); }

}  /* extern "C" */
