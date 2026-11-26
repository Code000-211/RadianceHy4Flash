/* ovt_unit_b.cpp -- the unit of tests/override's kernel library no override rebuilds. */
#include "ovt.h"

extern "C" int ovt_launch_b(const RadArgs*, RadStream) { return OVT_ANSWER(3); }
