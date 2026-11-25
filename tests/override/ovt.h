/* ovt.h -- what the units of tests/override's kernel library share (CMakeLists.txt there). */
#ifndef RAD_TEST_OVT_H
#define RAD_TEST_OVT_H

#include "rad_abi.h"

/* OVT_VARIANT is 1 in the library and 2 where a unit is rebuilt into an override, so a hook's
 * answer says which build of it ran: variant * 100 + the hook's own number. */
#ifndef OVT_VARIANT
#define OVT_VARIANT 1
#endif
#define OVT_ANSWER(n) (OVT_VARIANT * 100 + (n))

#ifdef __cplusplus
extern "C" {
#endif

/* ovt_rows.cpp: state of the library outside its units. */
extern int ovt_launches;

/* ovt_unit_a.cpp */
int     ovt_launch_a(const RadArgs*, RadStream);
int64_t ovt_scratch_a(const RadArgs*);
int     ovt_helper_a(void);

/* ovt_unit_b.cpp */
int ovt_launch_b(const RadArgs*, RadStream);

#ifdef __cplusplus
}
#endif

#endif /* RAD_TEST_OVT_H */
