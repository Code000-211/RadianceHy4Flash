/* thread_pin.h -- put the step and rank threads on one CCD (C-03). */
#pragma once

namespace rad {

/* Pin the calling thread to physical cores [slot, slot + cores) of the first CCD, spreading the
 * ranks over one CCD and leaving the rest of the machine to the HTTP workers. Off unless
 * RADIANCE_PIN_CCD names a CCD index or is 1 (then slot selects by thread). Returns 0 when the
 * thread is pinned, 1 when pinning is off, and a negative errno when the platform refused it --
 * never fatal: a deployment that cannot pin runs exactly as before. */
int pin_thread_to_ccd(int slot);

}  /* namespace rad */
