/* tests/r4d_emu/model_stubs.cpp -- the two libr4d entry points r4d_model_bf16.hip names but that
 * live in units the emulator does not build: the peer copy r4d_rad_cast falls back to across two
 * devices (r4d_ar_oneshot_2rank_exact.hip) and the f32 encoder's rope (r4d_audio.hip).
 *
 * No test in tests/r4d_emu takes either path -- the emulator is one device and every emulated rope
 * is bf16 -- so they answer RAD_E_UNSUPPORTED, which makes a launch that ever does reach one fail
 * loudly instead of running something made up. They are needed only because the emulator links the
 * units' object files whole. */
#include "rad_abi.h"

extern "C" {

int r4d_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                   long stream) {
    (void)dst; (void)src; (void)dpitch; (void)spitch; (void)width; (void)height; (void)stream;
    return -100;   /* what the real copy answers when the card refuses: RAD_E_DEVICE at the caller */
}

int r4d_rad_encoder_rope(const RadArgs* a, RadStream s) {
    (void)a; (void)s;
    return RAD_E_UNSUPPORTED;
}

}  /* extern "C" */
