/* tests/halo_emu: libhalo's sample_chain (halo_sample_chain_f32.hip) under r4d_emu's sampler test:
 * both thread orders against the serial host oracle, and the f32-atomic negative control. */
#define SAMPLE_HALO 1
#include "../r4d_emu/test_sample.cpp"
