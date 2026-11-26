/* w2.h -- "Weg 2" (abi/rad_w2.h), the runtime's half: the switch, the budget table and the state a
 * Ctx holds while it is on (Ctx::w2_, w2.cpp).
 *
 * RADIANCE_W2=1 turns it on (default off: Ctx::w2_open is never called, rad_expert_w2 returns null
 * and no block issues a W2 operand). With it on:
 *   RADIANCE_W2_MIN_ROWS  the lowest pass height that takes the form (default 16: the verify
 *                         passes of several sequences; one sequence's passes and a no-drafter
 *                         decode stay on the one-launch form, whose tail launches would cost them
 *                         more than their short window returns);
 *   RADIANCE_W2_KMAX      the most units a layer's list copies (default RAD_W2_MAX_COPY; the
 *                         budget by distinct routed experts applies below it);
 *   RADIANCE_W2_WIN_US0 / RADIANCE_W2_WIN_US_EXPERT   the window's line (rad_w2_window_us);
 *   RADIANCE_W2_RING      `vram` (default) or `uncached`: the ring's memory (W2Config::ring_cached). */
#pragma once
#include "rad_w2.h"

#include <cstdint>

namespace rad {

struct W2Config {
    bool   on = false;
    int    min_rows = 16;
    int    kmax = RAD_W2_MAX_COPY;
    double win_us0 = RAD_W2_WIN_US0;
    double win_us_expert = RAD_W2_WIN_US_EXPERT;
    /* The ring in ordinary (cached) device memory, with a system-scope acquire on each layer's
     * first tail launch so no cache holds a slot's previous contents: the default, because a
     * routed GEMM reads its weights slower from uncached device memory (rad-kbench, moe_down_dec
     * with every expert there: +17..26 % a launch at 1 to 32 rows; measurements/W2-prototype.md).
     * RADIANCE_W2_RING=uncached takes uncached device memory instead, as the prefetch's ring,
     * which needs no acquire. */
    bool   ring_cached = true;
    /* w2-explore (abi/rad_w2.h RadW2Layer::form / spin / gu_first): RADIANCE_W2_FORM=inline,
     * RADIANCE_W2_SPIN=<polls>, RADIANCE_W2_GU_FIRST=<bytes of a unit's gate_up part>. */
    int    form = 0;
    int    spin = 0;
    int64_t gu_first = 0;
};
/* From the environment (`get` is std::getenv, or a test's). */
W2Config w2_config(const char* (*get)(const char*));
W2Config w2_config();

/* w2-explore: RADIANCE_W2_AB = reserve (1) / queue (2) / noreserve (3), else 0: which part of the
 * switch's idle cost a run carries (engine_bringup.cpp). */
int w2_ab();

/* The budget table a layer's control block carries: out[d] = the longest list that lands inside
 * the window of a layer routing d distinct experts, d in [0, RAD_W2_MAX_EXPERT]. */
void w2_fill_kmax(const W2Config& cfg, uint8_t* out);

/* Device bytes W2 takes for `layers` routed layers whose largest expert unit is `unit` bytes (what
 * Ctx::w2_open allocates on the card), each allocation rounded to the card's 2 MiB grain. */
int64_t w2_device_bytes(int64_t unit, int64_t layers);

}  /* namespace rad */
