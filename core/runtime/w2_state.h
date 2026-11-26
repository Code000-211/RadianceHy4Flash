/* w2_state.h -- W2's host state, held by the Ctx (Ctx::w2_). See w2.h and abi/rad_w2.h. */
#pragma once
#include "w2.h"
#include "../device/device.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace rad {

struct W2State {
    W2Config     cfg;
    int          device = 0;
    char*        ring   = nullptr;     /* RAD_W2_GROUPS x RAD_W2_MAX_COPY slots of `slot` bytes */
    bool         ring_cached = false;  /* W2Config::ring_cached: ordinary device memory */
    uint32_t*    done_base = nullptr;  /* the uncached allocation the done words are in, where not the ring's */
    uint64_t     slot   = 0;
    RadW2Pools   pools{};
    RadW2Layer*  layers = nullptr;     /* device, [n_layers] */
    uint32_t*    head   = nullptr;     /* device */
    RadW2Stats*  stats  = nullptr;     /* device */
    RadW2Box*    box    = nullptr;     /* pinned host, [RAD_W2_BOXES] */
    uint32_t*    done   = nullptr;     /* device, behind the ring's last slot: a word a group */
    RadW2Stats*  stats_h = nullptr;    /* pinned host, read at the end */
    int          n_layers = 0;
    std::vector<uint8_t> bound;        /* per layer: its control block is written */
    uint64_t     epoch = 0;            /* layers bound so far: part of a pass's key */

    /* The host thread: each post, once the plan has written it, to the DMA queue. It polls while a
     * pass that carries the form ran within the last kActive, and sleeps otherwise. */
    std::thread              thr;
    std::mutex               mu;
    std::condition_variable  cv;
    std::atomic<bool>        stop{false};
    std::atomic<int64_t>     active_until{0};      /* steady_clock ns */
    std::atomic<bool>        asleep{true};
    std::atomic<int64_t>     lists{0}, dropped{0}, units{0};
};

}  /* namespace rad */
