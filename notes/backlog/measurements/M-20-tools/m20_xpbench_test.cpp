/* m20_xpbench_test.cpp -- M-20 design measurement (scratch, not committed as code): the expert
 * prefetch's own DMA path (rad_xp_list on the card's second SDMA engine, the queue xprefetch.cpp
 * uses) timed for routed `down` / `gate_up` / whole-unit copies from pinned host memory into an
 * uncached device ring, on one card and on both cards at once (TP2 shares the host side), plus the
 * host round trip the prefetch's serving thread adds (device writes a pinned word, a host thread
 * sees it and hands the list to the queue).
 *
 * Env: M20_DN (down bytes), M20_GU (gate_up bytes), M20_REPS (default 60). */
#include "rad_test.h"
#include "host_kargs_main.h"

#include "rad_internal.h"
#include "device/aql.h"
#include "device/device.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

int64_t envi(const char* k, int64_t d) {
    const char* v = std::getenv(k);
    return v && *v ? std::atoll(v) : d;
}

struct Res { double p10, p50, p90; };
Res stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[(size_t)std::min<double>(v.size() - 1, f * (v.size() - 1) + 0.5)]; };
    return { q(0.1), q(0.5), q(0.9) };
}

struct Card {
    int dev = 0;
    uint8_t* host = nullptr;   /* pinned source pool */
    uint8_t* ring = nullptr;   /* uncached device ring */
    uint32_t* done = nullptr;  /* pinned host word */
    int64_t pool = 0, ringb = 0;
    bool open(int d, int64_t pool_b, int64_t ring_b) {
        dev = d; pool = pool_b; ringb = ring_b;
        if (rad_dev_set(d) < 0) return false;
        host = (uint8_t*)rad_dev_alloc(pool_b, RAD_MEM_HOST_PINNED);
        ring = (uint8_t*)rad_dev_alloc_uncached(ring_b);
        done = (uint32_t*)rad_dev_alloc(4096, RAD_MEM_HOST_PINNED);
        if (!host || !ring || !done) return false;
        for (int64_t i = 0; i < pool_b; i += 4096) host[i] = (uint8_t)i;
        *(volatile uint32_t*)done = 0;
        return true;
    }
    /* One list of n copies of `b` bytes; returns microseconds from hand-over to the done word. */
    double list(int n, int64_t b, uint32_t val, int salt) {
        std::vector<uint64_t> src(n), dst(n), by(n, (uint64_t)b);
        const int64_t sstride = (b + 65535) / 65536 * 65536;
        const int64_t nsrc = std::max<int64_t>(1, pool / sstride), ndst = std::max<int64_t>(1, ringb / sstride);
        for (int i = 0; i < n; ++i) {
            src[i] = (uint64_t)(uintptr_t)(host + ((i * 7 + salt * 13) % nsrc) * sstride);
            dst[i] = (uint64_t)(uintptr_t)(ring + (i % ndst) * sstride);
        }
        const auto t0 = clk::now();
        if (rad_xp_list(dev, nullptr, n, src.data(), dst.data(), by.data(), done, val, RAD_XP_NO_YIELD) != RAD_OK)
            return -1;
        while (*(volatile uint32_t*)done != val) {}
        return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    }
};

}  // namespace

TEST(m20_xp_dma_bench) {
    if (rad_dev_count() < 1 || !rad_xp_supported(0)) SKIP("no prefetch DMA queue");
    const int64_t dn = envi("M20_DN", 608256), gu = envi("M20_GU", 891290), reps = envi("M20_REPS", 60);
    const int ncard = std::min(2, rad_dev_count());
    std::vector<Card> c(ncard);
    for (int d = 0; d < ncard; ++d) REQUIRE(c[d].open(d, (int64_t)512 << 20, (int64_t)64 << 20));
    for (int d = 0; d < ncard; ++d) REQUIRE(rad_xp_supported(d));
    uint32_t val = 1;
    struct Row { const char* what; int64_t b; };
    const Row rows[] = { { "down", dn }, { "gate_up", gu }, { "unit", dn + gu } };
    const int ns[] = { 1, 2, 4, 8, 16, 32 };
    /* warm */
    for (int d = 0; d < ncard; ++d) for (int i = 0; i < 20; ++i) c[d].list(4, dn, val++, i);

    for (int mode = 0; mode < (ncard > 1 ? 2 : 1); ++mode) {
        for (const Row& r : rows)
            for (int n : ns) {
                std::vector<double> t0v, t1v;
                for (int k = 0; k < reps; ++k) {
                    if (mode == 0) {
                        t0v.push_back(c[0].list(n, r.b, val++, k));
                    } else {
                        std::atomic<int> go{0};
                        double t1 = 0;
                        const uint32_t v1 = val++;
                        std::thread th([&] { rad_dev_set(1); go.fetch_add(1); while (go.load() < 2) {}
                                             t1 = c[1].list(n, r.b, v1, k); });
                        const uint32_t v0 = val++;
                        rad_dev_set(0);
                        go.fetch_add(1);
                        while (go.load() < 2) {}
                        const double t0 = c[0].list(n, r.b, v0, k);
                        th.join();
                        t0v.push_back(t0);
                        t1v.push_back(t1);
                    }
                }
                const Res a = stats(t0v);
                const double gbs = (double)n * r.b / (a.p50 * 1e-6) / 1e9;
                if (mode == 0)
                    std::printf("M20XP one-card %-7s bytes=%lld n=%2d  us p10/p50/p90 = %8.1f %8.1f %8.1f  "
                                "per-copy p50 %6.1f us  %5.1f GB/s\n", r.what, (long long)r.b, n, a.p10, a.p50,
                                a.p90, a.p50 / n, gbs);
                else {
                    const Res b = stats(t1v);
                    const double gbs1 = (double)n * r.b / (b.p50 * 1e-6) / 1e9;
                    std::printf("M20XP two-cards %-7s bytes=%lld n=%2d  card0 p50 %8.1f us (%5.1f GB/s) p90 %8.1f | "
                                "card1 p50 %8.1f us (%5.1f GB/s) p90 %8.1f\n", r.what, (long long)r.b, n, a.p50, gbs,
                                a.p90, b.p50, gbs1, b.p90);
                }
            }
    }
    /* large copy: link bandwidth */
    for (int k = 0; k < 2; ++k) {
        std::vector<double> v;
        for (int i = 0; i < 10; ++i) v.push_back(c[0].list(1, (int64_t)48 << 20, val++, i));
        const Res a = stats(v);
        std::printf("M20XP one-card 48MiB  us p50 %.1f  %.1f GB/s\n", a.p50, (48.0 * (1 << 20)) / (a.p50 * 1e-6) / 1e9);
    }
    /* host round trip: the device writes a pinned word (a stream memset), a host thread sees it and
     * hands a list of n down copies to the queue -- the prefetch's serving-thread path. */
    {
        RadStream s = nullptr;
        REQUIRE_EQ(rad_dev_set(0), RAD_OK);
        REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
        uint32_t* box = (uint32_t*)rad_dev_alloc(4096, RAD_MEM_HOST_PINNED);
        REQUIRE(box != nullptr);
        for (int n : { 1, 4, 8 }) {
            std::vector<double> seen, total;
            for (int k = 0; k < reps; ++k) {
                *(volatile uint32_t*)box = 0;
                std::atomic<bool> armed{false};
                double t_seen = 0, t_done = 0;
                clk::time_point t0;
                const uint32_t v = val++;
                std::thread th([&] {
                    rad_dev_set(0);
                    armed = true;
                    while (*(volatile uint32_t*)box == 0) {}
                    const auto ts = clk::now();
                    std::vector<uint64_t> src(n), dst(n), by(n, (uint64_t)dn);
                    for (int i = 0; i < n; ++i) {
                        src[i] = (uint64_t)(uintptr_t)(c[0].host + (int64_t)(i + k) * (1 << 20));
                        dst[i] = (uint64_t)(uintptr_t)(c[0].ring + (int64_t)i * (1 << 20));
                    }
                    rad_xp_list(0, nullptr, n, src.data(), dst.data(), by.data(), c[0].done, v, RAD_XP_NO_YIELD);
                    while (*(volatile uint32_t*)c[0].done != v) {}
                    const auto td = clk::now();
                    t_seen = std::chrono::duration<double, std::micro>(ts - t0).count();
                    t_done = std::chrono::duration<double, std::micro>(td - t0).count();
                });
                while (!armed) {}
                t0 = clk::now();
                rad_memset_async(box, 1, 4, s);
                th.join();
                rad_stream_sync(s);
                seen.push_back(t_seen);
                total.push_back(t_done);
            }
            const Res a = stats(seen), b = stats(total);
            std::printf("M20XP roundtrip n=%d down: device word seen by host p50 %.1f us (p90 %.1f); copies done "
                        "p50 %.1f us (p90 %.1f) after the memset was issued\n", n, a.p50, a.p90, b.p50, b.p90);
        }
        rad_stream_destroy(s);
    }
}

RAD_TEST_MAIN_HOST_KARGS()
