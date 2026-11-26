/* cpu_stream.cpp -- quick multithreaded STREAM-like host memory bandwidth (read and copy),
 * for notes/backlog/measurements/pcie-h2d.md. Build: g++ -O3 -march=native -pthread cpu_stream.cpp
 * Args: [GiB per array, default 2] [reps, default 5]. Threads are spread over the CCDs first
 * (EPYC 7402P: 4 CCDs x 6 cores, CPUs 0-23 cores, 24-47 their SMT siblings).
 * Read: each thread sums its slice (4 accumulators). Copy: memcpy of its slice (glibc uses
 * non-temporal stores at this size); copy GB/s counts read + write bytes. GB = 1e9 bytes. */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int cpu_for(int i) {
    if (i >= 24) return 24 + cpu_for(i - 24);
    return (i % 4) * 6 + (i / 4) % 6;
}

int main(int argc, char** argv) {
    const size_t gib = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const size_t bytes = gib << 30;
    auto* a = (uint64_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    auto* b = (uint64_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(a, bytes, MADV_HUGEPAGE);
    madvise(b, bytes, MADV_HUGEPAGE);
    const size_t n = bytes / 8;
    std::vector<int> Ts = { 1, 2, 4, 8, 12, 16, 24, 48 };
    if (const char* e = std::getenv("CS_THREADS")) Ts = { std::atoi(e) };   /* one count only (background load) */
    const int modes = std::getenv("CS_READONLY") ? 1 : 2;
    for (int T : Ts) {
        for (int mode = 0; mode < modes; ++mode) {
            std::vector<double> gbs;
            std::atomic<uint64_t> sink{0};
            for (int r = 0; r < reps + 1; ++r) {
                std::atomic<int> ready{0};
                std::atomic<bool> go{false};
                std::vector<std::thread> th;
                std::vector<double> t0(T), t1(T);
                for (int t = 0; t < T; ++t)
                    th.emplace_back([&, t] {
                        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu_for(t), &cs);
                        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
                        const size_t lo = n * t / T, hi = n * (t + 1) / T;
                        if (r == 0) { for (size_t i = lo; i < hi; i += 512) { a[i] = i; b[i] = 0; } }
                        ready++;
                        while (!go) {}
                        t0[t] = now_s();
                        if (mode == 0) {
                            uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                            for (size_t i = lo; i + 3 < hi; i += 4) { s0 += a[i]; s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3]; }
                            sink += s0 + s1 + s2 + s3;
                        } else {
                            std::memcpy(b + lo, a + lo, (hi - lo) * 8);
                        }
                        t1[t] = now_s();
                    });
                while (ready < T) {}
                go = true;
                for (auto& x : th) x.join();
                if (r == 0) continue;  /* first pass faults pages in */
                const double dt = *std::max_element(t1.begin(), t1.end()) - *std::min_element(t0.begin(), t0.end());
                gbs.push_back((mode == 0 ? bytes : 2.0 * bytes) / dt / 1e9);
                if (std::getenv("CS_THREADS")) { std::printf("CPUSTREAM-rep %s threads=%d t=%.3f GB/s %.1f\n", mode ? "copy" : "read", T, now_s(), gbs.back()); std::fflush(stdout); }
            }
            std::sort(gbs.begin(), gbs.end());
            std::printf("CPUSTREAM %-4s threads=%2d GiB=%zu  GB/s med %6.1f min %6.1f max %6.1f\n", mode ? "copy" : "read", T,
                        gib, gbs[gbs.size() / 2], gbs.front(), gbs.back());
            std::fflush(stdout);
        }
    }
    return 0;
}
