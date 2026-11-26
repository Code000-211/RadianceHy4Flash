/* region_probe.cpp -- does the bandwidth depend on where host pages sit physically?
 * The container hides physical addresses (/proc/self/pagemap reports PFN 0 without CAP_SYS_ADMIN),
 * so this probes by bandwidth instead: it maps G GiB of THP-backed anonymous memory in 1 GiB chunks
 * and, per chunk, measures (a) CPU read bandwidth with T threads spread over the CCDs and
 * (b) host->device bandwidth with both cards at once (each card reads its own 512 MiB half of the
 * chunk, SDMA engine 0 via ROCr, 256 MiB copies, and the zero-copy read kernel), the chunk pinned
 * the engine's way (hsa_amd_memory_lock_to_pool on the fine-grained host pool). A 4-way-interleaved
 * region (the top ~64 GB on this host's 4 x 32 GB channels) would show as a second cluster.
 *
 * Build: hipcc -O3 -std=c++20 --offload-arch=gfx1201 region_probe.cpp -o region_probe -lhsa-runtime64 -pthread
 * Args: GiB [threads=16] [reps=3] */
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <string>
#include <thread>
#include <vector>

#define CHK(x) do { if ((x) != 0) { std::fprintf(stderr, "fail %s line %d\n", #x, __LINE__); std::exit(1); } } while (0)

static size_t thp_bytes(void* p, size_t len) {
    FILE* f = std::fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    char line[512]; uintptr_t lo = (uintptr_t)p, hi = lo + len; bool in = false; size_t total = 0;
    while (std::fgets(line, sizeof line, f)) {
        uintptr_t a, b;
        if (line[0] != ' ' && std::sscanf(line, "%lx-%lx ", &a, &b) == 2) { in = (a < hi && b > lo); continue; }
        size_t kb;
        if (in && std::sscanf(line, "AnonHugePages: %zu kB", &kb) == 1) total += kb << 10;
    }
    std::fclose(f);
    return total;
}
static double now_s() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int cpu_for(int i) { return i >= 24 ? 24 + cpu_for(i - 24) : (i % 4) * 6 + (i / 4) % 6; }

static hsa_agent_t g_cpu{};
static std::vector<hsa_agent_t> g_gpu;
static hsa_amd_memory_pool_t g_fine{};
static hsa_status_t acb(hsa_agent_t a, void*) {
    hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    if (t == HSA_DEVICE_TYPE_CPU && !g_cpu.handle) g_cpu = a;
    if (t == HSA_DEVICE_TYPE_GPU) g_gpu.push_back(a);
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t pcb(hsa_amd_memory_pool_t p, void*) {
    hsa_amd_segment_t s; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &s);
    uint32_t f = 0; hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &f);
    if (s == HSA_AMD_SEGMENT_GLOBAL && (f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) &&
        !(f & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) && !g_fine.handle) g_fine = p;
    return HSA_STATUS_SUCCESS;
}

typedef unsigned int v4u __attribute__((ext_vector_type(4)));
__global__ void stream_read(const v4u* __restrict__ src, size_t n16, uint32_t* out) {
    v4u acc = {0, 0, 0, 0};
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) acc ^= __builtin_nontemporal_load(src + i);
    uint32_t r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) out[blockIdx.x] = r;
}

static double cpu_read(const uint64_t* p, size_t bytes, int T) {
    std::atomic<int> ready{0}; std::atomic<bool> go{false}; std::atomic<uint64_t> sink{0};
    std::vector<double> t0(T), t1(T); std::vector<std::thread> th;
    const size_t n = bytes / 8;
    for (int t = 0; t < T; ++t) th.emplace_back([&, t] {
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu_for(t), &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
        const size_t lo = n * t / T, hi = n * (t + 1) / T;
        ready++; while (!go) {}
        t0[t] = now_s();
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (size_t i = lo; i + 3 < hi; i += 4) { s0 += p[i]; s1 += p[i + 1]; s2 += p[i + 2]; s3 += p[i + 3]; }
        sink += s0 + s1 + s2 + s3; t1[t] = now_s();
    });
    while (ready < T) {}
    go = true;
    for (auto& x : th) x.join();
    return bytes / (*std::max_element(t1.begin(), t1.end()) - *std::min_element(t0.begin(), t0.end())) / 1e9;
}

int main(int argc, char** argv) {
    const int G = argc > 1 ? std::atoi(argv[1]) : 32, T = argc > 2 ? std::atoi(argv[2]) : 16, R = argc > 3 ? std::atoi(argv[3]) : 3;
    const size_t C = (size_t)1 << 30, half = C / 2, big = (size_t)256 << 20;
    int nd = 0; CHK(hipGetDeviceCount(&nd)); nd = std::min(nd, 2);
    std::vector<char*> dst(nd); std::vector<hipStream_t> st(nd); std::vector<uint32_t*> out(nd);
    for (int d = 0; d < nd; ++d) { CHK(hipSetDevice(d)); CHK(hipMalloc((void**)&dst[d], big)); CHK(hipMalloc((void**)&out[d], 1 << 20)); CHK(hipStreamCreate(&st[d])); }
    CHK(hsa_init()); hsa_iterate_agents(acb, nullptr); hsa_amd_agent_iterate_memory_pools(g_cpu, pcb, nullptr);
    /* HIP device order == HSA GPU order here (checked by h2d_bench's BDF match on this host) */
    std::vector<hsa_signal_t> sig(nd);
    for (int d = 0; d < nd; ++d) hsa_signal_create(0, 0, nullptr, &sig[d]);

    std::vector<char*> chunk(G);
    for (int c = 0; c < G; ++c) {
        void* m = mmap(nullptr, C + ((size_t)2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { std::printf("PROBE mmap failed at chunk %d\n", c); return 1; }
        char* p = (char*)(((uintptr_t)m + ((2 << 20) - 1)) & ~(uintptr_t)((2 << 20) - 1));
        madvise(p, C, MADV_HUGEPAGE);
        for (size_t o = 0; o < C; o += 4096) p[o] = (char)o;
        chunk[c] = p;
    }
    std::printf("PROBE %d chunks of 1 GiB mapped (THP), T=%d threads, %d reps (median)\n", G, T, R);
    const int passes = std::getenv("PROBE_PASSES") ? std::atoi(std::getenv("PROBE_PASSES")) : 1;
    std::vector<size_t> thp(G);
    for (int c = 0; c < G; ++c) thp[c] = thp_bytes(chunk[c], C) >> 20;
    const int npass = std::getenv("PROBE_SKIPSCAN") ? 0 : passes;
    for (int pass = 0; pass < npass; ++pass)
    for (int c = 0; c < G; ++c) {
        std::vector<double> cr;
        for (int r = 0; r < R; ++r) cr.push_back(cpu_read((const uint64_t*)chunk[c], C, T));
        std::sort(cr.begin(), cr.end());
        void* on = nullptr;
        if (hsa_amd_memory_lock_to_pool(chunk[c], C, g_gpu.data(), nd, g_fine, 0, &on) != HSA_STATUS_SUCCESS || on != chunk[c]) {
            std::printf("PROBE chunk %d lock failed\n", c); continue;
        }
        double res[2][2] = {};
        for (int path = 0; path < 2; ++path) {
            std::vector<double> agg;
            for (int r = 0; r < R + 1; ++r) {
                std::atomic<int> rdy{0}; std::vector<double> t0(nd), t1(nd); std::vector<std::thread> th;
                for (int d = 0; d < nd; ++d) th.emplace_back([&, d] {
                    CHK(hipSetDevice(d));
                    const char* src = chunk[c] + d * half;
                    rdy++; while (rdy < nd) {}
                    t0[d] = now_s();
                    if (path == 0) {
                        hsa_signal_store_relaxed(sig[d], 1);
                        hsa_amd_memory_async_copy_on_engine(dst[d], g_gpu[d], src, g_cpu, big, 0, nullptr, sig[d], HSA_AMD_SDMA_ENGINE_0, true);
                        while (hsa_signal_wait_scacquire(sig[d], HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {}
                    } else {
                        hipLaunchKernelGGL(stream_read, dim3(1024), dim3(256), 0, st[d], (const v4u*)src, big / 16, out[d]);
                        CHK(hipStreamSynchronize(st[d]));
                    }
                    t1[d] = now_s();
                });
                for (auto& x : th) x.join();
                if (r == 0) continue;
                agg.push_back(big * nd / (*std::max_element(t1.begin(), t1.end()) - *std::min_element(t0.begin(), t0.end())) / 1e9);
            }
            std::sort(agg.begin(), agg.end());
            res[path][0] = agg[agg.size() / 2]; res[path][1] = agg.front();
        }
        hsa_amd_memory_unlock(chunk[c]);
        std::printf("PROBE pass %d chunk %3d thp %4zu MiB  cpu-read T=%d %6.1f GB/s (min %6.1f)  | 2-card sdma-e0 agg %5.1f (min %5.1f)  2-card kern agg %5.1f (min %5.1f)\n",
                    pass, c, thp[c], T, cr[cr.size() / 2], cr.front(), res[0][0], res[0][1], res[1][0], res[1][1]);
        std::fflush(stdout);
    }
    /* pairs: card0 reads chunk a, card1 reads chunk b (b = -1: card0 alone). Shows whether two chunks
     * draw on separate memory channels (the aggregate adds) or the same ones (it does not). */
    if (const char* pe = std::getenv("PROBE_PAIRS")) {
        std::string spec = pe; size_t pos = 0;
        while (pos < spec.size()) {
            size_t e = spec.find(';', pos); if (e == std::string::npos) e = spec.size();
            int a = -1, b = -1; std::sscanf(spec.substr(pos, e - pos).c_str(), "%d:%d", &a, &b); pos = e + 1;
            if (a < 0 || a >= G || b >= G) continue;
            const int use = b < 0 ? 1 : nd;
            std::vector<int> ch = { a }; if (b >= 0 && b != a) ch.push_back(b);
            for (int x : ch) { void* on = nullptr; hsa_amd_memory_lock_to_pool(chunk[x], C, g_gpu.data(), nd, g_fine, 0, &on); }
            double res[2][3] = {};
            for (int path = 0; path < 2; ++path) {
                std::vector<double> agg, c0, c1;
                for (int r = 0; r < 6; ++r) {
                    std::atomic<int> rdy{0}; std::vector<double> t0(use), t1(use); std::vector<std::thread> th;
                    for (int d = 0; d < use; ++d) th.emplace_back([&, d] {
                        CHK(hipSetDevice(d));
                        const char* src = (d == 0 ? chunk[a] : chunk[b]) + (a == b ? d * half : 0);
                        rdy++; while (rdy < use) {}
                        t0[d] = now_s();
                        if (path == 0) {
                            hsa_signal_store_relaxed(sig[d], 1);
                            hsa_amd_memory_async_copy_on_engine(dst[d], g_gpu[d], src, g_cpu, big, 0, nullptr, sig[d], HSA_AMD_SDMA_ENGINE_0, true);
                            while (hsa_signal_wait_scacquire(sig[d], HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {}
                        } else {
                            hipLaunchKernelGGL(stream_read, dim3(1024), dim3(256), 0, st[d], (const v4u*)src, big / 16, out[d]);
                            CHK(hipStreamSynchronize(st[d]));
                        }
                        t1[d] = now_s();
                    });
                    for (auto& x : th) x.join();
                    if (r == 0) continue;
                    agg.push_back(big * use / (*std::max_element(t1.begin(), t1.end()) - *std::min_element(t0.begin(), t0.end())) / 1e9);
                    c0.push_back(big / (t1[0] - t0[0]) / 1e9);
                    c1.push_back(use > 1 ? big / (t1[1] - t0[1]) / 1e9 : 0);
                }
                std::sort(agg.begin(), agg.end()); std::sort(c0.begin(), c0.end()); std::sort(c1.begin(), c1.end());
                res[path][0] = agg[agg.size() / 2]; res[path][1] = c0[c0.size() / 2]; res[path][2] = c1[c1.size() / 2];
            }
            for (int x : ch) hsa_amd_memory_unlock(chunk[x]);
            std::printf("PAIR card0<-chunk %2d card1<-%s%2d | sdma-e0 agg %5.1f (card0 %5.1f card1 %5.1f) | kern agg %5.1f (card0 %5.1f card1 %5.1f)\n",
                        a, b < 0 ? "none " : "chunk ", b < 0 ? 0 : b, res[0][0], res[0][1], res[0][2], res[1][0], res[1][1], res[1][2]);
            std::fflush(stdout);
        }
    }
    for (int c = 0; c < G; ++c) munmap(chunk[c], C);
    return 0;
}
