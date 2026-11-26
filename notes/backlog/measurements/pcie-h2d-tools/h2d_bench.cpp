/* h2d_bench.cpp -- standalone host->device bandwidth benchmark for the radiance PCIe question
 * (notes/backlog/measurements/pcie-h2d.md). Not engine code; builds with plain hipcc:
 *
 *   hipcc -O3 -std=c++20 --offload-arch=gfx1201 h2d_bench.cpp -o h2d_bench -lhsa-runtime64 -pthread
 *
 * Every point: one card alone and both cards at once (one thread per card, a spin barrier before
 * each repetition so both start together), R repetitions (default 7), median / min / max GB/s
 * (GB = 1e9 bytes) of the whole list, wall clock from the first issue to completion seen by host.
 *
 * Host buffer kinds (one 512 MiB source pool per card, separate per card):
 *   hipmalloc    hipHostMalloc(Portable|Mapped)                 (coherent, as HIP allocates it)
 *   hipmalloc_nc hipHostMalloc(Portable|Mapped|NonCoherent)
 *   reg4k        mmap private + MADV_NOHUGEPAGE + hipHostRegister (4 KiB pages)
 *   regthp       mmap private 2 MiB aligned + MADV_HUGEPAGE + hipHostRegister (THP)
 *   hugetlb      mmap MAP_HUGETLB (2 MiB hugetlbfs) + hipHostRegister, skipped when the pool is empty
 *   engine       the engine's pin_pieces() style (core/device/aql.cpp): mmap private, 2 MiB aligned,
 *                MADV_HUGEPAGE, MADV_DONTFORK, prefault, hsa_amd_memory_lock_to_pool(host fine-grained
 *                pool, all cards) -- HIP never sees it, so only the hsa / kernel paths use it
 * Copy paths:
 *   hip1   hipMemcpyAsync on one stream          hip2  the same list alternated over two streams
 *   e0/e1  hsa_amd_memory_async_copy_on_engine on SDMA engine 0 / 1 (ROCr's own SDMA queues)
 *   e01    the list alternated over SDMA engine 0 and 1
 *   kern   zero-copy: a streaming-read kernel reads the pinned host range itself (one launch a copy)
 * Cases:   big = 1 x 256 MiB, l058 = 400 x 608256 B, l143 = 160 x 1499546 B (scattered sources)
 *
 * Args: --hosts a,b --paths a,b --cases a,b --sets 0;1;01 --reps N
 * Two-process mode: --shm NAME --rank I --nproc 2 (each process sees one card through
 * ROCR_VISIBLE_DEVICES; a barrier in /dev/shm/NAME syncs every repetition; rank 0 prints the
 * aggregate from CLOCK_MONOTONIC stamps both ranks publish). */
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define HIPCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)
#define HSACHK(x) do { hsa_status_t s_ = (x); if (s_ != HSA_STATUS_SUCCESS) { const char* m_ = nullptr; \
    hsa_status_string(s_, &m_); std::fprintf(stderr, "HSA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, m_ ? m_ : "?"); std::exit(1); } } while (0)

static constexpr size_t kPool = (size_t)512 << 20;
static constexpr size_t kDst = (size_t)272 << 20;
static constexpr size_t k2M = (size_t)2 << 20;

static double now_s() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------- HSA agents and pools */
struct Hsa {
    hsa_agent_t cpu{};
    std::vector<hsa_agent_t> gpu;      /* in HIP device order (matched by BDF) */
    hsa_amd_memory_pool_t host_fine{}, host_coarse{};
    bool have_coarse = false;
} H;

static hsa_status_t agent_cb(hsa_agent_t a, void* v) {
    hsa_device_type_t t;
    hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
    auto* all = (std::vector<hsa_agent_t>*)v;
    if (t == HSA_DEVICE_TYPE_CPU && H.cpu.handle == 0) H.cpu = a;
    if (t == HSA_DEVICE_TYPE_GPU) all->push_back(a);
    return HSA_STATUS_SUCCESS;
}
static hsa_status_t pool_cb(hsa_amd_memory_pool_t p, void*) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t fl = 0;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fl);
    if ((fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) && !(fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) &&
        H.host_fine.handle == 0)
        H.host_fine = p;
    if ((fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) && !H.have_coarse) { H.host_coarse = p; H.have_coarse = true; }
    return HSA_STATUS_SUCCESS;
}
static void hsa_setup(int ndev) {
    HSACHK(hsa_init());   /* HIP already initialised ROCr; this only takes a reference */
    std::vector<hsa_agent_t> all;
    HSACHK(hsa_iterate_agents(agent_cb, &all));
    HSACHK(hsa_amd_agent_iterate_memory_pools(H.cpu, pool_cb, nullptr));
    for (int d = 0; d < ndev; ++d) {
        int bus = 0, devn = 0, dom = 0;
        HIPCHK(hipDeviceGetAttribute(&bus, hipDeviceAttributePciBusId, d));
        HIPCHK(hipDeviceGetAttribute(&devn, hipDeviceAttributePciDeviceId, d));
        HIPCHK(hipDeviceGetAttribute(&dom, hipDeviceAttributePciDomainId, d));
        bool found = false;
        for (hsa_agent_t a : all) {
            uint32_t bdf = 0;
            hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf);
            if ((int)(bdf >> 8) == bus && (int)((bdf >> 3) & 31) == devn) { H.gpu.push_back(a); found = true; break; }
        }
        if (!found) { std::fprintf(stderr, "no HSA agent for HIP device %d\n", d); std::exit(1); }
        uint32_t mask = 0;
        hsa_amd_memory_copy_engine_status(H.gpu[d], H.cpu, &mask);
        std::printf("INFO dev %d pci %02x:%02x  SDMA engines free host->dev mask 0x%x\n", d, bus, devn, mask);
    }
}

/* ---------------------------------------------------------------- host buffers */
static size_t thp_bytes(void* p, size_t len) {
    /* AnonHugePages of the smaps entries that overlap [p, p+len) */
    FILE* f = std::fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    char line[512];
    uintptr_t lo = (uintptr_t)p, hi = lo + len;
    bool in = false;
    size_t total = 0;
    while (std::fgets(line, sizeof line, f)) {
        uintptr_t a, b;
        if (std::sscanf(line, "%lx-%lx ", &a, &b) == 2 && std::strchr(line, '-') && line[0] != ' ') {
            in = (a < hi && b > lo);
            continue;
        }
        size_t kb;
        if (in && std::sscanf(line, "AnonHugePages: %zu kB", &kb) == 1) total += kb << 10;
        if (in && std::sscanf(line, "ShmemPmdMapped: %zu kB", &kb) == 1) total += kb << 10;
        if (in && std::sscanf(line, "Private_Hugetlb: %zu kB", &kb) == 1) total += kb << 10;
        if (in && std::sscanf(line, "Shared_Hugetlb: %zu kB", &kb) == 1) total += kb << 10;
    }
    std::fclose(f);
    return total;
}

static void prefault(char* p, size_t len) {
    const int T = 4;
    std::vector<std::thread> th;
    const size_t per = (len + T - 1) / T;
    for (int t = 0; t < T; ++t)
        th.emplace_back([=] { for (size_t o = t * per; o < std::min(len, (t + 1) * per); o += 4096) p[o] = (char)(o >> 12); });
    for (auto& x : th) x.join();
}

static char* mmap_aligned(size_t len, int extra_flags) {
    void* m = mmap(nullptr, len + k2M, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | extra_flags, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    uintptr_t lo = (uintptr_t)m, at = (lo + k2M - 1) & ~(uintptr_t)(k2M - 1);
    if (at > lo) munmap(m, at - lo);
    if (lo + k2M > at) munmap((void*)(at + len), lo + k2M - at);
    return (char*)at;
}

struct HostBuf { char* p = nullptr; size_t len = 0; std::string kind; bool hip_known = false; };

static bool host_alloc(const std::string& kind, HostBuf& hb, int ndev) {
    hb.kind = kind; hb.len = kPool;
    if (kind == "hipmalloc" || kind == "hipmalloc_nc") {
        unsigned fl = hipHostMallocPortable | hipHostMallocMapped | (kind == "hipmalloc_nc" ? hipHostMallocNonCoherent : 0);
        HIPCHK(hipHostMalloc((void**)&hb.p, kPool, fl));
        prefault(hb.p, kPool);
        hb.hip_known = true;
    } else if (kind == "reg4k" || kind == "regthp") {
        hb.p = mmap_aligned(kPool, 0);
        if (!hb.p) return false;
        madvise(hb.p, kPool, kind == "reg4k" ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
        prefault(hb.p, kPool);
        HIPCHK(hipHostRegister(hb.p, kPool, hipHostRegisterPortable | hipHostRegisterMapped));
        hb.hip_known = true;
    } else if (kind == "hugetlb") {
        void* m = mmap(nullptr, kPool, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (m == MAP_FAILED) { std::printf("INFO host=hugetlb unavailable: mmap(MAP_HUGETLB 512 MiB): %s\n", std::strerror(errno)); return false; }
        hb.p = (char*)m;
        prefault(hb.p, kPool);
        HIPCHK(hipHostRegister(hb.p, kPool, hipHostRegisterPortable | hipHostRegisterMapped));
        hb.hip_known = true;
    } else if (kind == "engine") {
        hb.p = mmap_aligned(kPool, 0);
        if (!hb.p) return false;
        madvise(hb.p, kPool, MADV_HUGEPAGE);
        madvise(hb.p, kPool, MADV_DONTFORK);
        prefault(hb.p, kPool);
        void* on = nullptr;
        HSACHK(hsa_amd_memory_lock_to_pool(hb.p, kPool, H.gpu.data(), ndev, H.host_fine, 0, &on));
        if (on != hb.p) { std::printf("INFO engine lock mapped at %p not in place %p\n", on, (void*)hb.p); return false; }
    } else {
        std::fprintf(stderr, "unknown host kind %s\n", kind.c_str()); std::exit(2);
    }
    std::printf("INFO host=%-12s base %p  huge-page backed %zu of %zu MiB\n", kind.c_str(), (void*)hb.p,
                thp_bytes(hb.p, kPool) >> 20, kPool >> 20);
    return true;
}
static void host_free(HostBuf& hb) {
    if (!hb.p) return;
    if (hb.kind.rfind("hipmalloc", 0) == 0) HIPCHK(hipHostFree(hb.p));
    else if (hb.kind == "engine") { hsa_amd_memory_unlock(hb.p); munmap(hb.p, hb.len); }
    else { HIPCHK(hipHostUnregister(hb.p)); munmap(hb.p, hb.len); }
    hb.p = nullptr;
}

/* ---------------------------------------------------------------- the zero-copy read kernel */
typedef unsigned int v4u __attribute__((ext_vector_type(4)));
__global__ void stream_read(const v4u* __restrict__ src, size_t n16, const uint8_t* tail, int ntail, uint32_t* out) {
    v4u acc = {0, 0, 0, 0};
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n16; i += 4 * stride) {
        v4u a = __builtin_nontemporal_load(src + i), b = __builtin_nontemporal_load(src + i + stride);
        v4u c = __builtin_nontemporal_load(src + i + 2 * stride), d = __builtin_nontemporal_load(src + i + 3 * stride);
        acc ^= a ^ b ^ c ^ d;
    }
    for (; i < n16; i += stride) acc ^= __builtin_nontemporal_load(src + i);
    uint32_t r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (blockIdx.x == 0 && threadIdx.x < (unsigned)ntail) r ^= tail[threadIdx.x];
    if (r == 0x9e3779b9u) out[blockIdx.x] = r;   /* keeps the loads alive */
}

/* ---------------------------------------------------------------- per-card context */
struct Case { std::string name; size_t bytes; int n; };
static const Case kCases[] = { { "big", (size_t)256 << 20, 1 }, { "l058", 608256, 400 }, { "l143", 1499546, 160 } };

struct Card {
    int dev = 0;
    char* dst = nullptr;
    uint32_t* out = nullptr;
    hipStream_t s0{}, s1{};
    hsa_signal_t sig{};
    HostBuf hb;
    int cus = 64;
    void open(int d) {
        dev = d;
        HIPCHK(hipSetDevice(d));
        HIPCHK(hipMalloc((void**)&dst, kDst));
        HIPCHK(hipMalloc((void**)&out, 1 << 20));
        HIPCHK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
        HIPCHK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
        HSACHK(hsa_signal_create(0, 0, nullptr, &sig));
        HIPCHK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, d));
    }
    /* runs one list; returns seconds */
    double run(const std::string& path, const Case& c) {
        HIPCHK(hipSetDevice(dev));
        const size_t sstride = (c.bytes + 65535) / 65536 * 65536;
        const size_t nsrc = std::max<size_t>(1, (hb.len - c.bytes) / sstride + 1);
        auto src_at = [&](int i) { return hb.p + (c.n == 1 ? 0 : ((size_t)(i * 7) % nsrc) * sstride); };
        auto dst_at = [&](int i) { return dst + (c.n == 1 ? 0 : (size_t)i * sstride % (kDst - sstride)); };
        const double t0 = now_s();
        if (path == "hip1" || path == "hip2") {
            for (int i = 0; i < c.n; ++i)
                HIPCHK(hipMemcpyAsync(dst_at(i), src_at(i), c.bytes, hipMemcpyHostToDevice, (path == "hip2" && (i & 1)) ? s1 : s0));
            HIPCHK(hipStreamSynchronize(s0));
            if (path == "hip2") HIPCHK(hipStreamSynchronize(s1));
        } else if (path == "e0" || path == "e1" || path == "e01") {
            hsa_signal_store_relaxed(sig, c.n);
            for (int i = 0; i < c.n; ++i) {
                hsa_amd_sdma_engine_id_t e = path == "e0" ? HSA_AMD_SDMA_ENGINE_0 : path == "e1" ? HSA_AMD_SDMA_ENGINE_1
                                           : ((i & 1) ? HSA_AMD_SDMA_ENGINE_1 : HSA_AMD_SDMA_ENGINE_0);
                HSACHK(hsa_amd_memory_async_copy_on_engine(dst_at(i), H.gpu[dev], src_at(i), H.cpu, c.bytes, 0, nullptr, sig, e, true));
            }
            while (hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0) {}
        } else if (path == "kern") {
            for (int i = 0; i < c.n; ++i) {
                const char* s = src_at(i);
                const size_t n16 = c.bytes / 16;
                const int ntail = (int)(c.bytes % 16);
                const int blocks = (int)std::min<size_t>((size_t)cus * 16, std::max<size_t>(1, (n16 + 1023) / 1024));
                hipLaunchKernelGGL(stream_read, dim3(blocks), dim3(256), 0, s0, (const v4u*)s, n16,
                                   (const uint8_t*)(s + n16 * 16), ntail, out);
            }
            HIPCHK(hipStreamSynchronize(s0));
        } else {
            std::fprintf(stderr, "unknown path %s\n", path.c_str()); std::exit(2);
        }
        return now_s() - t0;
    }
};

/* ---------------------------------------------------------------- barrier (threads or processes) */
struct Shared {
    std::atomic<int> count, gen;
    double t0[2][64], t1[2][64];
};
static void barrier(Shared* s, int nparty) {
    const int g = s->gen.load(std::memory_order_acquire);
    if (s->count.fetch_add(1, std::memory_order_acq_rel) == nparty - 1) {
        s->count.store(0, std::memory_order_relaxed);
        s->gen.store(g + 1, std::memory_order_release);
    } else {
        while (s->gen.load(std::memory_order_acquire) == g) {}
    }
}

struct Stat { double med, mn, mx; };
static Stat stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return { v[v.size() / 2], v.front(), v.back() };
}

static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> r; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, d)) if (!x.empty()) r.push_back(x);
    return r;
}

int main(int argc, char** argv) {
    std::string hosts = "hipmalloc,hipmalloc_nc,reg4k,regthp,hugetlb,engine", paths = "hip1,hip2,e0,e1,e01,kern",
                cases = "big,l058,l143", sets = "0;1;01", shm;
    int reps = 7, rank = 0, nproc = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--hosts") hosts = v; else if (k == "--paths") paths = v; else if (k == "--cases") cases = v;
        else if (k == "--sets") sets = v; else if (k == "--reps") reps = std::atoi(v.c_str());
        else if (k == "--shm") shm = v; else if (k == "--rank") rank = std::atoi(v.c_str());
        else if (k == "--nproc") nproc = std::atoi(v.c_str());
    }
    reps = std::min(reps, 64);
    int ndev = 0;
    HIPCHK(hipGetDeviceCount(&ndev));
    ndev = std::min(ndev, 2);
    for (int d = 0; d < ndev; ++d) { HIPCHK(hipSetDevice(d)); HIPCHK(hipFree(nullptr)); }
    hsa_setup(ndev);
    std::vector<Card> card(ndev);
    for (int d = 0; d < ndev; ++d) card[d].open(d);

    Shared* sh = nullptr;
    if (!shm.empty()) {
        int fd = shm_open(shm.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd < 0 || ftruncate(fd, sizeof(Shared)) != 0) { perror("shm"); return 1; }
        sh = (Shared*)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
    } else {
        sh = new Shared();
        sh->count = 0; sh->gen = 0;
    }

    for (const std::string& hk : split(hosts, ',')) {
        bool ok = true;
        for (int d = 0; d < ndev && ok; ++d) { HIPCHK(hipSetDevice(d)); ok = host_alloc(hk, card[d].hb, ndev); }
        if (!ok) { for (int d = 0; d < ndev; ++d) host_free(card[d].hb); continue; }
        for (const std::string& path : split(paths, ',')) {
            if ((path == "hip1" || path == "hip2") && !card[0].hb.hip_known) continue;
            for (const std::string& cn : split(cases, ',')) {
                const Case* c = nullptr;
                for (const Case& x : kCases) if (x.name == cn) c = &x;
                if (!c) continue;
                for (const std::string& set : split(sets, ';')) {
                    std::vector<int> devs;
                    for (char ch : set) if (ch - '0' < ndev) devs.push_back(ch - '0');
                    if (devs.empty() || (sh && !shm.empty() && devs.size() != 1)) continue;
                    const int np = shm.empty() ? (int)devs.size() : nproc;
                    std::vector<std::vector<double>> gbs(devs.size());
                    std::vector<double> agg, per_us;
                    auto worker = [&](int slot) {
                        Card& cd = card[devs[slot]];
                        const int who = shm.empty() ? slot : rank;
                        for (int w = 0; w < 2; ++w) { barrier(sh, np); cd.run(path, *c); }
                        for (int r = 0; r < reps; ++r) {
                            barrier(sh, np);
                            const double a = now_s();
                            const double t = cd.run(path, *c);
                            sh->t0[who][r] = a; sh->t1[who][r] = a + t;
                            gbs[slot].push_back((double)c->bytes * c->n / t / 1e9);
                        }
                        barrier(sh, np);
                    };
                    std::vector<std::thread> th;
                    for (size_t s = 1; s < devs.size(); ++s) th.emplace_back(worker, (int)s);
                    worker(0);
                    for (auto& x : th) x.join();
                    const double bytes = (double)c->bytes * c->n;
                    for (int r = 0; r < reps; ++r) {
                        double lo = 1e300, hi = 0;
                        for (int p = 0; p < np; ++p) { lo = std::min(lo, sh->t0[p][r]); hi = std::max(hi, sh->t1[p][r]); }
                        agg.push_back(bytes * np / (hi - lo) / 1e9);
                    }
                    std::string tag = shm.empty() ? ("devs=" + set) : ("proc" + std::to_string(rank) + "/" + std::to_string(nproc));
                    std::printf("H2D %-8s host=%-12s path=%-4s case=%-4s bytes=%-9zu n=%-3d", tag.c_str(), hk.c_str(),
                                path.c_str(), c->name.c_str(), c->bytes, c->n);
                    for (size_t s = 0; s < devs.size(); ++s) {
                        Stat st = stat(gbs[s]);
                        std::printf(" | card%d GB/s med %5.2f min %5.2f max %5.2f per-copy %7.1f us", devs[s], st.med, st.mn,
                                    st.mx, bytes / c->n / (st.med * 1e9) * 1e6);
                    }
                    if (np > 1 && rank == 0) { Stat a = stat(agg); std::printf(" | aggregate med %5.2f min %5.2f max %5.2f", a.med, a.mn, a.mx); }
                    std::printf("\n");
                    std::fflush(stdout);
                }
            }
        }
        for (int d = 0; d < ndev; ++d) host_free(card[d].hb);
    }
    if (!shm.empty() && rank == 0) shm_unlink(shm.c_str());
    return 0;
}
