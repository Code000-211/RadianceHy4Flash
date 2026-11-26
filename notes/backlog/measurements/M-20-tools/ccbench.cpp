// ccbench: an in-place (zero-copy) gate_up-sized read by a kernel, concurrent with an SDMA copy of
// down-sized pieces, on one card or on both cards at once (TP2). Host buffers: 2 MiB aligned,
// MADV_HUGEPAGE, prefaulted, hipHostRegister (THP-backed like the engine pool).
// ccbench REPS
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <chrono>
#define HIPCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)
typedef unsigned int v4u __attribute__((ext_vector_type(4)));
static const size_t G = 891136, D = 608768, HB = (size_t)512 << 20, DST = (size_t)64 << 20;

// one "expert" = 96 blocks x 128 threads reading G bytes (the decode moe_iqr grid for one expert's gate_up at TP2)
__global__ void ipread(const char* base, const size_t* offs, size_t bytes, uint32_t* out) {
    const int e = blockIdx.y;
    const v4u* src = (const v4u*)(base + offs[e]);
    const size_t n16 = bytes / 16, stride = (size_t)gridDim.x * blockDim.x;
    v4u acc = {0, 0, 0, 0};
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) acc ^= __builtin_nontemporal_load(src + i);
    uint32_t r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9e3779b9u) out[blockIdx.x] = r;
}

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Bar { std::atomic<int> c{0}, g{0}; int n; void wait() { int gg = g.load(); if (c.fetch_add(1) == n - 1) { c = 0; g.fetch_add(1); } else while (g.load() == gg) {} } };

struct Card {
    int dev; char* hb; char* dst; uint32_t* out; size_t* doffs; hipStream_t s0, s1; hipEvent_t a0, a1, b0, b1;
    unsigned rng = 12345;
    void open(int d) {
        dev = d; HIPCHK(hipSetDevice(d));
        void* p = mmap(nullptr, HB + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        hb = (char*)(((uintptr_t)p + (2 << 20) - 1) & ~(uintptr_t)((2 << 20) - 1));
        madvise(hb, HB, MADV_HUGEPAGE); std::memset(hb, 1 + d, HB);
        HIPCHK(hipHostRegister(hb, HB, hipHostRegisterDefault));
        HIPCHK(hipMalloc((void**)&dst, DST)); HIPCHK(hipMalloc((void**)&out, 1 << 16)); HIPCHK(hipMalloc((void**)&doffs, 64 * 8));
        HIPCHK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking)); HIPCHK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
        for (auto* e : {&a0, &a1, &b0, &b1}) HIPCHK(hipEventCreate(e));
    }
    size_t roff(size_t len) { rng = rng * 1103515245u + 12345u; return ((size_t)(rng >> 4) % ((HB - len) >> 16)) << 16; }
    // mode 1 kernel only, 2 copy only, 3 both (copy enqueued first, kernel right after); ng experts, nd downs
    void run(int mode, int ng, int nd, double& tk, double& tc, double& tm) {
        HIPCHK(hipSetDevice(dev));
        size_t ho[64]; for (int i = 0; i < ng; ++i) ho[i] = roff(G);
        HIPCHK(hipMemcpy(doffs, ho, ng * 8, hipMemcpyHostToDevice));
        size_t so[64]; for (int i = 0; i < nd; ++i) so[i] = roff(D);
        const double t0 = now_us();
        if (mode & 2) { HIPCHK(hipEventRecord(b0, s1)); for (int i = 0; i < nd; ++i) HIPCHK(hipMemcpyAsync(dst + (size_t)i * (1 << 20), hb + so[i], D, hipMemcpyHostToDevice, s1)); HIPCHK(hipEventRecord(b1, s1)); }
        if (mode & 1) { HIPCHK(hipEventRecord(a0, s0)); hipLaunchKernelGGL(ipread, dim3(96, ng), dim3(128), 0, s0, (const char*)hb, (const size_t*)doffs, G, out); HIPCHK(hipEventRecord(a1, s0)); }
        HIPCHK(hipStreamSynchronize(s0)); HIPCHK(hipStreamSynchronize(s1));
        tm = now_us() - t0; float f;
        tk = tc = 0;
        if (mode & 1) { HIPCHK(hipEventElapsedTime(&f, a0, a1)); tk = f * 1e3; }
        if (mode & 2) { HIPCHK(hipEventElapsedTime(&f, b0, b1)); tc = f * 1e3; }
    }
};

static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 100, ndev = 0; HIPCHK(hipGetDeviceCount(&ndev));
    std::vector<Card> cards(std::min(ndev, 2)); for (int i = 0; i < (int)cards.size(); ++i) cards[i].open(i);
    struct Cfg { int ng, nd; } cfgs[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {3, 1}, {8, 8}};
    for (int both = 0; both <= 1; ++both) {
        int nc = both ? (int)cards.size() : 1;
        for (auto cf : cfgs) for (int mode = 1; mode <= 3; ++mode) {
            std::vector<std::vector<double>> K(nc), C(nc), M(nc);
            Bar bar; bar.n = nc;
            auto body = [&](int c) {
                for (int r = 0; r < reps + 5; ++r) {
                    double tk, tc, tm; bar.wait(); cards[c].run(mode, cf.ng, cf.nd, tk, tc, tm);
                    if (r >= 5) { K[c].push_back(tk); C[c].push_back(tc); M[c].push_back(tm); }
                }
            };
            std::vector<std::thread> th; for (int c = 0; c < nc; ++c) th.emplace_back(body, c); for (auto& t : th) t.join();
            std::printf("CC %s ng=%d nd=%d mode=%s |", both ? "two-cards" : "one-card ", cf.ng, cf.nd, mode == 1 ? "kern" : mode == 2 ? "copy" : "both");
            for (int c = 0; c < nc; ++c) std::printf(" card%d kern %7.1f copy %7.1f wall %7.1f us |", c, med(K[c]), med(C[c]), med(M[c]));
            std::printf("\n"); std::fflush(stdout);
        }
    }
}
