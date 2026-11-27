/* w2.cpp -- "Weg 2", the runtime's half: Ctx::w2_*. See w2.h and abi/rad_w2.h. */
#include "ctx.h"
#include "w2_state.h"
#include "rad_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cpuid.h>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <x86intrin.h>
#include <pthread.h>

namespace rad {

W2Config w2_config(const char* (*get)(const char*)) {
    W2Config c;
    auto num = [&](const char* k, double d) {
        const char* v = get(k);
        if (!v || !*v) return d;
        char* end = nullptr;
        const double x = std::strtod(v, &end);
        return end && end != v && std::isfinite(x) ? x : d;
    };
    const char* on = get("RADIANCE_W2");
    c.on = !(on && !std::strcmp(on, "0"));                   /* on unless RADIANCE_W2=0 */
    c.min_rows = (int)std::clamp(num("RADIANCE_W2_MIN_ROWS", c.min_rows), 1.0, 65536.0);
    c.kmax = (int)std::clamp(num("RADIANCE_W2_KMAX", c.kmax), 0.0, (double)RAD_W2_MAX_COPY);
    c.win_us0 = std::clamp(num("RADIANCE_W2_WIN_US0", c.win_us0), 0.0, 1e6);
    c.win_us_expert = std::clamp(num("RADIANCE_W2_WIN_US_EXPERT", c.win_us_expert), 0.0, 1e6);
    const char* ring = get("RADIANCE_W2_RING");
    c.ring_cached = !(ring && !std::strcmp(ring, "uncached"));
    return c;
}
W2Config w2_config() {
    return w2_config([](const char* k) -> const char* { return std::getenv(k); });
}

void w2_fill_kmax(const W2Config& cfg, uint8_t* out) {
    for (int d = 0; d <= RAD_W2_MAX_EXPERT; ++d)
        out[d] = (uint8_t)rad_w2_kmax(cfg.win_us0 + cfg.win_us_expert * (double)d, cfg.kmax);
}

int64_t w2_device_bytes(int64_t unit, int64_t layers) {
    constexpr int64_t kGrain = (int64_t)2 << 20;
    auto grains = [](int64_t b) { return (b + kGrain - 1) / kGrain * kGrain; };
    return grains((int64_t)RAD_W2_GROUPS * RAD_W2_MAX_COPY * unit + 4096) +
           grains(layers * (int64_t)sizeof(RadW2Layer)) + grains(4) + grains((int64_t)sizeof(RadW2Stats));
}

namespace {

/* A pool class wider than this holds whole tables, not experts (xprefetch.cpp's bound). */
constexpr uint64_t kW2MaxUnit = (uint64_t)64 << 20;
/* Lists the queue may be behind before a post is dropped: its copies would land too late. */
constexpr int64_t kW2Behind = 2;
/* How long the serving thread keeps polling after the last pass that carried the form. */
constexpr int64_t kW2ActiveNs = 50'000'000;
constexpr uint32_t kW2NapCycles = 100000;      /* ~25 us of MWAITX */
/* The done words behind the ring: one a group, a cache line apart (in uint32_t). */
constexpr int64_t kW2DoneStride = 16;
constexpr int64_t kW2DoneBytes  = 4096;

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
bool have_mwaitx() {
    static const bool have = [] {
        unsigned a, b, c, d;
        return __get_cpuid(0x80000001u, &a, &b, &c, &d) && ((c >> 29) & 1u);
    }();
    return have;
}
__attribute__((target("mwaitx"))) void arm(const volatile void* line) {
    _mm_monitorx(const_cast<void*>(line), 0, 0);
}
__attribute__((target("mwaitx"))) void nap() { _mm_mwaitx(2 /* timer on */, 0, kW2NapCycles); }

/* Waits until done() holds, `line` being where the awaited store lands; false once `until` (ns)
 * passed or *stop was set. The clock is read every 64 naps or 1024 pauses. */
template <class Done>
bool wait_for(const volatile void* line, Done done, int64_t until, const std::atomic<bool>& stop) {
    if (done()) return true;
    const bool mw = have_mwaitx();
    const uint32_t mask = mw ? 63u : 1023u;
    for (uint32_t k = 1;; ++k) {
        if (mw) {
            arm(line);
            if (done()) return true;
            nap();
        } else {
            _mm_pause();
        }
        if (done()) return true;
        if ((k & mask) == 0 && (stop.load(std::memory_order_relaxed) || now_ns() > until)) return false;
    }
}

}  // namespace

void Ctx::w2_free(W2State& x) {
    if (x.ring) {
        if (x.ring_cached) rad_dev_free(x.ring, RAD_MEM_DEVICE);
        else rad_dev_free_uncached(x.ring);
    }
    if (x.done_base) rad_dev_free_uncached(x.done_base);
    for (void* p : {(void*)x.layers, (void*)x.head, (void*)x.stats})
        if (p) rad_dev_free(p, RAD_MEM_DEVICE);
    for (void* p : {(void*)x.box, (void*)x.stats_h})
        if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
}

/* At bringup, after placement has made its pinned pools: allocates everything, or nothing and logs
 * why W2 is off. */
void Ctx::w2_open(const W2Config& cfg, const uint64_t* lo, const uint64_t* hi, const uint64_t* stride, int n) {
    if (w2_ || !cfg.on || device_is_host() || route_.empty()) return;
    if (!rad_xp_supported(device_)) {
        RAD_INFO("rank %d: W2 off: the device backend has no DMA engine to spare for it", rank_);
        return;
    }
    auto x = std::make_unique<W2State>();
    x->cfg = cfg;
    x->device = device_;
    for (int i = 0; i < n && x->pools.n < RAD_W2_MAX_POOLS; ++i) {
        if (stride[i] == 0 || stride[i] > kW2MaxUnit || hi[i] <= lo[i]) continue;
        x->pools.lo[x->pools.n] = lo[i];
        x->pools.hi[x->pools.n] = hi[i];
        x->pools.stride[x->pools.n] = stride[i];
        x->slot = std::max(x->slot, stride[i]);
        ++x->pools.n;
    }
    if (x->pools.n == 0) {
        RAD_INFO("rank %d: W2 off: no expert sits in a pinned pool", rank_);
        return;
    }
    const int64_t nl = (int64_t)route_.size();
    const int64_t ring = (int64_t)RAD_W2_GROUPS * RAD_W2_MAX_COPY * (int64_t)x->slot;
    if (rad_dev_set(device_) < 0) return;
    /* The DMA engine writes the ring and kernels read it, so no cache may hold a slot's previous
     * contents: either the ring is ordinary device memory and each layer's first tail launch
     * acquires at system scope (w2_tail; the default, W2Config::ring_cached), or it is uncached
     * device memory, as the prefetch's. The groups' done words are uncached either way, a cache
     * line each: the list's fences write one, the tail's blocks read it from the card and not
     * across the link. */
    x->ring_cached = cfg.ring_cached;
    if (x->ring_cached) {
        x->ring = (char*)rad_dev_alloc(ring, RAD_MEM_DEVICE);
        x->done_base = (uint32_t*)rad_dev_alloc_uncached(kW2DoneBytes);
        x->done = x->done_base;
    } else {
        x->ring = (char*)rad_dev_alloc_uncached(ring + kW2DoneBytes);
        x->done = x->ring ? (uint32_t*)(x->ring + ring) : nullptr;
    }
    x->layers  = (RadW2Layer*)rad_dev_alloc(nl * (int64_t)sizeof(RadW2Layer), RAD_MEM_DEVICE);
    x->head    = (uint32_t*)rad_dev_alloc(4, RAD_MEM_DEVICE);
    x->stats   = (RadW2Stats*)rad_dev_alloc((int64_t)sizeof(RadW2Stats), RAD_MEM_DEVICE);
    x->box     = (RadW2Box*)rad_dev_alloc((int64_t)sizeof(RadW2Box) * RAD_W2_BOXES, RAD_MEM_HOST_PINNED);
    x->stats_h = (RadW2Stats*)rad_dev_alloc((int64_t)sizeof(RadW2Stats), RAD_MEM_HOST_PINNED);
    const bool ok = x->ring && x->done && x->layers && x->head && x->stats && x->box && x->stats_h &&
                    rad_memset_async(x->done, 0, kW2DoneBytes, stream_) >= 0 &&
                    rad_memset_async(x->layers, 0, nl * (int64_t)sizeof(RadW2Layer), stream_) >= 0 &&
                    rad_memset_async(x->head, 0, 4, stream_) >= 0 &&
                    rad_memset_async(x->stats, 0, sizeof(RadW2Stats), stream_) >= 0 &&
                    rad_stream_sync(stream_) >= 0;
    if (!ok) {
        const char* why = rad_dev_last_error();
        RAD_WARN("rank %d: W2 off: %s", rank_, why && *why ? why : "no memory for it");
        w2_free(*x);
        return;
    }
    std::memset((void*)x->box, 0, sizeof(RadW2Box) * RAD_W2_BOXES);
    std::memset(x->stats_h, 0, sizeof(RadW2Stats));
    x->n_layers = (int)nl;
    x->bound.assign((size_t)nl, 0);
    W2State* xs = x.get();
    x->thr = std::thread([this, xs] { w2_serve(*xs); });
    uint8_t km[RAD_W2_MAX_EXPERT + 1];
    w2_fill_kmax(cfg, km);
    RAD_INFO("rank %d: W2 on: resident experts first on passes of %d rows and up, at most %d whole units a "
             "routed layer by DMA (budget at 50 / 100 / 200 routed experts: %d / %d / %d), ring %s%s", rank_,
             cfg.min_rows, cfg.kmax, (int)km[50], (int)km[100], (int)km[200], humanb(ring).c_str(),
             x->ring_cached ? " of ordinary device memory, read behind a system-scope acquire" : " (uncached)");
    w2_ = std::move(x);
}

void Ctx::w2_close() {
    if (!w2_) return;
    W2State& x = *w2_;
    /* Nothing may touch the ring when it is freed: issued tails read it and queued lists write it. */
    if (stream_) rad_stream_sync(stream_);
    if (lane1_) rad_stream_sync(lane1_);
    {
        std::lock_guard<std::mutex> lk(x.mu);
        x.stop = true;
    }
    x.cv.notify_all();
    if (x.thr.joinable()) x.thr.join();
    const auto t0 = std::chrono::steady_clock::now();
    while (rad_xp_behind(x.device) > 0) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
            RAD_WARN("rank %d: the W2 DMA queue still has %lld lists after 5 s; its ring is freed anyway", rank_,
                     (long long)rad_xp_behind(x.device));
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    RadW2Stats s{};
    if (w2_stats(&s) && s.plans) {
        std::string by;
        for (int k = 1; k <= RAD_W2_MAX_COPY; ++k)
            if (s.posted_n[k])
                by += fmt(" %d:%u/%u", k, s.landed_n[k], s.posted_n[k] * (uint32_t)k);
        RAD_INFO("rank %d: W2: units landed / posted by list length:%s; %u misses past the tail's %d left to the "
                 "first pass", rank_, by.c_str(), s.spilled, RAD_W2_TAIL_EXPERTS);
        RAD_INFO("rank %d: W2: %u routed layers planned, %u misses, %u units posted, %u read from the ring and %u "
                 "in place by the tail (%.1f%% landed); %lld lists, %lld posts dropped", rank_, s.plans, s.misses,
                 s.posted, s.landed, s.late, s.landed + s.late ? 100.0 * s.landed / (s.landed + s.late) : 0.0,
                 (long long)x.lists.load(), (long long)x.dropped.load());
    }
    w2_free(x);
    w2_.reset();
}

/* The device counters, read back on the compute stream (a host wait: for the end and for tests). */
bool Ctx::w2_stats(RadW2Stats* out) {
    if (!w2_ || !out) return false;
    W2State& x = *w2_;
    if (rad_memcpy_async(x.stats_h, x.stats, sizeof(RadW2Stats), stream_) < 0 || rad_stream_sync(stream_) < 0)
        return false;
    *out = *x.stats_h;
    return true;
}

uint64_t Ctx::w2_epoch() const { return w2_ ? w2_->epoch : 0; }

/* In front of a layer's first tail launch. With the ring in ordinary device memory that launch
 * acquires at system scope: the slots' lines the caches still hold are from the tail that read
 * this group three routed layers ago, and the DMA engine has rewritten the slots since. Nothing
 * reads a slot between this acquire and the tail's own reads behind its done word. */
void Ctx::w2_tail() {
    if (w2_ && w2_->ring_cached) rad_dev_acquire_next();
}

/* A pass that carries the form is about to run (issued or played): the serving thread polls. */
void Ctx::w2_kick() {
    if (!w2_) return;
    W2State& x = *w2_;
    x.active_until.store(now_ns() + kW2ActiveNs, std::memory_order_relaxed);
    if (x.asleep.load(std::memory_order_relaxed)) {
        { std::lock_guard<std::mutex> lk(x.mu); }
        x.cv.notify_one();
    }
}

/* Host thread: hands each post to the DMA queue once the plan has written it. Posts are numbered
 * from 1 in stream order and post s sits in box s % RAD_W2_BOXES. */
void Ctx::w2_serve(W2State& x) {
    pthread_setname_np(pthread_self(), "rad-w2");
    uint32_t next = 1;
    for (;;) {
        if (now_ns() > x.active_until.load(std::memory_order_relaxed)) {
            std::unique_lock<std::mutex> lk(x.mu);
            x.asleep = true;
            x.cv.wait(lk, [&] { return x.stop.load() || now_ns() <= x.active_until.load(); });
            x.asleep = false;
        }
        if (x.stop) return;
        volatile RadW2Box* b = x.box + next % RAD_W2_BOXES;
        const int64_t until = x.active_until.load(std::memory_order_relaxed);
        if (!wait_for(&b->seq, [&] { return (int32_t)(b->seq - next) >= 0; }, until, x.stop)) continue;
        std::atomic_thread_fence(std::memory_order_acquire);
        if (b->seq != next) {
            /* The plans ran a ring ahead of this thread: take up at the newest post. */
            uint32_t newest = next;
            for (int i = 0; i < RAD_W2_BOXES; ++i) {
                const uint32_t s = x.box[i].seq;
                if ((int32_t)(s - newest) > 0) newest = s;
            }
            x.dropped.fetch_add((int64_t)(newest - next), std::memory_order_relaxed);
            next = newest;
            continue;
        }
        const uint32_t seq = next++;
        uint64_t src[RAD_W2_MAX_COPY], dst[RAD_W2_MAX_COPY], bytes[RAD_W2_MAX_COPY];
        /* The post's stores cross the link in no promised order: it is whole once the sum over
         * what is read is the sum the plan wrote. A post the plan left a job out of (the unit moved
         * under it) never is, and is dropped. */
        RadW2Box snap;
        auto whole = [&] {
            std::atomic_thread_fence(std::memory_order_acquire);
            std::memcpy(&snap, (const void*)b, sizeof snap);
            return snap.seq == seq && snap.n >= 1 && snap.n <= RAD_W2_MAX_COPY &&
                   rad_w2_box_sum(seq, snap.n, snap.group, snap.jobs) == snap.sum;
        };
        if (!wait_for(&b->sum, whole, now_ns() + 200'000, x.stop)) {
            x.dropped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const int n = (int)snap.n;
        const uint32_t g = snap.group;
        bool got = g < RAD_W2_GROUPS;
        for (int i = 0; i < n && got; ++i) {
            src[i] = snap.jobs[i].src;
            dst[i] = snap.jobs[i].dst;
            bytes[i] = snap.jobs[i].bytes;
            const uint64_t r0 = (uint64_t)(uintptr_t)x.ring, r1 = r0 + (uint64_t)RAD_W2_GROUPS * RAD_W2_MAX_COPY * x.slot;
            if (!bytes[i] || bytes[i] > x.slot || dst[i] < r0 || dst[i] + bytes[i] > r1) got = false;
        }
        if (!got || rad_xp_behind(x.device) > kW2Behind) {
            x.dropped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        /* The done word counts the units as they land (RAD_W2_DONE), so a tail that comes before
         * the list's end still reads from the ring what is there. */
        if (rad_xp_list(x.device, nullptr, n, src, dst, bytes, x.done + g * kW2DoneStride, RAD_W2_DONE(seq, 0),
                        RAD_XP_DONE_EACH) == RAD_OK) {
            x.lists.fetch_add(1, std::memory_order_relaxed);
            x.units.fetch_add(n, std::memory_order_relaxed);
        } else {
            x.dropped.fetch_add(1, std::memory_order_relaxed);
            RAD_WARN("rank %d: the W2 DMA queue refused a list", rank_);
        }
    }
}

/* The layer's control block for a pass of `rows` rows, or null: W2 off, the pass too narrow, or the
 * layer's tables not bound yet. `first[x]` is place x's first weight (its gate_up codes), whose
 * table entry says where the unit is. Binds the layer at the first call that can. */
void* Ctx::w2_layer(int layer, const rad_weight* first, int64_t n_place, int64_t rows) {
    if (!w2_) return nullptr;
    W2State& x = *w2_;
    if (rows < x.cfg.min_rows || layer < 0 || layer >= x.n_layers || !first || n_place <= 0 ||
        n_place > RAD_W2_MAX_EXPERT)
        return nullptr;
    if (!x.bound[(size_t)layer]) {
        auto L = std::make_unique<RadW2Layer>();
        std::memset(L.get(), 0, sizeof(RadW2Layer));
        for (int64_t p = 0; p < n_place; ++p) {
            const rad_weight w = first[p];
            if (!w || (size_t)w >= wmemb_head_.size() || wmemb_head_[w] < 0) continue;
            const WMemb& m = wmemb_[(size_t)wmemb_head_[w]];
            if (m.next >= 0) continue;                      /* in several tables: left resident */
            const WeightTable& t = wtabs_[(size_t)m.slot];
            if (!t.dev) return nullptr;                     /* not bound yet: a later pass */
            L->ent[p] = (const uint64_t*)((void**)t.dev + m.entry);
        }
        const int g = layer % RAD_W2_GROUPS;
        L->magic = RAD_W2_MAGIC;
        L->ring = (uint64_t)(uintptr_t)(x.ring + (size_t)g * RAD_W2_MAX_COPY * x.slot);
        L->slot = x.slot;
        L->done = x.done + g * kW2DoneStride;
        L->box = (RadW2Box*)rad_dev_device_ptr(x.box);
        L->head = x.head;
        L->stats = x.stats;
        L->n_expert = (uint32_t)n_place;
        L->group = (uint32_t)g;
        L->pools = x.pools;
        w2_fill_kmax(x.cfg, L->kmax);
        /* Once per layer, at an issue: this host wait is off the step path, and off the tape. */
        if (rec_) tape_pause(true);
        const bool wrote = rad_memcpy_async(x.layers + layer, L.get(), sizeof(RadW2Layer), stream_) >= 0 &&
                           rad_stream_sync(stream_) >= 0;
        if (rec_) tape_pause(false);
        if (!wrote) return nullptr;
        x.bound[(size_t)layer] = 1;
        ++x.epoch;
    }
    if (rec_) rec_->w2 = true;
    w2_kick();
    return x.layers + layer;
}

}  /* namespace rad */
