// r4d_w2.h -- "Weg 2" device code (abi/rad_w2.h): the plan the router's sort ends with, and what a
// routed GEMM's block asks about its expert.
//
// Nothing here runs unless an op was handed a W2 operand; the kernels that call these are
// instantiations of their own (W2 true), so the launches of a build with the switch off are the
// instantiations they were.
#pragma once
#include <hip/hip_runtime.h>
#include "rad_w2.h"

namespace r4d_w2 {

#if defined(__HIP__)
/* A store that writes through to pinned host memory, and the load that reads it back there. */
template <class T>
__device__ __forceinline__ void sys_store(T* p, T v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
template <class T>
__device__ __forceinline__ T sys_load(const T* p) {
    return __hip_atomic_load((T*)p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
#else
/* The host SIMT emulator (tests/r4d_emu): one memory, no caches. */
template <class T> inline void sys_store(T* p, T v) { *p = v; }
template <class T> inline T sys_load(const T* p) { return *p; }
#endif

/* The pool a pointer sits in, or -1: an expert whose first weight is there is a miss. */
__device__ __forceinline__ int pool_of(const RadW2Pools& pl, uint64_t p) {
    int pi = -1;
    for (int i = 0; i < pl.n && i < RAD_W2_MAX_POOLS; ++i)
        if (p >= pl.lo[i] && p < pl.hi[i]) pi = i;
    return pi;
}

/* LDS the plan needs: a flag a place, and three counters. */
struct PlanLds {
    uint8_t miss[RAD_W2_MAX_EXPERT];
    int32_t n_distinct, n_miss;
    uint32_t seq, k;
    uint64_t term[RAD_W2_MAX_COPY];      /* each job's part of the post's sum */
};

/* The plan, at the end of the router's one workgroup of `nthr` threads: `cnt[x]` is place x's slot
 * count in LDS (the sort's), `n_local` the places it has. Marks every place in L->mask, counts the routed experts and the
 * misses, leaves the misses past RAD_W2_TAIL_EXPERTS to the first pass (in place order), takes the
 * first kmax[distinct] misses in place order and posts their pool slots. One
 * post a call at most; a layer without a miss, or with a budget of none, posts nothing and leaves
 * seq 0, which is what makes its tail launches return at once (every mask byte is 0 or 1 and no
 * block of the tail finds a copy).
 *
 * The caller has passed a barrier since `cnt` was written. */
__device__ __forceinline__ void plan(RadW2Layer* L, const int32_t* cnt, PlanLds& s, uint32_t nthr,
                                     uint32_t n_local) {
    const uint32_t tid = threadIdx.x;
    uint32_t n = L->n_expert < RAD_W2_MAX_EXPERT ? L->n_expert : RAD_W2_MAX_EXPERT;
    if (n > n_local) n = n_local;
    if (tid == 0) {
        s.n_distinct = 0;
        s.n_miss = 0;
        s.seq = 0;
        s.k = 0;
    }
    __syncthreads();
    for (uint32_t x = tid; x < n; x += nthr) {
        uint8_t m = 0;
        if (cnt[x] > 0) {
            atomicAdd(&s.n_distinct, 1);
            const uint64_t* ent = L->ent[x];
            if (ent && pool_of(L->pools, *(const volatile uint64_t*)ent) >= 0) {
                m = 1;
                atomicAdd(&s.n_miss, 1);
            }
        }
        s.miss[x] = m;
    }
    __syncthreads();
    /* The tail serves the first RAD_W2_TAIL_EXPERTS misses in place order (its grid's bound); the
     * rest stay in the first pass. */
    for (uint32_t x = tid; x < n; x += nthr) {
        if (!s.miss[x]) continue;
        uint32_t j = 0;
        for (uint32_t u = 0; u < x; ++u) j += s.miss[u] != 0;
        if (j >= (uint32_t)RAD_W2_TAIL_EXPERTS) s.miss[x] = 2;
    }
    __syncthreads();
    for (uint32_t x = tid; x < n; x += nthr)
        if (s.miss[x] == 2) s.miss[x] = 0;
    __syncthreads();
    const uint32_t n_miss = (uint32_t)s.n_miss, n_distinct = (uint32_t)s.n_distinct;
    if (tid == 0) {
        uint32_t k = L->kmax[n_distinct <= RAD_W2_MAX_EXPERT ? n_distinct : RAD_W2_MAX_EXPERT];
        if (k > RAD_W2_MAX_COPY) k = RAD_W2_MAX_COPY;
        if (k > n_miss) k = n_miss;
        uint32_t seq = 0;
        if (k) {
            /* The rank's posts are numbered in stream order: one router at a time runs on it. */
            seq = *L->head + 1u;
            if (seq == 0) seq = 1u;
            *L->head = seq;
        }
        s.seq = seq;
        s.k = k;
        for (uint32_t j = 0; j < RAD_W2_MAX_COPY; ++j) s.term[j] = 0;
    }
    __syncthreads();
    const uint32_t k = s.k, seq = s.seq;
    RadW2Box* box = k ? L->box + seq % RAD_W2_BOXES : nullptr;
    for (uint32_t x = tid; x < n; x += nthr) {
        uint8_t m = s.miss[x];
        if (m && k) {
            /* The miss's rank in place order. */
            uint32_t j = 0;
            for (uint32_t u = 0; u < x; ++u) j += s.miss[u];
            if (j < k) {
                const uint64_t p = *(const volatile uint64_t*)L->ent[x];
                const int pi = pool_of(L->pools, p);
                if (pi >= 0) {
                    const uint64_t lo = L->pools.lo[pi], st = L->pools.stride[pi];
                    const uint64_t base = lo + ((p - lo) / st) * st;
                    const uint64_t dst = L->ring + (uint64_t)j * L->slot;
                    L->copy[j].base = base;
                    L->copy[j].dst = dst;
                    L->copy[j].bytes = st;
                    sys_store(&box->jobs[j].src, base);
                    sys_store(&box->jobs[j].dst, dst);
                    sys_store(&box->jobs[j].bytes, st);
                    s.term[j] = rad_w2_job_term(j, base, dst, st);
                    m = (uint8_t)(RAD_W2_COPY0 + j);
                } else {
                    /* Moved onto the card between the two reads: no job, and no term in the sum;
                     * the host's sum over the box's stale words then differs, it drops the post,
                     * and the tail reads every miss in place. */
                    L->copy[j].base = 0;
                    L->copy[j].dst = 0;
                    L->copy[j].bytes = 0;
                    s.term[j] = 0x5EED5EED5EED5EEDull + j;
                }
            }
        }
        L->mask[x] = m;
    }
    __syncthreads();
    if (tid == 0) {
        L->n_miss = n_miss;
        L->n_copy = k;
        L->n_distinct = n_distinct;
        L->seq = seq;
        RadW2Stats* st = L->stats;
        if (st) {
            st->plans += 1u;
            st->misses += n_miss;
            st->posted += k;
            st->spilled += n_miss > (uint32_t)RAD_W2_TAIL_EXPERTS ? n_miss - (uint32_t)RAD_W2_TAIL_EXPERTS : 0u;
            st->posted_n[k] += 1u;
        }
        if (k) {
            uint64_t sum = rad_w2_box_base(seq, k, L->group);
            for (uint32_t j = 0; j < k; ++j) sum ^= s.term[j];
            sys_store(&box->n, k);
            sys_store(&box->group, L->group);
            sys_store(&box->sum, sum);
            sys_store(&box->seq, seq);
        }
    }
}

/* The tail's run search: whether place x counts (a miss). The first pass counts every place. */
__device__ __forceinline__ bool in_tail(const RadW2Layer* L, uint32_t x) {
    return x < RAD_W2_MAX_EXPERT && L->mask[x] != RAD_W2_RESIDENT;
}

/* What a routed GEMM's block does with its expert `e` (a place) under W2. `tail` false is the
 * first pass: true (run it) for a resident expert. `tail` true: true for a miss. */
__device__ __forceinline__ bool mine(const RadW2Layer* L, uint32_t e, bool tail) {
    const uint32_t m = e < RAD_W2_MAX_EXPERT ? (uint32_t)L->mask[e] : 0u;
    return (m != RAD_W2_RESIDENT) == tail;
}

/* The tail: where a miss's weight is read. `p` is its table entry's value. The ring's copy if the
 * unit was posted, its list has landed and `p` still lies in the pool slot the copy was made from;
 * `p` itself otherwise (RAD_W2_DONE: the units of a list land one by one). Block-uniform (every thread of a block asks about the same expert). */
__device__ __forceinline__ const void* where(const RadW2Layer* L, uint32_t e, const void* p,
                                             bool* from_ring = nullptr) {
    if (from_ring) *from_ring = false;
    const uint32_t m = e < RAD_W2_MAX_EXPERT ? (uint32_t)L->mask[e] : 0u;
    if (m < RAD_W2_COPY0) return p;
    const uint32_t j = m - RAD_W2_COPY0;
    if (j >= RAD_W2_MAX_COPY) return p;
    const uint64_t base = L->copy[j].base, dst = L->copy[j].dst, bytes = L->copy[j].bytes;
    const uint64_t u = (uint64_t)p;
    if (!bytes || u < base || u - base >= bytes) return p;
    /* The list writes the word behind every unit: this one has landed once the word is this
     * post's and counts past it. */
    const uint32_t seq = L->seq;
    if (!seq) return p;
    const uint32_t d = sys_load(L->done);
    if ((d >> RAD_W2_DONE_BITS) != (seq & (0xffffffffu >> RAD_W2_DONE_BITS)) ||
        (d & ((1u << RAD_W2_DONE_BITS) - 1u)) <= j)
        return p;
    if (from_ring) *from_ring = true;
    return (const void*)(dst + (u - base));
}

}  // namespace r4d_w2
