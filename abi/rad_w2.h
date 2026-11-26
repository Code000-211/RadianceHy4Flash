/* rad_w2.h -- "Weg 2": resident experts first, budgeted whole-unit copies for the misses (issue #309,
 * notes/backlog/design/M-20.md §5-§7). What the kernels (libr4d), the runtime (core/runtime/w2.cpp)
 * and the MoE block (arch/common/rad_block_moe_fp8.h) share. Prototype, behind RADIANCE_W2
 * (default off: no block issues a W2 operand, and every launch is the one it was).
 *
 * A routed layer under expert-tiered placement reads its off-card experts ("misses") from the
 * pinned host pool across the link, inside the gate_up and down GEMMs, where every resident
 * expert's rows wait for them. With W2 on, a wide pass (a verify pass) runs the layer as
 *
 *   router            sorts the slots as before and, in the same launch (r4d_w2::plan), marks each
 *                     routed expert resident / miss in `mask`, picks at most `kmax[distinct]`
 *                     misses and posts their whole units (gate_up + down, one pool slot) as a
 *                     copy list into a pinned mailbox;
 *   host thread       sees the mailbox and hands the list to the card's prefetch DMA queue
 *                     (rad_xp_list), which counts the units into `done` as they land;
 *   gate_up, quant, down over the RESIDENT experts only (the misses' blocks return behind their
 *                     search), so the link belongs to the copies;
 *   the tail          gate_up, quant, down again, the same op at the same pass height M (the same
 *                     kernel instance and tile geometry, so every expert's sums are the ones of
 *                     the one-launch form) over a grid that holds the misses' tiles alone: a miss
 *                     reads its unit from the ring if its copy has landed and from the pool in
 *                     place otherwise. A layer without a miss still launches its tail (a recorded
 *                     pass is one tape), whose blocks find no tile and return.
 *
 * Results never change: every expert's rows are computed by the same kernel instance from the same
 * bytes, once per pass (the quantiser between the GEMMs runs twice over the pass and writes the
 * resident rows' codes twice, the same bytes). A ring pointer is used only while the expert's table
 * entry still names the pool slot the copy was made from (a heat-engine move wins) and only once
 * `done` says that unit of that post has landed.
 *
 * This header is C, read by device code and host code alike. */
#ifndef RAD_W2_H
#define RAD_W2_H

#include <stdint.h>

/* The few functions device code calls too (the plan builds a post's sum). */
#if defined(__HIP__) || defined(__CUDACC__)
#define RAD_W2_FN static inline __attribute__((host, device))
#else
#define RAD_W2_FN static inline
#endif

#define RAD_W2_MAGIC      0x3257444152ull   /* "RADW2" */
#define RAD_W2_MAX_EXPERT 512
#define RAD_W2_MAX_COPY   8      /* ring slots a layer: the most whole units one layer's list copies */
#define RAD_W2_GROUPS     3      /* a group's slots are reused three routed layers later */
#define RAD_W2_BOXES      16     /* the mailbox ring: posts the host thread may be behind */
#define RAD_W2_MAX_POOLS  4
/* The most misses a layer's tail serves; a layer with more keeps the rest in the first pass, read
 * in place there as without W2. It bounds the tail launches' grid: these experts' rows make at
 * most RAD_W2_TAIL_EXPERTS + T / 16 run tiles (rad_w2_tail_tiles), a seventh of the first pass's
 * bound at a verify pass of 48 rows, and a block that finds no tile still pays its search. */
#define RAD_W2_TAIL_EXPERTS 16

/* A group's done word: the post's number above its low RAD_W2_DONE_BITS bits, which count the
 * units of that post's list that have landed (the DMA list writes it behind every unit). Unit j of
 * post `seq` is on the card once the word is rad_w2_done(seq, j + 1) or more of the same post. */
#define RAD_W2_DONE_BITS 5
#define RAD_W2_DONE(seq, landed) (((uint32_t)(seq) << RAD_W2_DONE_BITS) + (uint32_t)(landed))

/* `mask[x]` of place x (the index the layer's expert offsets and weight tables use). */
#define RAD_W2_RESIDENT 0u       /* in VRAM, or not routed: the first pass runs it */
#define RAD_W2_INPLACE  1u       /* a miss the tail reads in place */
#define RAD_W2_COPY0    2u       /* RAD_W2_COPY0 + j: a miss whose unit was posted to ring slot j */

typedef struct RadW2Pools {
    uint64_t lo[RAD_W2_MAX_POOLS], hi[RAD_W2_MAX_POOLS], stride[RAD_W2_MAX_POOLS];
    int32_t  n, pad;
} RadW2Pools;

/* One post, in pinned host memory. The plan's stores cross the link in no promised order and it
 * does not wait for them (a fence a layer costs the router several microseconds), so the post
 * carries a sum over everything in it (rad_w2_box_sum): the host thread takes a post whose number
 * it sees only once the sum it computes is the sum it reads. */
typedef struct RadW2Job {
    uint64_t src, dst, bytes, pad;
} RadW2Job;
typedef struct RadW2Box {
    RadW2Job jobs[RAD_W2_MAX_COPY];
    uint32_t n, group;
    uint32_t seq, pad;
    uint64_t sum;
} RadW2Box;
RAD_W2_FN uint64_t rad_w2_job_term(uint32_t j, uint64_t src, uint64_t dst, uint64_t bytes) {
    uint64_t t = (src + dst * 31u + bytes * 131u + (uint64_t)j + 1u) * 0xD6E8FEB86659FD93ull;
    return t ^ (t >> 29);
}
RAD_W2_FN uint64_t rad_w2_box_base(uint32_t seq, uint32_t n, uint32_t group) {
    return (uint64_t)seq * 0x9E3779B97F4A7C15ull + (uint64_t)n * 0xBF58476D1CE4E5B9ull + (uint64_t)group + 1u;
}
/* The sum of a post of `n` jobs: its header's term, and every job's XORed in. */
RAD_W2_FN uint64_t rad_w2_box_sum(uint32_t seq, uint32_t n, uint32_t group, const RadW2Job* jobs) {
    uint64_t s = rad_w2_box_base(seq, n, group);
    for (uint32_t j = 0; j < n && j < RAD_W2_MAX_COPY; ++j)
        s ^= rad_w2_job_term(j, jobs[j].src, jobs[j].dst, jobs[j].bytes);
    return s;
}

/* Device counters, summed over the layers of a rank. */
typedef struct RadW2Stats {
    uint32_t plans;       /* router launches that planned */
    uint32_t misses;      /* routed experts found in the pool */
    uint32_t posted;      /* units posted for a copy */
    uint32_t landed;      /* posted units the tail's gate_up read from the ring */
    uint32_t late;        /* posted units it read in place: the copy had not landed, or the unit moved */
    uint32_t spilled;     /* misses past RAD_W2_TAIL_EXPERTS, left to the first pass */
    uint32_t pad[2];
    /* By the layer's list length: lists posted, and units of such lists read from the ring. */
    uint32_t posted_n[RAD_W2_MAX_COPY + 1], landed_n[RAD_W2_MAX_COPY + 1];
    uint32_t pad2[2];
} RadW2Stats;

typedef struct RadW2Copy {
    uint64_t base, dst, bytes;   /* the pool slot, its ring slot, the slot's size */
} RadW2Copy;

/* A routed layer's control block, device memory, one a rank. The first part is written once when
 * the layer is bound, the second by every plan. */
typedef struct RadW2Layer {
    uint64_t        magic;
    uint64_t        ring;                 /* the layer's group of RAD_W2_MAX_COPY ring slots (device) */
    uint64_t        slot;                 /* bytes a ring slot */
    const uint32_t* done;                 /* the group's done word (uncached device memory): RAD_W2_DONE */
    RadW2Box*       box;                  /* the mailbox ring (pinned host, device address) */
    uint32_t*       head;                 /* the rank's post counter (device) */
    RadW2Stats*     stats;
    uint32_t        n_expert;             /* places the plan looks at (the regular experts) */
    uint32_t        group;
    /* w2-explore. `form` 0: the prototype's (resident launches, then the tail's). 1, "inline": one
     * gate_up and one down launch whose blocks are renumbered so that every tile of a posted
     * (copied) expert runs behind every other tile, reading the ring where the unit has landed
     * and the pool in place otherwise; no tail launch. The blocks that find no tile then all sit
     * at the launch's end. 2, "inline2": the launch's rows (column groups) keep their order and
     * their empty blocks; the posted experts' tiles of every column group take the last empty
     * blocks of the last rows. 3, "inline3": the one launch's own numbering, a posted expert's
     * block reading the ring where the unit has landed by then. `spin`: how often a block of a posted
     * expert re-reads the done word before it gives up and reads in place. `gu_first` 1: the
     * list copies every unit's first `split` bytes (the gate_up tables) before any unit's rest,
     * so the done word counts 2 n ranges. */
    uint32_t        form, spin, gu_first;
    /* 1: a miss that is not posted stays in the first launch (read in place there, as without
     * W2) instead of in the tail, where its link read is the layer's critical path. */
    uint32_t        inplace_first;
    uint64_t        split;
    RadW2Pools      pools;
    /* The most units a layer's list may hold, by the number of distinct routed experts (which is
     * what sets the window from the router's end to the resident down GEMM's end). */
    uint8_t         kmax[RAD_W2_MAX_EXPERT + 1];
    uint8_t         pad0[7];
    /* Where place x's first weight (the gate_up codes) has its table entry; null for none. */
    const uint64_t* ent[RAD_W2_MAX_EXPERT];

    /* ---- written by the plan */
    uint32_t        seq;                  /* this pass's post, 0 for none */
    uint32_t        n_miss, n_copy, n_distinct;
    RadW2Copy       copy[RAD_W2_MAX_COPY];
    uint8_t         mask[RAD_W2_MAX_EXPERT];
    /* w2-explore, form 4 ("inline4"): the place of posted unit j (0xffff: none). The one launch
     * keeps its numbering and its search; a posted expert's blocks return where they stand, and
     * the launch's last blocks that found no tile (the end of its last rows) run the posted
     * experts' tiles, which they find from this list without a search. */
    uint16_t        cp_place[RAD_W2_MAX_COPY];
    uint32_t        ntb;                  /* the posted experts' run tiles (16 rows), a bound */
    uint32_t        pad2;
} RadW2Layer;

/* ------------------------------------------------------------------ the budget (host)
 *
 * Measured on the local lane at TP2, both cards copying at once (measurements/M-20.md, "whole
 * units (1.43 MiB), both cards"): a list of n whole units takes 90.6 us at n = 1 and about 83 us a
 * unit from n = 2 (172.4, 339.4, 665.4 us at 2, 4, 8), and the host hand-over adds about 10 us.
 * The window is the router's end to the end of the resident experts' down GEMM. */
#define RAD_W2_HANDOVER_US 10.0
#define RAD_W2_UNIT1_US    90.6
#define RAD_W2_UNIT_US     83.0

/* Microseconds a list of n whole units takes from the plan's post to its last byte. */
static inline double rad_w2_list_us(int n) {
    if (n <= 0) return 0.0;
    return RAD_W2_HANDOVER_US + (n == 1 ? RAD_W2_UNIT1_US : RAD_W2_UNIT_US * (double)n);
}
/* The longest list that lands inside `window_us`, at most `cap` (and RAD_W2_MAX_COPY). */
static inline int rad_w2_kmax(double window_us, int cap) {
    int k = 0;
    if (cap > RAD_W2_MAX_COPY) cap = RAD_W2_MAX_COPY;
    while (k < cap && rad_w2_list_us(k + 1) <= window_us) ++k;
    return k;
}
/* The window by distinct routed experts, a line through the lane's four shapes (design/M-20.md §2,
 * §3: router end to down start plus the no-miss down): no-spec c1 16.9 experts 66 us, DFly2 c1
 * 51.9 / 154, no-spec c8 99.2 / 281, DFly2 c8 203.2 / 620. */
#define RAD_W2_WIN_US0        10.0
#define RAD_W2_WIN_US_EXPERT   3.0
static inline double rad_w2_window_us(int distinct) {
    return RAD_W2_WIN_US0 + RAD_W2_WIN_US_EXPERT * (double)distinct;
}

/* Run tiles (16 sorted rows of one expert, from the run's own first row) the tail's experts can
 * make in a pass of T sorted rows: the tail launches' grid along x, less the one block past them. */
static inline uint32_t rad_w2_tail_tiles(uint32_t T) {
    return (uint32_t)RAD_W2_TAIL_EXPERTS + T / 16u;
}

#endif /* RAD_W2_H */
