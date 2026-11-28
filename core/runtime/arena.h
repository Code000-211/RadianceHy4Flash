/* arena.h -- one contiguous allocation, sized by the buffer plan at load and never allocated from
 * at runtime.
 *
 * There is deliberately no allocate(): an allocator on the step path is the dynamic behaviour this
 * engine exists to remove (spec §3.1). Offsets come from the buffer plan, which computes liveness
 * over the declared op list for the worst step; the arena only owns the block they index into.
 *
 * Scratch sits at the tail and is shared by every op on a stream, which is safe because launches
 * on one stream execute in order. A context's second lane is a second stream, so a program that
 * declares one gets a second region per lane (Program::scratch_regions). A lane used without that
 * declaration shares region 0, and Ctx::issue orders a scratch user behind an unordered one on the
 * other lane. Program::scratch_bytes is the largest single kernel's requirement; a kernel asking
 * for more is refused by name in prepare.
 */
#pragma once
#include "../rad_core.h"

namespace rad {

class Arena {
public:
    Arena() = default;
    ~Arena() { release(); }
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    /* `kind` is a RAD_MEM_* class: RAD_MEM_DEVICE for the device arena, RAD_MEM_HOST for the
     * second arena a program with RAD_DOMAIN_HOST buffers gets. `regions` scratch regions sit at the
     * tail, one per lane: region 0 of `scratch_bytes`, the others of `scratch_bytes1` (-1: the same;
     * Program::scratch_bytes_lane1). */
    int  init(int64_t plan_bytes, int64_t scratch_bytes, int kind, int regions = 1,
              int64_t scratch_bytes1 = -1);
    void release();

    char*   base()          const { return base_; }
    /* Region `r`, or region 0 when the arena holds fewer. */
    void*   scratch(int r = 0) const {
        if (!base_) return nullptr;
        return base_ + scratch_off_ +
               (r > 0 && r < scratch_regions_ ? scratch_pitch_ + (r - 1) * scratch_pitch1_ : 0);
    }
    /* Region 0's size (the program's largest kernel), or region `r`'s. */
    int64_t scratch_bytes() const { return scratch_bytes_; }
    int64_t scratch_bytes(int r) const {
        return r > 0 && r < scratch_regions_ ? scratch_bytes1_ : scratch_bytes_;
    }
    int     scratch_regions() const { return scratch_regions_; }
    int64_t bytes()         const { return bytes_; }
    bool    live()          const { return base_ != nullptr; }

private:
    char*   base_ = nullptr;
    int     kind_ = RAD_MEM_DEVICE;
    int64_t bytes_ = 0;
    int64_t scratch_off_ = 0;
    int64_t scratch_bytes_ = 0;
    int64_t scratch_pitch_ = 0;     /* scratch_bytes_ padded to RAD_ALIGN_UNIT */
    int64_t scratch_bytes1_ = 0;    /* each region past 0 */
    int64_t scratch_pitch1_ = 0;
    int     scratch_regions_ = 1;
};


/* The arena's size, computed without allocating it. Shared by Ctx::bind_arena (allocates at the
 * end of startup) and the VRAM budget resolver (subtracts it before the pools are allocated), so
 * the two cannot drift apart and overcommit the card.
 *
 * `dev_out` is the plan_bytes the device arena's init() receives; `total_out` is what it
 * allocates, plan and scratch both padded to RAD_ALIGN_UNIT as Arena::init pads them.
 * Returns the total. */
int64_t arena_plan_bytes(const Program& P, int64_t* dev_out, int64_t* total_out);

/* The host activation arena's size (the RAD_DOMAIN_HOST buffers), which Ctx::bind_arena pins and
 * maps at the end of startup; 0 when the program declares none. */
int64_t arena_host_bytes(const Program& P);

/* The arena level a step of `rows` tokens runs at: the smallest level that holds it, or 0 (the
 * full plan) when none does. `level_rows[k]` is the largest step level k serves; entry 0 is the
 * full plan and is never chosen by size. */
int arena_level_for(const std::vector<int64_t>& level_rows, int64_t rows);

/* A smaller arena level as the prefill stager's ring arithmetic needs it: the largest step it
 * holds, and the highest offset its own buffers reach (ArenaLevel::end). */
struct ArenaStepLevel {
    int64_t rows = 0;
    int64_t end = 0;
};

/* Where the prefill stager's ring is below the full plan (place/stager.h, "the region"): the
 * offset, in bytes from the arena base, above the highest smaller level a staged step can run at
 * -- a level that holds a step of `arm_rows` rows, the smallest prompt chunk the stager arms for
 * there, or the span's shared level (`span_rows`, 0 without one), which keeps the place for every
 * step it holds. One place serves all of them, so it is above the highest. -1 when no level
 * qualifies; the caller adds the guard and keeps the place under the full plan's own region. */
int64_t arena_ring_above(const std::vector<ArenaStepLevel>& levels, int64_t arm_rows,
                         int64_t span_rows);

/* The second place's offset from the arena base: above `above` (arena_ring_above) and the guard
 * between a level's buffers and what lies over them, on an alignment unit, and never past
 * `stage_off`, where the region itself starts. -1 when `above` is. */
int64_t arena_ring_offset(int64_t above, int64_t guard, int64_t stage_off);

/* The bytes at the top of an arena of `top` bytes that lie above the second place (at `low_off`,
 * `stage_bytes` long, its end on an alignment unit): what a step staged there, and every step of
 * the span's shared level, leaves lent. 0 without the place, or when it reaches the top. */
int64_t arena_lent_above_ring(int64_t top, int64_t low_off, int64_t stage_bytes);

/* What a step may leave of the arena lent to the expert slab. A step staged in the ring's second
 * place copies through it, so that place must not be lent, whatever level the step runs at; every
 * other step lends what its level does. `ring_lent` is what is above that place, 0 when it
 * reaches the top of the arena. */
int64_t arena_step_lent(int64_t level_lent, int64_t ring_lent, bool staged);

/* How long the arena's floor holds after the last step that reached it (Engine::arena_tick). */
constexpr uint64_t kArenaFloorNs = 2000000000ull;

/* Whether a step below the full plan still finds the prefill stager's ring in the region at the
 * top of the arena (place/stager.h, "a smaller step behind a staged one"). The arena's floor is
 * still the full plan's (`floor_rows` against `full_rows`: a step that large ran inside the floor's
 * time, or a staged step is not collected yet), and the expert slab holds no more of the arena
 * than the full plan leaves it (`full_lent`, which is nothing), neither as the loan's
 * target (`loan`) nor in slots it has yet to give back (`loan_held`). Every loan is counted from the top down and
 * the region is the top, so any byte lent past the full plan's is a byte of it. */
bool arena_ring_kept(int64_t floor_rows, int64_t full_rows, int64_t loan, int64_t loan_held,
                     int64_t full_lent);

/* Whether the arena's floor falls to the latest step's size, so that the loan may grow
 * (Engine::arena_tick): that step is under the floor, the floor was last reached more than
 * kArenaFloorNs ago (`since_ns`), and the step is not one staged below the full plan that may
 * still be copying (`ring_going`: the copies go into arena memory the loan would grow over, and
 * a step issued ahead is not through them when the step before it ends). Once such a step has
 * been collected the floor is the one the steps would have left without it
 * (Engine::arena_collected). */
bool arena_floor_falls(int64_t last_rows, int64_t floor_rows, bool ring_going, uint64_t since_ns);

/* The arena's floor, and when it was last reached. */
struct ArenaFloor {
    int64_t  rows = 0;
    uint64_t since_ns = 0;
};

/* The floor after a step of `rows` (its level's rows) is set at `now_ns`: a step at least as
 * large as the floor reaches it, and the floor's time starts again; a smaller one leaves it
 * (Engine::arena_step). The engine keeps two of these while a step staged below the full plan
 * pins the floor there: the pinned one, and the one the steps would have left without the pin,
 * which it puts back once that step is collected (Engine::arena_collected). */
ArenaFloor arena_floor_reach(ArenaFloor floor, int64_t rows, uint64_t now_ns);

/* Whether a step proposes swaps behind its pass (Engine::tiering_dispatch). A pass the stager
 * staged from min_rows() does not, as before: it is bound by its own copies on the mover's
 * transfer stream, and its histograms are credited for the next step that swaps. A pass staged
 * only by the arm floor (`below_min`: a prompt chunk of arm_rows() or more in a pass under
 * min_rows(), PrefillStager::staged_below_min) does, as that pass did before the stager armed
 * for it: with those two steps silent, resumed turns of 252-379 rows behind them in the same
 * server ran 6-12 ms slower (notes/backlog/measurements/M-19.md). An unstaged step always does. */
bool stage_swaps(bool staged, bool below_min);

/* Where the stager's ring is for a step, on one rank. */
enum class ArenaRing {
    None,    /* the step's level lends both places: nothing is staged */
    Top,     /* the region at the top of the arena */
    Low,     /* the second place, above the smaller levels' buffers */
};

/* What Engine::arena_step knows of a step and of one rank's arena when it decides that. */
struct ArenaStep {
    bool    full_plan = false;    /* the step runs at the full plan */
    bool    span_level = false;   /* ... or at the span's shared level */
    bool    has_low = false;      /* the rank's ring has a second place */
    bool    arms = false;         /* the stager arms for the step below the full plan */
    bool    adopts = false;       /* ... or would take layers staged ahead for it */
    int64_t floor_rows = 0;       /* the arena's floor, and the full plan's rows */
    int64_t full_rows = 0;
    int64_t loan = 0;             /* the arena's loan to the slab, and how far the slab reaches */
    int64_t loan_held = 0;
    int64_t full_lent = 0;        /* what the full plan lends, what the step's level lends, */
    int64_t level_lent = 0;       /* and what is above the ring's second place */
    int64_t ring_lent = 0;
};

/* The decision: where the ring is, what the step may leave lent (the loan is recalled to it
 * before the step where it is higher), and whether the step pins the floor at the full plan
 * while it may be copying (`holds`: it is staged below the full plan).
 *
 *   the full plan               the region; it lends nothing
 *   the span's shared level     the second place, which the level never lends
 *   a step the stager arms for, or one it would take staged layers in, while the region is
 *   still unlent (arena_ring_kept)
 *                               the region, where the last pass left the ring; nothing recalled
 *   a step the stager arms for, the region lent
 *                               the second place, recalled for the step (arena_step_lent)
 *   any other step              no ring; it lends what its level does */
struct ArenaStepRing {
    ArenaRing ring = ArenaRing::None;
    int64_t   allow = 0;
    bool      holds = false;
};
ArenaStepRing arena_step_ring(const ArenaStep& s);
}  /* namespace rad */
