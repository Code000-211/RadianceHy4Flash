/* stager.h -- the prefill stager: routed layers' non-resident experts copied onto the card ahead
 * of the layers that read them.
 *
 * ---- why prefill needs it ---------------------------------------------------------------------
 *
 * An expert the slab has no room for lives in the pinned host pool, and a GEMM that routes to it
 * reads it across the link in place (Site::DeviceZeroCopy). That suits decode, which reads a few
 * experts per step from the host. A prefill chunk routes to nearly every expert of every layer, so
 * each layer's whole pooled part (~85 MB per card on Qwen3.8-Flash-Next) crosses the link inside
 * its two GEMMs, on the critical path, while the rest of the layer leaves the link idle. With
 * r4d_selftest --perfmoe, a 2048-token gate_up GEMM takes 1.53 ms with every expert on the card
 * and 4.34 ms with the coldest 13% in host memory.
 *
 * So for large enough steps the stager copies the pooled experts of the layers ahead into a VRAM
 * ring while earlier layers run, publishes the copies for each layer's ops, and publishes the host
 * addresses again once those ops are issued. The bytes are the same, so output is bit-identical.
 *
 * ---- the region --------------------------------------------------------------------------------
 *
 * One region, two layers' share of the experts the budget leaves off the card, at the top of the
 * activation arena (Program::arena_stage_off, sized by vram_budget_resolve). The full plan lends
 * nothing, so the region is a full-plan step's ring. A step smaller than the full plan lends
 * that top to the expert slab.
 *
 * So the ring has a second place for a smaller step: right above the buffers of the largest
 * smaller arena level a staged step can run at (arena_ring_above). The engine keeps that place
 * out of the slab's loan for a step that is staged there, taking it back first where it is lent
 * (Engine::arena_step, arena_step_lent); every other step at those levels lends it as before. A
 * pass whose ring is in the other place than the last one's moves it first (rebase). The engine
 * says where the ring is, per pass (set_ring).
 *
 * Under --prefill-span the full plan is sized for a spanned step, and a step of one interval (a
 * chunk that shares its step, a short prompt) runs at the level of that size
 * (Engine::probe_arena_levels). That level keeps the second place for every step it holds, so
 * everything over it, the span's extra rows included, is lent to the slab while such steps run.
 *
 * ---- which passes ------------------------------------------------------------------------------
 *
 * A pass of min_rows() tokens or more, one past every level that lends the region, always finds
 * the ring and is staged. Below that a pass is staged when it carries a prompt chunk of
 * arm_rows() rows or more (stage_arms). The copies cost the same whatever the rows, a routed
 * layer's whole pooled part, while reading in place costs by the row, so there is a size from
 * which they pay: on Hy4-flash at TP2 a 720-row last chunk ran its routed GEMMs 58% cheaper
 * staged and a lone 723-row prompt 41%, while steps of 250-380 rows lost more to the copies than
 * their GEMMs gained (notes/backlog/closed/M-03b.md). A decode step's rows are one a sequence,
 * and it is staged only from min_rows().
 *
 * ---- a smaller step behind a staged one --------------------------------------------------------
 *
 * The slab does not get the top of the arena back the moment steps turn small: the loan regrows
 * only once no full-plan step has run for a while (Engine::arena_tick). A prompt's last chunk, or
 * a conversation's next turn, therefore often runs at a smaller level while the region itself is
 * still unlent. Such a step keeps the ring where it is, in the region, instead of moving it to
 * the second place: nothing is taken back from the slab, and the first layers the pass before
 * staged for it (below) are adopted where they lie rather than withdrawn with the move. A step
 * staged below the full plan keeps the loan off its ring while it copies and no longer: behind
 * it the loan is what it would be had the step run unstaged (Engine::arena_collected). The
 * region therefore serves the chunks and turns that arrive while the full-plan step before them
 * still holds the loan back, and a step staged in the second place gives the place back to the
 * slab once it is collected; the next such step takes it again.
 *
 * Layers staged ahead also outlive a next chunk that turns out too small to arm. While the ring
 * is still where they are, that pass takes them as they are -- it waits on their copies, reads
 * them and gives each back in order -- and stages nothing more itself; they are not withdrawn
 * unread. A decode step still returns them: it is replayed from a tape, and a pass the stager
 * takes part in is issued live.
 *
 * ---- why a ring, not two buffers ---------------------------------------------------------------
 *
 * Layers leave unequal parts off the card: at depth, with the KV cache's loan taken back, the last
 * third of a Qwen3.8-Flash-Next stack pools about twice the median layer. With two fixed buffers,
 * a light layer's copy finishes early and the engine idles, and a heavy layer's copy delays every
 * layer after it. The ring packs each layer at its own size, so copies run ahead through light
 * layers and heavy ones start early. A layer staged beyond the next one takes at most half the
 * ring, so the next layer always has the room two buffers would give it.
 *
 * ---- order and safety --------------------------------------------------------------------------
 *
 *   the pass's first op     stage the first routed layer the pass issues, and the ones after it
 *                           as far as the pace and the ring allow
 *   a layer's first op      wait on its copies; publish the host addresses of earlier layers
 *                           again; stage further layers into the freed room
 *   a layer's last op       record "free" behind it on the op's stream
 *   the pass's end          the same for anything still staged; and, when the next step continues
 *                           a prompt, stage the first layers of its trunk pass
 *
 * A layer's place in the ring is counted in bytes handed out since bring-up, so a copy finds the
 * places it overwrites from the count alone. Before a copy is issued, the mover's stream waits on
 * the "free" event of every earlier place it laps (recorded behind that place's last reader, in
 * this pass or an earlier one). The layer's ops wait on the copies' event through one stream wait
 * before its first op: copies are published settled (see Mover::stage_units), and only the
 * layer's own ops resolve them.
 *
 * ---- how far ahead -----------------------------------------------------------------------------
 *
 * A layer's copies are released once the compute stream reaches the MoE of the layer kLead before
 * it (an event recorded at that layer's first op), not as soon as the ring has room: copies that
 * land long before use slow the layers they overlap. On Qwen3.8-Flash-Next TP2, unpaced copies
 * cost a 26K-token prompt 0.6% and one layer ahead cost a 160K-token prompt 2.4% (its heavy late
 * layers waited); three ahead covered the late layers at the least cost to short prompts.
 *
 * ---- the next step's first layers --------------------------------------------------------------
 *
 * A step carrying a prompt chunk is issued only after the previous step is read back, so its first
 * layer would otherwise wait for all its own copies. When the next step continues a prompt
 * (set_next_prefill), this pass's end stages the next trunk pass's first layers, paced on this
 * pass's last layers as if the passes were one, and publishes them at once. The next trunk pass
 * adopts them when armed; an unarmed trunk pass publishes their host addresses again in
 * begin_pass, which the run phase calls before updating the weight tables. A draft pass in
 * between reads none of the trunk's layers and leaves them.
 *
 * ---- trunk only --------------------------------------------------------------------------------
 *
 * A draft head's routed layer is not staged. The head runs in its own passes; over a prefill chunk
 * (the history pass) it needs its experts only on a prompt's last chunk, for the few rows that
 * draft, so its pooled experts are read in place as at decode. The head's layer is numbered past
 * the trunk's (RadModelMeta::n_layers) and left out at bring-up; a draft pass is never armed.
 */
#pragma once
#include "../runtime/stager.h"
#include "mover.h"
#include "planner.h"

#include <deque>
#include <map>
#include <vector>

namespace rad {

/* One routed layer a stager serves: its units and the span of ops that read them. */
struct StagedLayer {
    int32_t              layer = -1;
    std::vector<int32_t> units;
    int32_t              first_op = -1, last_op = -1;
};

class PrefillStager final : public OpStager {
public:
    ~PrefillStager() override;

    /* Binds the routed layers of `plan` and the ops that read them, with `bytes` of device memory
     * at `region` as the ring -- or at `low`, the ring's second place above the smaller arena
     * levels' buffers, when there is one (null otherwise). `min_rows` is the smallest pass it
     * arms for whatever the pass carries, and `arm_rows` the smallest prompt chunk it arms for
     * below that (0: none). Leaves enabled() false when no routed layer has a pooled unit to
     * stage. */
    int init(const Program& prog, const Plan& plan, Mover* mover, char* region, char* low,
             int64_t bytes, int64_t min_rows, int64_t arm_rows = 0);

    /* Per pass, before release: where the ring is for the step -- the region or `low`, as the
     * engine decided (Engine::arena_step, arena_step_ring) -- or null when the step's level lends
     * both. */
    void set_ring(char* base) { want_ = base; }
    /* Where set_ring last said it is, or null. */
    char*   ring() const { return want_; }
    /* The ring's two places: the region, and above the smaller levels' buffers (the same one
     * when there is no such place). */
    char*   ring_full() const { return region_; }
    char*   ring_low() const { return low_; }
    /* Per step, before release: whether the next step continues a prompt in a pass this stager
     * arms for, so this step's trunk pass stages that pass's first layers -- and from which layer
     * it starts, which inside a chunk group (S-02) is the next sub-step's layer_lo, not 0. */
    void set_next_prefill(bool more, int32_t first_layer = 0) {
        next_prefill_ = more;
        arm_first_ = more ? first_layer : 0;
    }

    bool    enabled() const { return !layers_.empty(); }
    int64_t ring_bytes() const { return ring_; }
    /* The smallest pass, in tokens, the stager arms for whatever it carries: the smallest that
     * always finds the ring unlent. The scheduler keeps a held-back prefill chunk at or above it
     * (Scheduler::set_stage_rows). */
    int64_t min_rows() const { return min_rows_; }
    /* The smallest prompt chunk, in tokens, it arms for below min_rows() (see "which passes"); 0
     * when it arms for none. Smaller steps read pooled experts in place. */
    int64_t arm_rows() const { return arm_rows_; }
    /* Whether it arms for `b` below min_rows(), given a ring: a trunk pass that carries a prompt
     * chunk of at least arm_rows() tokens. The engine asks before it decides where the ring is
     * and what the step leaves lent, and begin_pass asks again. */
    bool    arms_for(const RadBatch* b) const { return enabled() && stage_arms(b, arm_rows_); }
    /* Whether layers are staged ahead that `b`, a prompt chunk of any size, would take if the
     * ring is still where they are (see "a smaller step behind a staged one"). */
    bool    adopts(const RadBatch* b) const { return n_early_ > 0 && stage_arms(b, 1); }
    size_t  n_layers() const { return layers_.size(); }
    /* Whether the last trunk pass armed (copied its pooled experts through the ring). A draft
     * pass leaves it unchanged. */
    bool    staged() const { return staged_; }
    /* ... and only because it carried a prompt chunk of arm_rows() or more: a pass under
     * min_rows(), which nothing staged before M-19. The engine proposes swaps behind such a pass
     * as it did then (stage_swaps). */
    bool    staged_below_min() const { return staged_below_min_; }
    static constexpr int32_t lead() { return kLead; }

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  after_op(rad_op h, RadStream s) override;
    int  end_pass(RadStream s) override;

private:
    using Layer = StagedLayer;
    std::vector<Layer>   layers_;      /* the routed layers, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */
    std::vector<int32_t> last_of_;     /* by op: the layer whose last op it is, or -1 */

    /* One layer's place in the ring. [beg, end) is in bytes handed out since bring-up; the region
     * offset is beg modulo its size, and a place never straddles the region's end. */
    struct Place {
        int32_t  li = -1;                      /* index into layers_ */
        int64_t  beg = 0, end = 0;
        RadEvent done = nullptr;               /* behind its copies, on the mover stream */
        RadEvent free = nullptr;               /* behind its last reader */
        bool     free_rec = false;
        /* Its last op has been issued at least once and `free` is recorded behind it. A step
         * longer than the MoE block's pass of rows issues a layer's ops once per pass, so the last
         * op can repeat: each repeat moves `free` behind itself, and the copies are published away
         * only at the next layer's first op. */
        bool     read = false;
        bool     retired = false;              /* its units are published at their pool slots */
        std::vector<int32_t> staged;
    };
    /* More places than a two-share ring can hold at the smallest per-layer share; a full table
     * only stops copies running further ahead. */
    static constexpr int kPlaces = 16;
    Place    place_[kPlaces];
    std::deque<int32_t>  ring_order_;  /* places in the ring, oldest first; retired ones stay until lapped */
    std::vector<int32_t> spare_;       /* places not in the ring */

    Mover*   mover_ = nullptr;
    char*    region_ = nullptr;
    char*    low_ = nullptr;           /* the ring's place at the span's shared level, or null */
    char*    base_ = nullptr;          /* where the places in the ring are */
    char*    want_ = nullptr;          /* where this pass wants it (set_ring) */
    bool     rebase_ = false;          /* the pass moves the ring before its first copy */
    RadEvent moved_t_ = nullptr;       /* the move's two waits: behind the mover's stream, */
    RadEvent moved_c_ = nullptr;       /* and behind the compute stream */
    int64_t  ring_ = 0;
    int64_t  head_ = 0;                /* bytes handed out since bring-up */
    int64_t  min_rows_ = 0;
    int64_t  arm_rows_ = 0;

    /* The pace (see the file header): layer k's copies wait on the compute-stream event recorded
     * at the first op of layer k - kLead. A layer within kLead of the pass's first routed layer
     * goes at once. */
    static constexpr int32_t kLead = 3;
    static constexpr int kGates = 64;
    RadEvent gate_[kGates] = {};
    int32_t  gate_li_[kGates] = {};    /* the layer each gate was recorded at this pass, or -1 */
    int32_t  first_li_ = -1;           /* the pass's first routed layer */

    bool     armed_ = false;
    /* The pass is too small to arm and only takes the layers staged ahead for it: it waits on
     * them, reads them and retires them as an armed pass does, and stages none itself. */
    bool     take_only_ = false;
    /* Whether the last trunk pass armed to stage, as its begin_pass decided; end_pass leaves it,
     * since the engine asks after the pass has ended. */
    bool     staged_ = false;
    bool     staged_below_min_ = false;
    bool     next_prefill_ = false;
    bool     first_op_ = false;        /* the pass has issued nothing yet */
    int32_t  next_ = 0;                /* the next layer to stage this pass */
    /* The next trunk pass's first layer as the engine named it (set_next_prefill). */
    int32_t  arm_first_ = 0;
    /* The next trunk pass's first layers, staged at the last pass's end: the layers
     * [early_first_, early_first_ + n_early_) -- [0, n_early_) for an unsliced pass -- which the
     * pass that begins there adopts. */
    int32_t  early_first_ = 0;
    int32_t  n_early_ = 0;

    enum { kStaged, kNone, kLater };
    int  stage_ahead(int32_t cur, bool* resync);
    int  stage_layer(int32_t li, bool now, RadEvent gate, int* got);
    void withdraw_early();
    int  take(int64_t want, bool partial, int32_t* out);
    int  recycle_front();
    int  rebase(RadStream s);
    void retire(Place& p);
};

/* ---- the file stager: routed layers read from the container -------------------------------------
 *
 * A routed unit on the file tier (--weights-disk-tier) has no address of its own, and a grouped
 * GEMM is handed every expert's pointer at issue. So on every pass, decode included, each routed
 * layer's file-tier units are read into one of the mover's two file buffers before the layer's
 * first expert op and published there, then published as null when the next routed layer starts.
 *
 * ---- which experts -------------------------------------------------------------------------------
 *
 * Only the experts the layer's rows route to are read. Their counts are known only once the
 * layer's router has run, so a layer is read at its own trigger, its first expert op: the counts
 * are copied back behind the router on the same stream (one drain of the stream), and a unit
 * counted zero is published at the mover's zero region instead of being read. The read cannot
 * start earlier: counts taken before the router are another layer's, or the last pass's, and a
 * truly routed expert they count zero would be computed from zeros. The card waits for every
 * read either way, so reading at the trigger costs what reading the next layer ahead behind the
 * same drain did.
 *
 * A layer the count source cannot give counts for (the first pass, before any report) is read
 * whole, and since that read needs no counts it goes ahead of time as before: at the pass's
 * first op for the first routed layer, at a layer's trigger for the next one, into the buffer
 * just released.
 *
 * So is every layer of a pass large enough that its rows route nearly every expert, kWholeRows
 * rows an expert on average by the model's expert count and top-k (whole_rows()). An exact read
 * takes the same bytes there, and each layer costs the card's work plus the drive's, since the
 * card idles behind the drain while the drive reads. A whole read needs no counts, so nothing
 * drains the stream, and the drive reads the next layer while the card computes the one before.
 * At 64 rows an expert, one routed an eighth as often as the average is left out of about one
 * pass in 3000, and read for nothing then. Smaller passes skip enough to pay for the drain: at
 * decode nearly every file-tier expert.
 *
 *   the pass's first op      read the first routed layer the pass issues, if it is read whole
 *   a layer's first op       release the layer before (record its buffer free on the stream,
 *                            publish null), read this layer unless read ahead, wait on its
 *                            copies, and read the next layer ahead if it is read whole -- its
 *                            copies queue behind the free event of the buffer it takes
 *   a repeat of that op      another slice of rows: read what its counts route that the earlier
 *                            slices' did not, and wait on those copies
 *   the pass's end           release whatever is still held
 *
 * A layer is released when the next one starts, not at its own last op: a routed block that slices
 * a large step into passes of `rows` tokens issues its expert ops once per slice, so a release at
 * the last op would leave the next slice reading null. Each slice routes its own rows, which is
 * why a repeat reads again.
 *
 * Every read that issued a copy is waited on, a hot-cache admission's or a pool deposit's as much
 * as a staged unit's: all are published settled, so the wait is the only thing between the GEMM
 * and the slot's previous bytes.
 *
 * The read blocks the issuing thread while the card runs what was issued before it; a reader
 * thread would gain little and add a second writer of the residency table.
 *
 * A layer is read ahead only when the pass will issue it (a bound learned per draft_pass from the
 * last pass of that kind), so a trunk pass does not read the MTP head's layer.
 *
 * ---- which units -------------------------------------------------------------------------------
 *
 * Under a dynamic plan the pool is a cache of the file tier: a deposit may relabel a pooled unit
 * back to the file tier (Mover::stage_file_units). A pooled unit the plan put in the pool is then
 * on the file tier too, so the stager covers every movable routed unit of a class with file-tier
 * units, not only the plan's file-tier ones, and tells the mover so (Mover::cover_file_units); a
 * trigger whose layer has nothing on the file tier skips the read and its drain. Where covering
 * them would interleave two layers' spans, only the plan's file-tier units are covered, and the
 * mover relabels none of the others. */
class FileStager final : public OpStager {
public:
    ~FileStager() override;

    /* Routing-aware reads: the engine binds a source of a layer's per-expert routing counts
     * (Ctx::route_counts_device). It returns the device pointer of `n_expert` i32 counts that the
     * layer's router rewrites every pass before the layer's first expert op, which is the
     * stager's trigger, or null when it has none (the layer has not reported yet) -- the stager
     * then reads the whole layer. Null `routes` reads whole layers. */
    using RouteCounts = const int32_t* (*)(void* ud, int32_t layer, int64_t* n_expert);

    /* Binds the routed layers with file-tier units and the ops that read them, over the mover's
     * two file buffers. Leaves enabled() false when no routed unit is on the file tier. Refuses a
     * plan whose routed file-tier layers are read by interleaved ops, since a file-tier unit has
     * no other address to fall back on. */
    int init(const Program& prog, const Plan& plan, Mover* mover,
             RouteCounts routes = nullptr, void* routes_ud = nullptr);

    bool    enabled() const { return !layers_.empty(); }
    size_t  n_layers() const { return layers_.size(); }
    int64_t buffer_bytes() const { return cap_; }
    /* Stream waits on the stager's copies so far, one per read that issued any. Tests read it. */
    int64_t waits() const { return waits_; }
    /* The smallest pass, in tokens, whose layers are read whole and ahead (see "which experts"),
     * or 0 when every pass is read exactly (no routing source, or the model's top-k unknown). */
    int64_t whole_rows() const { return whole_rows_; }
    static constexpr int64_t kWholeRows = 64;

    bool begin_pass(const RadBatch* b) override;
    int  before_op(rad_op h, RadStream s, bool* resync) override;
    int  end_pass(RadStream s) override;

private:
    using Layer = StagedLayer;
    int64_t              waits_ = 0;
    std::vector<Layer>   layers_;      /* the routed layers the stager covers, in first-op order */
    std::vector<int32_t> first_of_;    /* by op: the layer whose first op it is, or -1 */

    Mover*   mover_ = nullptr;
    char*    buf_[2] = { nullptr, nullptr };
    int64_t  cap_ = 0;
    RouteCounts routes_ = nullptr;     /* routing-aware reads; null reads whole layers */
    void*    routes_ud_ = nullptr;
    int32_t* scratch_ = nullptr;        /* pinned, for the trigger's blocking counts copy */
    int64_t  scratch_n_ = 0;
    /* Per buffer: what its layer's reads staged (nulled at release), skipped (likewise) and read
     * into hot-cache slots (whose publications outlive the buffer -- the slot is their address
     * until heat evicts one). */
    Mover::FileRead rd_[2];
    std::vector<int32_t> again_;        /* a repeat's candidates: the layer's skipped units */
    RadEvent done_[2] = { nullptr, nullptr };   /* behind a buffer's copies, on the mover stream */
    RadEvent free_[2] = { nullptr, nullptr };   /* behind the ops that read a buffer's layer */
    bool     free_rec_[2] = { false, false };
    bool     pending_[2] = { false, false };     /* copies no stream has waited on yet */
    int32_t  holds_[2] = { -1, -1 };             /* the layer a buffer holds, or -1 */
    int32_t  cur_ = -1;                          /* the layer whose ops are being issued */

    bool     armed_ = false;
    bool     first_op_ = false;
    int32_t  pass_kind_ = 0;
    int32_t  last_issued_ = -1;
    std::map<int32_t, int32_t> pass_last_;   /* by draft_pass: the last op such a pass issued */
    int64_t  whole_rows_ = 0;
    bool     whole_ = false;                 /* this pass reads every layer whole (whole_rows) */

    bool will_issue(int32_t li) const;
    /* Whether layer `li` is read with counts, at its own trigger; otherwise it is read whole,
     * and may be read ahead. */
    bool counted(int32_t li) const;
    /* The counts of the rows about to run, copied back behind the router on `s`, or null. */
    int  counts(int32_t li, RadStream s, const int32_t** ecnt, int32_t* n_expert);
    int  start(int32_t li, RadStream s, bool* resync);
    int  top_up(int32_t li, RadStream s, bool* resync);
    int  wait(int32_t b, RadStream s);
    int  finish(int32_t b, RadStream s);
};

}  /* namespace rad */
