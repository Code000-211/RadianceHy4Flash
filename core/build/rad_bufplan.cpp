/* rad_bufplan.cpp -- the buffer plan (spec §3.1).
 *
 * Activations are declared, liveness is computed over the declared op list, and non-overlapping
 * lifetimes are packed into one arena sized once, so the hot path never allocates.
 *
 * Two deliberate conservatisms (both in the spec): the arena is sized for max_tok, and liveness
 * covers the declared list, a superset of what any step issues. A plan correct for the worst step
 * is correct for every step.
 *
 * A buffer's life is a list of runs, not one interval (BufferInfo::runs). A buffer its plugin
 * declared rad_buf_full_write is rewritten whole by every op that writes it, so it is dead between
 * its use-runs: an attention scratch and a MoE scratch that never coexist inside a layer then share
 * one region instead of each holding bytes for the whole step. Every other buffer has the single
 * run [first_def, last_use].
 *
 * The packer is greedy interval colouring, largest first: each buffer takes the lowest aligned
 * offset that collides with no placed buffer whose life overlaps any of its runs. O(n^2) in the
 * buffer count, microseconds for a few hundred activations, and within a few percent of optimal on
 * real shapes (the one big activation goes first). Optimal packing is NP-hard and not worth a
 * solver.
 */
#include "rad_build.h"
#include "rad_internal.h"

#include <algorithm>
#include <climits>

namespace rad {

/* Every buffer starts at a 256-byte boundary: the container's sub-block alignment, and enough for
 * any RDNA4 vector load (dwordx4 is 16 B). A kernel needing more alignment for its scratch says so
 * through its scratch hook, which is sized separately. */
static constexpr int64_t kBufAlign = (int64_t)RAD_ALIGN_SUB;

namespace {

struct Item {
    size_t  idx;      /* index into Program::buffers */
    int64_t bytes;    /* already rounded up to kBufAlign */
    std::vector<BufRun> runs;   /* its life, at least one run, in order */
    int64_t off = -1;
};

bool overlaps(const Item& a, const Item& b) {
    for (const BufRun& x : a.runs)
        for (const BufRun& y : b.runs)
            if (x.lo <= y.hi && y.lo <= x.hi) return true;
    return false;
}

/* Lowest aligned offset that fits `want` without colliding with anything already placed
 * (off >= 0) that is alive at the same time. */
int64_t first_fit(const Item& want, const std::vector<Item>& placed) {
    std::vector<std::pair<int64_t, int64_t>> occ;
    occ.reserve(placed.size());
    for (const Item& p : placed)
        if (p.off >= 0 && overlaps(p, want)) occ.emplace_back(p.off, p.off + p.bytes);
    std::sort(occ.begin(), occ.end());

    int64_t off = 0;
    for (auto& [s, e] : occ) {
        if (s >= off + want.bytes) break;      /* the gap below this block is big enough */
        off = std::max(off, align_up(e, kBufAlign));
    }
    return off;
}

/* Places every item, largest first, ties by lifetime start then index, so the plan is identical
 * on every rank and every run. Returns the packed extent. */
int64_t pack(std::vector<Item>& items) {
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.bytes != b.bytes) return a.bytes > b.bytes;
        if (a.runs.front().lo != b.runs.front().lo) return a.runs.front().lo < b.runs.front().lo;
        return a.idx < b.idx;
    });
    int64_t end = 0;
    for (Item& it : items) {
        it.off = first_fit(it, items);
        end = std::max(end, it.off + it.bytes);
    }
    return end;
}

/* The most bytes live at one op index: a sweep over the run boundaries, so a buffer with many runs
 * costs its runs and not its op count. */
int64_t peak_live(const std::vector<Item>& items, int32_t last_op) {
    std::vector<std::pair<int32_t, int64_t>> ev;   /* (op index, +bytes at a run's start / -its end) */
    ev.reserve(items.size() * 2);
    for (const Item& it : items)
        for (const BufRun& r : it.runs) {
            if (r.hi < 0) continue;
            ev.emplace_back(r.lo, it.bytes);
            ev.emplace_back(r.hi + 1, -it.bytes);
        }
    std::sort(ev.begin(), ev.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;               /* a run's end before another's start: runs
                                                   * are inclusive, so an end sits at hi + 1 */
    });
    int64_t live = 0, peak = 0;
    for (const auto& [t, d] : ev) {
        if (t < 0 || t > last_op) continue;
        live += d;
        peak = std::max(peak, live);
    }
    return peak;
}

}  /* namespace */

int plan_buffers(Program& p, BufPlanReport* rep) {
    BufPlanReport r;
    int status = RAD_OK;

    const int32_t last_op = (int32_t)p.ops.size() - 1;
    const int32_t last   = last_op > 0 ? last_op : 0;

    std::vector<Item> transient, persist, host;
    for (size_t i = 1; i < p.buffers.size(); ++i) {
        BufferInfo& b = p.buffers[i];
        if (b.bytes <= 0) b.bytes = buffer_bytes(b.decl);
        if (b.bytes <= 0) {
            RAD_ERR("buffer plan: '%s' cannot be sized (dtype %u, rank %u)", b.name.c_str(),
                    b.decl.dtype, b.decl.rank);
            status = RAD_E_INVAL;
            continue;
        }

        Item it{ i, align_up(b.bytes, kBufAlign), b.runs };
        if (it.runs.empty() && b.first_def >= 0 && b.last_use >= b.first_def)
            it.runs.assign(1, BufRun{ b.first_def, b.last_use });
        if (it.runs.empty()) {
            /* No declared use: live for the whole program. */
            it.runs.assign(1, BufRun{ 0, last });
            if (b.decl.kind == RAD_BUF_TRANSIENT) ++r.n_unknown_live;
        }
        /* The buffer keeps the runs it was planned with, so the runtime guard in
         * core/runtime/issue.cpp and the level packing read the same life. */
        b.runs = it.runs;

        /* Each domain is coloured separately, with offsets relative to its own arena. One shared
         * offset space would leave both arenas sparse and spend VRAM on host activations.
         * Program::arena_bytes is the device high-water mark, host_arena_bytes the host one. */
        if (b.decl.domain == RAD_DOMAIN_HOST) host.push_back(it);
        else if (b.decl.kind == RAD_BUF_TRANSIENT) transient.push_back(it);
        else persist.push_back(it);
    }

    for (const Item& t : transient) r.sum_bytes += t.bytes;

    /* peak_live: what a perfect packer would reach (see BufPlanReport). */
    r.peak_live = peak_live(transient, last);

    /* PERSIST survives across steps and DERIVED is written by the core before any op runs, so
     * both are live for every op and share with nothing.
     *
     * They go at the bottom, transients above. A smaller step packs its transients smaller
     * (plan_level) and lends the unused top of the arena to the expert slab, so the fixed region
     * must sit where no level moves it: its contents outlive the step that wrote them. */
    int64_t off = 0;
    for (const Item& it : persist) {
        p.buffers[it.idx].arena_offset = off;
        off += it.bytes;
        off = align_up(off, kBufAlign);
    }
    const int64_t fixed = align_up(off, kBufAlign);
    r.persist_bytes = fixed;

    const int64_t tr_end = pack(transient);
    for (const Item& it : transient) p.buffers[it.idx].arena_offset = fixed + it.off;
    r.transient_bytes = tr_end;

    p.arena_fixed_bytes = fixed;
    p.arena_bytes = fixed + tr_end;
    p.arena_peak_live = r.peak_live;

    /* The host arena has its own base pointer; these offsets are relative to it. Same packer in
     * one pass: a host PERSIST is live for the whole program, so it shares with nothing. */
    const int64_t host_end = pack(host);
    for (const Item& it : host) p.buffers[it.idx].arena_offset = it.off;
    r.host_bytes = host_end;
    p.host_arena_bytes = host_end;

    if (rep) *rep = r;
    return status;
}

/* ------------------------------------------------------------------ a smaller step's plan */

int plan_level(const Program& full, const Program& probe, int64_t rows, ArenaLevel* out,
               std::string* why) {
    auto refuse = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_STATE;
    };
    if (probe.buffers.size() != full.buffers.size())
        return refuse(fmt("the sizing declare named %zu buffers and the real one %zu",
                          probe.buffers.size() - 1, full.buffers.size() - 1));

    ArenaLevel lv;
    lv.rows = rows;
    lv.offset.assign(full.buffers.size(), -1);
    lv.decl.assign(full.buffers.size(), RadBufDecl{});

    std::vector<Item> transient;
    for (size_t i = 1; i < full.buffers.size(); ++i) {
        const BufferInfo& fb = full.buffers[i];
        const BufferInfo& pb = probe.buffers[i];
        const RadBufDecl& f = fb.decl;
        const RadBufDecl& q = pb.decl;
        if (fb.name != pb.name || f.kind != q.kind || f.domain != q.domain ||
            f.dtype != q.dtype || f.rank != q.rank)
            return refuse(fmt("buffer %zu is '%s' in the real declare and '%s' in the sizing one",
                              i, fb.name.c_str(), pb.name.c_str()));
        /* Only the rows may shrink. Every other extent is a stride some kernel was resolved
         * against at the real size. */
        for (uint32_t d = 1; d < f.rank; ++d)
            if (f.shape[d] != q.shape[d])
                return refuse(fmt("buffer '%s' changes extent %u (%lld to %lld) with the step size",
                                  fb.name.c_str(), d, (long long)f.shape[d], (long long)q.shape[d]));
        if (f.rank > 0 && q.shape[0] > f.shape[0])
            return refuse(fmt("buffer '%s' is larger for a smaller step (%lld rows to %lld)",
                              fb.name.c_str(), (long long)f.shape[0], (long long)q.shape[0]));
        lv.decl[i] = q;
        /* The fixed region and the host arena are the same at every level. A persistent buffer
         * holds an earlier step's data, so it cannot move, and it keeps its full size: one sized
         * by the step (a per-token plane written outside the op order) is only smaller at a
         * smaller step, and the real extent covers it. */
        if (f.kind != RAD_BUF_TRANSIENT || f.domain == RAD_DOMAIN_HOST) {
            lv.decl[i] = f;
            lv.offset[i] = fb.arena_offset;
            continue;
        }
        /* The real declare's liveness: the op list is the same one the kernels were issued
         * against, so the runs are too — only the bytes change. */
        Item it{ i, align_up(buffer_bytes(q), kBufAlign), fb.runs };
        if (it.runs.empty() && fb.first_def >= 0 && fb.last_use >= fb.first_def)
            it.runs.assign(1, BufRun{ fb.first_def, fb.last_use });
        if (it.runs.empty())
            it.runs.assign(1, BufRun{ 0, std::max<int32_t>((int32_t)full.ops.size() - 1, 0) });
        transient.push_back(it);
    }

    const int64_t tr_end = pack(transient);
    for (const Item& it : transient) lv.offset[it.idx] = full.arena_fixed_bytes + it.off;
    lv.end = full.arena_fixed_bytes + tr_end;
    if (lv.end > full.arena_bytes)
        return refuse(fmt("packing the smaller buffers took %lld bytes, more than the full plan's "
                          "%lld", (long long)lv.end, (long long)full.arena_bytes));
    *out = std::move(lv);
    return RAD_OK;
}

}  /* namespace rad */
