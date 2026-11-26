/* r4d_selftest_w2.h -- r4d_selftest's W2 check (abi/rad_w2.h), included by r4d_selftest.cpp behind
 * its helpers (dalloc, st_memcpy, st_stream_sync, T, ok, fmt).
 *
 * A two-class routed GEMM's launch split as the MoE block splits it under W2: the first launch
 * with a `w2` operand, then the tail with `w2_tail`, over a control block written here. `ts` / `ps`
 * are the plain launch's nine operands and its parameters, `wbuf(e)` expert e's first weight
 * buffer and its bytes. Every routed expert is marked resident, a miss read in place, or a miss
 * with a posted copy in a "ring" (a copy of its buffer elsewhere on the card), in turn, up to the
 * RAD_W2_TAIL_EXPERTS misses a tail serves. Held:
 *   - the first launch writes the resident experts' rows and the rows past the live count, byte for
 *     byte the plain launch's, and leaves every miss's rows untouched;
 *   - the tail then completes them: every row the plain launch's bytes (same tiles, same sums);
 *   - with the first ring slot's bytes inverted the tail's rows of that expert differ (it did read
 *     the ring) and nothing else does; with the done word off the post's number they are the
 *     plain launch's again (a copy that has not landed is read in place);
 *   - gate_up's tail counts each posted unit once, as landed or late.
 * A form that does not read the operand (anything but the decode grid) runs every expert on both
 * launches; that is seen at the first launch and held to the plain bytes twice. `g_w2_split`
 * counts the launches that did split, `g_w2_whole` the ones that did not. */
#pragma once
#include "rad_w2.h"

#include <functional>
#include <memory>

static int g_w2_split = 0, g_w2_whole = 0;

static void moe_w2_split_case(const std::string& nm, const RadKernelInfo* r, std::vector<RadTensor> ts,
                              std::vector<RadParam> ps, void* d_y, const std::vector<int32_t>& off,
                              int64_t E, int64_t Tr, int64_t N, bool token_order,
                              const std::function<std::pair<void*, size_t>(int64_t)>& wbuf) {
    if (ts.size() != 9 || E > RAD_W2_MAX_EXPERT) return;
    const std::string what = nm + ": W2";
    const uint16_t poison = 0x7e57u;
    /* One launch over `*out` as the output's bytes before it (poison where empty). */
    auto run = [&](const std::vector<RadTensor>& t, std::vector<uint16_t>* out) {
        std::vector<uint16_t> before((size_t)(Tr * N), poison);
        if (out->size() == before.size()) before = *out;
        st_memcpy(d_y, before.data(), before.size() * 2, hipMemcpyHostToDevice);
        std::vector<RadTensor> tv = t;
        RadArgs a;
        std::memset(&a, 0, sizeof(a));
        a.t = tv.data(); a.n_t = (int)tv.size();
        a.p = ps.data(); a.n_p = (int)ps.size();
        a.world_size = 1;
        if (r->scratch) {
            const int64_t need = r->scratch(&a);
            if (need > 0) { a.scratch = dalloc((size_t)need); a.scratch_bytes = need; }
        }
        const int rc = r->launch(&a, (RadStream) nullptr);
        if (rc != RAD_OK || st_stream_sync(nullptr) != hipSuccess) return false;
        out->resize((size_t)(Tr * N));
        st_memcpy(out->data(), d_y, out->size() * 2, hipMemcpyDeviceToHost);
        return true;
    };
    std::vector<uint16_t> ref;
    if (!run(ts, &ref)) { ok(what.c_str(), false, "the plain launch failed"); return; }

    /* The control block. */
    auto L = std::make_unique<RadW2Layer>();
    std::memset(L.get(), 0, sizeof(RadW2Layer));
    size_t slot = 0;
    for (int64_t e = 0; e < E; ++e) slot = std::max(slot, wbuf(e).second);
    slot = (slot + 4095) / 4096 * 4096;
    void* d_ring = dalloc(slot * RAD_W2_MAX_COPY);
    void* d_done = dalloc(64);
    void* d_stat = dalloc(sizeof(RadW2Stats));
    void* d_L = dalloc(sizeof(RadW2Layer));
    if (!d_ring || !d_done || !d_stat || !d_L) { ok(what.c_str(), false, "hipMalloc failed"); return; }
    const uint32_t seq = 7;
    int n_copy = 0, n_miss = 0, n_res = 0, turn = 0;
    int64_t e_copy0 = -1;
    for (int64_t e = 0; e < E; ++e) {
        if (off[(size_t)e + 1] <= off[(size_t)e]) continue;
        /* at most RAD_W2_TAIL_EXPERTS misses: the tail's grid holds no more (the plan's rule) */
        const int kind = n_miss < RAD_W2_TAIL_EXPERTS ? turn++ % 3 : 0;
        if (kind == 0) { ++n_res; continue; }
        ++n_miss;
        if (kind == 2 && n_copy < RAD_W2_MAX_COPY) {
            const auto wb = wbuf(e);
            void* dst = (char*)d_ring + (size_t)n_copy * slot;
            st_memcpy(dst, wb.first, wb.second, hipMemcpyDeviceToDevice);
            L->copy[n_copy].base = (uint64_t)(uintptr_t)wb.first;
            L->copy[n_copy].dst = (uint64_t)(uintptr_t)dst;
            L->copy[n_copy].bytes = wb.second;
            if (n_copy == 0) e_copy0 = e;
            L->mask[e] = (uint8_t)(RAD_W2_COPY0 + n_copy++);
        } else {
            L->mask[e] = RAD_W2_INPLACE;
        }
    }
    if (!n_res || !n_copy) return;                       /* too few routed experts to split */
    L->magic = RAD_W2_MAGIC;
    L->ring = (uint64_t)(uintptr_t)d_ring;
    L->slot = slot;
    L->done = (const uint32_t*)d_done;
    L->stats = (RadW2Stats*)d_stat;
    L->n_expert = (uint32_t)E;
    L->seq = seq;
    L->n_miss = (uint32_t)n_miss;
    L->n_copy = (uint32_t)n_copy;
    const RadW2Stats z{};
    auto put = [&](uint32_t done) {
        st_memcpy(d_L, L.get(), sizeof(RadW2Layer), hipMemcpyHostToDevice);
        st_memcpy(d_done, &done, 4, hipMemcpyHostToDevice);
        st_memcpy(d_stat, &z, sizeof z, hipMemcpyHostToDevice);
    };
    put(RAD_W2_DONE(seq, n_copy));                        /* every unit of the list has landed */
    RadTensor none;
    std::memset(&none, 0, sizeof none);
    std::vector<RadTensor> t1 = ts, t2 = ts;
    t1.push_back(none);                                   /* split */
    t1.push_back(T(d_L, RAD_U8, { (int64_t)sizeof(RadW2Layer) }));
    t2.push_back(none);
    t2.push_back(none);                                   /* w2 */
    t2.push_back(T(d_L, RAD_U8, { (int64_t)sizeof(RadW2Layer) }));

    auto rows_differ = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, int64_t r0,
                           int64_t r1) {
        return std::memcmp(a.data() + r0 * N, b.data() + r0 * N, (size_t)((r1 - r0) * N) * 2) != 0;
    };
    const int64_t live = off[(size_t)E];
    std::vector<uint16_t> y1, y2;
    if (!run(t1, &y1)) { ok(what.c_str(), false, "the first launch failed"); return; }
    /* Did the form read the operand? A miss's rows are then as they were. */
    int64_t bad_res = 0, bad_miss = 0;
    for (int64_t e = 0; e < E; ++e) {
        const int64_t r0 = off[(size_t)e], r1 = off[(size_t)e + 1];
        if (r1 <= r0) continue;
        if (L->mask[e] == RAD_W2_RESIDENT) { bad_res += rows_differ(y1, ref, r0, r1); continue; }
        bool touched = false;
        for (int64_t i = r0 * N; i < r1 * N && !touched; ++i) touched = y1[(size_t)i] != poison;
        bad_miss += touched;
    }
    if (bad_miss == n_miss) {                             /* every expert ran: not the decode grid */
        ++g_w2_whole;
        y2 = y1;
        const bool ok1 = y1 == ref, ok2 = run(t2, &y2) && y2 == ref;
        ok((what + " operands on a form that does not split: the plain bytes twice").c_str(), ok1 && ok2);
        return;
    }
    ++g_w2_split;
    const bool tail_zero = live >= Tr || !rows_differ(y1, ref, live, Tr);
    ok((what + fmt(" first launch: %d resident experts' rows and the dropped rows written, %d misses' "
                   "rows untouched", n_res, n_miss)).c_str(),
       bad_res == 0 && bad_miss == 0 && tail_zero,
       fmt("%lld resident experts differ, %lld misses touched, dropped rows %s", (long long)bad_res,
           (long long)bad_miss, tail_zero ? "ok" : "differ"));
    y2 = y1;
    const bool ran = run(t2, &y2);
    ok((what + fmt(" tail: every row the plain launch's bytes (%d units from the ring, %d in place)",
                   n_copy, n_miss - n_copy)).c_str(), ran && y2 == ref);
    if (token_order) {
        RadW2Stats st{};
        st_memcpy(&st, d_stat, sizeof st, hipMemcpyDeviceToHost);
        ok((what + " tail: each posted unit counted once, as landed").c_str(),
           st.landed == (uint32_t)n_copy && st.late == 0,
           fmt("landed %u late %u of %d", st.landed, st.late, n_copy));
    }
    /* The ring is what a landed copy's expert is read from: invert slot 0's first bytes. */
    if (e_copy0 >= 0) {
        const size_t nb = std::min<size_t>(wbuf(e_copy0).second, 4096);
        std::vector<uint8_t> orig(nb), inv(nb);
        st_memcpy(orig.data(), d_ring, nb, hipMemcpyDeviceToHost);
        for (size_t i = 0; i < nb; ++i) inv[i] = (uint8_t)~orig[i];
        st_memcpy(d_ring, inv.data(), nb, hipMemcpyHostToDevice);
        put(RAD_W2_DONE(seq, n_copy));
        std::vector<uint16_t> y3 = y1, y4 = y1;
        const bool ran3 = run(t2, &y3);
        bool only = ran3, hit = false;
        for (int64_t e = 0; e < E && ran3; ++e) {
            const int64_t r0 = off[(size_t)e], r1 = off[(size_t)e + 1];
            if (r1 <= r0) continue;
            const bool d = rows_differ(y3, ref, r0, r1);
            if (e == e_copy0) hit = d;
            else if (d) only = false;
        }
        ok((what + " tail: a landed copy is read from the ring (its inverted bytes change that expert "
                   "alone)").c_str(), ran3 && hit && only);
        /* The list's first unit alone has landed: that expert from the ring (its inverted bytes),
         * every other in place. */
        if (n_copy > 1) {
            put(RAD_W2_DONE(seq, 1));
            std::vector<uint16_t> y5 = y1;
            const bool ran5 = run(t2, &y5);
            ok((what + " tail: of a list still copying, the units that landed are read from the ring").c_str(),
               ran5 && y5 == y3);
        }
        put(RAD_W2_DONE(seq + 1, n_copy));                /* another post's word: nothing of this one */
        const bool ran4 = run(t2, &y4);
        ok((what + " tail: a copy that has not landed is read in place").c_str(), ran4 && y4 == ref);
        if (token_order) {
            RadW2Stats st{};
            st_memcpy(&st, d_stat, sizeof st, hipMemcpyDeviceToHost);
            ok((what + " tail: each posted unit counted once, as late").c_str(),
               st.late == (uint32_t)n_copy && st.landed == 0,
               fmt("landed %u late %u of %d", st.landed, st.late, n_copy));
        }
        st_memcpy(d_ring, orig.data(), nb, hipMemcpyHostToDevice);
    }
}
