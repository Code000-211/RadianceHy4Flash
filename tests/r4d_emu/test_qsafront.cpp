/* tests/r4d_emu: the fused key side -- qsa_work_pool_tail (K-24) -- against the three ops it
 * replaces, on the host SIMT emulator: qsa_work, then idx_pool_gated, then qsa_tail_store, which is
 * what the plugin issues for the same step before the fusion.
 *
 *   A: qsa_work -> idx_pool_gated -> qsa_tail_store          (the three ops)
 *   B: qsa_work_pool_tail [-> qsa_tail_store]                (the fused op, and the tail store when
 *                                                             the plugin keeps it)
 *
 * Nothing in the fused op is approximate: it is the same gather, the same shared pool arithmetic
 * (r4d_idx_pool_body.h) and the same row copy, so the comparison is byte for byte -- every element
 * of every buffer both sides touch, not a tolerance. test_idxpool.cpp holds the arithmetic itself
 * to libref; this file holds the fusion to the three-op path.
 *
 * The one asymmetry is the tail, and it is the item's alias proof: the fused op stores a chunk's
 * rows itself only while the chunk's row count stays inside ring - ratio + 1 (a longer chunk's own
 * store could rewrite a row its own gather still has to read). The plugin decides the same bound
 * from T and S, so the fixtures run both sides of it:
 *
 *  - decode: one token a sequence, positions that close a group and positions that close none, so
 *    one or three of the ops have nothing to do -- the case the fusion is for;
 *  - a verify chunk exactly at the bound (rows == ring - ratio + 1): the fused op stores its rows
 *    and no qsa_tail_store follows;
 *  - a prefill chunk past the bound, alone and beside a short sequence: the fused op skips the long
 *    chunk's rows and qsa_tail_store runs, and the short chunk is stored by both with the same
 *    bytes, which the comparison of the final state pins down;
 *  - a dead tail slot, a dead key page and a page past the cache: no gather, or a gather with
 *    nothing to write;
 *  - M-RoPE positions, both when a group's first token is in this step and when it is not (its
 *    rotary components come from the plane's first row then);
 *  - a step with no key pages at all (cache absent), and both width forms (W < 16 one group a
 *    workgroup, W >= 16 four).
 *
 * No libref: the three ops it would compare against are qsa_work / idx_pool_gated / qsa_tail_store
 * themselves, emulated here. */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t kN = 128;      /* the indexer's key width, compiled in */
constexpr int64_t kRatio = 4;    /* tokens a group */

static size_t esz(uint32_t dt) {
    switch (dt) {
        case RAD_F32: case RAD_I32: return 4;
        case RAD_BF16: return 2;
        case RAD_I64: return 8;
        default: return 1;
    }
}

static RadParam pi(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
static RadParam pf(const char* k, double v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_F64;
    p.dval = v;
    return p;
}
static RadParam ps(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

/* A buffer of a fixture: what the op sees, and whether both sides write it (OUT), only read it
 * (IN) or neither (SKIP -- `stage` exists in the three-op path only). */
enum Kind { IN, OUT, SKIP };

struct Opd {
    uint32_t dt = RAD_BF16;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
    Kind kind = IN;
    int64_t elements() const {
        int64_t n = 1;
        for (int64_t e : shape) n *= e;
        return n;
    }
    RadTensor tensor(void* data) const {
        RadTensor t{};
        t.data = data;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t st = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[(size_t)i];
            t.stride[i] = st;
            st *= shape[(size_t)i];
        }
        return t;
    }
};

static Opd opd(uint32_t dt, std::vector<int64_t> shape, Kind kind = IN) {
    Opd o;
    o.dt = dt;
    o.shape = std::move(shape);
    o.kind = kind;
    o.bytes.assign((size_t)o.elements() * esz(dt), 0);
    return o;
}

static uint16_t f2b(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

/* A key that moves with both indices: two rows of a weight never agree, and no two groups' keys
 * collide by accident. */
static double sample(int64_t r, int64_t c, int64_t salt) {
    const uint32_t h = (uint32_t)(r * 1315423911u + c * 2654435761u + (uint32_t)salt * 40503u);
    const uint32_t m = (h ^ (h >> 15)) * 2246822519u;
    return (double)((int32_t)(m >> 8) % 2000) / 1000.0 - 0.5;
}

static void put(Opd& o, int64_t off, double v) {
    switch (o.dt) {
        case RAD_BF16: ((uint16_t*)o.bytes.data())[off] = f2b((float)v); break;
        case RAD_F32: ((float*)o.bytes.data())[off] = (float)v; break;
        case RAD_I32: ((int32_t*)o.bytes.data())[off] = (int32_t)v; break;
        default: o.bytes[(size_t)off] = (uint8_t)(int)v; break;
    }
}

static void fill_tensor(Opd& o, int64_t salt) {
    /* an absent or empty operand (the rotary plane of a gptj cell) */
    if (o.elements() <= 0) return;
    const RadTensor t = o.tensor(o.bytes.data());
    std::vector<int64_t> ix((size_t)t.rank, 0);
    for (;;) {
        int64_t off = 0;
        for (size_t i = 0; i < ix.size(); ++i) off += ix[i] * t.stride[i];
        double s = 0;
        for (size_t i = 0; i < ix.size(); ++i) s = s * 7.0 + sample(ix[i], (int64_t)i, salt);
        put(o, off, s);
        int64_t d = (int64_t)ix.size() - 1;
        for (; d >= 0; --d)
            if (++ix[(size_t)d] < t.shape[d]) break;
            else ix[(size_t)d] = 0;
        if (d < 0) break;
    }
}

/* ---- a fixture ------------------------------------------------------------------------------ */

enum {
    O_K = 0, O_POS, O_CU, O_STATE, O_SIDX, O_BT, O_WV, O_WG, O_APE, O_GAIN, O_ROPE, O_STAGE,
    O_PAGE, O_BPOS, O_NC, O_CACHE, O_POOL_OUT, O_COUNT
};

struct Sc {
    const char* what = "";
    int64_t seqs = 1, ring = 8, slots = 4, pages = 8, rd = 128;
    std::vector<int64_t> qlen;   /* tokens a sequence */
    std::vector<int64_t> p0;     /* the sequence's first position */
    std::vector<int64_t> slot;   /* its tail slot, -1 for none */
    bool imrope = false;
    bool cache = true;
    bool tail_op = false;        /* the plugin also issues qsa_tail_store after the fused op */
};

struct Call {
    RadLaunchFn           fn;
    std::vector<int>      ix;   /* the fixture buffer of each operand, -1 for an absent one */
    std::vector<RadParam> p;
};

/* The work count the plugin declares: one item a sequence at least, and every block a step can
 * close on top of that. */
static int64_t work_of(const Sc& s) {
    int64_t T = 0;
    for (int64_t q : s.qlen) T += q;
    return (T + kRatio - 1) / kRatio + s.seqs;
}

static std::vector<Opd> build(const Sc& s) {
    int64_t T = 0;
    for (int64_t q : s.qlen) T += q;
    const int64_t W = work_of(s);
    /* The block table's columns are block *positions* (`b = pos / ratio`), not work items: a
     * sequence whose first position is well past 0 opens no block below it. */
    int64_t btcols = W;
    for (size_t i = 0; i < s.qlen.size(); ++i)
        btcols = std::max<int64_t>(btcols, (s.p0[i] + s.qlen[i] + kRatio) / kRatio + 1);
    std::vector<Opd> o(O_COUNT);
    o[O_K]     = opd(RAD_BF16, { T, kN });
    o[O_POS]   = opd(RAD_I32, { T });
    o[O_CU]    = opd(RAD_I32, { s.seqs + 1 });
    o[O_STATE] = opd(RAD_BF16, { s.slots, 1, s.ring, kN }, OUT);
    o[O_SIDX]  = opd(RAD_I32, { s.seqs });
    o[O_BT]    = opd(RAD_I32, { s.seqs, btcols });
    o[O_WV]    = opd(RAD_F32, { kN, kN });
    o[O_WG]    = opd(RAD_F32, { kN, kN });
    o[O_APE]   = opd(RAD_F32, { kRatio, kN });
    o[O_GAIN]  = opd(RAD_F32, { kN });
    o[O_ROPE]  = opd(RAD_I32, { 3, s.imrope ? T : 0 });
    o[O_STAGE] = opd(RAD_BF16, { W * kRatio, kN }, SKIP);
    o[O_PAGE]  = opd(RAD_I32, { W }, OUT);
    o[O_BPOS]  = opd(RAD_I32, s.imrope ? std::vector<int64_t>{ W, 3 } : std::vector<int64_t>{ W },
                     OUT);
    o[O_NC]    = opd(RAD_I32, { s.seqs }, OUT);
    o[O_CACHE] = opd(RAD_BF16, { s.cache ? s.pages : 0, 1, kRatio, kN / kRatio }, OUT);
    /* The pool needs one of `cache`/`out` to write; with no key pages the fixture hands it `out`
     * and the case checks the work list alone. The fused op has no such operand, so neither side
     * is compared on it. */
    o[O_POOL_OUT] = opd(RAD_BF16, { s.cache ? 0 : W, kN }, SKIP);
    for (int i = 0; i < O_COUNT; ++i)
        if (i != O_PAGE && i != O_BPOS && i != O_NC)
            fill_tensor(o[i], 11 + i);
    /* The page, the position and the count buffers start as stale bytes: an op must write every
     * work item and every sequence, not only the live ones. */
    for (int i : { (int)O_PAGE, (int)O_BPOS, (int)O_NC })
        for (size_t j = 0; j < o[i].bytes.size(); ++j) o[i].bytes[j] = (uint8_t)(0x31 + (j * 11));

    {
        auto* cu = (int32_t*)o[O_CU].bytes.data();
        auto* po = (int32_t*)o[O_POS].bytes.data();
        int64_t t = 0;
        for (int64_t sq = 0; sq < s.seqs; ++sq) {
            cu[sq] = (int32_t)t;
            for (int64_t i = 0; i < s.qlen[(size_t)sq]; ++i)
                po[t + i] = (int32_t)(s.p0[(size_t)sq] + i);
            t += s.qlen[(size_t)sq];
        }
        cu[s.seqs] = (int32_t)t;
        auto* sx = (int32_t*)o[O_SIDX].bytes.data();
        for (int64_t sq = 0; sq < s.seqs; ++sq) sx[sq] = (int32_t)s.slot[(size_t)sq];
        /* The block table: a page for every block a sequence can close, one entry dead by value
         * (-1) and one past the cache, so the pool skips a cache write the gather still does. */
        auto* bt = (int32_t*)o[O_BT].bytes.data();
        for (int64_t sq = 0; sq < s.seqs; ++sq)
            for (int64_t b = 0; b < btcols; ++b) {
                int32_t v = (int32_t)((b * 3 + sq * 5 + 1) % s.pages);
                if (b == 6) v = -1;                      /* a group with no page at all */
                if (b == 7) v = (int32_t)(s.pages + 3);  /* a page past the cache */
                bt[sq * btcols + b] = v;
            }
        /* The rotary plane: a component per token that is not the position, so an op that reads
         * the wrong row is visible. */
        auto* rp = (int32_t*)o[O_ROPE].bytes.data();
        const int64_t pitch = s.imrope ? T : 0;
        for (int64_t j = 0; j < 3; ++j)
            for (int64_t i = 0; i < pitch; ++i) rp[j * pitch + i] = (int32_t)(j * 1000 + 7 * i + 3);
    }
    return o;
}

/* The `sections` text a call hands to an entry has to outlive the launch, so it is kept here
 * rather than in a temporary (only one case runs at a time). */
static std::string g_sections;

static std::vector<RadParam> key_params(const Sc& s, bool fused) {
    int64_t T = 0;
    for (int64_t q : s.qlen) T += q;
    std::vector<RadParam> p = { pi("M", T), pi("n", kN), pi("ratio", kRatio), pi("ring", s.ring),
                                pi("seqs", s.seqs), pi("work", work_of(s)), pi("rotary_dim", s.rd),
                                pf("eps", 1e-5), pf("theta", 10000.0),
                                ps("mode", s.imrope ? "imrope_gptj" : "gptj") };
    if (s.imrope) {
        /* Three sections summing to rd / 2; the middle one carries them all. */
        g_sections = "0 " + std::to_string(s.rd / 2) + " 0";
        p.push_back(ps("sections", g_sections.c_str()));
    }
    p.push_back(pi("block_size", kRatio));
    p.push_back(ps("dtype", "bf16"));
    (void)fused;
    return p;
}

static std::vector<Call> unfused(const Sc& s) {
    std::vector<Call> v;
    Call w;
    w.fn = r4d_rad_qsa_work;
    w.ix = { O_K, O_POS, O_CU, O_STATE, O_SIDX, O_BT, s.imrope ? O_ROPE : -1, O_STAGE, O_PAGE,
             O_BPOS, O_NC };
    w.p = key_params(s, false);
    v.push_back(w);
    Call p;
    p.fn = r4d_rad_idx_pool_gated;
    /* The pool takes `page` and `cache` together or neither (its entry's rule): with no key
     * pages there is nothing to select. */
    p.ix = { O_STAGE, O_WV, O_WG, O_APE, O_GAIN, O_BPOS, s.cache ? O_PAGE : -1,
             s.cache ? O_CACHE : -1, s.cache ? -1 : O_POOL_OUT };
    p.p = key_params(s, false);
    v.push_back(p);
    Call t;
    t.fn = r4d_rad_qsa_tail_store;
    t.ix = { O_K, O_POS, O_CU, O_STATE, O_SIDX };
    t.p = key_params(s, false);
    v.push_back(t);
    return v;
}

static std::vector<Call> fused(const Sc& s) {
    std::vector<Call> v;
    Call f;
    f.fn = r4d_rad_qsa_work_pool_tail;
    f.ix = { O_K, O_POS, O_CU, O_STATE, O_SIDX, O_BT, O_WV, O_WG, O_APE, O_GAIN,
             s.imrope ? O_ROPE : -1, O_PAGE, O_BPOS, O_NC, s.cache ? O_CACHE : -1 };
    f.p = key_params(s, true);
    v.push_back(f);
    if (s.tail_op) {
        Call t;
        t.fn = r4d_rad_qsa_tail_store;
        t.ix = { O_K, O_POS, O_CU, O_STATE, O_SIDX };
        t.p = key_params(s, true);
        v.push_back(t);
    }
    return v;
}

static RadTensor absent_tensor() {
    RadTensor t{};
    t.data = nullptr;
    return t;
}

static const char* op_name(RadLaunchFn fn) {
    if (fn == r4d_rad_qsa_work) return "qsa_work";
    if (fn == r4d_rad_idx_pool_gated) return "idx_pool_gated";
    if (fn == r4d_rad_qsa_tail_store) return "qsa_tail_store";
    if (fn == r4d_rad_qsa_work_pool_tail) return "qsa_work_pool_tail";
    return "an op";
}

static int run_chain(const std::vector<Opd>& fx, const std::vector<Call>& calls,
                     std::vector<std::vector<uint8_t>>& mem, const char* what) {
    mem.resize(fx.size());
    for (size_t i = 0; i < fx.size(); ++i) mem[i] = fx[i].bytes;
    for (const Call& c : calls) {
        std::vector<RadTensor> ts;
        for (int i : c.ix)
            ts.push_back(i < 0 ? absent_tensor() : fx[(size_t)i].tensor(mem[(size_t)i].data()));
        std::vector<RadParam> p = c.p;
        RadArgs a{};
        a.t = ts.data();
        a.n_t = (int)ts.size();
        a.p = p.data();
        a.n_p = (int)p.size();
        a.world_size = 1;
        const int rc = c.fn(&a, nullptr);
        if (rc != RAD_OK) {
            std::printf("FAIL %s: %s refused the fixture (rc=%d)\n", what, op_name(c.fn), rc);
            return rc;
        }
    }
    return RAD_OK;
}

static int g_cases, g_bad, g_only = -1;   /* argv[1]: run one case, to bisect a fixture */

/* Both paths over two copies of the same fixture, compared byte for byte. */
static void check(const Sc& s) {
    const int id = g_cases++;
    if (g_only >= 0 && g_only != id) return;
    const std::vector<Opd> fx = build(s);
    std::vector<std::vector<uint8_t>> a, b;
    if (run_chain(fx, unfused(s), a, s.what) != RAD_OK) { ++g_bad; return; }
    if (run_chain(fx, fused(s), b, s.what) != RAD_OK) { ++g_bad; return; }
    for (size_t i = 0; i < fx.size(); ++i) {
        if (fx[i].kind == SKIP) continue;
        if (a[i].size() != b[i].size()) {
            std::printf("FAIL %s: buffer %zu size\n", s.what, i);
            ++g_bad;
            return;
        }
        if (fx[i].kind == IN) {
            /* An input comes back as it went in, on both paths. */
            if (std::memcmp(a[i].data(), fx[i].bytes.data(), a[i].size()) ||
                std::memcmp(b[i].data(), fx[i].bytes.data(), b[i].size())) {
                std::printf("FAIL %s: input %zu changed\n", s.what, i);
                ++g_bad;
                return;
            }
            continue;
        }
        if (std::memcmp(a[i].data(), b[i].data(), a[i].size())) {
            /* The first byte that differs, with its index, so a wrong row is easy to find. */
            size_t j = 0;
            while (j < a[i].size() && a[i][j] == b[i][j]) ++j;
            std::printf("FAIL %s: output %zu differs at byte %zu (three-op %02x, fused %02x)\n",
                        s.what, i, j, a[i][j], b[i][j]);
            ++g_bad;
            return;
        }
    }
    std::printf("ok   qsa_work_pool_tail %s\n", s.what);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_only = std::atoi(argv[1]);
    /* Decode, a lone sequence, the step a group closes: two keys are still in the tail (the group's
     * first token is before this step) and the fourth is the step's own token. */
    check({ "decode 1 seq, a group closes (keys from the tail and the step)", 1, 8, 2, 8, 128,
            { 1 }, { 11 }, { 1 }, false, true, false });
    /* Decode, four sequences at positions that close nothing: all three ops find nothing to do
     * except the tail's rows. One sequence has no tail slot. */
    check({ "decode 4 seqs, nothing closes, one dead slot", 4, 8, 4, 8, 128,
            { 1, 1, 1, 1 }, { 10, 10, 10, 10 }, { 0, -1, 2, 3 }, false, true, false });
    /* A verify chunk exactly at the alias bound (5 rows == ring - ratio + 1): its rows are stored
     * by the fused op itself and no qsa_tail_store follows. */
    check({ "verify 1 seq at the bound (rows == ring - ratio + 1)", 1, 8, 2, 8, 128,
            { 5 }, { 8 }, { 1 }, false, true, false });
    /* A prefill chunk past the bound: the fused op skips the rows, qsa_tail_store stores them. */
    check({ "prefill 1 seq past the bound (qsa_tail_store follows)", 1, 8, 2, 8, 128,
            { 20 }, { 8 }, { 1 }, false, true, true });
    /* The wide form (W >= 16) past the bound: four groups a workgroup, several of them a group. */
    check({ "prefill 1 seq, wide form", 1, 8, 2, 8, 128, { 64 }, { 8 }, { 1 }, false, true, true });
    /* A chunk past the bound whose *first* group still reads the tail: this is why the bound is
     * there. Its rows are written by qsa_tail_store, after every gather of the step, and a fused
     * op that stored them itself would clobber rows (rings 0..2 here) its own first group reads. */
    check({ "prefill 1 seq past the bound, its first group reads the tail", 1, 8, 2, 8, 128,
            { 20 }, { 11 }, { 1 }, false, true, true });
    /* Seven rows over a ring of eight: the chunk's last token (position 16) shares row 0 with
     * position 8, and key 0 of the chunk's first group is position 8's -- so a fused op that
     * stored this chunk's rows itself would pool a different key than the three-op path. Both the
     * store and the gather fall in one workgroup, which makes it visible here; a store clobbering
     * a row that *another* workgroup reads is a race no emulator run can show, and it is the bound
     * written in the kernel's header, not this case, that keeps it out. */
    check({ "prefill 1 seq past the bound, its own store would take key 0's row", 1, 8, 2, 8, 128,
            { 7 }, { 10 }, { 1 }, false, true, true });
    /* A short sequence beside a long one: the short one's rows are stored here and again by
     * qsa_tail_store, with the same bytes. */
    check({ "prefill 2 seqs, short beside long", 2, 8, 4, 8, 128,
            { 2, 20 }, { 6, 100 }, { 0, 1 }, false, true, true });
    /* The wide form with the fused tail: sixteen one-token sequences. */
    check({ "decode 16 seqs, wide form, every row stored here", 16, 8, 20, 8, 128,
            { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            { 10, 11, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, 34, 37, 38 },
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, false, true, false });
    /* M-RoPE, the group's first token before this step and in it: the two branches of the
     * position rule. */
    check({ "M-RoPE, the group's first token before the step", 1, 8, 2, 8, 128,
            { 3 }, { 9 }, { 1 }, true, true, false });
    check({ "M-RoPE, the group's first token in the step", 1, 8, 2, 8, 128,
            { 5 }, { 8 }, { 1 }, true, true, false });
    /* A rotary narrower than the key, and no key pages at all. */
    check({ "narrow rotary, no cache", 1, 8, 2, 8, 64, { 5 }, { 8 }, { 1 }, false, false, false });
    check({ "prefill, no cache, past the bound", 2, 8, 4, 8, 64, { 20, 1 }, { 8, 30 }, { 0, 1 },
            false, false, true });
    /* A ring wider than the step's window (a larger speculative window). */
    check({ "decode, ring 12", 2, 12, 4, 8, 128, { 1, 1 }, { 22, 23 }, { 0, 1 }, false, true,
            false });

    std::printf("qsafront: %d cases, %d failed\n", g_cases, g_bad);
    return g_bad ? 1 : 0;
}
