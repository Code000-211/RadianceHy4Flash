/* derive.cpp -- the derivation table, one row per name.
 */
#include "derive.h"

#include "rad_core.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rad {

/* ------------------------------------------------------------------ seq_start_index */
/* [n_seq+1] -- where each sequence's tokens begin in this step's flat token array, and the total.
 * A copy of cu_seqlens, so a plugin can name it as an operand instead of reaching into RadBatch. */
static int d_seq_start_index(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    int64_t n = in.n_seq + 1;
    if (n > cap) return RAD_E_FULL;
    std::memcpy(out, in.cu_seqlens, (size_t)n * sizeof(int32_t));
    *n_out = n;
    return RAD_OK;
}

/* ------------------------------------------------------------------ conv_state_index */
/* [n_seq] -- the conv state cache entry this sequence's update reads from.
 *
 * The conv state is a rolling window of `width-1 + num_spec` entries read at the slot the last
 * accepted token left, so a speculative rejection moves the read offset instead of recomputing
 * (spec §10). Base is the sequence's state slot; offset is the cursor the scheduler advances by
 * 1 + num_accepted per verify. Both live in request bookkeeping, hence a scheduler derivation. */
static int d_conv_state_index(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    if (in.n_seq > cap) return RAD_E_FULL;
    if (!in.conv_slot) return RAD_E_STATE;
    const int32_t W = in.conv_window > 0 ? in.conv_window : 1;
    for (int64_t i = 0; i < in.n_seq; ++i) {
        const StepEntry& e = in.plan->e[(size_t)i];
        int32_t slot = in.conv_slot[i];
        out[i] = slot < 0 ? -1 : (int32_t)(slot * W + (e.conv_cursor % W));
    }
    *n_out = in.n_seq;
    return RAD_OK;
}

/* ------------------------------------------------------- the linear-state chunk walk */
/* Chunked linear-attention kernels tile each sequence in fixed chunks from the sequence's own
 * origin, not the step's token array. A chunk never crosses a sequence boundary, and interior
 * boundaries land on absolute multiples of the chunk length, or the carried recurrent state would
 * be that of a partial tile. Prefill chunks already end on the quantum (geometry.h); the walk is
 * still needed because a mixed step's decode rows start at arbitrary positions. */
template <bool WantSeq>
static int chunk_walk(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    const int64_t C = in.state_chunk > 0 ? in.state_chunk : 1;
    int64_t k = 0;

    if (!WantSeq) {
        if (cap < 1) return RAD_E_FULL;
        out[k++] = 0;
    }
    for (int64_t i = 0; i < in.n_seq; ++i) {
        int64_t b = in.cu_seqlens[i], e = in.cu_seqlens[i + 1];
        if (e <= b) continue;
        int64_t p0 = in.positions[b];
        /* First interior boundary: the next absolute multiple of C strictly after p0. */
        int64_t next = ((p0 / C) + 1) * C;
        for (int64_t abs = next; abs < p0 + (e - b); abs += C) {
            if (k >= cap) return RAD_E_FULL;
            out[k++] = WantSeq ? (int32_t)i : (int32_t)(b + (abs - p0));
        }
        if (k >= cap) return RAD_E_FULL;
        out[k++] = WantSeq ? (int32_t)i : (int32_t)e;
    }
    *n_out = k;
    return RAD_OK;
}

/* [n_chunks + 1] token indices: chunk c covers [out[c], out[c+1]). out[0] is 0 and the last entry is
 * n_tok, like cu_seqlens. */
static int d_state_chunk_bounds(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    return chunk_walk<false>(in, out, cap, n_out);
}

/* [n_chunks] -- which sequence each chunk belongs to. Derivable from the bounds, but one shared
 * pass here saves a per-chunk search in every layer. */
static int d_state_chunk_seq(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    return chunk_walk<true>(in, out, cap, n_out);
}

/* ------------------------------------------------------- projection_on_rows */
/* [n_tok] -- whether each token's residual stream takes the control vector this step.
 *
 * The switch itself is a per-request choice (SamplingParams::use_control_vector) and the step
 * kernel reads one row's flag, so the mask is expanded here once per step from the plan rather
 * than searched per layer: an architecture issues one projection op after every block's write, and
 * a per-layer search over the sequence bounds would repeat on every one of them.
 *
 * All zeros is the ordinary model, byte for byte: the op skips a zero row without writing, so a
 * request that did not ask for the vector computes what it computes with no vector loaded. */
static bool                 g_cvec_cast_enabled     = false;
static int64_t              g_cvec_cast_think_after = 24;
static std::vector<int32_t> g_cvec_cast_route_slots;

void cvec_set_cast_mode(bool enabled, int64_t think_after) {
    g_cvec_cast_enabled     = enabled;
    g_cvec_cast_think_after = std::max<int64_t>(0, think_after);
    g_cvec_cast_route_slots.clear();
}

std::vector<int32_t> cvec_take_cast_route_slots() {
    std::vector<int32_t> out;
    out.swap(g_cvec_cast_route_slots);
    return out;
}

CvecGate cvec_gate_from_env() {
    static const CvecGate env_g = []() {
        CvecGate out;
        if (const char* e = std::getenv("RADIANCE_CVEC_THINK_ONLY"))
            out.think_only = *e && std::strcmp(e, "0") != 0;
        if (const char* e = std::getenv("RADIANCE_CVEC_OUTPUT_ONLY"))
            out.output_only = *e && std::strcmp(e, "0") != 0;
        if (const char* e = std::getenv("RADIANCE_CVEC_PREFILL_LAST_ONLY"))
            out.prefill_last_only = *e && std::strcmp(e, "0") != 0;
        if (const char* e = std::getenv("RADIANCE_CVEC_THINK_AFTER"))
            if (*e) out.think_after = std::max<int64_t>(0, std::strtoll(e, nullptr, 10));
        if (const char* e = std::getenv("RADIANCE_CVEC_THINK_RAMP"))
            if (*e) out.think_ramp = std::max<int64_t>(0, std::strtoll(e, nullptr, 10));
        return out;
    }();
    CvecGate g = env_g;
    g.cast_mode        = g_cvec_cast_enabled;
    g.cast_think_after = g_cvec_cast_think_after;
    return g;
}

static ReasoningCount::State cvec_think_state(const Request* r, int64_t* body_out) {
    ReasoningCount rc;
    rc.start(r->sp.reasoning);
    for (int32_t tok : r->output) {
        rc.push(r->sp.reasoning, tok);
        if (rc.st == ReasoningCount::Done) break;
    }
    if (body_out) *body_out = rc.body;
    return rc.st;
}

bool cvec_req_active(const Request* r, const CvecGate& g) {
    if (!r || !r->sp.use_control_vector) return false;
    if (g.cast_mode) return true;
    if (g.output_only) {
        if (!r->sp.reasoning.known()) return true;
        const auto st = cvec_think_state(r, nullptr);
        return st == ReasoningCount::Done;
    }
    if ((!g.think_only && g.think_after <= 0 && g.think_ramp <= 0) || !r->sp.reasoning.known())
        return true;
    int64_t body = 0;
    const auto st = cvec_think_state(r, &body);
    if (st == ReasoningCount::Done) return false;
    if (g.think_after > 0 && body < g.think_after) return false;
    return true;
}

bool cvec_token_active(const Request* r, int64_t pos, const CvecGate& g) {
    if (!r) return false;
    const int64_t prompt_len = (int64_t) r->prompt.size();
    if (g.cast_mode) return pos >= prompt_len - 1;
    /* OUTPUT_ONLY, THINK_AFTER > 0, and THINK_RAMP > 0 leave all prompt/prefill tokens
     * (pos < prompt_len) unsteered even if recomputed after preemption. */
    if (g.output_only && pos < prompt_len) return false;
    if ((g.think_after > 0 || g.think_ramp > 0) && r->sp.reasoning.known() && pos < prompt_len)
        return false;
    if (g.prefill_last_only && pos < prompt_len - 1) return false;
    return true;
}

int32_t cvec_row_flag(const Request* r, int64_t pos, const CvecGate& g) {
    if (!cvec_token_active(r, pos, g)) return 0;
    if (g.cast_mode) {
        const int64_t prompt_len = (int64_t) r->prompt.size();
        if (pos == prompt_len - 1) return 1;  /* ROUTE_PREFILL_LAST */
        if (!r->sp.reasoning.known()) return 4;  /* OUTPUT (reasoning_effort="none") */
        int64_t body = 0;
        if (cvec_think_state(r, &body) == ReasoningCount::Done) return 4;  /* OUTPUT */
        const int64_t committed_last = prompt_len + (int64_t) r->output.size() - 1;
        const int64_t extra = std::max<int64_t>(0, pos - committed_last);
        return ((body + extra) < g.cast_think_after) ? 2 : 3;  /* THINK_EARLY / THINK_LATE */
    }
    if (g.output_only || g.think_ramp <= 0 || !r->sp.reasoning.known()) return 1;
    int64_t body = 0;
    if (cvec_think_state(r, &body) == ReasoningCount::Done) return 0;
    /* Advance body offset for speculative draft tokens beyond the last committed output token. */
    const int64_t committed_last = (int64_t) r->prompt.size() + (int64_t) r->output.size() - 1;
    const int64_t extra = std::max<int64_t>(0, pos - committed_last);
    const int64_t into = (body + extra) - g.think_after;
    if (into < 0) return 0;
    if (into >= g.think_ramp) return 1;
    const double frac = (double) (into + 1) / (double) (g.think_ramp + 1);
    const int32_t milli = (int32_t) std::clamp<long long>(std::llround(frac * 10000.0), 1LL, 9999LL);
    return -milli;
}

bool cvec_entry_has_active_rows(const StepEntry& e, const CvecGate& g) {
    if (!e.req || e.n_tokens <= 0 || !cvec_req_active(e.req, g)) return false;
    const int64_t end_pos = (int64_t) e.ctx_len + (int64_t) e.n_tokens - 1;
    return cvec_token_active(e.req, end_pos, g);
}

static int d_projection_on_rows(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* n_out) {
    if (in.n_tok > cap) return RAD_E_FULL;
    /* Zero first. The ranges below cover every row a step carries today, but the region is reused
     * across steps and a row outside them would otherwise keep the previous step's flag -- which
     * would steer a request that did not ask for it. The fill is a memset of a few KiB. */
    std::memset(out, 0, (size_t) in.n_tok * sizeof(int32_t));
    const CvecGate g = cvec_gate_from_env();
    static const bool cast_trace = []() {
        const char* e = std::getenv("RADIANCE_CAST_TRACE");
        return e && *e && e[0] != '0';
    }();
    for (int64_t i = 0; i < in.n_seq; ++i) {
        const StepEntry& e = in.plan->e[(size_t) i];
        if (!cvec_req_active(e.req, g)) continue;
        const int64_t b = in.cu_seqlens[i], end = in.cu_seqlens[i + 1];
        const int32_t slot = e.card >= 0 ? e.card : (e.slot >= 0 ? e.slot : 0);
        for (int64_t k = b; k < end && k < in.n_tok; ++k) {
            const int64_t pos = (int64_t) e.ctx_len + (k - b);
            const int32_t f = cvec_row_flag(e.req, pos, g);
            if (g.cast_mode && f > 0) {
                out[k] = (1 << 20) | ((slot & 0xFFF) << 8) | (f & 0xF);
                if (cast_trace && f == 1) g_cvec_cast_route_slots.push_back(slot);
            } else {
                out[k] = f;
            }
        }
    }
    *n_out = in.n_tok;
    return RAD_OK;
}

/* Queries keep their order within each sequence, so a compact suffix attends to precisely
 * the same keys (the causal kernels align Q to the end of K). Scoring can repeat sampled
 * rows and append them out of order. Leave the logits gather's original indices intact. */
template<bool anchors>
static int d_prefill_tail(const DeriveInput& in, int32_t* out, int64_t cap, int64_t* produced) {
    *produced = 0;
    const auto& p = *in.plan;
    if (p.phase == RAD_PHASE_DECODE || p.draft_pass || p.n_seq_decode == in.n_seq) return RAD_OK;
    if (cap < 1) return RAD_E_FULL;
    *produced = 1;
    out[0] = 0;
    if (!in.n_out) return RAD_OK;
    std::vector<uint8_t> keep(in.n_tok, 0);
    for (int64_t j = 0; j < in.n_out; ++j) {
        const int32_t r = in.out_ids[j];
        if (r < 0 || r >= in.n_tok) return RAD_E_INVAL;
        keep[r] = 1;
    }
    /* Every decode row, sampled or not: the compact pass reads its rows [0, DT) as the batch's
     * decode rows (block table, seqused, the stable instance), so they stay whole and in place
     * in both variants (the plugin checks it: hyv4flash.cpp, tail_keeps_decode). */
    std::fill_n(keep.begin(), p.n_tok_decode, 1);
    if (anchors) {
        for (int64_t s = p.n_seq_decode; s < in.n_seq; ++s)
            if (in.cu_seqlens[s + 1] > in.cu_seqlens[s])
                keep[in.cu_seqlens[s + 1] - 1] = 1;
    }
    const int64_t m = std::count(keep.begin(), keep.end(), 1);
    out[0] = m;
    if (m == in.n_tok) return RAD_OK;
    const int64_t size = 4 + 2 * in.n_seq - p.n_seq_decode + (in.rope_pos ? 5 : 2) * m;
    if (size + (in.ckpt_cut ? 3 : 0) > cap) return RAD_E_FULL;
    int32_t* cu = out + 2;
    int32_t* pf_cu = cu + in.n_seq + 1;
    int32_t* ids = pf_cu + in.n_seq - p.n_seq_decode + 1;
    int32_t* pos = ids + m;
    cu[0] = 0;
    int n = 0;
    for (int64_t s = 0; s < in.n_seq; ++s) {
        for (int r = in.cu_seqlens[s]; r < in.cu_seqlens[s + 1]; ++r) {
            if (!keep[r]) {
                if (n != cu[s]) { out[0] = in.n_tok; return RAD_OK; }
                continue;
            }
            ids[n] = r;
            pos[n] = in.positions[r];
            if (in.rope_pos)
                for (int j = 0; j < 3; ++j)
                    pos[(j + 1) * m + n] = in.rope_pos[j * in.n_tok + r];
            ++n;
        }
        cu[s + 1] = n;
    }
    out[1] = cu[p.n_seq_decode];
    for (int64_t i = p.n_seq_decode; i <= in.n_seq; ++i)
        pf_cu[i - p.n_seq_decode] = cu[i] - out[1];
    *produced = size;
    if (in.ckpt_cut) {
        /* A checkpoint cut belongs to a single prefill sequence. Translate it from
         * original rows into that sequence's retained suffix. */
        out[size] = 0;
        out[size + 1] = std::clamp<int64_t>(in.ckpt_cut - ids[0], 0, m);
        out[size + 2] = m;
        *produced += 3;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the table */
static const DeriveRow g_rows[] = {
    { "prefill_tail", d_prefill_tail<false>, "compact prefill query suffixes" },
    { "prefill_tail_anchors", d_prefill_tail<true>, "compact query suffixes including draft anchors" },
    { "seq_start_index",  d_seq_start_index,
      "[n_seq+1] first token index of each sequence in this step, and the total" },
    { "conv_state_index", d_conv_state_index,
      "[n_seq] conv rolling-window entry to read, base slot * window + accepted cursor" },
    { "state_chunk_bounds", d_state_chunk_bounds,
      "[n_chunks + 1] token indices bounding each linear-state chunk; never crosses a sequence" },
    { "state_chunk_seq",    d_state_chunk_seq,
      "[n_chunks] the sequence each chunk belongs to" },
    { "projection_on_rows", d_projection_on_rows,
      "[n_tok] 1 where this token's request applies the loaded control vector, 0 for the stock "
      "model" },
};

int derive_index(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < (int)(sizeof g_rows / sizeof g_rows[0]); ++i)
        if (std::strcmp(g_rows[i].name, name) == 0) return i;
    return -1;
}

const DeriveRow* derive_row(int i) {
    return (i >= 0 && i < (int)(sizeof g_rows / sizeof g_rows[0])) ? &g_rows[i] : nullptr;
}

int derive_count() { return (int)(sizeof g_rows / sizeof g_rows[0]); }

}  /* namespace rad */
