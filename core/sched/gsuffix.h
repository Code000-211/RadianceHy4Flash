/* gsuffix.h -- the global suffix cache for --draft-suffix (--draft-suffix-global-mib).
 *
 * SuffixDecoding (Oliaro et al., arXiv 2411.04975) keeps two suffix trees: one over the request's
 * own prompt and output (suffix.h here) and a global one over earlier responses, which pays off
 * when responses repeat across requests (an agent's tool calls, a file rewritten each round). This
 * is the global half, reduced to what a chain needs:
 *
 *   - for every context of k tokens, kmin <= k <= kmax (kmin = --draft-suffix-min-match, kmax =
 *     kmin + 3), how often each token followed it in the responses held, and in all;
 *   - a proposal follows, from the sequence's end, the most frequent follower of the longest
 *     context that occurred (ties to the lower token id), and goes on over its own tokens. Token j
 *     scores C_j = count(context, token) / (count(context) + 1), the path D_j = prod C_i, and the
 *     chain stops where D_j falls under --draft-suffix-min-score: the paper's frequency rule, with
 *     one unseen follower in the denominator so a context seen once is not a certainty.
 *   - the scheduler verifies it instead of the request's own chain when it expects more accepted
 *     tokens (sum of D) than that chain and than the head's verifies accept on average.
 *
 * Responses go in whole after their request ends, with the last kmax tokens of the prompt in front
 * as context (counted only where the follower is a response token). Each carries an LRFU score
 * (PcScore, mem/prefix.h): its insertion is one reference, and every verify that accepts rows of a
 * chain it started (the context node's latest response, `Out::src`) is another (credit()). Past
 * the size cap the lowest-scoring response's counts are taken out again; with no credits that is
 * the oldest. A save keeps only responses that were credited, plus at a stop those inserted
 * within `active_s`. Contexts are keyed by a 64-bit
 * hash, so a clash merges two contexts' counts. Every proposed token is verified, so a clash, a
 * stale count or an evicted response changes how many tokens a step lands, never which.
 *
 * Off the scheduler's path: submit() copies the tokens into a queue and returns; a worker thread
 * inserts them, a few hundred positions per hold of the write lock. propose() takes the read lock
 * only if it is free (try), and proposes nothing otherwise, so a lookup never waits on an insert.
 *
 * Persistence: save() writes the held responses (not the tables, rebuilt from them) to one file,
 * keyed by the tokenizer's identity and the context lengths, checksummed, written to a temporary
 * file and renamed. load() checks all of it and queues the responses for the worker, so startup
 * does not wait for the rebuild.
 */
#pragma once
#include "ngram.h"
#include "mem/prefix.h"   /* PcScore */

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rad {

class GlobalSuffix {
public:
    struct Config {
        int     kmin = 6, kmax = 9;
        int64_t cap_bytes = 0;      /* 0 is off */
        double  half_s = 6 * 3600.0;   /* the LRFU half-life */
    };
    ~GlobalSuffix() { stop(); }
    void configure(const Config& c);
    bool enabled() const { return cfg_.cap_bytes > 0; }
    /* The worker; without it submit() only queues (tests call drain()). */
    void start();
    void stop();

    /* A finished response: `ctx` the prompt's last tokens (at most kmax are used), `out` the
     * response. Copies and returns at once. */
    void submit(const int32_t* ctx, int64_t nctx, const int32_t* out, int64_t nout);
    /* Inserts everything queued now, on this thread (tests, and save()). */
    void drain();

    struct Out {
        int   n = 0;
        int   match = 0;          /* the context length the first token followed */
        float score = 0.0f;       /* D of the last token */
        float expect = 0.0f;      /* sum of D along the chain */
        bool  busy = false;       /* the write lock was held: nothing looked up */
        uint64_t src = 0;         /* the response the first token's context came from last */
    };
    /* `accepted` rows of a chain from response `src` were accepted: one reference to it. Queued,
     * applied by the inserter; never waits. */
    void credit(uint64_t src);
    Out propose(const TokenSeq& seq, int max_rows, float min_score, int32_t* out) const;

    /* Writes what earned it: credited responses, and with `final` those inserted within
     * `active_s` seconds. `written` gets how many. */
    int  save(const std::string& path, const std::string& key_text, bool final = true,
              int64_t active_s = 3600, int64_t* written = nullptr);
    /* RAD_OK and the responses queued, or an error and nothing loaded (a missing file is
     * RAD_E_NOTFOUND). */
    int  load(const std::string& path, const std::string& key_text, int64_t* n_seqs = nullptr);

    struct Stats {
        int64_t seqs = 0, tokens = 0, contexts = 0, bytes = 0, evicted = 0, inserted = 0;
        int64_t queued = 0, dropped = 0;
        int64_t credits = 0;
    };
    Stats stats() const;
    /* Whether response `id` (Out::src) is still held. Tests. */
    bool holds(uint64_t id) const { std::shared_lock lk(mu_); return by_id_.count(id) > 0; }
    /* Bumped by every insert and eviction: a save has something new when it moved. */
    uint64_t generation() const { return gen_.load(std::memory_order_relaxed); }

private:
    struct Seq {
        std::vector<int32_t> toks;
        int32_t nctx = 0;
        uint64_t id = 0;
        int64_t  born_ms = 0;
        PcScore  score;
    };
    struct Node {
        uint32_t total = 0;
        uint64_t last = 0;                              /* the latest response through it */
        std::vector<std::pair<int32_t, uint32_t>> f;   /* follower, count */
    };
    static uint64_t ctx_hash(const int32_t* t, int k);
    void count(const Seq& s, int64_t from, int64_t to, int delta);
    void insert_locked_steps(Seq s);
    void apply_credits();
    void evict_locked();
    void worker();
    int64_t bytes_locked() const;

    Config cfg_;
    mutable std::shared_mutex mu_;
    std::list<Seq> seqs_;
    std::unordered_map<uint64_t, std::list<Seq>::iterator> by_id_;
    uint64_t next_id_ = 1;
    int64_t credits_ = 0;
    std::vector<std::unordered_map<uint64_t, Node>> tab_;   /* by k - kmin */
    int64_t tokens_ = 0, followers_ = 0, nodes_ = 0;
    int64_t evicted_ = 0, inserted_ = 0;

    std::mutex imu_;       /* one inserter at a time: the worker, or drain() */
    std::mutex qmu_;
    std::condition_variable qcv_;
    std::deque<Seq> queue_;
    std::vector<uint64_t> credit_q_;
    int64_t dropped_ = 0;
    bool stopping_ = false;
    std::thread worker_;
    std::atomic<uint64_t> gen_{ 0 };
};

}  /* namespace rad */
