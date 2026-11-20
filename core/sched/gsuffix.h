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
 *     scores C_j = count(context, token) / count(context), the path D_j = prod C_i, and the chain
 *     stops where D_j falls under --draft-suffix-min-score: the paper's frequency rule.
 *
 * Responses go in whole after their request ends, with the last kmax tokens of the prompt in front
 * as context (counted only where the follower is a response token). They are held oldest first;
 * past the size cap the oldest response's counts are taken out again. Contexts are keyed by a 64-bit
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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
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
    };
    Out propose(const TokenSeq& seq, int max_rows, float min_score, int32_t* out) const;

    int  save(const std::string& path, const std::string& key_text);
    /* RAD_OK and the responses queued, or an error and nothing loaded (a missing file is
     * RAD_E_NOTFOUND). */
    int  load(const std::string& path, const std::string& key_text, int64_t* n_seqs = nullptr);

    struct Stats {
        int64_t seqs = 0, tokens = 0, contexts = 0, bytes = 0, evicted = 0, inserted = 0;
        int64_t queued = 0, dropped = 0;
    };
    Stats stats() const;
    /* Bumped by every insert and eviction: a save has something new when it moved. */
    uint64_t generation() const { return gen_.load(std::memory_order_relaxed); }

private:
    struct Seq { std::vector<int32_t> toks; int32_t nctx = 0; };
    struct Node {
        uint32_t total = 0;
        std::vector<std::pair<int32_t, uint32_t>> f;   /* follower, count */
    };
    static uint64_t ctx_hash(const int32_t* t, int k);
    void count(const Seq& s, int64_t from, int64_t to, int delta);
    void insert_locked_steps(Seq s);
    void evict_locked();
    void worker();
    int64_t bytes_locked() const;

    Config cfg_;
    mutable std::shared_mutex mu_;
    std::deque<Seq> seqs_;
    std::vector<std::unordered_map<uint64_t, Node>> tab_;   /* by k - kmin */
    int64_t tokens_ = 0, followers_ = 0, nodes_ = 0;
    int64_t evicted_ = 0, inserted_ = 0;

    std::mutex imu_;       /* one inserter at a time: the worker, or drain() */
    std::mutex qmu_;
    std::condition_variable qcv_;
    std::deque<Seq> queue_;
    int64_t dropped_ = 0;
    bool stopping_ = false;
    std::thread worker_;
    std::atomic<uint64_t> gen_{ 0 };
};

}  /* namespace rad */
