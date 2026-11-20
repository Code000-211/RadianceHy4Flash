/* suffix_test.cpp -- core/sched/suffix.h built incrementally: a chunk at a time, and taken over
 * from an earlier prompt's copy, the index holds exactly what one build at the first token does,
 * and proposes the same chains. */
#include "rad_test.h"

#include "sched/suffix.h"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace rad;

namespace {

/* Code-like text: a few hundred "lines" drawn from a small set with variations, so patterns
 * recur at every length the index looks at. */
std::vector<int32_t> code_like(int64_t n, uint32_t seed) {
    std::vector<std::vector<int32_t>> lines;
    uint32_t x = seed * 2654435761u + 1;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };
    for (int i = 0; i < 300; ++i) {
        std::vector<int32_t> l;
        const int len = 4 + (int)(rnd() % 20);
        for (int j = 0; j < len; ++j) l.push_back((int32_t)(rnd() % 3000));
        lines.push_back(l);
    }
    std::vector<int32_t> v;
    while ((int64_t)v.size() < n) {
        const auto& l = lines[(size_t)(rnd() % (rnd() % 7 == 0 ? 300 : 40))];
        v.insert(v.end(), l.begin(), l.end());
        if (rnd() % 5 == 0) v.push_back((int32_t)(rnd() % 50000));
    }
    v.resize((size_t)n);
    return v;
}

TokenSeq seq_of(const std::vector<int32_t>& p, const std::vector<int32_t>& o = {}) {
    TokenSeq s;
    s.a = p.data(); s.na = (int64_t)p.size();
    s.b = o.data(); s.nb = (int64_t)o.size();
    return s;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

TEST(an_index_built_a_chunk_at_a_time_equals_one_built_at_the_first_token) {
    for (uint32_t seed : { 1u, 2u, 3u }) {
        const auto prompt = code_like(20000, seed);
        std::vector<int32_t> out(prompt.begin() + 1000, prompt.begin() + 1007);
        for (int k : { 3, 6 }) {
            SuffixIndex once, chunked;
            int32_t a[8], b[8];
            for (int64_t at = 0; at < (int64_t)prompt.size(); at += 1088 + seed * 17)
                chunked.index(seq_of(prompt), 7, k, std::min<int64_t>(at, (int64_t)prompt.size()));
            chunked.index(seq_of(prompt), 7, k, (int64_t)prompt.size());
            /* The first token's commit, then a few more. */
            for (size_t n = 1; n <= out.size(); ++n) {
                std::vector<int32_t> o(out.begin(), out.begin() + (long)n);
                const auto ra = once.propose(seq_of(prompt, o), 7, k, 7, 0.0f, a);
                const auto rb = chunked.propose(seq_of(prompt, o), 7, k, 7, 0.0f, b);
                CHECK_EQ(ra.n, rb.n);
                CHECK_EQ(ra.match, rb.match);
                CHECK(ra.score == rb.score);
                for (int i = 0; i < ra.n && i < rb.n; ++i) CHECK_EQ(a[i], b[i]);
                CHECK(once.contents() == chunked.contents());
            }
        }
    }
}

TEST(an_index_taken_over_from_an_earlier_prompt_equals_a_fresh_one) {
    const auto all = code_like(30000, 9);
    std::vector<int32_t> turn1(all.begin(), all.begin() + 12000);
    std::vector<int32_t> turn2(all.begin(), all.begin() + 25000);   /* turn 1 plus what followed */
    SuffixIndex t1;
    t1.index(seq_of(turn1), 1, 6, (int64_t)turn1.size());
    SuffixIndex stash = t1;                                        /* the scheduler's copy */
    /* Turn 1 goes on decoding into its own index; the copy is untouched by it. */
    int32_t buf[8];
    std::vector<int32_t> o1(all.begin() + 12000, all.begin() + 12050);
    (void)t1.propose(seq_of(turn1, o1), 1, 6, 7, 0.3f, buf);
    stash.set_owner(2);
    SuffixIndex fresh;
    for (int64_t at = 8192; at < (int64_t)turn2.size(); at += 8192) {
        stash.index(seq_of(turn2), 2, 6, at);
        fresh.index(seq_of(turn2), 2, 6, at);
    }
    std::vector<int32_t> o2{ all[25000] };
    int32_t a[8], b[8];
    const auto ra = stash.propose(seq_of(turn2, o2), 2, 6, 7, 0.3f, a);
    const auto rb = fresh.propose(seq_of(turn2, o2), 2, 6, 7, 0.3f, b);
    CHECK_EQ(ra.n, rb.n);
    for (int i = 0; i < ra.n; ++i) CHECK_EQ(a[i], b[i]);
    CHECK(stash.contents() == fresh.contents());
    /* A shrinking or foreign sequence never reads a stale index: propose starts over. */
    SuffixIndex other = stash;
    std::vector<int32_t> shorter(all.begin(), all.begin() + 100);
    (void)other.propose(seq_of(shorter), 2, 6, 7, 0.3f, a);
    CHECK_EQ(other.indexed(), 100ll);
}

/* The costs the scheduler pays under its lock, printed for the record: the build at the first
 * token (before), a chunk's advance and the stash copy (after). */
TEST(perf_suffix_index_costs) {
    for (int64_t n : { (int64_t)20000, (int64_t)200000 }) {
        const auto p = code_like(n, 5);
        std::vector<int32_t> o{ 17 };
        int32_t buf[8];
        double once_ms = 1e9, chunk_ms = 1e9, first_ms = 1e9, sum_ms = 1e9, copy_ms = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            SuffixIndex a;
            auto t0 = std::chrono::steady_clock::now();
            (void)a.propose(seq_of(p, o), 1, 6, 7, 0.3f, buf);
            once_ms = std::min(once_ms, ms_since(t0));
            SuffixIndex b;
            double worst = 0, first = 0, sum = 0;
            for (int64_t at = 8192; at < n + 8192; at += 8192) {
                auto t1 = std::chrono::steady_clock::now();
                b.index(seq_of(p), 1, 6, std::min(at, n));
                const double d = ms_since(t1);
                sum += d;
                if (at == 8192) first = d; else worst = std::max(worst, d);
            }
            chunk_ms = std::min(chunk_ms, worst);
            first_ms = std::min(first_ms, first);
            sum_ms = std::min(sum_ms, sum);
            auto t2 = std::chrono::steady_clock::now();
            SuffixIndex c = b;
            copy_ms = std::min(copy_ms, ms_since(t2));
            CHECK(c.indexed() == n);
        }
        std::fprintf(stderr, "    %lld tokens: build at first token %.2f ms; per 8192-token chunk: "
                     "first %.2f ms, others at most %.2f ms, all %.2f ms; stash copy %.2f ms\n",
                     (long long)n, once_ms, first_ms, chunk_ms, sum_ms, copy_ms);
    }
}

RAD_TEST_MAIN()
