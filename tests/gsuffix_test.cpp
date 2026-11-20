/* gsuffix_test.cpp -- core/sched/gsuffix.h, the global suffix cache: its proposals equal a
 * brute-force count over the responses it holds, eviction takes the oldest out exactly, and its
 * file round-trips and refuses another tokenizer. */
#include "rad_test.h"

#include "sched/gsuffix.h"
#include "rad_internal.h"

#include <cstdio>
#include <map>
#include <thread>
#include <vector>

using namespace rad;

namespace {

struct Resp { std::vector<int32_t> ctx, out; };

std::vector<Resp> responses(int n, uint32_t seed) {
    uint32_t x = seed * 2654435761u + 7;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };
    std::vector<std::vector<int32_t>> phrases;
    for (int i = 0; i < 40; ++i) {
        std::vector<int32_t> p;
        for (int j = 0, l = 3 + (int)(rnd() % 12); j < l; ++j) p.push_back((int32_t)(rnd() % 200));
        phrases.push_back(p);
    }
    std::vector<Resp> v;
    for (int i = 0; i < n; ++i) {
        Resp r;
        for (int j = 0; j < 12; ++j) r.ctx.push_back((int32_t)(rnd() % 200));
        const int len = 20 + (int)(rnd() % 300);
        while ((int)r.out.size() < len) {
            const auto& p = phrases[rnd() % phrases.size()];
            r.out.insert(r.out.end(), p.begin(), p.end());
        }
        v.push_back(r);
    }
    return v;
}

/* The reference: counts over the raw tokens, contexts compared exactly. */
GlobalSuffix::Out brute(const std::vector<Resp>& held, int kmin, int kmax, std::vector<int32_t> seq,
                        int max_rows, float min_score, int32_t* out) {
    GlobalSuffix::Out r;
    if ((int)seq.size() < kmin) return r;
    float d = 1, sum = 0;
    int m = 0;
    for (; m < max_rows; ++m) {
        std::map<int32_t, uint32_t> fol;
        uint32_t total = 0;
        int used = 0;
        for (int k = std::min<int>(kmax, (int)seq.size()); k >= kmin && !total; --k) {
            for (const Resp& h : held) {
                std::vector<int32_t> s(h.ctx.end() - std::min<long>(kmax, (long)h.ctx.size()), h.ctx.end());
                const int nctx = (int)s.size();
                s.insert(s.end(), h.out.begin(), h.out.end());
                for (int e = std::max(nctx, k); e < (int)s.size(); ++e)
                    if (std::equal(s.begin() + e - k, s.begin() + e, seq.end() - k)) {
                        ++fol[s[(size_t)e]];
                        ++total;
                    }
            }
            used = k;
        }
        if (!total) break;
        int32_t tok = 0; uint32_t c = 0;
        for (const auto& f : fol) if (f.second > c) { tok = f.first; c = f.second; }
        const float p = (float)c / (float)(total + 1);
        if (d * p < min_score) break;
        d *= p; sum += d;
        out[m] = tok;
        if (m == 0) r.match = used;
        seq.push_back(tok);
    }
    r.n = m; r.score = m ? d : 0; r.expect = sum;
    return r;
}

TokenSeq seq_of(const std::vector<int32_t>& v) { TokenSeq s; s.a = v.data(); s.na = (int64_t)v.size(); return s; }

void check_same(GlobalSuffix& g, const std::vector<Resp>& held, int kmin, int kmax, uint32_t seed) {
    uint32_t x = seed;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; };
    int hits = 0;
    for (int q = 0; q < 300; ++q) {
        /* A query: the tail of a held response (it recurs) or noise. */
        std::vector<int32_t> s;
        const Resp& h = held[rnd() % held.size()];
        const size_t at = rnd() % h.out.size();
        s.assign(h.out.begin(), h.out.begin() + (long)at);
        if (rnd() % 4 == 0) s.push_back((int32_t)(rnd() % 200));
        if ((int)s.size() < kmin) continue;
        int32_t a[8], b[8];
        const float ms = (rnd() % 2) ? 0.0f : 0.2f;
        const auto ra = g.propose(seq_of(s), 7, ms, a);
        const auto rb = brute(held, kmin, kmax, s, 7, ms, b);
        CHECK_EQ(ra.n, rb.n);
        CHECK_EQ(ra.match, rb.match);
        CHECK(ra.expect == rb.expect);
        for (int i = 0; i < ra.n && i < rb.n; ++i) CHECK_EQ(a[i], b[i]);
        hits += ra.n > 0;
    }
    CHECK(hits > 50);
}

}  // namespace

TEST(global_proposals_equal_a_brute_force_count) {
    GlobalSuffix g;
    g.configure({ 3, 6, (int64_t)1 << 30 });
    const auto rs = responses(30, 1);
    for (const Resp& r : rs) g.submit(r.ctx.data(), (int64_t)r.ctx.size(), r.out.data(), (int64_t)r.out.size());
    g.drain();
    CHECK_EQ(g.stats().seqs, 30ll);
    check_same(g, rs, 3, 6, 11);
}

TEST(eviction_takes_the_oldest_responses_out_exactly) {
    GlobalSuffix big;
    big.configure({ 3, 6, (int64_t)1 << 30 });
    const auto rs = responses(60, 2);
    for (const Resp& r : rs) big.submit(r.ctx.data(), (int64_t)r.ctx.size(), r.out.data(), (int64_t)r.out.size());
    big.drain();
    const int64_t all = big.stats().bytes;
    GlobalSuffix g;
    g.configure({ 3, 6, all / 3 });
    for (const Resp& r : rs) g.submit(r.ctx.data(), (int64_t)r.ctx.size(), r.out.data(), (int64_t)r.out.size());
    g.drain();
    const auto st = g.stats();
    CHECK(st.evicted > 0);
    CHECK(st.bytes <= all / 3);
    CHECK_EQ(st.seqs + st.evicted, 60ll);
    std::vector<Resp> held(rs.end() - st.seqs, rs.end());
    check_same(g, held, 3, 6, 12);
}

TEST(the_file_round_trips_and_refuses_another_tokenizer_or_damage) {
    const std::string path = radtest::tmp_path("gsuffix.bin");
    GlobalSuffix g;
    g.configure({ 3, 6, (int64_t)1 << 30 });
    const auto rs = responses(20, 3);
    for (const Resp& r : rs) g.submit(r.ctx.data(), (int64_t)r.ctx.size(), r.out.data(), (int64_t)r.out.size());
    CHECK_OK(g.save(path, "vocab A\n"));
    GlobalSuffix h;
    h.configure({ 3, 6, (int64_t)1 << 30 });
    int64_t n = 0;
    CHECK_OK(h.load(path, "vocab A\n", &n));
    CHECK_EQ(n, 20ll);
    h.drain();
    check_same(h, rs, 3, 6, 13);
    GlobalSuffix k;
    k.configure({ 3, 6, (int64_t)1 << 30 });
    CHECK_EQ(k.load(path, "vocab B\n", &n), (int)RAD_E_FORMAT);
    k.drain();
    CHECK_EQ(k.stats().seqs, 0ll);
    GlobalSuffix l;
    l.configure({ 4, 7, (int64_t)1 << 30 });                  /* other context lengths */
    CHECK_EQ(l.load(path, "vocab A\n", &n), (int)RAD_E_FORMAT);
    FILE* f = std::fopen(path.c_str(), "r+b");
    REQUIRE(f);
    std::fseek(f, 200, SEEK_SET);
    std::fputc(0x55, f);
    std::fclose(f);
    CHECK_EQ(h.load(path, "vocab A\n", &n), (int)RAD_E_FORMAT);
    CHECK_EQ(h.load(path + ".missing", "vocab A\n", &n), (int)RAD_E_NOTFOUND);
}

TEST(lookups_beside_the_worker_never_wait_and_never_crash) {
    GlobalSuffix g;
    g.configure({ 3, 6, (int64_t)1 << 20 });
    g.start();
    const auto rs = responses(400, 4);
    std::thread t([&] {
        for (const Resp& r : rs) g.submit(r.ctx.data(), (int64_t)r.ctx.size(), r.out.data(), (int64_t)r.out.size());
    });
    int busy = 0, got = 0;
    for (int i = 0; i < 20000; ++i) {
        int32_t o[8];
        const auto r = g.propose(seq_of(rs[(size_t)i % rs.size()].out), 7, 0.0f, o);
        busy += r.busy;
        got += r.n > 0;
    }
    t.join();
    g.stop();
    g.drain();
    std::fprintf(stderr, "    %d lookups found a chain, %d found the cache busy\n", got, busy);
    CHECK(g.stats().bytes <= ((int64_t)1 << 20));
}

RAD_TEST_MAIN()
