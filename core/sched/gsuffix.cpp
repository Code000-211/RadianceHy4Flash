/* gsuffix.cpp -- see gsuffix.h. */
#include "gsuffix.h"

#include "mem/pcstore.h"     /* pc_checksum, pc_key_of */
#include "rad_internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rad {

namespace {
constexpr char kMagic[8] = { 'R', 'A', 'D', 'S', 'F', 'X', '0', '1' };
constexpr char kEnd[8]   = { 'R', 'A', 'D', 'S', 'F', 'E', 'N', 'D' };
constexpr int64_t kStep = 512;            /* positions a write-lock hold covers */
constexpr size_t  kQueueMax = 256;        /* responses waiting for the worker */
constexpr int64_t kNodeBytes = 72, kFollowerBytes = 8;

template <class T> void put(std::string& b, const T& v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
}  // namespace

void GlobalSuffix::configure(const Config& c) {
    stop();
    std::unique_lock lk(mu_);
    cfg_ = c;
    if (cfg_.kmin < 1) cfg_.kmin = 1;
    if (cfg_.kmax < cfg_.kmin) cfg_.kmax = cfg_.kmin;
    seqs_.clear();
    tab_.assign((size_t)(cfg_.kmax - cfg_.kmin + 1), {});
    tokens_ = followers_ = nodes_ = evicted_ = inserted_ = 0;
}

void GlobalSuffix::start() {
    if (!enabled() || worker_.joinable()) return;
    {
        std::lock_guard lk(qmu_);
        stopping_ = false;
    }
    worker_ = std::thread([this] { worker(); });
}

void GlobalSuffix::stop() {
    {
        std::lock_guard lk(qmu_);
        stopping_ = true;
    }
    qcv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

uint64_t GlobalSuffix::ctx_hash(const int32_t* t, int k) {
    uint64_t h = 0x84222325cbf29ce4ull ^ (uint64_t)k * 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < k; ++i) {
        h ^= (uint64_t)(uint32_t)t[i] + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 32;
    }
    return h;
}

void GlobalSuffix::count(const Seq& s, int64_t from, int64_t to, int delta) {
    const int32_t* t = s.toks.data();
    for (int64_t e = std::max<int64_t>(from, s.nctx); e < to; ++e)
        for (int k = cfg_.kmin; k <= cfg_.kmax; ++k) {
            if (e - k < 0) break;
            auto& tab = tab_[(size_t)(k - cfg_.kmin)];
            const uint64_t h = ctx_hash(t + e - k, k);
            if (delta > 0) {
                auto ins = tab.try_emplace(h);
                Node& n = ins.first->second;
                if (ins.second) ++nodes_;
                ++n.total;
                auto f = std::find_if(n.f.begin(), n.f.end(), [&](const auto& p) { return p.first == t[e]; });
                if (f == n.f.end()) { n.f.push_back({ t[e], 1u }); ++followers_; }
                else ++f->second;
            } else {
                auto it = tab.find(h);
                if (it == tab.end()) continue;
                Node& n = it->second;
                auto f = std::find_if(n.f.begin(), n.f.end(), [&](const auto& p) { return p.first == t[e]; });
                if (f != n.f.end() && --f->second == 0) { n.f.erase(f); --followers_; }
                if (n.total > 0) --n.total;
                if (n.total == 0) { tab.erase(it); --nodes_; }
            }
        }
}

int64_t GlobalSuffix::bytes_locked() const {
    return tokens_ * 4 + nodes_ * kNodeBytes + followers_ * kFollowerBytes;
}

void GlobalSuffix::insert_locked_steps(Seq s) {
    std::lock_guard ik(imu_);
    const int64_t n = (int64_t)s.toks.size();
    if (n <= s.nctx || n <= cfg_.kmin) return;
    {
        std::unique_lock lk(mu_);
        seqs_.push_back(std::move(s));
        tokens_ += n;
    }
    /* The deque's back is stable while this thread is the only writer. */
    for (int64_t at = 0; at < n; at += kStep) {
        std::unique_lock lk(mu_);
        count(seqs_.back(), at, std::min(n, at + kStep), +1);
    }
    {
        std::unique_lock lk(mu_);
        ++inserted_;
    }
    gen_.fetch_add(1, std::memory_order_relaxed);
    evict_locked();
}

void GlobalSuffix::evict_locked() {
    for (;;) {
        int64_t n;
        {
            std::unique_lock lk(mu_);
            if (seqs_.size() <= 1 || bytes_locked() <= cfg_.cap_bytes) return;
            n = (int64_t)seqs_.front().toks.size();
        }
        for (int64_t at = 0; at < n; at += kStep) {
            std::unique_lock lk(mu_);
            count(seqs_.front(), at, std::min(n, at + kStep), -1);
        }
        std::unique_lock lk(mu_);
        tokens_ -= n;
        seqs_.pop_front();
        ++evicted_;
        gen_.fetch_add(1, std::memory_order_relaxed);
    }
}

void GlobalSuffix::submit(const int32_t* ctx, int64_t nctx, const int32_t* out, int64_t nout) {
    if (!enabled() || nout <= 0) return;
    Seq s;
    const int64_t c = std::min<int64_t>(std::max<int64_t>(nctx, 0), cfg_.kmax);
    s.toks.reserve((size_t)(c + nout));
    if (c) s.toks.insert(s.toks.end(), ctx + nctx - c, ctx + nctx);
    s.toks.insert(s.toks.end(), out, out + nout);
    s.nctx = (int32_t)c;
    {
        std::lock_guard lk(qmu_);
        if (queue_.size() >= kQueueMax) { queue_.pop_front(); ++dropped_; }
        queue_.push_back(std::move(s));
    }
    qcv_.notify_one();
}

void GlobalSuffix::worker() {
    for (;;) {
        Seq s;
        {
            std::unique_lock lk(qmu_);
            qcv_.wait(lk, [this] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) return;            /* stopping, nothing left */
            s = std::move(queue_.front());
            queue_.pop_front();
        }
        insert_locked_steps(std::move(s));
    }
}

void GlobalSuffix::drain() {
    for (;;) {
        Seq s;
        {
            std::lock_guard lk(qmu_);
            if (queue_.empty()) return;
            s = std::move(queue_.front());
            queue_.pop_front();
        }
        insert_locked_steps(std::move(s));
    }
}

GlobalSuffix::Out GlobalSuffix::propose(const TokenSeq& seq, int max_rows, float min_score,
                                        int32_t* out) const {
    Out r;
    if (!enabled() || max_rows < 1 || seq.size() < cfg_.kmin) return r;
    std::shared_lock lk(mu_, std::try_to_lock);
    if (!lk.owns_lock()) { r.busy = true; return r; }
    if (seqs_.empty()) return r;
    /* The window: the sequence's last kmax tokens, then the proposal's own. */
    int32_t win[64];
    const int64_t L = seq.size();
    int w = (int)std::min<int64_t>(L, cfg_.kmax);
    for (int i = 0; i < w; ++i) win[i] = seq[L - w + i];
    float d = 1.0f, sum = 0.0f;
    int m = 0;
    for (; m < max_rows; ++m) {
        const Node* best = nullptr;
        int used = 0;
        for (int k = std::min(w, cfg_.kmax); k >= cfg_.kmin; --k) {
            const auto& tab = tab_[(size_t)(k - cfg_.kmin)];
            auto it = tab.find(ctx_hash(win + w - k, k));
            if (it == tab.end() || it->second.total == 0 || it->second.f.empty()) continue;
            best = &it->second;
            used = k;
            break;
        }
        if (!best) break;
        int32_t tok = 0;
        uint32_t c = 0;
        for (const auto& f : best->f)
            if (f.second > c || (f.second == c && f.first < tok)) { tok = f.first; c = f.second; }
        /* One more in the denominator: a context seen once is a guess, not a certainty, and a
         * small cache would otherwise propose every one-off continuation at full confidence. */
        const float p = (float)c / (float)(best->total + 1);
        if (d * p < min_score) break;
        d *= p;
        sum += d;
        out[m] = tok;
        if (m == 0) r.match = used;
        if (w < cfg_.kmax) win[w++] = tok;
        else { std::memmove(win, win + 1, (size_t)(w - 1) * sizeof(int32_t)); win[w - 1] = tok; }
    }
    r.n = m;
    r.score = m > 0 ? d : 0.0f;
    r.expect = sum;
    return r;
}

GlobalSuffix::Stats GlobalSuffix::stats() const {
    Stats s;
    {
        std::shared_lock lk(mu_);
        s.seqs = (int64_t)seqs_.size();
        s.tokens = tokens_;
        s.contexts = nodes_;
        s.bytes = bytes_locked();
        s.evicted = evicted_;
        s.inserted = inserted_;
    }
    std::lock_guard lk(const_cast<std::mutex&>(qmu_));
    s.queued = (int64_t)queue_.size();
    s.dropped = dropped_;
    return s;
}

int GlobalSuffix::save(const std::string& path, const std::string& key_text) {
    if (!enabled()) return RAD_E_STATE;
    drain();    /* what is queued is held too */
    std::string b;
    b.append(kMagic, 8);
    const BlockHash key = pc_key_of(key_text);
    {
        std::shared_lock lk(mu_);
        put(b, (uint32_t)1);
        put(b, key.hi); put(b, key.lo);
        put(b, (uint32_t)key_text.size());
        b += key_text;
        put(b, (int32_t)cfg_.kmin); put(b, (int32_t)cfg_.kmax);
        put(b, (uint64_t)seqs_.size());
        for (const Seq& s : seqs_) {
            put(b, s.nctx);
            put(b, (uint64_t)s.toks.size());
            b.append(reinterpret_cast<const char*>(s.toks.data()), s.toks.size() * 4);
        }
    }
    put(b, pc_checksum(b.data(), (int64_t)b.size()));
    b.append(kEnd, 8);
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
    if (fd < 0) return RAD_E_IO;
    const char* p = b.data();
    int64_t left = (int64_t)b.size();
    bool ok = true;
    while (left > 0) {
        const ssize_t w = ::write(fd, p, (size_t)left);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { ok = false; break; }
        p += w; left -= w;
    }
    ok = ok && ::fsync(fd) == 0;
    ::close(fd);
    if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
        RAD_WARN("draft-suffix: saving %s failed: %s; the previous file is kept", path.c_str(),
                 std::strerror(errno));
        ::unlink(tmp.c_str());
        return RAD_E_IO;
    }
    return RAD_OK;
}

int GlobalSuffix::load(const std::string& path, const std::string& key_text, int64_t* n_seqs) {
    if (n_seqs) *n_seqs = 0;
    if (!enabled()) return RAD_E_STATE;
    std::string b;
    {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return RAD_E_NOTFOUND;
        struct stat st {};
        if (fstat(fd, &st) != 0) { ::close(fd); return RAD_E_IO; }
        b.resize((size_t)st.st_size);
        int64_t got = 0;
        while (got < (int64_t)b.size()) {
            const ssize_t r = ::pread(fd, &b[(size_t)got], b.size() - (size_t)got, got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) break;
            got += r;
        }
        ::close(fd);
        if (got != (int64_t)b.size()) return RAD_E_IO;
    }
    if (b.size() < 32 || std::memcmp(b.data(), kMagic, 8) || std::memcmp(b.data() + b.size() - 8, kEnd, 8))
        return RAD_E_FORMAT;
    uint64_t sum = 0;
    std::memcpy(&sum, b.data() + b.size() - 16, 8);
    if (pc_checksum(b.data(), (int64_t)b.size() - 16) != sum) return RAD_E_FORMAT;
    size_t at = 8;
    const size_t end = b.size() - 16;
    bool bad = false;
    auto get = [&](void* v, size_t n) {
        if (at + n > end) { bad = true; std::memset(v, 0, n); return; }
        std::memcpy(v, b.data() + at, n);
        at += n;
    };
    uint32_t ver = 0, kl = 0;
    BlockHash key;
    get(&ver, 4); get(&key.hi, 8); get(&key.lo, 8); get(&kl, 4);
    if (bad || ver != 1 || at + kl > end || key != pc_key_of(key_text) ||
        b.compare(at, kl, key_text) != 0)
        return RAD_E_FORMAT;
    at += kl;
    int32_t kmin = 0, kmax = 0;
    uint64_t n = 0;
    get(&kmin, 4); get(&kmax, 4); get(&n, 8);
    if (bad || kmin != cfg_.kmin || kmax != cfg_.kmax) return RAD_E_FORMAT;
    std::vector<Seq> seqs;
    for (uint64_t i = 0; i < n && !bad; ++i) {
        Seq s;
        uint64_t len = 0;
        get(&s.nctx, 4); get(&len, 8);
        if (bad || len > (end - at) / 4) { bad = true; break; }
        s.toks.resize((size_t)len);
        get(s.toks.data(), (size_t)len * 4);
        seqs.push_back(std::move(s));
    }
    if (bad || at != end) return RAD_E_FORMAT;
    {
        std::lock_guard lk(qmu_);
        for (Seq& s : seqs) queue_.push_back(std::move(s));
    }
    qcv_.notify_one();
    if (n_seqs) *n_seqs = (int64_t)n;
    return RAD_OK;
}

}  /* namespace rad */
