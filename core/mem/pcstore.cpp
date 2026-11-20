/* pcstore.cpp -- see pcstore.h. */
#include "mem/pcstore.h"

#include "mem/hostarena.h"
#include "mem/kvtier.h"

#include "rad_internal.h"
#include "util/simd.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/uio.h>
#include <unistd.h>

namespace rad {

namespace {

constexpr char kSegMagic[8] = { 'R', 'A', 'D', 'P', 'C', 'S', 'G', '1' };
constexpr char kIdxMagic[8] = { 'R', 'A', 'D', 'P', 'C', 'I', 'X', '1' };
constexpr char kEndMagic[8] = { 'R', 'A', 'D', 'P', 'C', 'E', 'N', 'D' };
constexpr uint32_t kVersion = 1;
constexpr int64_t kSegHeader = 64;
constexpr int64_t kMiB = (int64_t)1 << 20;

inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

int64_t now_ms() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

double secs_since(uint64_t t0) { return (double)(rad_mono_ns() - t0) * 1e-9; }

template <class T> void put(std::string& b, const T& v) {
    b.append(reinterpret_cast<const char*>(&v), sizeof v);
}

struct Reader {
    const char* p; size_t n; size_t at = 0; bool bad = false;
    template <class T> T get() {
        T v{};
        if (at + sizeof v > n) { bad = true; return v; }
        std::memcpy(&v, p + at, sizeof v);
        at += sizeof v;
        return v;
    }
    const char* take(size_t k) {
        if (at + k > n) { bad = true; return nullptr; }
        const char* q = p + at;
        at += k;
        return q;
    }
};

std::string key_hex(const BlockHash& k) { return k.str(); }

std::string sanitize(const std::string& s) {
    std::string o;
    for (char c : s) o += (std::isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
    return o.empty() ? std::string("model") : o;
}

bool write_all(int fd, const void* p, int64_t n) {
    const char* c = static_cast<const char*>(p);
    while (n > 0) {
        const ssize_t w = ::write(fd, c, (size_t)std::min<int64_t>(n, (int64_t)1 << 30));
        if (w < 0) { if (errno == EINTR) continue; return false; }
        if (w == 0) { errno = EIO; return false; }
        c += w;
        n -= w;
    }
    return true;
}

bool read_file(const std::string& path, std::string* out) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st {};
    if (fstat(fd, &st) != 0) { ::close(fd); return false; }
    out->resize((size_t)st.st_size);
    int64_t got = 0;
    while (got < (int64_t)st.st_size) {
        const ssize_t r = ::pread(fd, &(*out)[(size_t)got], (size_t)(st.st_size - got), got);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; ::close(fd); return false; }
        got += r;
    }
    ::close(fd);
    return true;
}

void fsync_dir(const std::string& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) { (void)::fsync(fd); ::close(fd); }
}

/* The key text stored in an index, or "" when the file is not an index. */
std::string index_key_text(const std::string& path) {
    std::string b;
    if (!read_file(path, &b) || b.size() < 20 || std::memcmp(b.data(), kIdxMagic, 8)) return {};
    Reader r{ b.data(), b.size() };
    r.take(8);
    (void)r.get<uint32_t>();
    const uint32_t kl = r.get<uint32_t>();
    r.get<uint64_t>(); r.get<uint64_t>();
    r.get<int64_t>(); r.get<int64_t>(); r.get<int64_t>();
    r.get<uint32_t>(); r.get<uint32_t>(); r.get<uint64_t>(); r.get<uint64_t>(); r.get<int64_t>();
    const char* t = r.take(kl);
    return r.bad || !t ? std::string() : std::string(t, kl);
}

std::string first_difference(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        const size_t ie = std::min(a.find('\n', i), a.size()), je = std::min(b.find('\n', j), b.size());
        const std::string la = a.substr(i, ie - i), lb = b.substr(j, je - j);
        if (la != lb) return "'" + la + "' was '" + lb + "'";
        i = ie + 1;
        j = je + 1;
    }
    return "(the same text)";
}

void remove_store_dir(const std::string& d) {
    if (DIR* dp = ::opendir(d.c_str())) {
        while (dirent* e = ::readdir(dp)) {
            const std::string n = e->d_name;
            if (n == "index.rpx" || (n.rfind("seg-", 0) == 0) || n.find(".tmp") != std::string::npos)
                ::unlink((d + "/" + n).c_str());
        }
        ::closedir(dp);
    }
    ::rmdir(d.c_str());
}

}  /* namespace */

uint64_t pc_checksum(const void* data, int64_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t lane[8] = { 0xc2b2ae3d27d4eb4full, 0x9e3779b185ebca87ull, 0x165667b19e3779f9ull ^ (uint64_t)n,
                         0x85ebca77c2b2ae63ull, 0x27d4eb2f165667c5ull, 0x94d049bb133111ebull,
                         0x9e3779b97f4a7c15ull ^ (uint64_t)n, 0xff51afd7ed558ccdull };
    const size_t stripes = n > 0 ? (size_t)n / 64 : 0;
    if (stripes) simd::stripe_acc_u64x8(lane, p, stripes);
    uint64_t a = 0x6a09e667f3bcc908ull ^ (uint64_t)n;
    for (int j = 0; j < 8; ++j) a = mix64(a ^ lane[j]) + (uint64_t)j;
    for (size_t i = stripes * 64; i < (size_t)n; ++i) a = mix64(a ^ p[i]);
    return mix64(a);
}

BlockHash pc_key_of(const std::string& text) {
    BlockHash h;
    uint64_t a = 0x243f6a8885a308d3ull, b = 0x13198a2e03707344ull;
    const uint64_t s = pc_checksum(text.data(), (int64_t)text.size());
    for (unsigned char c : text) { a = mix64(a ^ c); b = mix64(b + c * 0x9e3779b97f4a7c15ull); }
    h.hi = mix64(a ^ s);
    h.lo = mix64(b + s);
    if (!h.valid()) h.lo = 1;
    return h;
}

PrefixStore::~PrefixStore() { close(); }

void PrefixStore::close_fds() {
    for (auto& kv : segs_) if (kv.second.fd >= 0) { ::close(kv.second.fd); kv.second.fd = -1; }
}

void PrefixStore::close() {
    close_fds();
    segs_.clear();
    ents_.clear();
    snaps_.clear();
    release_loaded();
    enabled_ = false;
}

std::string PrefixStore::seg_path(uint32_t id) const { return fmt("%s/seg-%u.rps", path_.c_str(), id); }

int64_t PrefixStore::disk_bytes() const {
    int64_t n = 0;
    for (const auto& kv : segs_) n += kv.second.bytes;
    return n;
}

int PrefixStore::seg_fd(uint32_t id) {
    auto it = segs_.find(id);
    if (it == segs_.end()) return -1;
    if (it->second.fd < 0) it->second.fd = ::open(seg_path(id).c_str(), O_RDONLY | O_CLOEXEC);
    return it->second.fd;
}

int PrefixStore::read_at(uint32_t seg, uint64_t off, void* dst, int64_t n) {
    const int fd = seg_fd(seg);
    if (fd < 0) return RAD_E_IO;
    char* c = static_cast<char*>(dst);
    while (n > 0) {
        const ssize_t r = ::pread(fd, c, (size_t)std::min<int64_t>(n, (int64_t)1 << 30), (off_t)off);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return RAD_E_IO;
        c += r; off += (uint64_t)r; n -= r;
    }
    return RAD_OK;
}

int PrefixStore::open(const Config& c) {
    close();
    cfg_ = c;
    key_ = pc_key_of(c.key_text);
    if (c.dir.empty() || c.entry_bytes <= 0 || c.block_tokens <= 0) return RAD_E_INVAL;
    if (::mkdir(c.dir.c_str(), 0700) != 0 && errno != EEXIST) {
        RAD_WARN("prefix store: cannot create %s: %s", c.dir.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    const std::string base = sanitize(c.name);
    path_ = c.dir + "/" + base + "." + key_hex(key_);
    /* Other keys of this name: a previous build or configuration, never read. */
    if (DIR* dp = ::opendir(c.dir.c_str())) {
        std::vector<std::string> stale;
        while (dirent* e = ::readdir(dp)) {
            const std::string n = e->d_name;
            if (n.size() == base.size() + 33 && n.compare(0, base.size() + 1, base + ".") == 0 &&
                c.dir + "/" + n != path_)
                stale.push_back(n);
        }
        ::closedir(dp);
        for (const std::string& n : stale) {
            const std::string d = c.dir + "/" + n;
            const std::string old = index_key_text(d + "/index.rpx");
            RAD_INFO("prefix store: %s was written under another key and is discarded: %s", d.c_str(),
                     old.empty() ? "no readable index" : first_difference(c.key_text, old).c_str());
            remove_store_dir(d);
        }
    }
    if (::mkdir(path_.c_str(), 0700) != 0 && errno != EEXIST) {
        RAD_WARN("prefix store: cannot create %s: %s", path_.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    const uint64_t t0 = rad_mono_ns();
    const int li = load_index();
    sweep(segs_);
    enabled_ = true;
    int64_t eb = 0;
    for (const auto& kv : segs_) eb += kv.second.bytes;
    report_ = fmt("prefix store: %s: %lld entries (%lld tokens) and %lld snapshots in %s on disk, "
                  "index read in %.3f s%s", path_.c_str(), (long long)ents_.size(),
                  (long long)(ents_.size() * (size_t)c.block_tokens), (long long)snaps_.size(),
                  humanb(eb).c_str(), secs_since(t0),
                  li == RAD_E_NOTFOUND ? " (none yet)" : li < 0 ? " (the index was unreadable and is discarded)" : "");
    return RAD_OK;
}

int PrefixStore::load_index() {
    std::string b;
    const std::string ip = path_ + "/index.rpx";
    if (::access(ip.c_str(), F_OK) != 0) return RAD_E_NOTFOUND;
    if (!read_file(ip, &b) || b.size() < 16 + 8 || std::memcmp(b.data(), kIdxMagic, 8) ||
        std::memcmp(b.data() + b.size() - 8, kEndMagic, 8)) {
        RAD_WARN("prefix store: %s is damaged; starting empty", ip.c_str());
        return RAD_E_FORMAT;
    }
    uint64_t sum = 0;
    std::memcpy(&sum, b.data() + b.size() - 16, 8);
    if (pc_checksum(b.data(), (int64_t)b.size() - 16) != sum) {
        RAD_WARN("prefix store: %s fails its checksum; starting empty", ip.c_str());
        return RAD_E_FORMAT;
    }
    Reader r{ b.data(), b.size() - 16 };
    r.take(8);
    const uint32_t ver = r.get<uint32_t>();
    const uint32_t kl = r.get<uint32_t>();
    BlockHash k;
    k.hi = r.get<uint64_t>();
    k.lo = r.get<uint64_t>();
    const int64_t eb = r.get<int64_t>(), sb = r.get<int64_t>(), bt = r.get<int64_t>();
    const uint32_t next = r.get<uint32_t>(), nseg = r.get<uint32_t>();
    const uint64_t ne = r.get<uint64_t>(), ns = r.get<uint64_t>();
    (void)r.get<int64_t>();
    const char* kt = r.take(kl);
    if (r.bad || ver != kVersion || k != key_ || !kt || std::string(kt, kl) != cfg_.key_text ||
        eb != cfg_.entry_bytes || sb != cfg_.snap_bytes || bt != cfg_.block_tokens) {
        RAD_WARN("prefix store: %s was written under another key or geometry; discarded", ip.c_str());
        return RAD_E_FORMAT;
    }
    std::unordered_map<uint32_t, Seg> segs;
    for (uint32_t i = 0; i < nseg; ++i) {
        const uint32_t id = r.get<uint32_t>();
        const int64_t bytes = r.get<int64_t>();
        segs[id].bytes = bytes;
    }
    if (r.bad) return RAD_E_FORMAT;
    /* Each segment must be whole and carry this key; a short or foreign one loses its items. */
    std::unordered_map<uint32_t, bool> good;
    for (auto& kv : segs) {
        const std::string sp = seg_path(kv.first);
        const int fd = ::open(sp.c_str(), O_RDONLY | O_CLOEXEC);
        bool ok = false;
        if (fd >= 0) {
            struct stat st {};
            char h[kSegHeader];
            ok = fstat(fd, &st) == 0 && st.st_size == kv.second.bytes &&
                 ::pread(fd, h, sizeof h, 0) == (ssize_t)sizeof h && !std::memcmp(h, kSegMagic, 8);
            if (ok) {
                uint32_t v = 0, id = 0; uint64_t hi = 0, lo = 0;
                std::memcpy(&v, h + 8, 4); std::memcpy(&id, h + 12, 4);
                std::memcpy(&hi, h + 16, 8); std::memcpy(&lo, h + 24, 8);
                ok = v == kVersion && id == kv.first && hi == key_.hi && lo == key_.lo;
            }
            ::close(fd);
        }
        if (!ok) RAD_WARN("prefix store: segment %s is missing, short or foreign; its items are dropped",
                          sp.c_str());
        good[kv.first] = ok;
    }
    const int64_t cut_ms = cfg_.ttl_s > 0 ? now_ms() - cfg_.ttl_s * 1000 : INT64_MIN;
    auto fits = [&](const PcLoc& l, int64_t n) {
        auto g = good.find(l.seg);
        return g != good.end() && g->second && (int64_t)l.off >= kSegHeader &&
               (int64_t)l.off + n <= segs[l.seg].bytes;
    };
    int64_t expired = 0;
    for (uint64_t i = 0; i < ne && !r.bad; ++i) {
        PcEntry e;
        e.key.hi = r.get<uint64_t>(); e.key.lo = r.get<uint64_t>();
        e.loc.seg = r.get<uint32_t>(); e.loc.off = r.get<uint64_t>(); e.loc.sum = r.get<uint64_t>();
        e.depth = r.get<int64_t>(); e.n_children = r.get<uint8_t>(); e.used_ms = r.get<int64_t>();
        const char* t = r.take((size_t)cfg_.block_tokens * 4);
        if (r.bad || !t) break;
        e.tokens.resize((size_t)cfg_.block_tokens);
        std::memcpy(e.tokens.data(), t, (size_t)cfg_.block_tokens * 4);
        if (!fits(e.loc, cfg_.entry_bytes)) continue;
        if (e.used_ms < cut_ms) { ++expired; continue; }
        EMeta m{ e.tokens, e.depth, e.n_children, e.used_ms, e.loc };
        ents_[e.key] = std::move(m);
        load_e_.push_back(std::move(e));
    }
    for (uint64_t i = 0; i < ns && !r.bad; ++i) {
        PcSnap s;
        s.key.hi = r.get<uint64_t>(); s.key.lo = r.get<uint64_t>();
        s.loc.seg = r.get<uint32_t>(); s.loc.off = r.get<uint64_t>(); s.loc.sum = r.get<uint64_t>();
        s.pos = r.get<int64_t>(); s.group = r.get<uint64_t>(); s.used_ms = r.get<int64_t>();
        if (r.bad) break;
        if (cfg_.snap_bytes <= 0 || !fits(s.loc, cfg_.snap_bytes)) continue;
        if (s.used_ms < cut_ms) { ++expired; continue; }
        snaps_[s.key] = SMeta{ s.pos, s.group, s.used_ms, s.loc };
        load_s_.push_back(s);
    }
    if (r.bad) {
        RAD_WARN("prefix store: %s ends early; starting empty", ip.c_str());
        ents_.clear(); snaps_.clear(); load_e_.clear(); load_s_.clear();
        return RAD_E_FORMAT;
    }
    for (auto& kv : segs) if (good[kv.first]) segs_[kv.first] = kv.second;
    next_seg_ = next;
    for (const auto& kv : segs_) next_seg_ = std::max(next_seg_, kv.first + 1);
    std::stable_sort(load_e_.begin(), load_e_.end(),
                     [](const PcEntry& a, const PcEntry& b) {
                         return a.used_ms != b.used_ms ? a.used_ms < b.used_ms : a.depth < b.depth; });
    std::stable_sort(load_s_.begin(), load_s_.end(),
                     [](const PcSnap& a, const PcSnap& b) { return a.used_ms < b.used_ms; });
    if (expired) RAD_INFO("prefix store: %lld item(s) older than --prefix-cache-ttl were not loaded",
                          (long long)expired);
    return RAD_OK;
}

/* Temporary files, and segments the index does not name (a flush that died before its index). */
void PrefixStore::sweep(const std::unordered_map<uint32_t, Seg>& keep) {
    DIR* dp = ::opendir(path_.c_str());
    if (!dp) return;
    std::vector<std::string> gone;
    while (dirent* e = ::readdir(dp)) {
        const std::string n = e->d_name;
        if (n.find(".tmp") != std::string::npos) { gone.push_back(n); continue; }
        unsigned id = 0;
        char tail[8] = {};
        if (std::sscanf(n.c_str(), "seg-%u.rp%1s", &id, tail) == 2 && !keep.count(id)) gone.push_back(n);
    }
    ::closedir(dp);
    for (const std::string& n : gone) ::unlink((path_ + "/" + n).c_str());
}

int64_t PrefixStore::read_entries(const BlockHash* keys, int64_t n, void* const* dst, char* ok) {
    const uint64_t t0 = rad_mono_ns();
    struct Want { uint32_t seg; uint64_t off; int64_t i; };
    std::vector<Want> w;
    w.reserve((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        ok[i] = 0;
        auto it = ents_.find(keys[i]);
        if (it != ents_.end()) w.push_back({ it->second.loc.seg, it->second.loc.off, i });
    }
    std::sort(w.begin(), w.end(), [](const Want& a, const Want& b) {
        return a.seg != b.seg ? a.seg < b.seg : a.off < b.off; });
    const int64_t eb = cfg_.entry_bytes;
    std::vector<char> got((size_t)n, 0);
    std::vector<iovec> iov;
    for (size_t a = 0; a < w.size();) {
        size_t z = a + 1;
        while (z < w.size() && z - a < 1024 && w[z].seg == w[a].seg &&
               w[z].off == w[z - 1].off + (uint64_t)eb)
            ++z;
        iov.clear();
        for (size_t k = a; k < z; ++k) iov.push_back({ dst[w[k].i], (size_t)eb });
        const int fd = seg_fd(w[a].seg);
        int64_t want = (int64_t)(z - a) * eb, done = 0;
        bool good = fd >= 0;
        size_t vi = 0;
        while (good && done < want) {
            const ssize_t r = ::preadv(fd, iov.data() + vi, (int)(iov.size() - vi),
                                       (off_t)(w[a].off + (uint64_t)done));
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { good = false; break; }
            done += r;
            /* Advance past whole and part iovecs. */
            int64_t left = r;
            while (left > 0 && vi < iov.size()) {
                if ((int64_t)iov[vi].iov_len <= left) { left -= (int64_t)iov[vi].iov_len; ++vi; }
                else {
                    iov[vi].iov_base = static_cast<char*>(iov[vi].iov_base) + left;
                    iov[vi].iov_len -= (size_t)left;
                    left = 0;
                }
            }
        }
        if (good) for (size_t k = a; k < z; ++k) got[(size_t)w[k].i] = 1;
        a = z;
    }
    int64_t n_ok = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!got[(size_t)i]) continue;
        auto it = ents_.find(keys[i]);
        if (pc_checksum(dst[i], eb) == it->second.loc.sum) { ok[i] = 1; ++n_ok; }
    }
    for (int64_t i = 0; i < n; ++i)
        if (!ok[i] && ents_.count(keys[i])) {
            ++st_.read_failures;
            RAD_WARN("prefix store: entry %s did not read back whole (I/O or checksum); dropped",
                     keys[i].str().c_str());
            ents_.erase(keys[i]);
        }
    ++st_.reads;
    st_.read_bytes += n_ok * eb;
    st_.read_s += secs_since(t0);
    return n_ok;
}

int PrefixStore::read_snap(const BlockHash& key, void* dst) {
    auto it = snaps_.find(key);
    if (it == snaps_.end() || cfg_.snap_bytes <= 0) return RAD_E_NOTFOUND;
    const uint64_t t0 = rad_mono_ns();
    int st = read_at(it->second.loc.seg, it->second.loc.off, dst, cfg_.snap_bytes);
    if (st == RAD_OK && pc_checksum(dst, cfg_.snap_bytes) != it->second.loc.sum) st = RAD_E_FORMAT;
    if (st != RAD_OK) {
        ++st_.read_failures;
        RAD_WARN("prefix store: snapshot %s at %lld did not read back whole (%s); dropped",
                 key.str().c_str(), (long long)it->second.pos,
                 st == RAD_E_FORMAT ? "checksum" : "I/O");
        snaps_.erase(it);
        return st;
    }
    ++st_.reads;
    st_.read_bytes += cfg_.snap_bytes;
    st_.read_s += secs_since(t0);
    return RAD_OK;
}

int PrefixStore::write_index(const std::unordered_map<BlockHash, EMeta, BlockHashHash>& e,
                             const std::unordered_map<BlockHash, SMeta, BlockHashHash>& s,
                             const std::unordered_map<uint32_t, Seg>& segs, uint32_t next_seg) {
    std::string b;
    b.reserve(256 + cfg_.key_text.size() + e.size() * (53 + (size_t)cfg_.block_tokens * 4) +
              s.size() * 60 + segs.size() * 12);
    b.append(kIdxMagic, 8);
    put(b, kVersion);
    put(b, (uint32_t)cfg_.key_text.size());
    put(b, key_.hi); put(b, key_.lo);
    put(b, cfg_.entry_bytes); put(b, cfg_.snap_bytes); put(b, cfg_.block_tokens);
    put(b, next_seg); put(b, (uint32_t)segs.size());
    put(b, (uint64_t)e.size()); put(b, (uint64_t)s.size());
    put(b, now_ms());
    b += cfg_.key_text;
    for (const auto& kv : segs) { put(b, kv.first); put(b, kv.second.bytes); }
    for (const auto& kv : e) {
        put(b, kv.first.hi); put(b, kv.first.lo);
        put(b, kv.second.loc.seg); put(b, kv.second.loc.off); put(b, kv.second.loc.sum);
        put(b, kv.second.depth); put(b, kv.second.n_children); put(b, kv.second.used_ms);
        b.append(reinterpret_cast<const char*>(kv.second.tokens.data()), (size_t)cfg_.block_tokens * 4);
    }
    for (const auto& kv : s) {
        put(b, kv.first.hi); put(b, kv.first.lo);
        put(b, kv.second.loc.seg); put(b, kv.second.loc.off); put(b, kv.second.loc.sum);
        put(b, kv.second.pos); put(b, kv.second.group); put(b, kv.second.used_ms);
    }
    put(b, pc_checksum(b.data(), (int64_t)b.size()));
    b.append(kEndMagic, 8);

    const std::string tmp = path_ + "/index.rpx.tmp", fin = path_ + "/index.rpx";
    const int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
    if (fd < 0) return RAD_E_IO;
    const bool ok = write_all(fd, b.data(), (int64_t)b.size()) && ::fsync(fd) == 0;
    ::close(fd);
    if (!ok || ::rename(tmp.c_str(), fin.c_str()) != 0) {
        const int e2 = errno;
        ::unlink(tmp.c_str());
        RAD_WARN("prefix store: writing %s failed: %s", fin.c_str(), std::strerror(e2));
        return RAD_E_IO;
    }
    fsync_dir(path_);
    return RAD_OK;
}

int PrefixStore::flush(const std::vector<PcPut>& ents, const std::vector<PcPut>& snaps, PcFlush* out) {
    PcFlush res;
    if (!enabled_) return RAD_E_STATE;
    const uint64_t t0 = rad_mono_ns();
    const int64_t eb = cfg_.entry_bytes, sb = cfg_.snap_bytes;

    std::unordered_map<BlockHash, EMeta, BlockHashHash> ne;
    std::unordered_map<BlockHash, SMeta, BlockHashHash> ns;
    struct Job { bool snap; BlockHash key; const void* bytes; };
    std::vector<Job> jobs;
    int64_t live = 0;
    /* Snapshots first in the budget: one is worth thousands of entries' prefill on a hybrid. */
    for (const PcPut& p : snaps) {
        if (sb <= 0 || ns.count(p.key)) continue;
        if (cfg_.cap_bytes > 0 && live + sb > cfg_.cap_bytes) { ++res.dropped; continue; }
        auto old = snaps_.find(p.key);
        SMeta m{ p.pos, p.group, p.used_ms, {} };
        if (!p.bytes) {
            if (old == snaps_.end()) { ++res.dropped; continue; }
            m.loc = old->second.loc;
        } else if (!p.fresh && old != snaps_.end()) {
            m.loc = old->second.loc;
        } else {
            jobs.push_back({ true, p.key, p.bytes });
        }
        ns[p.key] = m;
        live += sb;
    }
    for (const PcPut& p : ents) {
        if (ne.count(p.key)) continue;
        if (cfg_.cap_bytes > 0 && live + eb > cfg_.cap_bytes) { ++res.dropped; continue; }
        auto old = ents_.find(p.key);
        EMeta m;
        m.depth = p.depth; m.n_children = p.n_children; m.used_ms = p.used_ms;
        if (!p.bytes) {
            if (old == ents_.end()) { ++res.dropped; continue; }
            m.tokens = old->second.tokens;
            m.loc = old->second.loc;
        } else {
            if (!p.tokens) continue;
            m.tokens.assign(p.tokens, p.tokens + cfg_.block_tokens);
            if (!p.fresh && old != ents_.end() && old->second.tokens == m.tokens) m.loc = old->second.loc;
            else jobs.push_back({ false, p.key, p.bytes });
        }
        ne[p.key] = std::move(m);
        live += eb;
    }

    /* Compaction: dead bytes past the live ones (and past 256 MiB) rewrite everything. */
    int64_t kept_old = 0, old_total = 0;
    for (const auto& kv : segs_) old_total += kv.second.bytes - kSegHeader;
    for (const auto& kv : ne) if (kv.second.loc.seg) kept_old += eb;
    for (const auto& kv : ns) if (kv.second.loc.seg) kept_old += sb;
    const int64_t dead = old_total - kept_old;
    const bool compact = dead > 256 * kMiB && dead > kept_old;
    std::unordered_map<BlockHash, const void*, BlockHashHash> ram_e, ram_s;
    if (compact) {
        for (const PcPut& p : ents) if (p.bytes) ram_e[p.key] = p.bytes;
        for (const PcPut& p : snaps) if (p.bytes) ram_s[p.key] = p.bytes;
        for (auto& kv : ne) if (kv.second.loc.seg) {
            auto r = ram_e.find(kv.first);
            jobs.push_back({ false, kv.first, r == ram_e.end() ? nullptr : r->second });
        }
        for (auto& kv : ns) if (kv.second.loc.seg) {
            auto r = ram_s.find(kv.first);
            jobs.push_back({ true, kv.first, r == ram_s.end() ? nullptr : r->second });
        }
    }

    std::unordered_map<uint32_t, Seg> segs;
    uint32_t next = next_seg_;
    std::string seg_tmp, seg_fin;
    uint32_t new_id = 0;
    int64_t write_bytes = 0;
    for (const Job& j : jobs) write_bytes += j.snap ? sb : eb;
    auto fail = [&](const char* what, int e) {
        if (!seg_tmp.empty()) ::unlink(seg_tmp.c_str());
        if (!seg_fin.empty()) ::unlink(seg_fin.c_str());
        ++st_.flush_failures;
        RAD_WARN("prefix store: flush to %s failed (%s: %s); the previous store is kept",
                 path_.c_str(), what, std::strerror(e));
        if (out) *out = res;
        return RAD_E_IO;
    };
    if (!jobs.empty()) {
        struct statvfs vf {};
        if (::statvfs(path_.c_str(), &vf) == 0) {
            const int64_t avail = (int64_t)vf.f_bavail * (int64_t)vf.f_frsize;
            if (avail < write_bytes + 64 * kMiB) return fail("not enough free space", ENOSPC);
        }
        new_id = next++;
        seg_tmp = seg_path(new_id) + ".tmp";
        const int fd = ::open(seg_tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0) return fail("create", errno);
        char h[kSegHeader] = {};
        std::memcpy(h, kSegMagic, 8);
        std::memcpy(h + 8, &kVersion, 4);
        std::memcpy(h + 12, &new_id, 4);
        std::memcpy(h + 16, &key_.hi, 8);
        std::memcpy(h + 24, &key_.lo, 8);
        std::memcpy(h + 32, &eb, 8);
        std::memcpy(h + 40, &sb, 8);
        bool ok = write_all(fd, h, kSegHeader);
        int err = ok ? 0 : errno;
        int64_t off = kSegHeader;
        std::vector<char> bounce;
        std::vector<iovec> iov;
        int64_t pending = 0, written = 0;
        auto drain = [&]() {
            int64_t left = pending;
            size_t vi = 0;
            while (ok && left > 0) {
                if (fail_after_ >= 0 && written + left > fail_after_) { ok = false; err = ENOSPC; break; }
                const ssize_t w = ::writev(fd, iov.data() + vi, (int)std::min<size_t>(iov.size() - vi, 1024));
                if (w < 0 && errno == EINTR) continue;
                if (w <= 0) { ok = false; err = w < 0 ? errno : EIO; break; }
                left -= w; written += w;
                int64_t adv = w;
                while (adv > 0 && vi < iov.size()) {
                    if ((int64_t)iov[vi].iov_len <= adv) { adv -= (int64_t)iov[vi].iov_len; ++vi; }
                    else {
                        iov[vi].iov_base = static_cast<char*>(iov[vi].iov_base) + adv;
                        iov[vi].iov_len -= (size_t)adv;
                        adv = 0;
                    }
                }
            }
            iov.clear();
            pending = 0;
        };
        for (const Job& j : jobs) {
            if (!ok) break;
            const int64_t n = j.snap ? sb : eb;
            const void* src = j.bytes;
            if (!src) {
                /* Compaction of an item held only on disk: through a bounce buffer, checked. */
                drain();
                bounce.resize((size_t)n);
                const PcLoc& l = j.snap ? snaps_[j.key].loc : ents_[j.key].loc;
                if (read_at(l.seg, l.off, bounce.data(), n) != RAD_OK ||
                    pc_checksum(bounce.data(), n) != l.sum) {
                    /* Unreadable: leave it out rather than fail the whole flush. */
                    if (j.snap) { ns.erase(j.key); res.gone_snaps.push_back(j.key); }
                    else { ne.erase(j.key); res.gone_entries.push_back(j.key); }
                    continue;
                }
                src = bounce.data();
            }
            const uint64_t sum = pc_checksum(src, n);
            PcLoc loc{ new_id, (uint64_t)off, sum };
            if (j.snap) ns[j.key].loc = loc; else ne[j.key].loc = loc;
            iov.push_back({ const_cast<void*>(src), (size_t)n });
            pending += n;
            off += n;
            if (!j.bytes || iov.size() >= 1024 || pending >= 256 * kMiB) drain();
            res.written_bytes += n;
            ++res.written_items;
        }
        drain();
        if (ok && ::fsync(fd) != 0) { ok = false; err = errno; }
        ::close(fd);
        if (!ok) return fail("write", err);
        seg_fin = seg_path(new_id);
        if (::rename(seg_tmp.c_str(), seg_fin.c_str()) != 0) return fail("rename", errno);
        seg_tmp.clear();
        segs[new_id].bytes = off;
    }
    /* The old segments still referenced stay. */
    for (const auto& kv : ne) {
        auto s = segs_.find(kv.second.loc.seg);
        if (s != segs_.end()) segs[kv.second.loc.seg].bytes = s->second.bytes;
    }
    for (const auto& kv : ns) {
        auto s = segs_.find(kv.second.loc.seg);
        if (s != segs_.end()) segs[kv.second.loc.seg].bytes = s->second.bytes;
    }
    if (write_index(ne, ns, segs, next) != RAD_OK) return fail("index", errno);
    seg_fin.clear();

    /* Committed. */
    for (const auto& kv : ents_) if (!ne.count(kv.first)) res.gone_entries.push_back(kv.first);
    for (const auto& kv : snaps_) if (!ns.count(kv.first)) res.gone_snaps.push_back(kv.first);
    for (const PcPut& p : ents) if (p.bytes && ne.count(p.key)) res.stored_entries.push_back(p.key);
    for (const PcPut& p : snaps) if (p.bytes && ns.count(p.key)) res.stored_snaps.push_back(p.key);
    for (auto& kv : segs_) {
        if (segs.count(kv.first)) { segs[kv.first].fd = kv.second.fd; kv.second.fd = -1; continue; }
        if (kv.second.fd >= 0) ::close(kv.second.fd);
        kv.second.fd = -1;
        ::unlink(seg_path(kv.first).c_str());
        ++res.segs_deleted;
    }
    ents_ = std::move(ne);
    snaps_ = std::move(ns);
    segs_ = std::move(segs);
    next_seg_ = next;
    release_loaded();
    res.kept_items = (int64_t)(ents_.size() + snaps_.size());
    res.live_bytes = (int64_t)ents_.size() * eb + (int64_t)snaps_.size() * sb;
    res.compacted = compact;
    res.secs = secs_since(t0);
    ++st_.flushes;
    st_.written_bytes += res.written_bytes;
    st_.write_s += res.secs;
    if (out) *out = std::move(res);
    return RAD_OK;
}


/* ------------------------------------------------------------------ the cache's side */

namespace {
int64_t mono_to_unix_ms(uint64_t mono, uint64_t mono_now, int64_t unix_now) {
    return mono >= mono_now ? unix_now : unix_now - (int64_t)((mono_now - mono) / 1000000ull);
}
uint64_t unix_ms_to_mono(int64_t ms, uint64_t mono_now, int64_t unix_now) {
    const int64_t ago = unix_now > ms ? unix_now - ms : 0;
    const uint64_t d = (uint64_t)ago * 1000000ull;
    return d < mono_now ? mono_now - d : 1;
}
}  // namespace

int pc_flush_cache(PrefixCache& pc, IdleTiers& t, const HostArena* ck, PrefixStore& st,
                   int64_t ttl_s, PcFlush* out) {
    std::vector<PrefixCache::PersistEntry> ents;
    std::vector<PrefixCache::PersistSnap> snaps;
    pc.persist_export(&ents, &snaps);
    const uint64_t mono = rad_mono_ns();
    const int64_t unix_ms = now_ms();
    const uint64_t cut = ttl_s > 0 && mono > (uint64_t)ttl_s * 1000000000ull
                             ? mono - (uint64_t)ttl_s * 1000000000ull : 0;
    std::sort(ents.begin(), ents.end(), [](const auto& a, const auto& b) {
        return a.used_ns != b.used_ns ? a.used_ns > b.used_ns : a.depth < b.depth; });
    std::sort(snaps.begin(), snaps.end(), [](const auto& a, const auto& b) {
        return a.used_ns > b.used_ns; });
    std::vector<PcPut> pe, ps;
    pe.reserve(ents.size());
    for (const auto& e : ents) {
        if (e.used_ns < cut) continue;
        PcPut p;
        p.key = e.key;
        p.bytes = e.host_slot >= 0 ? t.host_ptr(e.host_slot) : nullptr;
        p.fresh = !e.disk;
        p.tokens = e.tokens;
        p.depth = e.depth;
        p.n_children = e.n_children;
        p.used_ms = mono_to_unix_ms(e.used_ns, mono, unix_ms);
        pe.push_back(p);
    }
    for (const auto& s : snaps) {
        if (s.used_ns < cut) continue;
        PcPut p;
        p.key = s.key;
        p.bytes = s.slot >= 0 && ck && ck->enabled() ? ck->at(s.slot) : nullptr;
        if (s.slot >= 0 && !p.bytes) continue;
        p.fresh = !s.disk;
        p.pos = s.pos;
        p.group = s.session;
        p.used_ms = mono_to_unix_ms(s.used_ns, mono, unix_ms);
        ps.push_back(p);
    }
    PcFlush f;
    const int rc = st.flush(pe, ps, &f);
    if (rc == RAD_OK) {
        pc.persist_stored(f.stored_entries, f.stored_snaps);
        /* What the store no longer holds, and what was offered from disk but left out. */
        std::vector<BlockHash> ge = f.gone_entries, gs = f.gone_snaps;
        for (const PcPut& p : pe) if (!st.has_entry(p.key)) ge.push_back(p.key);
        for (const PcPut& p : ps) if (!st.has_snap(p.key)) gs.push_back(p.key);
        pc.persist_gone(ge, gs);
    }
    if (out) *out = std::move(f);
    return rc;
}

int64_t pc_import(PrefixCache& pc, PrefixStore& st, int64_t* snaps) {
    const uint64_t mono = rad_mono_ns();
    const int64_t unix_ms = now_ms();
    int64_t n = 0, k = 0;
    for (const PcEntry& e : st.loaded_entries())
        if (pc.import_entry(e.key, e.tokens, e.depth, e.n_children,
                            unix_ms_to_mono(e.used_ms, mono, unix_ms)) == RAD_OK) ++n;
    /* One session a group: the retention policy sees each conversation's ladder again. Ids
     * from 2^62 never meet a request id. */
    std::unordered_map<uint64_t, uint64_t> remap;
    for (const PcSnap& s : st.loaded_snaps()) {
        auto it = remap.find(s.group);
        const uint64_t sid = it != remap.end() ? it->second
                                               : (remap[s.group] = (1ull << 62) + remap.size());
        if (pc.import_snapshot(s.key, s.pos, sid, unix_ms_to_mono(s.used_ms, mono, unix_ms)) == RAD_OK)
            ++k;
    }
    if (snaps) *snaps = k;
    st.release_loaded();
    return n;
}

int pc_page_in(PrefixCache& pc, IdleTiers& t, HostArena* ck, PrefixStore& st,
               const std::vector<int32_t>& tokens, const std::vector<MMContentKey>& mm,
               PcPageIn* out) {
    PcPageIn r;
    PrefixCache::PageIn plan;
    pc.plan_page_in(tokens, mm, &plan);
    if (plan.entries.empty() && !plan.has_snap) { if (out) *out = r; return RAD_OK; }
    const uint64_t t0 = rad_mono_ns();
    std::vector<BlockHash> keys;
    std::vector<int32_t> slots;
    std::vector<void*> dst;
    for (const BlockHash& h : plan.entries) {
        int32_t s = -1;
        if (t.page_in_slot(h, &s) != RAD_OK) break;     /* a full arena ends the chain here */
        keys.push_back(h);
        slots.push_back(s);
        dst.push_back(t.host_ptr(s));
    }
    if (!keys.empty()) {
        std::vector<char> ok(keys.size(), 0);
        st.read_entries(keys.data(), (int64_t)keys.size(), dst.data(), ok.data());
        const uint64_t now = rad_mono_ns();
        for (size_t i = 0; i < keys.size(); ++i) {
            t.page_in_done(keys[i], slots[i], ok[i] != 0, now);
            if (ok[i]) { ++r.entries; r.bytes += st.stats().reads >= 0 ? t.entry_bytes() : 0; }
            else { ++r.failed; pc.drop_offdevice(keys[i]); }
        }
    }
    if (plan.has_snap && ck && ck->enabled()) {
        int32_t s = -1;
        if (pc.snap_page_in_slot(plan.snap, &s) == RAD_OK && s >= 0) {
            const bool ok = st.read_snap(plan.snap, ck->at(s)) == RAD_OK;
            pc.snap_page_in_done(plan.snap, s, ok);
            if (ok) { r.snap = true; r.bytes += ck->slot_bytes(); } else ++r.failed;
        }
    }
    r.secs = secs_since(t0);
    if (out) *out = r;
    return RAD_OK;
}

}  /* namespace rad */
