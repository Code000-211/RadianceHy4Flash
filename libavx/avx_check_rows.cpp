/* avx_check_rows.cpp -- embed_lookup_q over a file-backed table, as a model serves it.
 *
 * The geometry table checks the op against libref over a heap buffer. A table in a container is
 * read differently: through a shared row cache, direct file reads for misses and a reader thread
 * one call ahead (avx_ngram.cpp). The failures there are about which bytes arrive (a slot claimed
 * twice, a row decoded from a slot taken over, a wait on an abandoned claim), and libref has no
 * such path, so the reference is the file's own bytes.
 *
 * Two threads act as ranks 0 and 1 of 2 over the same ids, released together, so each claims rows,
 * waits on the other's claims and meets its own claim again when an id repeats. Half the ids come
 * from a small hot set; the rest from a table four times the cache's 2^19 rows, so sets fill and
 * rows are evicted while the other thread may be reading them. Each call also passes the next
 * call's ids as `ahead`, so the reader thread races both ranks.
 *
 * Each output row is compared bit for bit with the file's bytes through the call's scale; a padding
 * id must give a zero row. Rows are 40 codes, so the AVX-512 decode runs one 32-code step plus a
 * masked tail. Direct reads align to the filesystem's reported DIO alignment (ext4, XFS) or to its
 * block size where none is reported (btrfs, tmpfs since Linux 6.6); the case reports which.
 *
 * Each table is written once per process under a name of its own (mkostemp), stays mapped for the
 * rest of the process and is unlinked once the gather has opened it: the ISA levels share the file
 * reads and differ only in the decode. */
#include "avx_check.h"

#include <barrier>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

#include <cstddef>

#include <fcntl.h>
#include <linux/filter.h>
#include <linux/io_uring.h>
#include <linux/seccomp.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

/* The errno of creating a ring as the gather does (avx_ngram.cpp: ReadRing::init), or 0. */
static int ring_probe() {
    io_uring_params p{};
    p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
    const long ring = syscall(__NR_io_uring_setup, 4096, &p);
    if (ring < 0) return errno;
    ::close((int)ring);
    return 0;
}

/* Ring memory can count against the locked-memory limit, which every process of the user shares,
 * so a run beside other io_uring users can be refused one (ENOMEM) for a while. That is the
 * machine, not the gather: the cases name it and skip. */
static const char* const kRingMemory = "io_uring ring memory refused (ENOMEM); the locked-memory "
                                       "limit is shared with this user's other processes";

/* A bf16 gather from a small table file in `dir`, as a process without io_uring serves it: every
 * miss is a synchronous pread (avx_ngram.cpp: ring_sync_mode). Run in the child that refuses the
 * ring, so it is the one place the fallback's reads are checked against the file's bytes. 0 when
 * every row is exact, 1 when a call fails or a row differs, 4 when `dir` cannot do direct I/O (the
 * gather opens its tables O_DIRECT, ring or not). */
static int pread_gather(const RadKernelInfo* r, const std::string& dir) {
    const int64_t n = 160, V = 8192, M = 256, rb = 2 * n;
    std::string path = dir + "/rad-avx-check-pread.XXXXXX";
    const int wfd = ::mkostemp(path.data(), O_CLOEXEC);
    if (wfd < 0) return 1;
    uint64_t x = 0x2545f4914f6cdd1dull;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    std::vector<uint64_t> words((size_t)(V * rb / 8));
    for (auto& w : words) w = rnd();
    const bool wrote = ::write(wfd, words.data(), words.size() * 8) == (ssize_t)(words.size() * 8) &&
                       ::fsync(wfd) == 0;
    ::close(wfd);
    const int dfd = wrote ? ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT) : -1;
    if (dfd < 0) { ::unlink(path.c_str()); return wrote ? 4 : 1; }
    ::close(dfd);
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    void* map = fd < 0 ? MAP_FAILED : mmap(nullptr, (size_t)(V * rb), PROT_READ, MAP_SHARED, fd, 0);
    if (fd >= 0) ::close(fd);
    if (map == MAP_FAILED) { ::unlink(path.c_str()); return 1; }
    const uint8_t* tab = static_cast<const uint8_t*>(map);

    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
        return t;
    };
    std::vector<int32_t> ids((size_t)M);
    std::vector<uint16_t> out((size_t)(M * n));
    std::vector<RadTensor> ts = {
        tensor(ids.data(), RAD_I32, {M}),
        tensor(const_cast<uint8_t*>(tab), RAD_BF16, {V, n}),
        tensor(nullptr, RAD_BF16, {1}),
        tensor(out.data(), RAD_BF16, {M, n}),
        tensor(nullptr, RAD_I32, {M}) };
    RadParam ps[3] = {};
    ps[0].key = "M";       ps[0].kind = RAD_P_INT; ps[0].ival = M;
    ps[1].key = "n_embd";  ps[1].kind = RAD_P_INT; ps[1].ival = n;
    ps[2].key = "n_vocab"; ps[2].kind = RAD_P_INT; ps[2].ival = V;
    RadArgs a{};
    a.t = ts.data(); a.n_t = (int)ts.size();
    a.p = ps; a.n_p = 3;
    a.rank = 0; a.world_size = 1;
    int rc = 0;
    /* Two calls over overlapping ids: the first misses everywhere, the second meets rows the first
     * published to the cache, a repeat inside the call and padding. */
    for (int c = 0; c < 2 && rc == 0; ++c) {
        for (int64_t m = 0; m < M; ++m) {
            const uint64_t v = rnd();
            ids[(size_t)m] = v % 13 == 0 ? -1 : (m > 0 && v % 7 == 0) ? ids[(size_t)m - 1]
                                                                       : (int32_t)((v >> 8) % 1024);
        }
        std::fill(out.begin(), out.end(), (uint16_t)0xdead);
        if (r->launch(&a, nullptr) != RAD_OK) { rc = 1; break; }
        for (int64_t m = 0; m < M && rc == 0; ++m) {
            const int32_t id = ids[(size_t)m];
            for (int64_t i = 0; i < n; ++i) {
                uint16_t want = 0;
                if (id >= 0) std::memcpy(&want, tab + ((int64_t)id * n + i) * 2, 2);
                if (out[(size_t)(m * n + i)] != want) { rc = 1; break; }
            }
        }
    }
    ::unlink(path.c_str());
    return rc;
}

/* embed_lookup_q's init must accept a process that cannot create an io_uring ring and say why and
 * what it does instead (Docker's default seccomp profile returns EPERM for io_uring_setup): since
 * the pread fallback (aa84ec2ad) the model starts there, its n-gram reads are only slower. A child
 * installs that refusal as a seccomp filter; init must then accept a bf16 gather (read through the
 * fallback) and an f32 one (read as memory), name the Docker option and the pread fallback on
 * stderr, and a bf16 gather from a file must return the file's rows exactly. This process's init
 * must be accepted whether a ring can be created here or not, as in a build sandbox. */
bool ngram_init_case(const Lib& lib, const std::string& dir) {
    const char* what = "embed_lookup_q's init";
    const RadKernelInfo* r = lib.row("embed_lookup_q");
    if (!r || !r->init || !r->launch) {
        std::printf("  %s: the row has no init or no launch\n", what);
        return false;
    }
    RadParam bf16[2]{}, f32[2]{};
    bf16[0].key = f32[0].key = "n_embd";
    bf16[0].kind = f32[0].kind = RAD_P_INT;
    bf16[0].ival = f32[0].ival = 160;
    bf16[1].key = f32[1].key = "dtype";
    bf16[1].kind = f32[1].kind = RAD_P_STR;
    bf16[1].sval = "bf16";
    f32[1].sval = "f32";

    int out[2];
    if (::pipe(out) != 0) {
        std::printf("  %s: pipe: %s\n", what, std::strerror(errno));
        return false;
    }
    std::fflush(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::dup2(out[1], 2);
        ::close(out[0]);
        sock_filter f[] = {
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_io_uring_setup, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EPERM & SECCOMP_RET_DATA)),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        };
        sock_fprog prog{ (unsigned short)(sizeof f / sizeof f[0]), f };
        if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
            ::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0)
            ::_exit(3);
        void* inst = nullptr;
        const int a = r->init(bf16, 2, 0, 1, &inst);
        const int b = r->init(f32, 2, 0, 1, &inst);
        std::fflush(nullptr);
        if (a != RAD_OK || b != RAD_OK) ::_exit(1);
        const int g = pread_gather(r, dir);
        std::fflush(nullptr);
        ::_exit(g == 0 ? 0 : g == 4 ? 4 : 2);
    }
    ::close(out[1]);
    std::string said;
    char buf[512];
    for (ssize_t k; (k = ::read(out[0], buf, sizeof buf)) > 0;) said.append(buf, (size_t)k);
    ::close(out[0]);
    int ws = 0;
    if (pid < 0 || ::waitpid(pid, &ws, 0) != pid || !WIFEXITED(ws)) {
        std::printf("  %s: the child did not run to the end\n", what);
        return false;
    }
    bool ok = true;
    const int code = WEXITSTATUS(ws);
    const bool named = said.find("--security-opt seccomp=unconfined") != std::string::npos &&
                       said.find("pread") != std::string::npos;
    if (code == 3) {
        std::printf("  %s, where io_uring is refused: SKIPPED -- no seccomp filter can be installed "
                    "here\n", what);
    } else if (code == 1 || code == 2 || !named) {
        std::printf("  %s, where io_uring is refused: FAIL -- %s; it said: %s\n", what,
                    code == 1 ? "the init refused" : code == 2 ? "the pread gather was wrong"
                                                               : "the reason or the fallback was not named",
                    said.c_str());
        ok = false;
    } else {
        std::printf("  %s, where io_uring is refused: accepted, naming the Docker option and the pread "
                    "fallback; %s\n", what,
                    code == 4 ? ("the gather is SKIPPED -- " + dir + " cannot do direct I/O").c_str()
                              : "a bf16 gather through pread returns the file's rows exactly");
    }

    /* The init tests the ring once per process; either way it accepts. */
    const int here = ring_probe();
    void* inst = nullptr;
    const int st = r->init(bf16, 2, 0, 1, &inst);
    const bool right = st == RAD_OK;
    std::printf("  %s, in this process (io_uring %s): %s\n", what, here == 0 ? "available" : "refused",
                right ? "accepted" : "FAIL");
    return ok && right;
}

/* A table file, written and mapped once per process. */
struct RowTable {
    enum { OK, SKIP, FAIL } state = FAIL;
    std::string path, note;       /* note: the skip or failure reason, or the read granule */
    const uint8_t* tab = nullptr;
    bool unlinked = false;
};

static RowTable* row_table(const std::string& dir, bool bf16, int64_t V, int64_t rb) {
    static std::map<std::pair<std::string, bool>, RowTable> tables;
    const auto key = std::make_pair(dir, bf16);
    if (auto it = tables.find(key); it != tables.end()) return &it->second;
    RowTable& t = tables[key];

    /* A name of its own, so a concurrent run cannot truncate the file under this one's mapping. */
    std::string path = dir + "/rad-avx-check-rows.XXXXXX";
    uint64_t x = 0x9e3779b97f4a7c15ull;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    {
        const int fd = ::mkostemp(path.data(), O_CLOEXEC);
        if (fd < 0) { t.note = "cannot create " + path; return &t; }
        std::vector<uint64_t> chunk(1 << 16);
        bool wrote = true;
        for (int64_t done = 0; done < V * rb && wrote; done += (int64_t)(chunk.size() * 8)) {
            for (auto& w : chunk) w = rnd();
            const size_t want = (size_t)std::min<int64_t>(V * rb - done, (int64_t)chunk.size() * 8);
            wrote = ::write(fd, chunk.data(), want) == (ssize_t)want;
        }
        wrote = wrote && ::fsync(fd) == 0;
        ::close(fd);
        if (!wrote) { ::unlink(path.c_str()); t.note = "short write"; return &t; }
    }
    /* The gather refuses a filesystem without direct I/O (the O_DIRECT open fails, as on tmpfs
     * before Linux 6.6, or it reports a zero DIO alignment), so the case is skipped there. Every
     * other filesystem must work, with or without a reported alignment. */
    bool reported = false;
    {
        const int dfd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        struct statx sx {};
        const bool dio = dfd >= 0 &&
                         ::statx(dfd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &sx) == 0 &&
                         !((sx.stx_mask & STATX_DIOALIGN) && sx.stx_dio_offset_align == 0);
        reported = (sx.stx_mask & STATX_DIOALIGN) != 0;
        if (dfd >= 0) ::close(dfd);
        if (!dio) {
            ::unlink(path.c_str());
            t.state = RowTable::SKIP;
            t.note = dir + " cannot do direct I/O";
            return &t;
        }
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    void* map = fd < 0 ? MAP_FAILED : mmap(nullptr, (size_t)(V * rb), PROT_READ, MAP_SHARED, fd, 0);
    if (fd >= 0) ::close(fd);
    if (map == MAP_FAILED) { ::unlink(path.c_str()); t.note = "mmap"; return &t; }
    /* The mapping stays for the life of the process. libavx resolves a table to its file once, by
     * the mapping's address range (avx_ngram.cpp: table_file), so no other file may be mapped at
     * these addresses while the process runs. */
    t.tab = static_cast<const uint8_t*>(map);
    t.path = path;
    t.note = reported ? "the alignment the filesystem reports"
                      : "no alignment reported, read by the block size";
    t.state = RowTable::OK;
    return &t;
}

bool row_gather_case(const Lib& lib, const std::string& dir, int64_t calls, uint64_t seed,
                     bool verbose, bool bf16, bool per_row) {
    const RadKernelInfo* r = lib.row("embed_lookup_q");
    const char* what = bf16 ? "embed_lookup_q from a file, bf16 table"
                            : (per_row ? "embed_lookup_q from a file, a scale a row"
                                       : "embed_lookup_q from a file");
    if (!r || !r->launch) {
        std::printf("  %s: no row\n", what);
        return false;
    }
    /* A bf16 table has twice the row bytes and no scale, so it exercises the cache and the reads
     * at a row size the E4M3 table never has. */
    /* Misses are io_uring reads. Where a sandbox forbids io_uring the gather refuses every call by
     * design, so the case is skipped with the reason. */
    if (const int e = ring_probe()) {
        std::printf("  %s: SKIPPED -- %s\n", what,
                    e == ENOMEM ? kRingMemory
                                : (std::string("io_uring is unavailable here (") +
                                   std::strerror(e) + ")").c_str());
        return true;
    }
    const int64_t n = 40, V = int64_t(1) << 21, M = 512, hot = 4096;
    const int64_t rb = bf16 ? 2 * n : n;
    const float scale = 0.375f;

    RowTable* rt = row_table(dir, bf16, V, rb);
    if (rt->state == RowTable::SKIP) {
        std::printf("  %s: SKIPPED -- %s\n", what, rt->note.c_str());
        return true;
    }
    if (rt->state != RowTable::OK) {
        std::printf("  %s: %s\n", what, rt->note.c_str());
        return false;
    }
    const char* granule = rt->note.c_str();
    const uint8_t* tab = rt->tab;
    uint64_t x = seed ? seed : 1;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };

    std::vector<int32_t> hotset((size_t)hot);
    for (auto& h : hotset) h = (int32_t)(rnd() % (uint64_t)V);
    std::vector<int32_t> ids((size_t)(calls * M));
    for (int64_t c = 0; c < calls; ++c)
        for (int64_t m = 0; m < M; ++m) {
            const uint64_t v = rnd();
            int32_t& id = ids[(size_t)(c * M + m)];
            if (m > 0 && v % 29 == 0)  id = ids[(size_t)(c * M + m - 1)];   /* repeated in a call */
            else if (v % 16 == 0)      id = -1;                             /* padding */
            else if (v % 16 < 8)       id = hotset[(v >> 8) % (uint64_t)hot];
            else                       id = (int32_t)((v >> 8) % (uint64_t)V);
        }
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b) lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * scale);
    /* Engram's table: one bf16 scale a row, resident beside the file-backed codes. */
    std::vector<uint16_t> row_scale;
    if (per_row) {
        row_scale.resize((size_t)V);
        for (auto& v : row_scale) v = rad_f32_to_bf16(0.05f + (float)(rnd() % 1000) / 500.0f);
    }

    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
        return t;
    };

    std::barrier<> go(2);
    int64_t bad[2] = {0, 0}, refused[2] = {0, 0}, failed[2] = {0, 0};
    /* Each rank thread makes its own ring on its first call. */
    auto rank = [&](int rk) {
        uint16_t sc = rad_f32_to_bf16(scale);
        std::vector<uint16_t> out((size_t)(M * n));
        std::vector<RadTensor> ts = {
            tensor(nullptr, RAD_I32, {M}),
            tensor(const_cast<uint8_t*>(tab), bf16 ? RAD_BF16 : RAD_F8E4M3, {V, n}),
            per_row ? tensor(row_scale.data(), RAD_BF16, {V})
                    : tensor(bf16 ? nullptr : &sc, RAD_BF16, {1}),
            tensor(out.data(), RAD_BF16, {M, n}),
            tensor(nullptr, RAD_I32, {M}) };
        RadParam ps[3] = {};
        ps[0].key = "M";       ps[0].kind = RAD_P_INT; ps[0].ival = M;
        ps[1].key = "n_embd";  ps[1].kind = RAD_P_INT; ps[1].ival = n;
        ps[2].key = "n_vocab"; ps[2].kind = RAD_P_INT; ps[2].ival = V;
        RadArgs a{};
        a.t = ts.data(); a.n_t = (int)ts.size();
        a.p = ps; a.n_p = 3;
        a.rank = rk; a.world_size = 2;
        for (int64_t c = 0; c < calls; ++c) {
            ts[0].data = &ids[(size_t)(c * M)];
            /* The last call passes no `ahead` (absent operand). */
            ts[4].data = c + 1 < calls ? &ids[(size_t)((c + 1) * M)] : nullptr;
            std::fill(out.begin(), out.end(), (uint16_t)0xdead);
            go.arrive_and_wait();
            if (const int rc = r->launch(&a, nullptr); rc != RAD_OK) {
                ++refused[rk];
                if (rc != RAD_E_UNSUPPORTED) ++failed[rk];
                continue;
            }
            for (int64_t m = 0; m < M; ++m) {
                const int32_t id = ids[(size_t)(c * M + m)];
                for (int64_t i = 0; i < n; ++i) {
                    uint16_t want = 0;
                    if (id >= 0 && bf16) std::memcpy(&want, tab + ((int64_t)id * n + i) * 2, 2);
                    else if (id >= 0 && per_row)
                        want = rad_f32_to_bf16(rad_fp8e4m3_to_f32(tab[(int64_t)id * n + i]) *
                                               rad_bf16_to_f32(row_scale[(size_t)id]));
                    else if (id >= 0)    want = lut[tab[(int64_t)id * n + i]];
                    if (out[(size_t)(m * n + i)] != want) { ++bad[rk]; break; }
                }
            }
        }
    };
    auto race = [&] {
        bad[0] = bad[1] = refused[0] = refused[1] = failed[0] = failed[1] = 0;
        std::thread t0(rank, 0), t1(rank, 1);
        t0.join();
        t1.join();
    };
    /* Calls refused as unsupported with every delivered row right mean a ring the kernel would
     * not make. Under contention for ring memory the case runs again on fresh threads, which make
     * fresh rings, up to three times; it is skipped by name only if a probe saw ENOMEM meanwhile.
     * Refusals without that evidence fail. */
    bool pressure = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
        race();
        const bool ring_refused = refused[0] + refused[1] && bad[0] + bad[1] == 0 &&
                                  failed[0] + failed[1] == 0;
        if (!ring_refused) break;
        if (ring_probe() == ENOMEM) pressure = true;
    }
    /* The gather holds its own descriptor from the first call on, so the name can go. */
    if (!rt->unlinked) {
        ::unlink(rt->path.c_str());
        rt->unlinked = true;
    }
    if (pressure && refused[0] + refused[1] && bad[0] + bad[1] + failed[0] + failed[1] == 0) {
        std::printf("  %s: SKIPPED -- %s\n", what, kRingMemory);
        return true;
    }

    const bool pass = bad[0] + bad[1] + refused[0] + refused[1] == 0;
    if (verbose || !pass)
        std::printf("  %s in %s (%s): %lld calls x 2 ranks, rows wrong %lld/%lld, "
                    "calls refused %lld/%lld %s\n", what, dir.c_str(), granule, (long long)calls,
                    (long long)bad[0], (long long)bad[1], (long long)refused[0],
                    (long long)refused[1], pass ? "ok" : "FAIL");
    else
        std::printf("  %s in %s (%s): %lld calls x 2 ranks, every row exact\n", what, dir.c_str(),
                    granule, (long long)calls);
    return pass;
}

/* A prefill's read-ahead, as the engine drives it (#355): each call gathers one chunk of 65,536
 * rows (4096 tokens of 16 heads) and passes the following eight chunks as `ahead`, the window
 * moving on by one chunk a call, with both ranks passing the same lists. Each row the calls gather
 * must be read from the file about once: each window adds one chunk to what the reader was handed,
 * and what is read ahead must still be in the cache when its chunk is gathered. Handing each window
 * over whole queued every chunk eight times and read ahead the cache's whole 2^19 rows, so rows
 * were evicted before their chunk came and read again, and the reader went on reading after the
 * last call. avx_ngram_reads counts the rows read from table files once the reader is idle; the
 * table is row_gather_case's, so some rows are already cached and read less than once. */
bool row_ahead_case(const Lib& lib, const std::string& dir, bool verbose) {
    const char* what = "embed_lookup_q's read-ahead over a prefill";
    const RadKernelInfo* r = lib.row("embed_lookup_q");
    auto reads = (uint64_t (*)())dlsym(lib.h, "avx_ngram_reads");
    if (!r || !r->launch || !reads) {
        std::printf("  %s: %s\n", what, !reads ? "no avx_ngram_reads export" : "no row");
        return false;
    }
    if (const int e = ring_probe()) {
        std::printf("  %s: SKIPPED -- %s\n", what,
                    e == ENOMEM ? kRingMemory
                                : (std::string("io_uring is unavailable here (") +
                                   std::strerror(e) + ")").c_str());
        return true;
    }
    const int64_t n = 40, V = int64_t(1) << 21, M = 65536, chunks = 12, window = 8;
    RowTable* rt = row_table(dir, false, V, n);
    if (rt->state == RowTable::SKIP) {
        std::printf("  %s: SKIPPED -- %s\n", what, rt->note.c_str());
        return true;
    }
    if (rt->state != RowTable::OK) {
        std::printf("  %s: %s\n", what, rt->note.c_str());
        return false;
    }
    const uint8_t* tab = rt->tab;
    const float scale = 0.375f;
    uint64_t x = 0x2545f4914f6cdd1dull;
    auto rnd = [&x]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    std::vector<int32_t> ids((size_t)(chunks * M));
    for (auto& id : ids) id = (int32_t)(rnd() % (uint64_t)V);
    std::vector<int32_t> distinct(ids);
    std::sort(distinct.begin(), distinct.end());
    const int64_t U = std::unique(distinct.begin(), distinct.end()) - distinct.begin();
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b) lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * scale);

    auto tensor = [](void* p, uint32_t dt, std::initializer_list<int64_t> shape) {
        RadTensor t{};
        t.data = p;
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int i = 0;
        for (int64_t s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = st; st *= t.shape[k]; }
        return t;
    };
    const uint64_t before = reads();
    std::barrier<> go(2);
    int64_t bad[2] = {0, 0}, refused[2] = {0, 0};
    auto rank = [&](int rk) {
        uint16_t sc = rad_f32_to_bf16(scale);
        std::vector<uint16_t> out((size_t)(M * n));
        RadParam ps[3] = {};
        ps[0].key = "M";       ps[0].kind = RAD_P_INT; ps[0].ival = M;
        ps[1].key = "n_embd";  ps[1].kind = RAD_P_INT; ps[1].ival = n;
        ps[2].key = "n_vocab"; ps[2].kind = RAD_P_INT; ps[2].ival = V;
        for (int64_t c = 0; c < chunks; ++c) {
            const int64_t na = std::min(window, chunks - 1 - c) * M;
            std::vector<RadTensor> ts = {
                tensor(&ids[(size_t)(c * M)], RAD_I32, {M}),
                tensor(const_cast<uint8_t*>(tab), RAD_F8E4M3, {V, n}),
                tensor(&sc, RAD_BF16, {1}), tensor(out.data(), RAD_BF16, {M, n}),
                tensor(na ? &ids[(size_t)((c + 1) * M)] : nullptr, RAD_I32, {na}) };
            RadArgs a{};
            a.t = ts.data(); a.n_t = (int)ts.size();
            a.p = ps; a.n_p = 3;
            a.rank = rk; a.world_size = 2;
            go.arrive_and_wait();
            if (r->launch(&a, nullptr) != RAD_OK) { ++refused[rk]; continue; }
            for (int64_t m = 0; m < M; ++m) {
                const uint8_t* row = tab + (int64_t)ids[(size_t)(c * M + m)] * n;
                for (int64_t i = 0; i < n; ++i)
                    if (out[(size_t)(m * n + i)] != lut[row[i]]) { ++bad[rk]; break; }
            }
        }
    };
    std::thread t0(rank, 0), t1(rank, 1);
    t0.join();
    t1.join();
    const uint64_t read = reads() - before;
    const bool exact = bad[0] + bad[1] + refused[0] + refused[1] == 0;
    const bool once = (int64_t)read <= U + U / 10;
    const bool pass = exact && once;
    std::printf("  %s in %s: %lld chunks of %lld rows x 2 ranks, %llu rows read for %lld distinct "
                "(%.2f each)%s%s\n", what, dir.c_str(), (long long)chunks, (long long)M,
                (unsigned long long)read, (long long)U, (double)read / (double)U,
                exact ? ", every row exact" : "", pass ? "" : " FAIL");
    if (!exact || verbose)
        std::printf("    rows wrong %lld/%lld, calls refused %lld/%lld\n", (long long)bad[0],
                    (long long)bad[1], (long long)refused[0], (long long)refused[1]);
    return pass;
}
