/* pcstore_test.cpp -- core/mem/pcstore: the prefix cache on disk across restarts.
 *
 * The store must never serve bytes another engine wrote (a key mismatch is discarded), never serve
 * bytes that changed on disk (a checksum mismatch is a miss), and a flush that fails at any point
 * (a full disk, a crash before the index) must leave the previous store readable.
 */
#include "rad_test.h"

#include "mem/pcstore.h"
#include "rad_internal.h"

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t EB = 1000;    /* entry bytes, deliberately not a multiple of 64 */
constexpr int64_t SB = 4096;    /* snapshot bytes */
constexpr int64_t BT = 4;       /* tokens an entry */

std::string fresh_dir(const char* leaf) {
    const std::string d = radtest::tmp_path(leaf);
    std::string cmd = "rm -rf '" + d + "'";
    (void)std::system(cmd.c_str());
    return d;
}

PrefixStore::Config cfg_of(const std::string& dir, const std::string& key = "model A\nbuild 1\n") {
    PrefixStore::Config c;
    c.dir = dir;
    c.name = "m";
    c.key_text = key;
    c.entry_bytes = EB;
    c.snap_bytes = SB;
    c.block_tokens = BT;
    return c;
}

BlockHash bh(uint64_t n) { return BlockHash{ n * 0x9e3779b97f4a7c15ull + 7, n + 1 }; }

struct Item {
    BlockHash key;
    std::vector<char> bytes;
    std::vector<int32_t> toks;
};

Item make(uint64_t n, int64_t sz) {
    Item it;
    it.key = bh(n);
    it.bytes.resize((size_t)sz);
    for (int64_t i = 0; i < sz; ++i) it.bytes[(size_t)i] = (char)((n * 131 + (uint64_t)i * 7) & 0xff);
    for (int i = 0; i < BT; ++i) it.toks.push_back((int32_t)(n * 10 + (uint64_t)i));
    return it;
}

std::vector<PcPut> puts_of(const std::vector<Item>& v, bool snap, bool fresh = true) {
    std::vector<PcPut> out;
    for (size_t i = 0; i < v.size(); ++i) {
        PcPut p;
        p.key = v[i].key;
        p.bytes = v[i].bytes.data();
        p.fresh = fresh;
        p.tokens = v[i].toks.data();
        p.depth = (int64_t)i;
        p.pos = 64 * (int64_t)(i + 1);
        p.group = 9;
        p.used_ms = 1000 + (int64_t)i;
        out.push_back(p);
    }
    (void)snap;
    return out;
}

int count_files(const std::string& d, const char* prefix) {
    int n = 0;
    if (DIR* dp = opendir(d.c_str())) {
        while (dirent* e = readdir(dp))
            if (std::strncmp(e->d_name, prefix, std::strlen(prefix)) == 0) ++n;
        closedir(dp);
    }
    return n;
}

}  // namespace

TEST(a_flushed_store_reads_back_every_byte_after_a_reopen) {
    const std::string d = fresh_dir("pcs_roundtrip");
    std::vector<Item> es, ss;
    for (uint64_t i = 0; i < 50; ++i) es.push_back(make(i, EB));
    for (uint64_t i = 0; i < 3; ++i) ss.push_back(make(100 + i, SB));
    {
        PrefixStore s;
        CHECK_OK(s.open(cfg_of(d)));
        PcFlush f;
        CHECK_OK(s.flush(puts_of(es, false), puts_of(ss, true), &f));
        CHECK_EQ(f.written_items, 53ll);
        CHECK_EQ(f.stored_entries.size(), 50u);
    }
    PrefixStore s;
    CHECK_OK(s.open(cfg_of(d)));
    CHECK_EQ(s.n_entries(), 50ll);
    CHECK_EQ(s.n_snaps(), 3ll);
    REQUIRE_EQ(s.loaded_entries().size(), 50u);
    /* Oldest first, tokens and metadata intact. */
    CHECK(s.loaded_entries()[0].key == es[0].key);
    CHECK(s.loaded_entries()[7].tokens == es[7].toks);
    CHECK_EQ(s.loaded_entries()[7].depth, 7ll);
    CHECK_EQ(s.loaded_snaps()[2].pos, 192ll);
    CHECK_EQ(s.loaded_snaps()[2].group, 9ull);
    /* Read in scrambled order: neighbours coalesce, every byte matches. */
    std::vector<BlockHash> keys;
    std::vector<std::vector<char>> buf(50, std::vector<char>((size_t)EB));
    std::vector<void*> dst;
    for (int i = 49; i >= 0; i -= 2) { keys.push_back(es[(size_t)i].key); dst.push_back(buf[keys.size() - 1].data()); }
    for (int i = 0; i < 50; i += 2) { keys.push_back(es[(size_t)i].key); dst.push_back(buf[keys.size() - 1].data()); }
    std::vector<char> ok(keys.size());
    CHECK_EQ(s.read_entries(keys.data(), (int64_t)keys.size(), dst.data(), ok.data()), 50ll);
    for (size_t k = 0; k < keys.size(); ++k) {
        const int i = k < 25 ? 49 - 2 * (int)k : 2 * (int)(k - 25);
        CHECK(ok[k]);
        CHECK(std::memcmp(buf[k].data(), es[(size_t)i].bytes.data(), EB) == 0);
    }
    std::vector<char> sb((size_t)SB);
    CHECK_OK(s.read_snap(ss[1].key, sb.data()));
    CHECK(std::memcmp(sb.data(), ss[1].bytes.data(), SB) == 0);
}

TEST(another_key_is_discarded_and_its_directory_removed) {
    const std::string d = fresh_dir("pcs_key");
    std::vector<Item> es{ make(1, EB), make(2, EB) };
    {
        PrefixStore s;
        CHECK_OK(s.open(cfg_of(d, "model A\nbuild 1\n")));
        CHECK_OK(s.flush(puts_of(es, false), {}, nullptr));
    }
    CHECK_EQ(count_files(d, "m."), 1);
    PrefixStore s;
    CHECK_OK(s.open(cfg_of(d, "model A\nbuild 2\n")));     /* one line differs */
    CHECK_EQ(s.n_entries(), 0ll);
    CHECK_EQ(count_files(d, "m."), 1);                     /* the old key's directory is gone */
    /* Another geometry under the same key text is refused too. */
    PrefixStore g;
    PrefixStore::Config c = cfg_of(d, "model A\nbuild 2\n");
    CHECK_OK(g.open(c));
    CHECK_OK(g.flush(puts_of(es, false), {}, nullptr));
    g.close();
    c.entry_bytes = EB + 64;
    CHECK_OK(g.open(c));
    CHECK_EQ(g.n_entries(), 0ll);
}

TEST(a_flipped_byte_is_a_miss_and_never_served) {
    const std::string d = fresh_dir("pcs_corrupt");
    std::vector<Item> es{ make(1, EB), make(2, EB), make(3, EB) };
    std::vector<Item> ss{ make(9, SB) };
    std::string path;
    {
        PrefixStore s;
        CHECK_OK(s.open(cfg_of(d)));
        CHECK_OK(s.flush(puts_of(es, false), puts_of(ss, true), nullptr));
        path = s.path();
    }
    /* The second entry's payload sits at 64 + snapshot + entry 0 or similar; flip a byte inside
     * the payload region past the header and the first item, wherever it lands. */
    const std::string seg = path + "/seg-1.rps";
    const int fd = ::open(seg.c_str(), O_RDWR);
    REQUIRE(fd >= 0);
    char c = 0;
    REQUIRE(::pread(fd, &c, 1, 64 + SB + EB + 10) == 1);
    c ^= 0x40;
    REQUIRE(::pwrite(fd, &c, 1, 64 + SB + EB + 10) == 1);
    ::close(fd);
    PrefixStore s;
    CHECK_OK(s.open(cfg_of(d)));
    std::vector<BlockHash> keys{ es[0].key, es[1].key, es[2].key };
    std::vector<std::vector<char>> buf(3, std::vector<char>((size_t)EB));
    void* dst[3] = { buf[0].data(), buf[1].data(), buf[2].data() };
    char ok[3];
    CHECK_EQ(s.read_entries(keys.data(), 3, dst, ok), 2ll);
    CHECK(ok[0] && !ok[1] && ok[2]);
    CHECK(!s.has_entry(es[1].key));                         /* forgotten */
    /* A damaged index starts empty rather than reading garbage. */
    s.close();
    const std::string ix = path + "/index.rpx";
    const int f2 = ::open(ix.c_str(), O_RDWR);
    REQUIRE(f2 >= 0);
    REQUIRE(::pwrite(f2, "X", 1, 100) == 1);
    ::close(f2);
    PrefixStore t;
    CHECK_OK(t.open(cfg_of(d)));
    CHECK_EQ(t.n_entries(), 0ll);
}

TEST(a_failed_flush_keeps_the_previous_store_and_leaves_no_files) {
    const std::string d = fresh_dir("pcs_full");
    std::vector<Item> a, b;
    for (uint64_t i = 0; i < 10; ++i) a.push_back(make(i, EB));
    for (uint64_t i = 10; i < 30; ++i) b.push_back(make(i, EB));
    PrefixStore s;
    CHECK_OK(s.open(cfg_of(d)));
    CHECK_OK(s.flush(puts_of(a, false), {}, nullptr));
    /* The disk fills halfway through the next flush. */
    std::vector<Item> all = a;
    all.insert(all.end(), b.begin(), b.end());
    s.test_fail_after(5 * EB);
    CHECK(s.flush(puts_of(all, false, false), {}, nullptr) < 0);
    s.test_fail_after(-1);
    CHECK_EQ(s.n_entries(), 10ll);
    CHECK_EQ(count_files(s.path(), "seg-"), 1);             /* no partial segment left */
    PrefixStore r;
    CHECK_OK(r.open(cfg_of(d)));
    CHECK_EQ(r.n_entries(), 10ll);
    std::vector<char> buf((size_t)EB);
    void* dst[1] = { buf.data() };
    char ok[1];
    CHECK_EQ(r.read_entries(&a[4].key, 1, dst, ok), 1ll);
    CHECK(std::memcmp(buf.data(), a[4].bytes.data(), EB) == 0);
}

TEST(a_crash_between_segment_and_index_leaves_the_old_index_and_sweeps_the_orphan) {
    const std::string d = fresh_dir("pcs_crash");
    std::vector<Item> a{ make(1, EB), make(2, EB) };
    std::string path;
    {
        PrefixStore s;
        CHECK_OK(s.open(cfg_of(d)));
        CHECK_OK(s.flush(puts_of(a, false), {}, nullptr));
        path = s.path();
    }
    /* What a crash leaves: a renamed segment the index never named, a half-written temp file,
     * and a truncated index temp. */
    {
        FILE* f = std::fopen((path + "/seg-7.rps").c_str(), "wb"); std::fputs("junk", f); std::fclose(f);
        f = std::fopen((path + "/seg-8.rps.tmp").c_str(), "wb"); std::fputs("junk", f); std::fclose(f);
        f = std::fopen((path + "/index.rpx.tmp").c_str(), "wb"); std::fputs("RADPC", f); std::fclose(f);
    }
    PrefixStore s;
    CHECK_OK(s.open(cfg_of(d)));
    CHECK_EQ(s.n_entries(), 2ll);
    CHECK_EQ(count_files(path, "seg-"), 1);
    CHECK_EQ(count_files(path, "index.rpx.tmp"), 0);
    /* A truncated segment (a copy cut short) loses its items, not the store. */
    s.close();
    CHECK(::truncate((path + "/seg-1.rps").c_str(), 64 + EB) == 0);
    PrefixStore t;
    CHECK_OK(t.open(cfg_of(d)));
    CHECK_EQ(t.n_entries(), 0ll);
}

TEST(an_unchanged_item_is_not_rewritten_and_dead_segments_go) {
    const std::string d = fresh_dir("pcs_incr");
    std::vector<Item> a;
    for (uint64_t i = 0; i < 8; ++i) a.push_back(make(i, EB));
    PrefixStore s;
    CHECK_OK(s.open(cfg_of(d)));
    PcFlush f1, f2, f3;
    CHECK_OK(s.flush(puts_of(a, false), {}, &f1));
    CHECK_EQ(f1.written_items, 8ll);
    /* Held on disk only (bytes null), or unchanged: nothing written. */
    std::vector<PcPut> p = puts_of(a, false, false);
    for (size_t i = 0; i < 4; ++i) p[i].bytes = nullptr;
    a.push_back(make(50, EB));
    p.push_back(puts_of({ a.back() }, false)[0]);
    CHECK_OK(s.flush(p, {}, &f2));
    CHECK_EQ(f2.written_items, 1ll);
    CHECK_EQ(count_files(s.path(), "seg-"), 2);
    /* Nothing offered: everything goes, so do both segments. */
    CHECK_OK(s.flush({}, {}, &f3));
    CHECK_EQ(f3.gone_entries.size(), 9u);
    CHECK_EQ(count_files(s.path(), "seg-"), 0);
}

TEST(the_cap_keeps_the_most_valuable_items) {
    const std::string d = fresh_dir("pcs_cap");
    std::vector<Item> a;
    for (uint64_t i = 0; i < 10; ++i) a.push_back(make(i, EB));
    PrefixStore::Config c = cfg_of(d);
    c.cap_bytes = 4 * EB + SB;      /* one snapshot and four entries */
    PrefixStore s;
    CHECK_OK(s.open(c));
    std::vector<Item> ss{ make(70, SB), make(71, SB) };
    PcFlush f;
    CHECK_OK(s.flush(puts_of(a, false), puts_of(ss, true), &f));
    CHECK_EQ(s.n_entries(), 4ll);
    CHECK_EQ(s.n_snaps(), 1ll);
    CHECK(s.has_entry(a[0].key) && s.has_entry(a[3].key) && !s.has_entry(a[4].key));
    CHECK(s.has_snap(ss[0].key));
}

TEST(the_checksum_sees_every_byte) {
    std::vector<char> v(4096 + 37, 1);
    const uint64_t h = pc_checksum(v.data(), (int64_t)v.size());
    for (size_t i : { (size_t)0, (size_t)63, (size_t)64, (size_t)4095, (size_t)4100 }) {
        v[i] ^= 1;
        CHECK(pc_checksum(v.data(), (int64_t)v.size()) != h);
        v[i] ^= 1;
    }
    CHECK(pc_key_of("a\n") != pc_key_of("b\n"));
}

RAD_TEST_MAIN()
