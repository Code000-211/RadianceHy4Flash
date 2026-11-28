/* table_fill_test.cpp -- the Engine's deferred Engram fill (B-01).
 *
 * copy_table reserves and publishes a table's memory as the load's walk meets its weight, starts
 * the read, and join_table_fills waits for every table it started once the container pass has run
 * beside them. These tests drive the same reserve/start/join/release path a --ngram-placement ram
 * server drives for the tables it reads for the cards: the locked copy (a host gather) and the
 * pinned one (a card's gather) on the host backend's memory, and the shared file (ram=DIR) the
 * engines on separate cards pin in place, which only a card's backend can map.
 */
#include "rad_test.h"
#include "engine_priv.h"
#include "device/device.h"
#include "format/radfile.h"
#include "format/table_copy.h"
#include "format/table_share.h"
#include "mem/vram_budget.h"
#include "place/planner.h"
#include "rad_runtime.h"

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {
/* The walk, copy_table and join_table_fills are private, as the other bringup tests reach them.
 * copy_table takes the share directory as the load passes it -- Config::ngram_share_dir_for the
 * layer, its own ram=DIR or the global one -- so a test says which layer it copies for. */
struct TableFillTestAccess {
    static void hold(Engine& e, std::unique_ptr<RadFile> f) { e.file_ = std::move(f); }
    static int  copy(Engine& e, const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                     std::string* why, const std::string& share_dir = std::string()) {
        return e.copy_table(src, n, foff, card, out, why, share_dir);
    }
    static int    join(Engine& e) { return e.join_table_fills(); }
    static size_t pending(Engine& e) { return e.pending_tables_.size(); }
    static size_t locked(Engine& e) { return e.table_copies_.size(); }
    static size_t pinned(Engine& e) { return e.card_copies_.size(); }
    static size_t shared(Engine& e) { return e.table_shares_.size(); }
    /* The per-layer disk=DIR path of load_weights: the table's own file, filled from the
     * container and mapped, replacing the container's bytes for the host gather. */
    static int    file_map(Engine& e, const uint8_t* src, int64_t n, int64_t foff, void** out,
                           std::string* why, const std::string& dir) {
        return e.table_file_map(src, n, foff, out, why, dir);
    }
    static size_t files(Engine& e) { return e.table_files_.size(); }
};

/* resolve_table_placement, driven the way Engine::bringup drives it: the facts, the table's
 * bytes, and the per-layer bytes the load measured (--ngram-placement-layers tests each asked
 * layer on its own). */
struct TableResolveTestAccess {
    static int resolve(Engine& e, const std::vector<VramFacts>& facts,
                       const std::vector<int64_t>& emb_bytes,
                       const std::vector<WeightFootprint>& wfs, int64_t kv_floor,
                       int64_t ngram_bytes, bool ngram_device, bool ngram_host, int64_t host_late,
                       const std::map<int, int64_t>& ngram_layer_bytes) {
        return e.resolve_table_placement(facts, emb_bytes, wfs, kv_floor, ngram_bytes,
                                         ngram_device, ngram_host, host_late, ngram_layer_bytes);
    }
    static const Config& cfg(Engine& e) { return e.cfg_; }
    static int64_t on_card(Engine& e) { return e.ngram_on_card_bytes_; }
};

/* The real load over the minimal state load_weights walks: one rank, the sentinel and one or
 * more reader-identified n-gram tables (read only by a host `embed_lookup_q`), each on the
 * mapped tier with its own host-gather unit, and the pools. A name that does not parse as
 * blk.<l>.engram_table (layer -1) drives the gate's global fallback; a name that does drives
 * the per-layer path. */
struct LoadGateTestAccess {
    static int prepare(Engine& e, const std::string& name, int64_t bytes) {
        return prepare_layers(e, { name }, { bytes });
    }
    static int prepare_layers(Engine& e, const std::vector<std::string>& names,
                              const std::vector<int64_t>& bytes) {
        EngineScope scope(e.owner_, "load-gate-test");
        auto rank = std::make_unique<Engine::Rank>();
        CtxDesc desc; desc.program = &rank->program;
        RAD_TRY(rank->ctx.init(desc));
        RAD_TRY(rank->pools.configure(e.cfg_, 0));
        Program& p = rank->program;
        p.weights.resize(1);                    /* index 0 is the sentinel */
        p.ctx.rank       = 0;
        p.ctx.world_size = 1;
        rank->plan.weight.resize(1);
        rank->plan.unit_of_weight = { -1 };
        for (size_t k = 0; k < names.size(); ++k) {
            WeightInfo w;
            w.name   = names[k];
            w.source = names[k];
            w.decl.dtype = RAD_U8;              /* one byte per element: bytes are exact */
            w.decl.rank  = 2;
            w.decl.shape[0] = bytes[k] / 2;
            w.decl.shape[1] = 1;
            w.decl.access = RAD_ACCESS_VOCAB;
            w.decl.group.layer  = -1;           /* the layer, if any, is in the name */
            w.decl.group.expert = -1;
            w.enc         = rad_enc_plain(RAD_BF16);
            w.enc_known   = true;
            w.n_sel       = 1;
            w.sel[0]      = 0;
            w.sel_rows[0] = bytes[k] / 2;
            w.sel_cols[0] = 1;
            w.stored_bytes  = bytes[k];
            w.logical_bytes = bytes[k];
            p.weights.push_back(w);
            OpInfo o;                           /* the reader that identifies the table */
            o.op      = "embed_lookup_q";
            o.index   = (int32_t)p.ops.size();
            o.weights = { (rad_weight)(k + 1) };
            p.ops.push_back(o);
            p.weights[k + 1].first_use_op = o.index;
            p.weights[k + 1].last_use_op  = o.index;
            /* The plan: the table on the mapped tier, read where it lies until the gate moves
             * it. */
            rank->plan.weight.resize(k + 2);
            rank->plan.weight[k + 1].tier  = Tier::Mapped;
            rank->plan.weight[k + 1].site  = Site::Host;
            rank->plan.weight[k + 1].unit  = (int32_t)rank->plan.units.size();
            rank->plan.weight[k + 1].bytes = bytes[k];
            MoveUnit mu;
            mu.tier    = Tier::Mapped;
            mu.site    = Site::Host;
            mu.bytes   = bytes[k];
            mu.weights = { (rad_weight)(k + 1) };
            mu.offset  = { 0 };
            rank->plan.units.push_back(mu);
            rank->plan.unit_of_weight.push_back((int32_t)rank->plan.units.size() - 1);
        }
        e.ranks_.push_back(std::move(rank));
        return RAD_OK;
    }
    static int   load(Engine& e) { return e.load_weights(); }
    static void* ptr(Engine& e, size_t i = 1) { return e.ranks_[0]->program.weights[i].ptr; }
    static void* published(Engine& e, size_t i = 1) {
        const WeightSlot* sl = e.ranks_[0]->mover.residency()->lookup((rad_weight)i);
        return sl ? sl->ptr : nullptr;
    }
};
}  // namespace rad

namespace {

/* Bytes no 16-byte shift of which can look like another: the fill's check samples them. */
std::vector<uint8_t> pattern_bytes(int64_t n, uint32_t seed) {
    std::vector<uint8_t> b((size_t)n);
    uint32_t x = seed | 1u;
    for (size_t i = 0; i < b.size(); ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        b[i] = (uint8_t)(x >> 11);
    }
    return b;
}

/* A container with three n-gram tables of different sizes, and the mapping the copies read from.
 * c comes first, so the 16-byte shift the failure case reads stays inside the container. */
struct Tables {
    std::string                   path;
    std::vector<uint8_t>          want[3];
    const uint8_t*                src[3] = {nullptr, nullptr, nullptr};
    int64_t                       off[3] = {0, 0, 0};
    int64_t                       n[3]   = {0, 0, 0};
    std::unique_ptr<rad::RadFile> file;

    int open() {
        file = std::make_unique<rad::RadFile>();
        if (file->open(path.c_str()) < 0) return -1;
        const char* names[] = {"ngram.c", "ngram.a", "ngram.b"};
        for (int i = 0; i < 3; ++i) {
            const RadFileEntry* e = file->find(names[i]);
            if (!e) return -1;
            const RadFilePlane& p = file->plane(*e, 0);
            src[i] = file->data(p);
            off[i] = (int64_t)p.offset;
            n[i] = (int64_t)p.bytes;
        }
        return 0;
    }
};

struct TempContainer {
    std::string path;
    ~TempContainer() { std::remove(path.c_str()); }
};

/* Writes the three tables; false when the writer refused. */
bool write_tables(const std::string& path, int64_t nc, int64_t na, int64_t nb, Tables* t) {
    t->path = path;
    t->n[0] = nc;
    t->n[1] = na;
    t->n[2] = nb;
    t->want[0] = pattern_bytes(nc, 0xc2b2ae35u);
    t->want[1] = pattern_bytes(na, 0x9e3779b9u);
    t->want[2] = pattern_bytes(nb, 0x85ebca6bu);
    rad::RadWriter writer;
    if (writer.begin(path.c_str()) < 0) return false;
    writer.set_arch("radtest");
    const char* names[] = {"ngram.c", "ngram.a", "ngram.b"};
    for (int i = 0; i < 3; ++i) {
        rad::RadWriter::Weight w;
        w.name = names[i];
        w.enc = rad_enc_plain(RAD_BF16);
        w.rank = 2;
        w.shape[0] = t->n[i] / 2;   /* bf16: two bytes a row */
        w.shape[1] = 1;
        const void* planes[] = {t->want[i].data()};
        if (writer.add_weight(w, planes) < 0) return false;
    }
    return writer.finish() == RAD_OK;
}

bool same_as(const void* got, const uint8_t* want, int64_t n) {
    return got && std::memcmp(got, want, (size_t)n) == 0;
}

/* Whether the shared file DIR/ngram-<id>.tab is there, whatever it holds. */
bool share_file_exists(const std::string& dir, const std::string& id) {
    return ::access((dir + "/ngram-" + id + ".tab").c_str(), F_OK) == 0;
}

/* Whether a copy failed on the locked-memory limit: the machine's setting, not a defect. Every copy
 * a case holds counts against the limit, so each is checked, not only the first (#235). */
bool memlock_short(int rc, const std::string& why) {
    return rc == RAD_E_NOMEM && why.find("mlock2") != std::string::npos;
}

}  // namespace

/* The locked copy (a host gather reads it): two tables started before either is waited on, one
 * copy for the same table a second rank asks for, publication only at the join, and a fill that
 * read the wrong bytes released and failing the load there. */
TEST(a_host_gathers_tables_read_before_the_join_and_are_published_after_it) {
    TempContainer tmp{radtest::tmp_path("table-fill-locked.rad")};
    const int64_t nc = 64ll << 10, na = 3ll << 20, nb = 1ll << 20;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    rad::Config config;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-locked-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    const int rc = rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], false, &outa, &why);
    /* A locked-memory limit below the tables is the machine's setting, not a defect. */
    if (memlock_short(rc, why)) SKIP("%s", why.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    const int rb = rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], false, &outb, &why);
    if (memlock_short(rb, why)) SKIP("%s", why.c_str());
    REQUIRE_EQ(rb, RAD_OK);
    CHECK(outa != nullptr && outa != (const void*)src[1]);
    CHECK(outb != nullptr && outb != (const void*)src[2]);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    /* Nothing is published before its bytes are in: the join is what publishes. */
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(0));

    /* The same table for another rank is the one copy, not a second read. */
    void* again = nullptr;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], false, &again, &why),
               RAD_OK);
    CHECK(again == outa);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(2));
    /* Byte for byte, not the fill's samples: the tables are the container's bytes. */
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));

    /* A fill that read the wrong bytes is not published: the join fails the load and frees it. */
    void* bad = nullptr;
    const int rbad =
        rad::TableFillTestAccess::copy(engine, src[0], n[0], off[0] + 16, false, &bad, &why);
    if (memlock_short(rbad, why)) SKIP("%s", why.c_str());
    REQUIRE_EQ(rbad, RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(1));
    CHECK_EQ(rad::TableFillTestAccess::join(engine), (int)RAD_E_IO);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(2));   /* the failed one is not */
    /* Joined and cleared: a second join is nothing left to do. */
    CHECK_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
}

/* The pinned copy (a card reads it over the link): the same deferral, with the copies published
 * to the engine's pinned ones only at the join. */
TEST(a_cards_tables_read_before_the_join_and_are_published_after_it) {
    TempContainer tmp{radtest::tmp_path("table-fill-pinned.rad")};
    const int64_t nc = 64ll << 10, na = 3ll << 20, nb = 1ll << 20;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    rad::Config config;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-pinned-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], true, &outa, &why),
               RAD_OK);
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], true, &outb, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(0));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(0));
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));
}

/* --ngram-placement ram=DIR (the third way a table is held): the file is filled by the engine that
 * finds it unready and marked ready at the join, checked and pinned without a read by the next
 * engine on it, and given back — mapping and file both — when the fill that wrote it failed. A
 * table under kTableShareMin keeps its private pinned copy even under ram=DIR. */
TEST(a_shared_table_is_filled_once_and_pinned_by_the_next_engine) {
    /* A file is pinned where it lies (rad_dev_map_file), which only a card's backend can do: the
     * unit entry runs this binary on the host backend and skips here, the _card entry runs it. */
    if (rad::device_is_host())
        SKIP("the shared placement pins a file in place, which the host backend cannot do");

    TempContainer tmp{radtest::tmp_path("table-fill-shared.rad")};
    /* c and a are at the share minimum, so ram=DIR shares them; b is under it. c comes first, so
     * the 16-byte shift the failure case reads stays inside the container. */
    const int64_t nc = rad::kTableShareMin, na = rad::kTableShareMin, nb = 64ll << 10;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const uint8_t* const* src = t.src;
    const int64_t* n = t.n;
    const int64_t* off = t.off;

    const std::string dir = radtest::tmp_path("ngram-share");
    REQUIRE(::mkdir(dir.c_str(), 0777) == 0);   /* an empty DIR is the placement's own */
    const std::string ida = rad::table_share_id(tmp.path, off[1], n[1]);
    /* Table 0 read 16 bytes late: a wrong read is what the check has to catch, and the share's id
     * names the range that was read, so the file the fill makes carries the shifted offset. */
    const int64_t bad_off = off[0] + 16;
    const std::string idc = rad::table_share_id(tmp.path, bad_off, n[0]);
    const std::string idb = rad::table_share_id(tmp.path, off[2], n[2]);
    REQUIRE(!ida.empty() && !idc.empty() && !idb.empty() && ida != idc);

    rad::Config config;
    config.ngram_share_dir = dir;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-shared-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    void* outa = nullptr;
    void* outb = nullptr;
    std::string why;
    const int rc = rad::TableFillTestAccess::copy(engine, src[1], n[1], off[1], true, &outa, &why);
    /* A card whose locked-memory limit is under the table is the machine's setting, not a defect. */
    if (rc == RAD_E_NOMEM && why.find("locking piece") != std::string::npos) SKIP("%s", why.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    CHECK(outa != nullptr && outa != (const void*)src[1]);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(1));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(0));
    /* The file is this engine's to fill: it is not a table another engine may reuse yet. */
    CHECK(!rad::table_share_is_ready(dir, ida, n[1]));
    CHECK(share_file_exists(dir, ida));

    /* Under kTableShareMin: the private pinned copy, and no file in DIR. */
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[2], n[2], off[2], true, &outb, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(2));
    CHECK(!share_file_exists(dir, idb));

    REQUIRE_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::pinned(engine), size_t(2));
    CHECK_EQ(rad::TableFillTestAccess::shared(engine), size_t(1));
    CHECK(same_as(outa, src[1], n[1]));
    CHECK(same_as(outb, src[2], n[2]));
    /* The join is what marks the file ready, for the engines that come after this one. */
    CHECK(rad::table_share_is_ready(dir, ida, n[1]));

    /* The next engine on the same container and DIR checks the file's samples and pins it, with
     * nothing to read: the copy is its own the moment it is made. */
    {
        Tables next;
        next.path = tmp.path;
        REQUIRE_EQ(next.open(), 0);
        rad::Engine engine2(config);
        rad::EngineScope owner2(engine2.owner(), "table-fill-shared-test-reuse");
        rad::TableFillTestAccess::hold(engine2, std::move(next.file));
        void* again = nullptr;
        REQUIRE_EQ(rad::TableFillTestAccess::copy(engine2, next.src[1], next.n[1], next.off[1], true,
                                                 &again, &why),
                   RAD_OK);
        CHECK_EQ(rad::TableFillTestAccess::pending(engine2), size_t(0));
        CHECK_EQ(rad::TableFillTestAccess::pinned(engine2), size_t(1));
        CHECK_EQ(rad::TableFillTestAccess::shared(engine2), size_t(1));
        CHECK(again != nullptr && again != (const void*)next.src[1]);
        CHECK(same_as(again, next.src[1], next.n[1]));
    }

    /* A ready file whose table no longer matches the container (a filler killed mid-write after
     * it was marked ready, or a corrupted tmpfs) is not read: the engine lets go of its mapping
     * and its lock and fails, and the file stays for the holder that filled it. */
    {
        const int fd = ::open((dir + "/ngram-" + ida + ".tab").c_str(), O_RDWR | O_CLOEXEC);
        REQUIRE(fd >= 0);
        uint8_t wrong = 0;
        REQUIRE_EQ(::pread(fd, &wrong, 1, rad::kTableShareHeader), (ssize_t)1);
        wrong = (uint8_t)(wrong ^ 0xff);
        REQUIRE_EQ(::pwrite(fd, &wrong, 1, rad::kTableShareHeader), (ssize_t)1);
        ::close(fd);

        Tables t3;
        t3.path = tmp.path;
        REQUIRE_EQ(t3.open(), 0);
        rad::Engine engine3(config);
        rad::EngineScope owner3(engine3.owner(), "table-fill-shared-test-mismatch");
        rad::TableFillTestAccess::hold(engine3, std::move(t3.file));
        void* stale = nullptr;
        why.clear();
        CHECK_EQ(rad::TableFillTestAccess::copy(engine3, t3.src[1], t3.n[1], t3.off[1], true, &stale,
                                                &why),
                 (int)RAD_E_IO);
        CHECK(why.find("does not match") != std::string::npos);
        CHECK_EQ(rad::TableFillTestAccess::pending(engine3), size_t(0));
        CHECK_EQ(rad::TableFillTestAccess::pinned(engine3), size_t(0));
        CHECK_EQ(rad::TableFillTestAccess::shared(engine3), size_t(0));
        CHECK(share_file_exists(dir, ida));
    }

    /* A fill that read the wrong bytes releases its mapping and the file: nothing half-filled is
     * left for the engine that comes next, and the load fails. */
    void* bad = nullptr;
    REQUIRE_EQ(rad::TableFillTestAccess::copy(engine, src[0], n[0], bad_off, true, &bad, &why),
               RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(1));
    CHECK(share_file_exists(dir, idc));
    CHECK_EQ(rad::TableFillTestAccess::join(engine), (int)RAD_E_IO);
    CHECK_EQ(rad::TableFillTestAccess::pending(engine), size_t(0));
    CHECK_EQ(rad::TableFillTestAccess::shared(engine), size_t(1));   /* the failed one is not */
    CHECK(!rad::table_share_is_ready(dir, idc, n[0]));
    CHECK(!share_file_exists(dir, idc));
    /* Joined and cleared: a second join is nothing left to do. */
    CHECK_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
}

/* --ngram-placement-layers through resolve_table_placement, where a named tier is tested on its
 * own bytes (engine_bringup.cpp): the ram layer whose memory fits resolves ram under a global
 * disk, a vram layer is refused when no kernel gathers the table on a card and again when the
 * card cannot hold it, a disk layer is refused when no kernel gathers it on the host, and
 * ngram_on_card_bytes_ counts every table that ends up beside the other weights: a global vram
 * table whole, and a vram layer's bytes under a global that keeps the rest off the card. */
TEST(the_resolver_honours_and_refuses_the_per_layer_placement_asks) {
    const int64_t MiB = (int64_t)1 << 20;
    const int64_t ngram = 10 * MiB;
    const std::map<int, int64_t> layer_bytes = {{2, 2 * MiB}, {5, 3 * MiB}};
    auto room_facts = [] {
        rad::VramFacts f{};
        f.capacity   = 16LL << 30;   /* a card with room: the trial's own facts (mem_test) */
        f.free       = f.capacity;
        f.arena      = 64LL << 20;
        f.kv_ceiling = 300LL << 20;
        return std::vector<rad::VramFacts>{ f };
    };
    const std::vector<int64_t> no_emb;
    const std::vector<rad::WeightFootprint> no_wfs;

    /* A ram layer under a global disk, the copy pinned for a card kernel (RLIMIT_MEMLOCK does
     * not count a driver's userptr mapping): the lockable check admits the layer's bytes, the
     * layer resolves ram, the layers the spec left out keep the global disk, and no byte of
     * the table sits beside the other weights. */
    {
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_layer_asked = {{2, rad::TablePlacement::Ram}};
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-ram-layer");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, room_facts(), no_emb, no_wfs, 0, ngram, true, true, 0, layer_bytes);
        REQUIRE_EQ(rc, RAD_OK);
        const rad::Config& c = rad::TableResolveTestAccess::cfg(engine);
        CHECK(c.ngram_effective(2) == rad::TablePlacement::Ram);
        CHECK(c.ngram_effective(5) == rad::TablePlacement::Disk);   /* unasked: the global */
        CHECK(c.ngram_resolved == rad::TablePlacement::Disk);
        CHECK_EQ(rad::TableResolveTestAccess::on_card(engine), (int64_t)0);
    }

    /* A vram layer with no card kernel: refused before any fit test, as a named tier is. */
    {
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_layer_asked = {{2, rad::TablePlacement::Vram}};
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-vram-layer-no-kernel");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, room_facts(), no_emb, no_wfs, 0, ngram, false, true, 0, layer_bytes);
        CHECK_EQ(rc, (int)RAD_E_INVAL);
    }

    /* A vram layer the card cannot hold -- the KV floor of one --max-model-len sequence leaves
     * nothing for it: refused, and the server would stop on this status. */
    {
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_layer_asked = {{2, rad::TablePlacement::Vram}};
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-vram-layer-misfit");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, room_facts(), no_emb, no_wfs, (int64_t)1 << 40, ngram, true, true, 0,
            layer_bytes);
        CHECK_EQ(rc, (int)RAD_E_NOMEM);
    }

    /* A disk layer with no host gatherer: the container has no reader, refused. (The global
     * ram ask resolves first -- a pinned copy, which the memory admits -- so the refusal here
     * is the layer's own.) */
    {
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Ram;
        config.ngram_layer_asked = {{2, rad::TablePlacement::Disk}};
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-disk-layer-no-host-kernel");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, room_facts(), no_emb, no_wfs, 0, ngram, true, false, 0, layer_bytes);
        CHECK_EQ(rc, (int)RAD_E_INVAL);
    }

    /* The card's bytes, global: a vram table counts whole. */
    {
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Vram;
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-vram-global");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, room_facts(), no_emb, no_wfs, 0, ngram, true, true, 0, layer_bytes);
        REQUIRE_EQ(rc, RAD_OK);
        CHECK(rad::TableResolveTestAccess::cfg(engine).ngram_resolved ==
              rad::TablePlacement::Vram);
        CHECK_EQ(rad::TableResolveTestAccess::on_card(engine), ngram);
    }

    /* The card's bytes, layered: a vram layer adds its own bytes under a global disk that
     * keeps the rest of the table off the card. */
    {
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_layer_asked = {{2, rad::TablePlacement::Vram}};
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-vram-layer-accounting");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, room_facts(), no_emb, no_wfs, 0, ngram, true, true, 0, layer_bytes);
        REQUIRE_EQ(rc, RAD_OK);
        const rad::Config& c = rad::TableResolveTestAccess::cfg(engine);
        CHECK(c.ngram_effective(2) == rad::TablePlacement::Vram);
        CHECK(c.ngram_resolved == rad::TablePlacement::Disk);
        CHECK_EQ(rad::TableResolveTestAccess::on_card(engine), (int64_t)2 * MiB);
    }

    /* Last, because a machine whose limits refuse it skips the rest: the same ram ask with a
     * host gather, where the copy is locked and RLIMIT_MEMLOCK counts it. A pass resolves the
     * layer exactly as the pinned case did; a refusal is the machine's setting, not a defect. */
    {
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_layer_asked = {{2, rad::TablePlacement::Ram}};
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-ram-layer-locked");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, room_facts(), no_emb, no_wfs, 0, ngram, false, true, 0, layer_bytes);
        if (rc == RAD_E_NOMEM)
            SKIP("the machine cannot lock the layer's table (RLIMIT_MEMLOCK/MemAvailable)");
        REQUIRE_EQ(rc, RAD_OK);
        CHECK(rad::TableResolveTestAccess::cfg(engine).ngram_effective(2) ==
              rad::TablePlacement::Ram);
    }
}

/* The per-layer ram=DIR / disk=DIR directories are the resolver's answer and the load's
 * operand: resolve fills Config's ngram_share_dir_for/ngram_disk_dir_for per layer, and
 * load_weights hands exactly those to copy_table and table_file_map (engine_bringup.cpp).
 * Driven here through the same access with a two-layer spec under a global disk: the disk
 * layer's table becomes its own file under its directory, filled from the container and
 * mapped for the host gather; the ram layer's directory waits for the pinned copy only a
 * card's backend can make, which is the next test. */
TEST(the_per_layer_dirs_are_the_ones_the_resolver_hands_the_load) {
    TempContainer tmp{radtest::tmp_path("table-fill-layers.rad")};
    /* Both asked tables at the share minimum, the size the load copies rather than reads in
     * place; c is the unasked table, which takes the global disk and no directory. */
    const int64_t nc = 64ll << 10, na = rad::kTableShareMin, nb = rad::kTableShareMin;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);

    const std::string ram_dir  = radtest::tmp_path("ngram-layer-ram");
    const std::string disk_dir = radtest::tmp_path("ngram-layer-disk");
    REQUIRE(::mkdir(ram_dir.c_str(), 0777) == 0);    /* empty DIRs are the spec's own */
    REQUIRE(::mkdir(disk_dir.c_str(), 0777) == 0);

    rad::Config config;
    config.ngram_placement = rad::TablePlacement::Disk;
    config.ngram_layer_asked = {{1, rad::TablePlacement::Ram}, {2, rad::TablePlacement::Disk}};
    config.ngram_layer_share_dir[1] = ram_dir;
    config.ngram_layer_disk_dir[2]  = disk_dir;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-layers-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    rad::VramFacts f{};
    f.capacity = 16LL << 30; f.free = f.capacity; f.arena = 64LL << 20; f.kv_ceiling = 300LL << 20;
    const int rc = rad::TableResolveTestAccess::resolve(
        engine, { f }, {}, {}, 0, nc + na + nb, true, true, 0, {{1, na}, {2, nb}});
    REQUIRE_EQ(rc, RAD_OK);
    const rad::Config& c = rad::TableResolveTestAccess::cfg(engine);
    CHECK(c.ngram_effective(1) == rad::TablePlacement::Ram);
    CHECK(c.ngram_share_dir_for(1) == ram_dir);
    CHECK(c.ngram_effective(2) == rad::TablePlacement::Disk);
    CHECK(c.ngram_disk_dir_for(2) == disk_dir);
    CHECK(c.ngram_effective(0) == rad::TablePlacement::Disk);   /* unasked: the global */
    CHECK(c.ngram_share_dir_for(0).empty());                    /* no directory of its own */
    CHECK(c.ngram_disk_dir_for(0).empty());

    /* disk=DIR through the access, with the directory the resolver named for THIS layer: the
     * table's own file under it, filled from the container and mapped -- the gather reads the
     * file's bytes, not the container's range. */
    void* m = nullptr;
    std::string why;
    const int fm = rad::TableFillTestAccess::file_map(engine, t.src[2], t.n[2], t.off[2], &m,
                                                      &why, c.ngram_disk_dir_for(2));
    REQUIRE_EQ(fm, RAD_OK);
    CHECK(m != nullptr && m != (const void*)t.src[2]);
    CHECK(same_as(m, t.want[2].data(), t.n[2]));
    CHECK_EQ(rad::TableFillTestAccess::files(engine), size_t(1));
    const std::string id2 = rad::table_share_id(tmp.path, t.off[2], t.n[2]);
    REQUIRE(!id2.empty());
    CHECK(::access((disk_dir + "/ngram-" + id2 + ".bin").c_str(), F_OK) == 0);
    /* The ram layer's directory holds nothing yet: its copy is the one a card's backend pins
     * in place -- the next test's subject -- and the disk layer's file landed in its own
     * directory, not the ram layer's. */
    CHECK(!share_file_exists(ram_dir, rad::table_share_id(tmp.path, t.off[1], t.n[1])));
    CHECK(::access((ram_dir + "/ngram-" + id2 + ".bin").c_str(), F_OK) != 0);
}

/* The ram layer's directory under the same spec: copy_table, handed the resolver's
 * ngram_share_dir_for(layer), fills the shared file THERE -- a table the card pins in place,
 * which only a card's backend can map. The join publishes it, and the bytes are the
 * container's. */
TEST(the_per_layer_ram_dir_holds_the_shared_file_the_card_pins) {
    if (rad::device_is_host())
        SKIP("the shared placement pins a file in place, which the host backend cannot do");
    TempContainer tmp{radtest::tmp_path("table-fill-layers-ram.rad")};
    const int64_t nc = 64ll << 10, na = rad::kTableShareMin, nb = rad::kTableShareMin;
    Tables t;
    REQUIRE(write_tables(tmp.path, nc, na, nb, &t));
    REQUIRE_EQ(t.open(), 0);
    const std::string ram_dir = radtest::tmp_path("ngram-layer-ram2");
    REQUIRE(::mkdir(ram_dir.c_str(), 0777) == 0);

    rad::Config config;
    config.ngram_placement = rad::TablePlacement::Disk;
    config.ngram_layer_asked = {{1, rad::TablePlacement::Ram}};
    config.ngram_layer_share_dir[1] = ram_dir;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "table-fill-layers-ram-test");
    rad::TableFillTestAccess::hold(engine, std::move(t.file));

    rad::VramFacts f{};
    f.capacity = 16LL << 30; f.free = f.capacity; f.arena = 64LL << 20; f.kv_ceiling = 300LL << 20;
    REQUIRE_EQ(rad::TableResolveTestAccess::resolve(
                   engine, { f }, {}, {}, 0, nc + na, true, true, 0, {{1, na}}),
               RAD_OK);
    const rad::Config& c = rad::TableResolveTestAccess::cfg(engine);
    CHECK(c.ngram_effective(1) == rad::TablePlacement::Ram);
    CHECK(c.ngram_share_dir_for(1) == ram_dir);

    void* out = nullptr;
    std::string why;
    const int rc = rad::TableFillTestAccess::copy(engine, t.src[1], t.n[1], t.off[1], true, &out,
                                                  &why, c.ngram_share_dir_for(1));
    /* A card whose locked-memory limit is under the table is the machine's setting, not a
     * defect. */
    if (rc == RAD_E_NOMEM && why.find("locking piece") != std::string::npos)
        SKIP("%s", why.c_str());
    REQUIRE_EQ(rc, RAD_OK);
    const std::string id1 = rad::table_share_id(tmp.path, t.off[1], t.n[1]);
    REQUIRE(!id1.empty());
    CHECK(share_file_exists(ram_dir, id1));
    CHECK_EQ(rad::TableFillTestAccess::join(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::shared(engine), size_t(1));
    CHECK(out != nullptr && out != (const void*)t.src[1]);
    CHECK(same_as(out, t.want[1].data(), t.n[1]));
}

/* ======================================================== the load gate, through load_weights */

/* The mapped-tier gate of load_weights (engine_bringup.cpp), driven by the real load rather than
 * through the copy access: a reader-identified n-gram table whose name the layer parsing does not
 * reach (not blk.<l>.engram_table -- rad_block_ple.h's, e.g.) takes the GLOBAL resolved
 * placement, exactly as before --ngram-placement-layers existed. The resolver budgeted the
 * unnamed table's copy; the gate may not keep it in the container. */
namespace {

struct GateContainer {
    std::string          path;
    std::vector<uint8_t> want;
    const uint8_t*       src = nullptr;
    int64_t              off = 0, n = 0;
    std::unique_ptr<rad::RadFile> file;

    bool write(const std::string& p, int64_t bytes, const std::string& name) {
        path = p;
        n    = bytes;
        want = pattern_bytes(bytes, 0x1234abcdu);
        rad::RadWriter writer;
        if (writer.begin(path.c_str()) < 0) return false;
        writer.set_arch("radtest");
        rad::RadWriter::Weight w;
        w.name = name;
        w.enc  = rad_enc_plain(RAD_BF16);
        w.rank = 2;
        w.shape[0] = bytes / 2;   /* bf16: two bytes a row */
        w.shape[1] = 1;
        const void* planes[] = { want.data() };
        if (writer.add_weight(w, planes) < 0) return false;
        return writer.finish() == RAD_OK;
    }
    int open(const std::string& name) {
        file = std::make_unique<rad::RadFile>();
        if (file->open(path.c_str()) < 0) return -1;
        const RadFileEntry* e = file->find(name);
        if (!e) return -1;
        const RadFilePlane& p = file->plane(*e, 0);
        src = file->data(p);
        off = (int64_t)p.offset;
        if ((int64_t)p.bytes != n) return -1;
        return 0;
    }
};

/* A container holding several tables, for the named-layer cases: each table's bytes, plane
 * offset and mapped pointer, indexed in the order written. */
struct GateMulti {
    std::string                       path;
    std::vector<std::vector<uint8_t>> want;
    std::vector<const uint8_t*>       src;
    std::vector<int64_t>              off, n;
    std::unique_ptr<rad::RadFile>     file;

    bool write(const std::string& p,
               const std::vector<std::pair<std::string, int64_t>>& ws) {
        path = p;
        rad::RadWriter writer;
        if (writer.begin(path.c_str()) < 0) return false;
        writer.set_arch("radtest");
        for (size_t k = 0; k < ws.size(); ++k) {
            want.push_back(pattern_bytes(ws[k].second, 0x5eed0000u + (unsigned)k));
            rad::RadWriter::Weight w;
            w.name = ws[k].first;
            w.enc  = rad_enc_plain(RAD_BF16);
            w.rank = 2;
            w.shape[0] = ws[k].second / 2;   /* bf16: two bytes a row */
            w.shape[1] = 1;
            const void* planes[] = { want[k].data() };
            if (writer.add_weight(w, planes) < 0) return false;
        }
        return writer.finish() == RAD_OK;
    }
    int open(const std::vector<std::string>& names) {
        file = std::make_unique<rad::RadFile>();
        if (file->open(path.c_str()) < 0) return -1;
        for (size_t k = 0; k < names.size(); ++k) {
            const RadFileEntry* e = file->find(names[k]);
            if (!e) return -1;
            const RadFilePlane& p = file->plane(*e, 0);
            src.push_back(file->data(p));
            off.push_back((int64_t)p.offset);
            n.push_back((int64_t)p.bytes);
            if (n[k] != (int64_t)want[k].size()) return -1;
        }
        return 0;
    }
};

}  // namespace

/* Finding 1 (round 3): the global ram copy reaches every reader-identified table, not only the
 * ones named blk.<l>.engram_table. The unnamed table under the global ram flag gets the copy the
 * resolver budgeted -- published through the residency table, byte-identical -- and under a
 * global disk with no directory it stays where it lies. */
TEST(an_unnamed_ngram_table_takes_the_global_ram_copy_through_the_load) {
    const std::string name = "ple.ple_table.emb";   /* ngram_table_layer() answers -1 for it */
    const int64_t n = 3ll << 20;

    {   /* ram: the gate hands the copy, the join publishes it, the gather reads the copy. */
        GateContainer c;
        REQUIRE(c.write(radtest::tmp_path("load-gate-ram.rad"), n, name));
        REQUIRE_EQ(c.open(name), 0);
        rad::Config config;
        config.model = c.path;
        config.ngram_placement = rad::TablePlacement::Ram;
        config.ngram_resolved  = rad::TablePlacement::Ram;   /* what the resolve confirmed */
        config.vram_weights_mib = 64; config.vram_kv_mib = 1; config.host_pool_mib = 64;
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "load-gate-ram");
        REQUIRE_EQ(rad::LoadGateTestAccess::prepare(engine, name, n), RAD_OK);
        rad::TableFillTestAccess::hold(engine, std::move(c.file));
        const int rc = rad::LoadGateTestAccess::load(engine);
        if (rc == RAD_E_NOMEM) SKIP("the machine cannot lock the table (RLIMIT_MEMLOCK)");
        REQUIRE_EQ(rc, RAD_OK);
        CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(1));
        void* const p = rad::LoadGateTestAccess::ptr(engine);
        CHECK(p != nullptr && p != (void*)c.src);
        CHECK(same_as(p, c.want.data(), n));
        CHECK(rad::LoadGateTestAccess::published(engine) == p);
    }

    {   /* The control: a global disk with no directory leaves the same table in the container. */
        GateContainer c;
        REQUIRE(c.write(radtest::tmp_path("load-gate-disk.rad"), n, name));
        REQUIRE_EQ(c.open(name), 0);
        rad::Config config;
        config.model = c.path;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_resolved  = rad::TablePlacement::Disk;
        config.vram_weights_mib = 64; config.vram_kv_mib = 1; config.host_pool_mib = 64;
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "load-gate-disk-none");
        REQUIRE_EQ(rad::LoadGateTestAccess::prepare(engine, name, n), RAD_OK);
        rad::TableFillTestAccess::hold(engine, std::move(c.file));
        REQUIRE_EQ(rad::LoadGateTestAccess::load(engine), RAD_OK);
        CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(0));
        CHECK_EQ(rad::TableFillTestAccess::files(engine), size_t(0));
        CHECK(rad::LoadGateTestAccess::ptr(engine) == (void*)c.src);
    }
}

/* Finding 1 (round 3), the disk=DIR arm: the unnamed table under the global disk=DIR takes its
 * own file under the global directory -- the load's gate, not the resolver's per-layer answer. */
TEST(an_unnamed_ngram_table_takes_the_global_disk_dir_file_through_the_load) {
    const std::string name = "ple.ple_table.emb";
    const int64_t n = rad::kTableShareMin;   /* below it the gate keeps the container */
    GateContainer c;
    REQUIRE(c.write(radtest::tmp_path("load-gate-dird.rad"), n, name));
    REQUIRE_EQ(c.open(name), 0);
    const std::string dir = radtest::tmp_path("load-gate-dird");
    REQUIRE(::mkdir(dir.c_str(), 0777) == 0);

    rad::Config config;
    config.model = c.path;
    config.ngram_placement = rad::TablePlacement::Disk;
    config.ngram_resolved  = rad::TablePlacement::Disk;
    config.ngram_disk_dir  = dir;            /* the global disk=DIR */
        config.vram_weights_mib = 64; config.vram_kv_mib = 1; config.host_pool_mib = 64;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "load-gate-dird");
    REQUIRE_EQ(rad::LoadGateTestAccess::prepare(engine, name, n), RAD_OK);
    rad::TableFillTestAccess::hold(engine, std::move(c.file));
    REQUIRE_EQ(rad::LoadGateTestAccess::load(engine), RAD_OK);
    CHECK_EQ(rad::TableFillTestAccess::files(engine), size_t(1));
    void* const p = rad::LoadGateTestAccess::ptr(engine);
    CHECK(p != nullptr && p != (void*)c.src);
    CHECK(same_as(p, c.want.data(), n));
    const std::string id = rad::table_share_id(config.model, c.off, c.n);
    REQUIRE(!id.empty());
    CHECK(::access((dir + "/ngram-" + id + ".bin").c_str(), F_OK) == 0);
}

/* Finding 1 (round 4): the gate asks the LAYER, not the global resolve, for a table the naming
 * reaches. Under a global disk, blk.1's ram and blk.2's disk=DIR are the named layers' own
 * placements -- neither takes the ngram_resolved fallback: the ram layer gets the locked copy
 * (published through the residency table, byte-identical), the disk layer its own file under
 * its own directory. Under the global fallback blk.1 would sit in the container (disk, no
 * directory) and blk.2's file would wait for a global directory that was never named. */
TEST(named_layers_take_their_own_placement_through_the_load) {
    const std::vector<std::string> names = { "blk.1.engram_table.weight",
                                             "blk.2.engram_table.weight" };
    const int64_t n1 = 3ll << 20;                 /* the ram layer: a locked copy */
    const int64_t n2 = rad::kTableShareMin;       /* the disk=DIR layer: its own file */
    GateMulti c;
    REQUIRE(c.write(radtest::tmp_path("load-gate-named.rad"),
                    { { names[0], n1 }, { names[1], n2 } }));
    REQUIRE_EQ(c.open(names), 0);
    const std::string dir2 = radtest::tmp_path("load-gate-named-disk2");
    REQUIRE(::mkdir(dir2.c_str(), 0777) == 0);

    rad::Config config;
    config.model = c.path;
    config.ngram_placement = rad::TablePlacement::Disk;   /* global: disk, no directory */
    config.ngram_resolved  = rad::TablePlacement::Disk;
    config.ngram_layer_asked    = { { 1, rad::TablePlacement::Ram },
                                    { 2, rad::TablePlacement::Disk } };
    config.ngram_layer_resolved = { { 1, rad::TablePlacement::Ram },
                                    { 2, rad::TablePlacement::Disk } };
    config.ngram_layer_disk_dir = { { 2, dir2 } };        /* the 2=disk=DIR */
    config.vram_weights_mib = 64; config.vram_kv_mib = 1; config.host_pool_mib = 64;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "load-gate-named");
    REQUIRE_EQ(rad::LoadGateTestAccess::prepare_layers(engine, names, { n1, n2 }), RAD_OK);
    rad::TableFillTestAccess::hold(engine, std::move(c.file));
    const int rc = rad::LoadGateTestAccess::load(engine);
    if (rc == RAD_E_NOMEM) SKIP("the machine cannot lock the table (RLIMIT_MEMLOCK)");
    REQUIRE_EQ(rc, RAD_OK);
    /* One locked copy and one table file: the two named layers' own paths, and no third. */
    CHECK_EQ(rad::TableFillTestAccess::locked(engine), size_t(1));
    CHECK_EQ(rad::TableFillTestAccess::files(engine), size_t(1));
    /* blk.1: the ram path -- the copy, not the container the global disk leaves. */
    void* const p1 = rad::LoadGateTestAccess::ptr(engine, 1);
    CHECK(p1 != nullptr && p1 != (void*)c.src[0]);
    CHECK(same_as(p1, c.want[0].data(), n1));
    CHECK(rad::LoadGateTestAccess::published(engine, 1) == p1);
    /* blk.2: its own file under the NAMED directory -- the global names none, so the
     * fallback would have left the table in the container. */
    void* const p2 = rad::LoadGateTestAccess::ptr(engine, 2);
    CHECK(p2 != nullptr && p2 != (void*)c.src[1]);
    CHECK(same_as(p2, c.want[1].data(), n2));
    const std::string id2 = rad::table_share_id(config.model, c.off[1], c.n[1]);
    REQUIRE(!id2.empty());
    CHECK(::access((dir2 + "/ngram-" + id2 + ".bin").c_str(), F_OK) == 0);
}

/* Finding 1 (round 4), the refusal: a named ram layer whose copy cannot be made stops startup
 * -- ngram_layer_asked.count(ng_layer) -- because the operator asked for that layer in
 * memory. The global placement being disk is no licence to demote it to the container the way
 * auto's failed copy is. RLIMIT_MEMLOCK under the table is the machine's own way of failing
 * the locked copy. */
TEST(a_named_ram_layer_whose_copy_cannot_be_made_refuses_startup) {
    const std::string name = "blk.1.engram_table.weight";
    const int64_t n = 3ll << 20;
    GateContainer c;
    REQUIRE(c.write(radtest::tmp_path("load-gate-refuse.rad"), n, name));
    REQUIRE_EQ(c.open(name), 0);
    rad::Config config;
    config.model = c.path;
    config.ngram_placement = rad::TablePlacement::Disk;
    config.ngram_resolved  = rad::TablePlacement::Disk;
    config.ngram_layer_asked    = { { 1, rad::TablePlacement::Ram } };
    config.ngram_layer_resolved = { { 1, rad::TablePlacement::Ram } };
    config.vram_weights_mib = 64; config.vram_kv_mib = 1; config.host_pool_mib = 64;
    rad::Engine engine(config);
    rad::EngineScope owner(engine.owner(), "load-gate-refuse");
    REQUIRE_EQ(rad::LoadGateTestAccess::prepare(engine, name, n), RAD_OK);
    rad::TableFillTestAccess::hold(engine, std::move(c.file));
    struct rlimit was, low;
    REQUIRE_EQ(::getrlimit(RLIMIT_MEMLOCK, &was), 0);
    if ((was.rlim_cur != RLIM_INFINITY && was.rlim_cur <= (rlim_t)(1ll << 20)) ||
        (was.rlim_max != RLIM_INFINITY && was.rlim_max < (rlim_t)(1ll << 20)))
        SKIP("the machine already caps memlock under the table");
    low.rlim_cur = (rlim_t)(1ll << 20);   /* under the table's 4 MiB aligned copy */
    low.rlim_max = was.rlim_max;          /* the hard limit untouched */
    REQUIRE_EQ(::setrlimit(RLIMIT_MEMLOCK, &low), 0);
    const int rc = rad::LoadGateTestAccess::load(engine);
    (void)::setrlimit(RLIMIT_MEMLOCK, &was);
    CHECK_EQ(rc, (int)RAD_E_NOMEM);
}

/* Finding 3 (round 4): the vram fit check charges the card only for the tables the global
 * placement actually puts there, exactly as the ram check adds the global share only under a
 * global ram. `1=vram` under a global disk is tested on layer 1's bytes alone: a card that
 * holds that table and not the whole passes the ask. The old accounting added the unasked
 * layers whatever the global said and refused this boot. */
TEST(a_vram_layer_under_a_global_disk_is_fitted_on_its_own_bytes) {
    const int64_t MiB = (int64_t)1 << 20;
    /* A card's claim is its capacity less the driver drift (16 MiB), the headroom (96) and
     * the arena: 2 GiB leaves 1872 MiB. Layer 1's 1 GiB fits; the whole 2 GiB table does not. */
    auto card = [](int64_t cap) {
        rad::VramFacts f{};
        f.capacity   = cap;
        f.free       = cap;
        f.arena      = 64 * MiB;
        f.kv_ceiling = 300 * MiB;
        return std::vector<rad::VramFacts>{ f };
    };
    const std::vector<int64_t> no_emb;
    const std::vector<rad::WeightFootprint> no_wfs;
    {   /* The card holds the named layer's table and not the global's: the ask passes. */
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_layer_asked = { { 1, rad::TablePlacement::Vram } };
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-vram-layer-own-bytes");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, card(2048 * MiB), no_emb, no_wfs, 0, 2048 * MiB, true, true, 0,
            { { 1, 1024 * MiB } });
        REQUIRE_EQ(rc, RAD_OK);
        const rad::Config& c = rad::TableResolveTestAccess::cfg(engine);
        CHECK(c.ngram_effective(1) == rad::TablePlacement::Vram);
        CHECK(c.ngram_resolved == rad::TablePlacement::Disk);
        CHECK_EQ(rad::TableResolveTestAccess::on_card(engine), (int64_t)1024 * MiB);
    }
    {   /* The ask itself does not fit: the fix is not a rubber stamp. */
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        config.ngram_layer_asked = { { 1, rad::TablePlacement::Vram } };
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-vram-layer-still-refused");
        const int rc = rad::TableResolveTestAccess::resolve(
            engine, card(2048 * MiB), no_emb, no_wfs, 0, 2049 * MiB, true, true, 0,
            { { 1, 2048 * MiB } });
        CHECK_EQ(rc, (int)RAD_E_NOMEM);
    }
}

/* Finding 2 (round 3): the snapshot rad_ngram_layer_resolved answers is one resolve pass's, not
 * a running history. Resolve with a per-layer answer, then resolve naming nothing: the layer
 * falls back to the global placement. Before the clear it kept the earlier engine's answer, and
 * the per-step prefetch gate read it. */
TEST(a_resolve_without_per_layer_answers_clears_the_snapshot) {
    const int64_t MiB = (int64_t)1 << 20;
    auto room_facts = [] {
        rad::VramFacts f{};
        f.capacity   = 16LL << 30;
        f.free       = f.capacity;
        f.arena      = 64LL << 20;
        f.kv_ceiling = 300LL << 20;
        return std::vector<rad::VramFacts>{ f };
    };
    {   /* An engine that names layer 3 ram: the view answers ram for it, disk elsewhere. */
        rad::Config config;
        config.ngram_placement   = rad::TablePlacement::Disk;
        config.ngram_layer_asked = {{3, rad::TablePlacement::Ram}};
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-snapshot-keep");
        REQUIRE_EQ(rad::TableResolveTestAccess::resolve(
                       engine, room_facts(), {}, {}, 0, 10 * MiB, true, true, 0,
                       {{3, 4 * MiB}}),
                   RAD_OK);
        CHECK_EQ(rad_ngram_layer_resolved(3), RAD_TABLE_PLACE_RAM);
        CHECK_EQ(rad_ngram_layer_resolved(4), RAD_TABLE_PLACE_DISK);
    }
    {   /* A later resolve that names no layer: layer 3 is no one's ram layer any more. */
        rad::Config config;
        config.ngram_placement = rad::TablePlacement::Disk;
        rad::Engine engine(config);
        rad::EngineScope owner(engine.owner(), "resolve-snapshot-clear");
        REQUIRE_EQ(rad::TableResolveTestAccess::resolve(
                       engine, room_facts(), {}, {}, 0, 10 * MiB, true, true, 0, {}),
                   RAD_OK);
        CHECK_EQ(rad_ngram_layer_resolved(3), RAD_TABLE_PLACE_DISK);
        CHECK_EQ(rad_ngram_layer_resolved(-1), RAD_TABLE_PLACE_DISK);
    }
}

RAD_TEST_MAIN()

