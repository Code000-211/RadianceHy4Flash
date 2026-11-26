/* pcstore.h -- the prefix cache kept across restarts (--prefix-cache-dir).
 *
 * At runtime nothing changes: the cache lives in VRAM, the host tier (--prefix-cache-mib) and the
 * snapshot store (--checkpoint-slots), as kvtier.h describes. This store is written from those
 * host copies on a clean stop (and, with --prefix-cache-flush-s, whenever the server has been idle
 * that long with something new), and read back at the next start: the index at once, an entry's
 * bytes only when a request's chain reaches it (Engine::pcache_page_in). A conversation then
 * resumes after a restart for the cost of a disk read instead of a prefill.
 *
 * ============================== IT HOLDS CONVERSATION CONTENT ON DISK, UNENCRYPTED ==============
 * A KV block is a lossless function of the prompt, and its token ids are stored beside it so a hit
 * can be verified. The directory must be treated as sensitive as the conversations are; startup
 * says so whenever --prefix-cache-dir is set.
 * ================================================================================================
 *
 * Safety. The bytes are only meaningful to an engine that would compute exactly those bytes: the
 * same container, build, plugins, kernels and their tuned choices, KV encoding, tensor-parallel
 * layout, block and slot geometry, checkpoint geometry and model-relevant options. The engine
 * spells all of that out as text (Engine::pcache_key_text); its 128-bit hash is the store's key.
 * Every file carries the key and a file whose key is not exactly this engine's is discarded, never
 * read. Every payload carries a 64-bit checksum, checked on every read; a mismatch drops the entry
 * (a miss, never a wrong hit). Writes go to a temporary file, are synced, and are renamed into
 * place, the index last, so a crash or a full disk at any point leaves the previous store intact.
 *
 * Layout: DIR/<name>.<key hex>/ holds `index.rpx` and the segment files `seg-<n>.rps` it names. A
 * flush appends one segment with what changed since the last flush, writes a new index, and
 * deletes segments nothing references any more; when dead bytes outgrow the live ones the flush
 * rewrites everything into one segment (compaction). Another key's directory under the same name
 * is a previous build or configuration of this model and is removed at open (its first differing
 * key line is logged), so one directory serves one server.
 *
 * Not thread-safe: the engine calls it from its step thread only.
 */
#pragma once
#include "mem/prefix.h"
#include "rad_core.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* 64-bit content checksum (XXH3-style lanes, util/simd.h). */
uint64_t pc_checksum(const void* p, int64_t n);
/* The 128-bit key of a key text. */
BlockHash pc_key_of(const std::string& text);

struct PcLoc {
    uint32_t seg = 0;
    uint64_t off = 0;
    uint64_t sum = 0;
};

/* One attention entry on disk: the cache's Entry and the tiers' clock and depth. */
struct PcEntry {
    BlockHash key;
    std::vector<int32_t> tokens;
    int64_t depth = 0;
    uint8_t n_children = 0;
    int64_t used_ms = 0;          /* unix ms of the last use */
    PcScore score;
    PcLoc   loc;
};

/* One linear-state / window snapshot on disk. `group` is the session the snapshots were indexed
 * under, so the retention policy sees one ladder per conversation after a restart. */
struct PcSnap {
    BlockHash key;
    int64_t  pos = 0;
    uint64_t group = 0;
    int64_t  used_ms = 0;
    PcScore  score;
    PcLoc    loc;
};

/* A flush's input: one live item of the cache, most valuable first within its kind. `bytes` is
 * its host copy, or null when it is held only on disk (it must be in the index then). `fresh`
 * means the host copy is not known to equal what the index holds. */
struct PcPut {
    BlockHash key;
    const void* bytes = nullptr;
    bool fresh = true;
    /* entries */
    const int32_t* tokens = nullptr;
    int64_t depth = 0;
    uint8_t n_children = 0;
    /* snapshots */
    int64_t pos = 0;
    uint64_t group = 0;
    int64_t used_ms = 0;
    PcScore score;
    /* Write it if it is not on disk yet. False leaves a new item off disk (admission control);
     * one the index already holds with these bytes stays either way. */
    bool admit = true;
};

/* What one flush may do. */
struct PcFlushOpts {
    int64_t write_budget = -1;    /* payload bytes this flush may write, -1 unbounded */
    bool    compact = true;       /* rewrite everything when dead bytes outweigh the live */
};

struct PcFlush {
    int64_t written_items = 0, written_bytes = 0;   /* payloads written this flush */
    int64_t kept_items = 0, live_bytes = 0;         /* in the new index */
    int64_t dropped = 0;                            /* left out: the cap, or not offered */
    int64_t rejected = 0;                           /* new, not admitted (PcPut::admit) */
    int64_t deferred = 0;                           /* admitted, over the write budget */
    int64_t segs_deleted = 0;
    bool    compacted = false;
    double  secs = 0;
    /* Items the new index does not hold, which a disk-only record must not outlive. */
    std::vector<BlockHash> gone_entries, gone_snaps;
    /* Items the new index holds with exactly the offered bytes. */
    std::vector<BlockHash> stored_entries, stored_snaps;
};

class PrefixStore {
public:
    struct Config {
        std::string dir;
        std::string name;             /* the served model, one directory per server */
        std::string key_text;
        int64_t entry_bytes = 0;      /* one host slot */
        int64_t snap_bytes = 0;       /* one snapshot store slot, 0 without */
        int64_t block_tokens = 0;
        int64_t cap_bytes = 0;        /* live payload bytes, 0 unbounded */
        int64_t ttl_s = 0;            /* items unused longer are not loaded, 0 keeps all */
    };
    ~PrefixStore();
    /* Opens (creating) the directory and loads a valid index of this key. Leftover temporary
     * files, unreferenced segments and other keys' directories under this name are removed. A
     * missing, foreign or damaged index is not an error: the store starts empty. */
    int  open(const Config& c);
    void close();
    bool enabled() const { return enabled_; }
    const std::string& path() const { return path_; }
    std::string report() const { return report_; }

    /* What open() loaded, oldest use first. Valid until the first flush. */
    const std::vector<PcEntry>& loaded_entries() const { return load_e_; }
    const std::vector<PcSnap>&  loaded_snaps() const { return load_s_; }
    void release_loaded() { std::vector<PcEntry>().swap(load_e_); std::vector<PcSnap>().swap(load_s_); }

    bool has_entry(const BlockHash& k) const { return ents_.count(k) > 0; }
    bool has_snap(const BlockHash& k) const { return snaps_.count(k) > 0; }
    int64_t n_entries() const { return (int64_t)ents_.size(); }
    int64_t n_snaps() const { return (int64_t)snaps_.size(); }
    int64_t disk_bytes() const;    /* every segment */

    /* Reads entries into `dst[i]` (entry_bytes each) and checks each checksum; `ok[i]` says
     * which came back whole. Neighbours on disk are read in one call. An entry that fails is
     * forgotten here (the next flush leaves it out). Returns how many came back. */
    int64_t read_entries(const BlockHash* keys, int64_t n, void* const* dst, char* ok);
    /* One snapshot into `dst` (snap_bytes). RAD_OK, or an error and the snapshot is forgotten. */
    int read_snap(const BlockHash& key, void* dst);

    /* Writes the cache's live set: `ents` and `snaps` each most valuable first. What is not
     * offered, or does not fit the cap, leaves the index. On any failure (a full disk, an I/O
     * error) the previous index and its segments stay as they were and an error is returned. */
    int flush(const std::vector<PcPut>& ents, const std::vector<PcPut>& snaps, PcFlush* out,
              const PcFlushOpts& opts = PcFlushOpts{});

    struct Stats {
        int64_t flushes = 0, flush_failures = 0, reads = 0, read_bytes = 0, read_failures = 0;
        int64_t written_bytes = 0;
        double  read_s = 0, write_s = 0;
    };
    const Stats& stats() const { return st_; }

    /* Tests: fail a payload write once this many bytes of the flush are written (-1 never). */
    void test_fail_after(int64_t bytes) { fail_after_ = bytes; }

private:
    struct Seg { int64_t bytes = 0; int fd = -1; };
    struct EMeta { std::vector<int32_t> tokens; int64_t depth = 0; uint8_t n_children = 0;
                   int64_t used_ms = 0; PcScore score; PcLoc loc; };
    struct SMeta { int64_t pos = 0; uint64_t group = 0; int64_t used_ms = 0; PcScore score; PcLoc loc; };

    int  load_index();
    int  write_index(const std::unordered_map<BlockHash, EMeta, BlockHashHash>& e,
                     const std::unordered_map<BlockHash, SMeta, BlockHashHash>& s,
                     const std::unordered_map<uint32_t, Seg>& segs, uint32_t next_seg);
    int  seg_fd(uint32_t id);
    std::string seg_path(uint32_t id) const;
    void close_fds();
    void sweep(const std::unordered_map<uint32_t, Seg>& keep);
    int  read_at(uint32_t seg, uint64_t off, void* dst, int64_t n);

    Config      cfg_;
    BlockHash   key_;
    bool        enabled_ = false;
    std::string path_;
    std::string report_;
    std::unordered_map<BlockHash, EMeta, BlockHashHash> ents_;
    std::unordered_map<BlockHash, SMeta, BlockHashHash> snaps_;
    std::unordered_map<uint32_t, Seg> segs_;
    uint32_t    next_seg_ = 1;
    std::vector<PcEntry> load_e_;
    std::vector<PcSnap>  load_s_;
    Stats       st_;
    int64_t     fail_after_ = -1;
};

/* ------------------------------------------------------------------ the cache's side
 *
 * The glue between the store and a configured PrefixCache with its IdleTiers and snapshot store,
 * shared by the engine and the tests. All run on the step thread under the scheduler's lock. */
class IdleTiers;
class HostArena;

/* The store's policy (--prefix-cache-dir-*). What a flush keeps is ordered by LRFU score, and a
 * cap or budget takes the lowest first. A new item is admitted to disk only when it has earned it:
 * reused at least once, or part of a long conversation (a chain of `long_tokens` or more, its
 * snapshots at or past that depth); at a clean stop (`final`) also anything used within
 * `active_s`, so the conversations in progress resume. */
struct PcPolicy {
    int64_t ttl_s = 0;
    double  half_s = 6 * 3600.0;
    int64_t active_s = 3600;      /* 0: the stop admits nothing extra */
    int64_t long_tokens = 32768;  /* 0: no length rule */
    bool    final = false;
    PcFlushOpts opts;
};

/* Writes the cache's live set (what has a host copy or is held on disk; snapshots in `ck`) by
 * the policy, leaving out what --prefix-cache-ttl has expired. */
int pc_flush_cache(PrefixCache& pc, IdleTiers& t, const HostArena* ck, PrefixStore& st,
                   const PcPolicy& pol, PcFlush* out);
/* Imports what the store loaded into an empty cache: entries as disk records, snapshots on disk
 * only. Returns the entries imported; `snaps` the snapshots. */
int64_t pc_import(PrefixCache& pc, PrefixStore& st, int64_t* snaps);
struct PcPageIn {
    int64_t entries = 0, bytes = 0, failed = 0;
    bool    snap = false;
    double  secs = 0;
};
/* Reads the disk records on this prompt's chain into host slots and the snapshot a hit would
 * resume from into the snapshot store, so the restore and lookup that follow see them. */
int pc_page_in(PrefixCache& pc, IdleTiers& t, HostArena* ck, PrefixStore& st,
               const std::vector<int32_t>& tokens, const std::vector<MMContentKey>& mm,
               PcPageIn* out);

}  /* namespace rad */
