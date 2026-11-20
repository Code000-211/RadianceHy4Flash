/* engine_pcache.cpp -- the prefix cache across restarts (--prefix-cache-dir, core/mem/pcstore.h):
 * the store's key, its opening and import at startup, the lazy read before a restore, and the
 * flushes (at a clean stop, and when idle every --prefix-cache-flush-s).
 */
#include "engine.h"
#include "engine_priv.h"
#include "diag/diag.h"
#include "mem/pcstore.h"
#include "runtime/tierexec.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

extern char** environ;

namespace rad {

namespace {

std::string file_id(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return path + " (missing)";
    return fmt("%s size=%lld mtime=%lld.%09ld ino=%llu", path.c_str(), (long long)st.st_size,
               (long long)st.st_mtim.tv_sec, (long)st.st_mtim.tv_nsec,
               (unsigned long long)st.st_ino);
}

/* A container is tens of gigabytes: its identity is its stat and the hash of its first and last
 * MiB (the header and the tensor table), not of its contents. */
std::string container_id(const std::string& path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) return "model " + path + " (not a file)";
    if (S_ISDIR(st.st_mode)) {
        std::set<std::string> rows;
        if (DIR* d = ::opendir(path.c_str())) {
            while (dirent* e = ::readdir(d)) {
                if (e->d_name[0] == '.') continue;
                rows.insert(file_id(path + "/" + e->d_name));
            }
            ::closedir(d);
        }
        std::string s = "model dir " + path + "\n";
        for (const std::string& r : rows) s += "  " + r + "\n";
        return s;
    }
    std::string s = "model " + file_id(path);
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        const int64_t mib = (int64_t)1 << 20;
        std::vector<char> b((size_t)mib);
        const ssize_t a = ::pread(fd, b.data(), b.size(), 0);
        s += fmt(" head=%016llx", (unsigned long long)pc_checksum(b.data(), a > 0 ? a : 0));
        const int64_t off = std::max<int64_t>(0, (int64_t)st.st_size - mib);
        const ssize_t z = ::pread(fd, b.data(), b.size(), off);
        s += fmt(" tail=%016llx", (unsigned long long)pc_checksum(b.data(), z > 0 ? z : 0));
        ::close(fd);
    }
    return s + "\n";
}

/* Every executable file mapped into the process: the binary, the kernel and architecture plugins,
 * the runtime and libm. A rebuild or an upgrade of any of them may round the KV differently. */
std::string mapped_code() {
    std::set<std::string> paths;
    if (FILE* f = std::fopen("/proc/self/maps", "r")) {
        char line[4096];
        while (std::fgets(line, sizeof line, f)) {
            char perms[8] = {};
            if (std::sscanf(line, "%*s %7s", perms) != 1 || !std::strchr(perms, 'x')) continue;
            const char* p = std::strchr(line, '/');
            if (!p) continue;
            std::string path(p);
            while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
            if (path.find(" (deleted)") != std::string::npos) continue;
            paths.insert(path);
        }
        std::fclose(f);
    }
    std::string s;
    for (const std::string& p : paths) s += "code " + file_id(p) + "\n";
    return s;
}

/* The tuned kernel choices (tunecache.h) move without any binary changing. */
std::string tune_files(const std::string& home) {
    std::string s;
    const std::string dir = home + "/tune";
    std::set<std::string> names;
    if (DIR* d = ::opendir(dir.c_str())) {
        while (dirent* e = ::readdir(d)) if (e->d_name[0] != '.') names.insert(e->d_name);
        ::closedir(d);
    }
    for (const std::string& n : names) {
        std::string body;
        if (FILE* f = std::fopen((dir + "/" + n).c_str(), "rb")) {
            char buf[65536];
            size_t k;
            while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) body.append(buf, k);
            std::fclose(f);
        }
        s += fmt("tune %s/%s %016llx\n", dir.c_str(), n.c_str(),
                 (unsigned long long)pc_checksum(body.data(), (int64_t)body.size()));
    }
    return s;
}

/* The options that change what a step computes for the same tokens. Scheduling, memory and
 * sampling options do not: the cache already serves KV across them within one process. */
const char* const kKeyFlags[] = {
    "--model", "--kernels", "--radiance-home", "--kv-cache-dtype", "--state-cache-dtype",
    "--kv-resident", "--tp", "--tp-attn-split", "--tp-wire", "--tp-wire-min-kb", "--p2p",
    "--placement", "--rope-scaling", "--max-model-len", "--max-num-batched-tokens",
    "--checkpoint-interval", "--prefix-resume", "--prefill-tail-snapshot", "--prefill-span",
    "--prefill-span-pass", "--deterministic", "--control-vector", "--lora", "--lora-dir",
    "--max-loras", "--max-lora-rank", "--lora-target-modules", "--embedding-placement",
    "--ngram-placement", "--vision-tower-placement", "--image-min-pixels", "--image-max-pixels",
    "--image-tokens", "--mm-max-patches", "--video-min-pixels", "--video-max-pixels",
    "--video-fps", "--video-max-frames", "--video-min-frames", "--video-max-frame-tokens",
    "--max-source-pixels", "--num-speculative-tokens", "--generation-config",
};

bool env_counts(const char* kv) {
    static const char* const kPrefixes[] = { "RADIANCE_", "R4D_", "RAD_", "HSA_", "HIP_", "AMD_", "ROCBLAS_" };
    static const char* const kSkip[] = { "HIP_VISIBLE_DEVICES=", "RADIANCE_LOG", "RAD_TEST_" };
    bool hit = false;
    for (const char* p : kPrefixes) hit = hit || std::strncmp(kv, p, std::strlen(p)) == 0;
    for (const char* p : kSkip) if (std::strncmp(kv, p, std::strlen(p)) == 0) return false;
    return hit;
}

}  // namespace

std::string Engine::pcache_key_text() {
    std::string s = fmt("radiance prefix store 1\nradiance %s abi %u\n", RAD_VERSION, (unsigned)RAD_ABI_VERSION);
    s += container_id(cfg_.model);
    s += mapped_code();
    s += tune_files(cfg_.radiance_home);
    std::set<std::string> env;
    for (char** e = environ; e && *e; ++e) if (env_counts(*e)) env.insert(*e);
    for (const std::string& e : env) s += "env " + e + "\n";
    for (const diag::ConfigRow& r : diag::config_rows(cfg_))
        for (const char* f : kKeyFlags)
            if (r.flag == f) s += "opt " + r.flag + "=" + r.value + "\n";
    s += fmt("tp %d\n", cfg_.tp);
    for (size_t i = 0; i < dev_ids_.size(); ++i)
        s += fmt("card %zu %s\n", i, dev_ids_[i].name.c_str());
    s += fmt("geometry quantum=%lld attn_block=%lld state_chunk=%lld interval=%lld max_tok=%lld "
             "exact=%d cvec=%016llx\n", (long long)geo_.quantum, (long long)geo_.attn_block,
             (long long)geo_.state_chunk, (long long)geo_.checkpoint_interval,
             (long long)cfg_.max_tok, cfg_.prefix_exact ? 1 : 0, (unsigned long long)cvec_salt_);
    const PrefixCache& pc = ranks_[0]->prefix;
    s += fmt("cache block=%lld groups=", (long long)pc.block_size());
    for (int32_t g : pc.entry_groups()) s += fmt("%d,", g);
    s += "\n";
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const KVManager& kv = ranks_[r]->kv;
        s += fmt("rank %zu ckpt=%lld win=%d\n", r, (long long)kv.checkpoint_bytes(),
                 kv.has_window_snapshot() ? 1 : 0);
        for (const KVGroupPlan& p : kv.plans())
            s += fmt("  group %d %s kind=%d layers=%lld block=%lld bpb=%lld bps=%lld conv=%lld "
                     "win=%lld zero=%d rep=%d drafter=%d\n", p.index, p.name.c_str(), p.kind,
                     (long long)p.n_layers, (long long)p.block_size, (long long)p.bytes_per_block,
                     (long long)p.bytes_per_state, (long long)p.conv_slots,
                     (long long)p.window_blocks, p.zero ? 1 : 0, p.replicated ? 1 : 0,
                     p.drafter ? 1 : 0);
    }
    std::vector<const KVManager*> rank_kv;
    for (auto& r : ranks_) rank_kv.push_back(&r->kv);
    TierExec::SlotLayout layout;
    if (TierExec::slot_layout(rank_kv, pc.entry_groups(), &layout) == RAD_OK) {
        s += fmt("slot bytes=%lld rep=%lld own=", (long long)layout.slot_bytes, (long long)layout.rep_bytes);
        for (size_t i = 0; i < layout.own_off.size(); ++i)
            s += fmt("%lld+%lld,", (long long)layout.own_off[i], (long long)layout.own_bytes[i]);
        s += "\n";
    }
    s += fmt("ckpt store slot=%lld\n", (long long)(ck_store_.enabled() ? ck_store_.slot_bytes() : 0));
    return s;
}

int Engine::pcache_open() {
    if (cfg_.prefix_cache_dir.empty()) return RAD_OK;
    Rank& lead = *ranks_[0];
    PrefixCache& pc = lead.prefix;
    const char* why = !cfg_.prefix_cache || !pc.enabled() ? "prefix caching is off"
                      : !lead.tierx.enabled() || !lead.tiers.host_on()
                          ? "it needs the host tier (--prefix-cache-mib), which is off"
                      : !pc.demotable() ? "a cached KV group streams, so no entry has a host copy"
                                        : nullptr;
    if (why) {
        RAD_WARN("--prefix-cache-dir %s: off, %s", cfg_.prefix_cache_dir.c_str(), why);
        return RAD_OK;
    }
    RAD_WARN("--prefix-cache-dir %s holds conversation content (KV and token ids) unencrypted; "
             "treat the directory as sensitive", cfg_.prefix_cache_dir.c_str());
    auto st = std::make_unique<PrefixStore>();
    const uint64_t tk = rad_mono_ns();
    PrefixStore::Config c;
    c.dir = cfg_.prefix_cache_dir;
    c.name = cfg_.served_model_name;
    c.key_text = pcache_key_text();
    c.entry_bytes = lead.tiers.entry_bytes();
    c.snap_bytes = ck_store_.enabled() ? ck_store_.slot_bytes() : 0;
    c.block_tokens = pc.block_size();
    c.cap_bytes = cfg_.prefix_cache_dir_mib > 0
                      ? cfg_.prefix_cache_dir_mib << 20
                      : lead.tiers.arena().cap_bytes() + (ck_store_.enabled() ? ck_store_.cap_bytes() : 0);
    c.ttl_s = cfg_.prefix_cache_ttl;
    const double key_s = (double)(rad_mono_ns() - tk) * 1e-9;
    if (const int rc = st->open(c); rc < 0) {
        RAD_WARN("--prefix-cache-dir %s: the store did not open (%s); off",
                 cfg_.prefix_cache_dir.c_str(), rad_strerror(rc));
        return RAD_OK;
    }
    const uint64_t t0 = rad_mono_ns();
    int64_t ns = 0;
    const int64_t ne = pc_import(pc, *st, &ns);
    RAD_INFO("%s", st->report().c_str());
    RAD_INFO("prefix store: key in %.3f s; %lld entries and %lld snapshots indexed in %.3f s, read "
             "on first use; up to %s kept, written at a stop%s", key_s, (long long)ne, (long long)ns,
             (double)(rad_mono_ns() - t0) * 1e-9, humanb(c.cap_bytes).c_str(),
             cfg_.prefix_cache_flush_s > 0
                 ? fmt(" and when idle every %lld s", (long long)cfg_.prefix_cache_flush_s).c_str()
                 : "");
    pstore_ = std::move(st);
    pstore_last_ns_ = rad_mono_ns();
    pstore_seen_ = pcache_activity();
    return RAD_OK;
}

int64_t Engine::pcache_activity() const {
    const Rank& lead = *ranks_[0];
    return lead.tiers.stats().to_host + lead.prefix.ckpt_commits() + lead.tiers.stats().paged_in;
}

void Engine::pcache_page_in(const Request& req) {
    if (!pstore_) return;
    Rank& lead = *ranks_[0];
    std::vector<MMContentKey> mm;
    media_keys(req, &mm, cvec_salt_for(req.sp.use_control_vector));
    PcPageIn r;
    (void)pc_page_in(lead.prefix, lead.tiers, ck_store_.enabled() ? &ck_store_ : nullptr, *pstore_,
                     req.prompt, mm, &r);
    if (r.entries > 0 || r.snap || r.failed > 0)
        RAD_INFO("prefix store: request %llu read %lld entries (%lld tokens)%s, %s, from disk in "
                 "%.3f s%s", (unsigned long long)req.id, (long long)r.entries,
                 (long long)(r.entries * lead.prefix.block_size()),
                 r.snap ? " and a snapshot" : "", humanb(r.bytes).c_str(), r.secs,
                 r.failed ? fmt("; %lld failed and were dropped", (long long)r.failed).c_str() : "");
}

void Engine::pcache_flush(bool final) {
    if (!pstore_) return;
    Rank& lead = *ranks_[0];
    const uint64_t t0 = rad_mono_ns();
    if (final) {
        /* Every finished turn's copy lands first, so what is only in VRAM gets a host copy. */
        (void)rad_dev_set(lead.device);
        for (auto& rk : ranks_) (void)rad_stream_sync(rk->ctx.stream());
        for (int round = 0; round < 1024 && lead.tiers.writeback_waiting(); ++round)
            if (!flush_copies()) break;
        if (pump_ && !pump_->copy.empty()) (void)flush_copies();
    }
    const double land_s = (double)(rad_mono_ns() - t0) * 1e-9;
    PcFlush f;
    int rc;
    {
        const auto lk = sched_.hold();
        rc = pc_flush_cache(lead.prefix, lead.tiers, ck_store_.enabled() ? &ck_store_ : nullptr,
                            *pstore_, cfg_.prefix_cache_ttl, &f);
    }
    pstore_last_ns_ = rad_mono_ns();
    pstore_seen_ = pcache_activity();
    if (rc < 0) return;   /* the store said why */
    if (final || f.written_items > 0 || f.segs_deleted > 0)
        RAD_INFO("prefix store: %s: wrote %lld item(s), %s, in %.2f s%s; it holds %lld item(s), "
                 "%s live, %s on disk%s%s", final ? "at stop" : "idle flush",
                 (long long)f.written_items, humanb(f.written_bytes).c_str(), f.secs,
                 final ? fmt(" (copies landed in %.2f s first)", land_s).c_str() : "",
                 (long long)f.kept_items, humanb(f.live_bytes).c_str(),
                 humanb(pstore_->disk_bytes()).c_str(),
                 f.compacted ? ", compacted" : "",
                 f.dropped ? fmt(", %lld left out by the cap", (long long)f.dropped).c_str() : "");
}

void Engine::gsuffix_open() {
    GlobalSuffix& g = sched_.global_suffix();
    if (!g.enabled()) return;
    const std::string dir = !cfg_.draft_suffix_dir.empty() ? cfg_.draft_suffix_dir : cfg_.prefix_cache_dir;
    RAD_INFO("draft-suffix: a global cache of earlier responses, up to %lld MiB, beside each "
             "request's own index%s", (long long)cfg_.draft_suffix_global_mib,
             dir.empty() ? " (not kept across restarts: no --draft-suffix-dir)" : "");
    if (dir.empty() || !vocab_) return;
    if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
        RAD_WARN("draft-suffix: cannot create %s: %s", dir.c_str(), std::strerror(errno));
        return;
    }
    /* Token ids mean something only under one vocabulary: the key is every piece. */
    const uint64_t t0 = rad_mono_ns();
    std::string pieces;
    const int32_t nv = vocab_->n_tokens();
    for (int32_t i = 0; i < nv; ++i) { pieces += vocab_->text(i); pieces += '\0'; }
    gsuffix_key_ = fmt("radiance suffix cache 1\nvocab kind=%u n=%d pieces=%016llx\nk=%d..%d\n", vocab_->kind(), nv,
                       (unsigned long long)pc_checksum(pieces.data(), (int64_t)pieces.size()),
                       cfg_.draft_suffix_min_match,
                       std::min(cfg_.draft_suffix_min_match + (int)SuffixIndex::kExtra, (int)SuffixIndex::kMaxK));
    std::string name;
    for (char c : cfg_.served_model_name) name += std::isalnum((unsigned char)c) || c == '-' || c == '_' || c == '.' ? c : '_';
    gsuffix_path_ = dir + "/" + (name.empty() ? std::string("model") : name) + ".suffix";
    int64_t n = 0;
    const int rc = g.load(gsuffix_path_, gsuffix_key_, &n);
    if (rc == RAD_OK)
        RAD_INFO("draft-suffix: %lld earlier response(s) read from %s in %.3f s, indexed in the "
                 "background", (long long)n, gsuffix_path_.c_str(), (double)(rad_mono_ns() - t0) * 1e-9);
    else if (rc != RAD_E_NOTFOUND)
        RAD_WARN("draft-suffix: %s does not match this tokenizer or is damaged; starting empty",
                 gsuffix_path_.c_str());
    gsuffix_saved_gen_ = g.generation();
    gsuffix_last_ns_ = rad_mono_ns();
}

void Engine::gsuffix_save(bool final) {
    GlobalSuffix& g = sched_.global_suffix();
    if (gsuffix_path_.empty() || !g.enabled()) return;
    if (!final) {
        const uint64_t now = rad_mono_ns();
        if (cfg_.prefix_cache_flush_s <= 0 ||
            now - gsuffix_last_ns_ < (uint64_t)cfg_.prefix_cache_flush_s * 1000000000ull) return;
        gsuffix_last_ns_ = now;
    }
    if (g.generation() == gsuffix_saved_gen_ && g.stats().queued == 0) return;
    const uint64_t t0 = rad_mono_ns();
    if (g.save(gsuffix_path_, gsuffix_key_) != RAD_OK) return;
    gsuffix_saved_gen_ = g.generation();
    const GlobalSuffix::Stats st = g.stats();
    RAD_INFO("draft-suffix: %lld response(s), %lld tokens, saved to %s in %.3f s%s",
             (long long)st.seqs, (long long)st.tokens, gsuffix_path_.c_str(),
             (double)(rad_mono_ns() - t0) * 1e-9, final ? " at stop" : "");
}

void Engine::pcache_tick(uint64_t now_ns) {
    gsuffix_save(false);
    if (!pstore_ || cfg_.prefix_cache_flush_s <= 0) return;
    if (now_ns - pstore_last_ns_ < (uint64_t)cfg_.prefix_cache_flush_s * 1000000000ull) return;
    if (pcache_activity() == pstore_seen_) { pstore_last_ns_ = now_ns; return; }
    /* Copies still on their way are written by the next tick. */
    if (pump_ && !pump_->copy.empty()) return;
    pcache_flush(false);
}

}  /* namespace rad */
