/* rad_kernel_override.cpp -- a kernel library made of another one, with a few of its translation
 * units rebuilt (rad_add_kernel_override, cmake/RadAddPlugin.cmake; docs/PLUGIN.md §14).
 *
 * The file this is linked into holds the rebuilt units and names the base library
 * (RAD_OVERRIDE_BASE, an absolute path, which the link also records as a dependency so the units'
 * references to the rest of the base resolve there). It answers every kernel-library export with
 * the base's own answer: the same plugin name, schemas, rows in the same order with the same
 * constraints and priorities, and the same per-row flags, so selection, tuning and chaining are
 * the base's. The one difference is the hooks: a row's hook the base takes from a symbol this file
 * also defines is the one defined here.
 *
 * The file carries the base's file name and sits in a home ahead of the base's on the search
 * path, where it shadows it (Registry::load_dirs), so no configuration names it.
 *
 * It refuses to open when no row's hook comes from it: such an override changes nothing, and a
 * measurement taken over it would be the base's under another name.
 *
 * A running engine takes a rebuild of this file without a restart (rad_kernel_reload, rad_abi.h):
 * the file as it now stands on disk is loaded again under a name of its own, its rows are read as
 * they were at load, and only their hooks are put into the rows served here, which the engine holds
 * pointers to. The base is not loaded again: the rebuilt units resolve into the one already mapped.
 */
#include "rad_abi.h"
#include "rad_abi_layout.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef RAD_OVERRIDE_BASE
#error "RAD_OVERRIDE_BASE names the library this one overrides (rad_add_kernel_override sets it)"
#endif

namespace {

/* A handle taken to look symbols up and given back at once. One kept on this file would stop the
 * loader's dlclose from ever unloading it, and one kept on the base would leave the base mapped,
 * bound to this file, after this file is gone. The base stays mapped for as long as this file is,
 * as its dependency. RTLD_NOLOAD, because both are mapped by the time any of this runs. */
struct Mapped {
    void* h;
    explicit Mapped(const char* path) : h(dlopen(path, RTLD_NOW | RTLD_NOLOAD)) {}
    ~Mapped() { if (h) dlclose(h); }
    Mapped(const Mapped&) = delete;
    Mapped& operator=(const Mapped&) = delete;
};

/* The image a code address lies in, or null. */
const void* image_of(const void* p) {
    Dl_info d{};
    return p && dladdr(p, &d) ? d.dli_fbase : nullptr;
}

/* ------------------------------------------------------------------ the base's exports */
/* Found by name in the base's own image; resolve() calls none of them. */
struct Base {
    std::string err;            /* why nothing can be served, or empty */
    std::string self_path;
    const void* self_image = nullptr;

    uint32_t             (*abi)() = nullptr;
    uint64_t             (*layout)() = nullptr;
    const RadPluginInfo* (*info)() = nullptr;
    int                  (*schema_count)() = nullptr;
    const RadOpSchema*   (*schema_at)(int) = nullptr;
    int                  (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    RadPluginOpenFn       open = nullptr;
    RadPluginCloseFn      close = nullptr;
    RadPluginFreedFn      freed = nullptr;
    RadPluginReferenceFn  reference = nullptr;
    RadKernelReferenceFn  kernel_reference = nullptr;
    RadKernelConcurrentFn concurrent = nullptr;
    RadKernelCollectiveFn collective = nullptr;
    RadKernelChainFn      chain = nullptr;
};

/* This thread is reading the base's rows (build(), below). The base came in as this file's
 * dependency, so a call one of its exports makes to another -- libr4d's rad_kernel_at asks
 * rad_kernel_count -- may be bound to this file's export of that name. While the rows are being
 * read there is nothing here to answer from, and the base is asking itself: it gets its own
 * answer. */
thread_local bool t_reading = false;

template <class F> void find(F* out, void* h, const char* name) {
    *out = reinterpret_cast<F>(dlsym(h, name));
}

Base resolve() {
    Base b;
    Dl_info me{};
    if (!dladdr(reinterpret_cast<const void*>(&image_of), &me) || !me.dli_fname) {
        b.err = "cannot find its own image";
        return b;
    }
    b.self_path = me.dli_fname;
    b.self_image = me.dli_fbase;
    /* The link names the base as a dependency (rad_add_kernel_override). A base that is not
     * mapped was linked some other way, and its rows would be pointers into a library nothing
     * holds. */
    Mapped base(RAD_OVERRIDE_BASE);
    if (!base.h) {
        b.err = "the library it overrides, " RAD_OVERRIDE_BASE ", is not mapped as its "
                "dependency: it has to be linked with rad_add_kernel_override";
        return b;
    }
    /* Looked up through the base's handle, so these are the base's definitions and not this
     * file's exports of the same names. */
    find(&b.abi, base.h, "rad_plugin_abi_version");
    find(&b.layout, base.h, "rad_plugin_abi_layout");
    find(&b.info, base.h, "rad_plugin_info");
    find(&b.schema_count, base.h, "rad_kernel_schema_count");
    find(&b.schema_at, base.h, "rad_kernel_schema_at");
    find(&b.count, base.h, "rad_kernel_count");
    find(&b.at, base.h, "rad_kernel_at");
    find(&b.open, base.h, "rad_plugin_open");
    find(&b.close, base.h, "rad_plugin_close");
    find(&b.freed, base.h, "rad_plugin_freed");
    find(&b.reference, base.h, "rad_plugin_reference");
    find(&b.kernel_reference, base.h, "rad_kernel_reference");
    find(&b.concurrent, base.h, "rad_kernel_concurrent");
    find(&b.collective, base.h, "rad_kernel_collective");
    find(&b.chain, base.h, "rad_kernel_chain");
    if (!b.abi || !b.layout || !b.info)
        b.err = "the library it overrides is not a radiance plugin";
    else if (!b.schema_count || !b.schema_at || !b.count || !b.at)
        b.err = "the library it overrides is missing rad_kernel_schema_count/at or "
                "rad_kernel_count/at: it is not a kernel library";
    return b;
}

const Base& base() {
    static const Base b = resolve();
    return b;
}

/* ------------------------------------------------------------------ the rows served */
/* Every hook of a row: what a rebuilt unit may replace. Every other field of a row is the base's
 * declaration. */
#define OV_HOOKS(X) \
    X(tune_valid) X(init) X(fini) X(launch) X(scratch) X(layout) X(relayout) X(opd_shape) \
    X(unrelayout) X(replaces)

struct Override {
    std::string   err;
    RadPluginInfo info{};
    std::string   version;
    std::vector<RadKernelInfo> rows;      /* the base's, hooks replaced; never resized once built */
    std::vector<std::string>   replaced;  /* the rows with a hook from this file */
    /* The file whose hooks the rows hold: this one as it was loaded, then each reload's. A reload
     * of a file that is still this one is refused. */
    struct stat   file{};
};

/* If `*hook` is not this file's, and this file defines the symbol the base takes it from, make it
 * this file's. Returns whether the hook is this file's afterwards. The base's reference may
 * already have been bound here by the dynamic linker (the base came in as this file's dependency
 * and the symbol is preemptible), which counts the same. */
template <class F> bool take(F* hook, void* self, const void* self_image) {
    const void* p = reinterpret_cast<const void*>(*hook);
    if (!p) return false;
    Dl_info d{};
    if (!dladdr(p, &d)) return false;
    if (d.dli_fbase == self_image) return true;
    if (!d.dli_sname || d.dli_saddr != p) return false;   /* not an exported symbol's start */
    void* q = dlsym(self, d.dli_sname);   /* this file first, then what it depends on */
    if (!q || image_of(q) != self_image) return false;
    *hook = reinterpret_cast<F>(q);
    return true;
}

/* Fills `o` where it stands: `info` points into it. */
void build(Override& o) {
    const Base& b = base();
    if (!b.err.empty()) {
        o.err = b.err;
        return;
    }
    Mapped self(b.self_path.c_str());
    if (!self.h) {
        o.err = std::string("cannot reopen itself: ") + dlerror();
        return;
    }
    struct Reading {
        Reading() { t_reading = true; }
        ~Reading() { t_reading = false; }
    } reading;

    /* The rows are copied as this file's headers lay them out, so the base must be built against
     * the same ones. */
    if (b.abi() != RAD_ABI_VERSION || b.layout() != rad_abi_layout()) {
        o.err = "the library it overrides was built against another ABI than its units; rebuild "
                "one of them";
        return;
    }
    const RadPluginInfo* bi = b.info();
    if (!bi || bi->kind != RAD_PLUGIN_KERNEL) {
        o.err = "the library it overrides is not a kernel library";
        return;
    }
    if (stat(b.self_path.c_str(), &o.file) != 0) o.file = {};
    o.info = *bi;
    o.version = std::string(bi->version ? bi->version : "") + "+override";
    o.info.version = o.version.c_str();

    const int n = b.count();
    o.rows.reserve(n > 0 ? (size_t)n : 0);
    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* k = b.at(i);
        if (!k) {
            o.err = "the library it overrides has no row at an index below its own count";
            return;
        }
        RadKernelInfo r = *k;
        /* Every hook, not only the launch: a unit that defines a row's scratch or init hook
         * beside its kernel must serve the pair. */
        bool mine = false;
#define OV_TAKE(hook) mine |= take(&r.hook, self.h, b.self_image);
        OV_HOOKS(OV_TAKE)
#undef OV_TAKE
        if (mine) o.replaced.emplace_back(r.name ? r.name : "?");
        o.rows.push_back(r);
    }
}

Override& ov() {
    struct Built {
        Override o;
        Built() { build(o); }
    };
    static Built b;
    return b.o;
}

/* ------------------------------------------------------------------ a reload */
/* A rebuild of this file read by rad_kernel_reload's PREPARE and not yet committed: its rows, in
 * this file's order. Touched only with every engine using the rows quiet, so by one thread. */
struct Staged {
    bool active = false;
    struct stat file{};
    std::vector<const RadKernelInfo*> rows;
};
Staged g_staged;

/* Every rebuilt copy loaded, never closed: a committed copy's hooks are the rows', and the code of
 * one that was not may still be registered with the device by address (docs/PLUGIN.md §14). */
std::vector<void*> g_copies;

bool same_file(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec;
}

/* Everything of a row but its hooks: the declaration, which a rebuild of units does not change. */
bool same_declaration(const RadKernelInfo& a, const RadKernelInfo& b) {
    return a.name == b.name && a.op == b.op && a.family == b.family && a.computes == b.computes &&
           a.shape == b.shape && a.dtypes == b.dtypes && a.domain == b.domain &&
           a.priority == b.priority && a.constraints == b.constraints &&
           a.n_constraints == b.n_constraints && a.tunables == b.tunables &&
           a.n_tunables == b.n_tunables && a.capabilities == b.capabilities;
}

/* Whether a hook of `r` lies in the image `image`. */
bool hook_in(const RadKernelInfo& r, const void* image) {
    bool in = false;
#define OV_IN(h) in = in || (r.h && image_of(reinterpret_cast<const void*>(r.h)) == image);
    OV_HOOKS(OV_IN)
#undef OV_IN
    return in;
}

bool same_hooks(const RadKernelInfo& a, const RadKernelInfo& b) {
    bool same = true;
#define OV_SAME(h) same = same && a.h == b.h;
    OV_HOOKS(OV_SAME)
#undef OV_SAME
    return same;
}

int refuse(char* why, int why_bytes, int st, const char* fmt, ...)
    __attribute__((format(printf, 4, 5)));
int refuse(char* why, int why_bytes, int st, const char* fmt, ...) {
    if (why && why_bytes > 0) {
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(why, (size_t)why_bytes, fmt, ap);
        va_end(ap);
    }
    return st;
}

/* The file at `path` copied to a new file beside it, whose name ends in no ".so", so a scan of the
 * directory never loads it. The copy's path, or empty with `err` set. */
std::string copy_beside(const std::string& path, struct stat* st, std::string* err) {
    const int in = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (in < 0 || fstat(in, st) != 0) {
        *err = std::strerror(errno);
        if (in >= 0) close(in);
        return std::string();
    }
    const size_t slash = path.find_last_of('/');
    std::string tmp = slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
    tmp += "." + path.substr(slash == std::string::npos ? 0 : slash + 1) + ".reload-XXXXXX";
    std::vector<char> name(tmp.begin(), tmp.end());
    name.push_back('\0');
    const int out = mkstemp(name.data());
    if (out < 0) {
        *err = std::string("cannot create a file beside it: ") + std::strerror(errno);
        close(in);
        return std::string();
    }
    char buf[1 << 16];
    bool ok = true;
    for (;;) {
        const ssize_t n = read(in, buf, sizeof buf);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { ok = n == 0; break; }
        for (ssize_t off = 0; ok && off < n;) {
            const ssize_t w = write(out, buf + off, (size_t)(n - off));
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) ok = false;
            else off += w;
        }
        if (!ok) break;
    }
    if (!ok) *err = std::string("cannot copy it: ") + std::strerror(errno);
    close(in);
    if (close(out) != 0 && ok) {
        ok = false;
        *err = std::string("cannot copy it: ") + std::strerror(errno);
    }
    if (!ok) {
        unlink(name.data());
        return std::string();
    }
    return std::string(name.data());
}

template <class F> F find_in(void* h, const char* name) {
    return reinterpret_cast<F>(dlsym(h, name));
}

int prepare(char* why, int why_bytes) {
    const Base& b = base();
    Override& o = ov();
    if (!o.err.empty()) return refuse(why, why_bytes, RAD_E_STATE, "%s", o.err.c_str());
    g_staged = Staged{};

    struct stat st{};
    std::string err;
    const std::string copy = copy_beside(b.self_path, &st, &err);
    if (copy.empty())
        return refuse(why, why_bytes, RAD_E_IO, "cannot read %s: %s", b.self_path.c_str(),
                      err.c_str());
    if (same_file(st, o.file)) {
        unlink(copy.c_str());
        return refuse(why, why_bytes, RAD_E_STATE,
                      "%s has not changed since its hooks were loaded: there is nothing to reload",
                      b.self_path.c_str());
    }
    /* The copy's base is the base mapped already, by its path, so nothing of the base loads
     * again. Its name goes once its rows are read: what a launch runs is mapped by then. */
    void* h = dlopen(copy.c_str(), RTLD_NOW | RTLD_LOCAL);
    struct Unlink {
        const std::string& p;
        ~Unlink() { unlink(p.c_str()); }
    } unlink_copy{ copy };
    if (!h) {
        const char* e = dlerror();
        return refuse(why, why_bytes, RAD_E_IO, "the rebuilt %s does not load: %s",
                      b.self_path.c_str(), e ? e : "(no reason given)");
    }
    g_copies.push_back(h);

    auto abi    = find_in<uint32_t (*)(void)>(h, "rad_plugin_abi_version");
    auto layout = find_in<uint64_t (*)(void)>(h, "rad_plugin_abi_layout");
    auto over   = find_in<const char* (*)(void)>(h, "rad_override_base");
    auto failed = find_in<const char* (*)(void)>(h, "rad_override_error");
    auto count  = find_in<int (*)(void)>(h, "rad_kernel_count");
    auto at     = find_in<const RadKernelInfo* (*)(int)>(h, "rad_kernel_at");
    if (!abi || !layout || !over || !failed || !count || !at)
        return refuse(why, why_bytes, RAD_E_INVAL,
                      "the rebuilt %s is not an override built by rad_add_kernel_override",
                      b.self_path.c_str());
    if (abi() != RAD_ABI_VERSION || layout() != rad_abi_layout())
        return refuse(why, why_bytes, RAD_E_ABI,
                      "the rebuilt %s was built against another ABI than the one running",
                      b.self_path.c_str());
    if (std::strcmp(over(), RAD_OVERRIDE_BASE) != 0)
        return refuse(why, why_bytes, RAD_E_INVAL,
                      "the rebuilt %s overrides %s, and the one running overrides %s: a reload "
                      "keeps the library under it",
                      b.self_path.c_str(), over(), RAD_OVERRIDE_BASE);
    if (const char* e = failed(); e && *e)
        return refuse(why, why_bytes, RAD_E_INVAL, "the rebuilt %s: %s", b.self_path.c_str(), e);

    const int n = count();
    if (n != (int)o.rows.size())
        return refuse(why, why_bytes, RAD_E_INVAL,
                      "the rebuilt %s serves %d rows and the one running %zu", b.self_path.c_str(),
                      n, o.rows.size());
    const void* copy_image = image_of(reinterpret_cast<const void*>(count));
    Staged s;
    int changed = 0;
    bool own = false;
    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !same_declaration(*k, o.rows[(size_t)i]))
            return refuse(why, why_bytes, RAD_E_INVAL,
                          "row %d (%s) of the rebuilt %s is declared differently from the one "
                          "running: only hooks are reloaded",
                          i, o.rows[(size_t)i].name ? o.rows[(size_t)i].name : "?",
                          b.self_path.c_str());
        changed += !same_hooks(*k, o.rows[(size_t)i]);
        own = own || hook_in(*k, copy_image);
        s.rows.push_back(k);
    }
    /* As at open: a rebuild none of whose rows takes a hook from it is no override. */
    if (!own || changed == 0)
        return refuse(why, why_bytes, RAD_E_INVAL,
                      "the rebuilt %s defines no hook of any row of %s, so it would serve the "
                      "library's own hooks",
                      b.self_path.c_str(), RAD_OVERRIDE_BASE);
    s.active = true;
    s.file = st;
    g_staged = std::move(s);
    return changed;
}

void commit() {
    const Base& b = base();
    Override& o = ov();
    if (!g_staged.active) return;
    std::vector<std::string> names;
    for (size_t i = 0; i < o.rows.size(); ++i) {
        const RadKernelInfo& s = *g_staged.rows[i];
        RadKernelInfo& r = o.rows[i];
        if (same_hooks(s, r)) continue;
#define OV_PUT(h) r.h = s.h;
        OV_HOOKS(OV_PUT)
#undef OV_PUT
        names.emplace_back(r.name ? r.name : "?");
    }
    o.file = g_staged.file;
    g_staged = Staged{};
    std::string list;
    for (size_t i = 0; i < names.size() && i < 12; ++i) list += (i ? ", " : "") + names[i];
    if (names.size() > 12) list += ", ...";
    std::fprintf(stderr, "rad-override: %s reloaded: %zu of the %zu rows of %s take new hooks: %s\n",
                 b.self_path.c_str(), names.size(), o.rows.size(), RAD_OVERRIDE_BASE,
                 list.c_str());
}

}  /* namespace */

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }

const RadPluginInfo* rad_plugin_info(void) {
    const Base& b = base();
    if (t_reading) return b.info();
    Override& o = ov();
    if (!o.err.empty()) {
        std::fprintf(stderr, "rad-override: %s: %s\n", b.self_path.c_str(), o.err.c_str());
        return nullptr;
    }
    return &o.info;
}

int rad_plugin_open(void) {
    const Base& b = base();
    if (t_reading) return b.open ? b.open() : RAD_OK;
    Override& o = ov();
    if (!o.err.empty()) return RAD_E_INVAL;
    if (o.replaced.empty()) {
        std::fprintf(stderr,
                     "rad-override: %s defines no hook of any row of %s, so it would serve that "
                     "library unchanged -- refused. A row's hook is taken by its symbol name: the "
                     "units must define the functions the base's rows name.\n",
                     b.self_path.c_str(), RAD_OVERRIDE_BASE);
        return RAD_E_INVAL;
    }
    const int st = b.open ? b.open() : RAD_OK;
    if (st < 0) return st;
    std::string names;
    for (size_t i = 0; i < o.replaced.size() && i < 12; ++i)
        names += (i ? ", " : "") + o.replaced[i];
    if (o.replaced.size() > 12) names += ", ...";
    std::fprintf(stderr,
                 "rad-override: %s serves %zu of the %zu rows of %s with its own hooks: %s\n",
                 b.self_path.c_str(), o.replaced.size(), o.rows.size(), RAD_OVERRIDE_BASE,
                 names.c_str());
    return st;
}

/* From here on the answers are the base's own, asked of the base: they do not depend on the rows
 * built above, so they hold while those are being read as well. */
void rad_plugin_close(void) {
    const Base& b = base();
    if (b.close) b.close();
}

void rad_plugin_freed(const void* p, int64_t bytes) {
    const Base& b = base();
    if (b.freed) b.freed(p, bytes);
}

int rad_plugin_reference(void) {
    const Base& b = base();
    return b.reference ? b.reference() : 0;
}

int rad_kernel_schema_count(void) {
    const Base& b = base();
    return b.schema_count ? b.schema_count() : 0;
}

const RadOpSchema* rad_kernel_schema_at(int i) {
    const Base& b = base();
    return b.schema_at ? b.schema_at(i) : nullptr;
}

/* The per-row answers are the base's at the same index: the rows are its rows in its order. Where
 * the base exports none, the answer is what the loader assumes for a library without the export. */
int rad_kernel_concurrent(int i) {
    const Base& b = base();
    return b.concurrent ? b.concurrent(i) : 0;
}

int rad_kernel_collective(int i) {
    const Base& b = base();
    return b.collective ? b.collective(i) : 0;
}

int rad_kernel_chain(int i) {
    const Base& b = base();
    return b.chain ? b.chain(i) : 0;
}

int rad_kernel_reference(int i) {
    const Base& b = base();
    if (b.kernel_reference) return b.kernel_reference(i);
    return b.reference ? b.reference() : 0;
}

/* The rows: the base's with this file's hooks, except to the base while it is being read. */
int rad_kernel_count(void) {
    const Base& b = base();
    if (t_reading) return b.count ? b.count() : 0;
    Override& o = ov();
    return o.err.empty() ? (int)o.rows.size() : 0;
}

const RadKernelInfo* rad_kernel_at(int i) {
    const Base& b = base();
    if (t_reading) return b.at ? b.at(i) : nullptr;
    Override& o = ov();
    if (!o.err.empty() || i < 0 || (size_t)i >= o.rows.size()) return nullptr;
    return &o.rows[(size_t)i];
}

/* A rebuild of this file under a running engine (rad_abi.h, RadKernelReloadFn). */
int rad_kernel_reload(int step, char* why, int why_bytes) {
    if (why && why_bytes > 0) why[0] = '\0';
    switch (step) {
    case RAD_RELOAD_PREPARE: return prepare(why, why_bytes);
    case RAD_RELOAD_COMMIT:
        if (!g_staged.active)
            return refuse(why, why_bytes, RAD_E_STATE, "no reload was prepared");
        commit();
        return RAD_OK;
    case RAD_RELOAD_ABORT: g_staged = Staged{}; return RAD_OK;
    default: return refuse(why, why_bytes, RAD_E_INVAL, "no reload step %d", step);
    }
}

const RadKernelInfo* rad_kernel_reload_at(int i) {
    if (!g_staged.active || i < 0 || (size_t)i >= g_staged.rows.size()) return nullptr;
    return g_staged.rows[(size_t)i];
}

/* Not the ABI's: what a running override asks of a rebuild of itself at PREPARE. */
const char* rad_override_base(void) { return RAD_OVERRIDE_BASE; }

const char* rad_override_error(void) { return ov().err.c_str(); }

}  /* extern "C" */

RAD_PLUGIN_ABI_LAYOUT_EXPORT
