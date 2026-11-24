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
 */
#include "rad_abi.h"
#include "rad_abi_layout.h"

#include <dlfcn.h>

#include <cstdio>
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
struct Override {
    std::string   err;
    RadPluginInfo info{};
    std::string   version;
    std::vector<RadKernelInfo> rows;      /* the base's, hooks replaced; never resized once built */
    std::vector<std::string>   replaced;  /* the rows with a hook from this file */
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
        mine |= take(&r.tune_valid, self.h, b.self_image);
        mine |= take(&r.init, self.h, b.self_image);
        mine |= take(&r.fini, self.h, b.self_image);
        mine |= take(&r.launch, self.h, b.self_image);
        mine |= take(&r.scratch, self.h, b.self_image);
        mine |= take(&r.layout, self.h, b.self_image);
        mine |= take(&r.relayout, self.h, b.self_image);
        mine |= take(&r.opd_shape, self.h, b.self_image);
        mine |= take(&r.unrelayout, self.h, b.self_image);
        mine |= take(&r.replaces, self.h, b.self_image);
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

}  /* extern "C" */

RAD_PLUGIN_ABI_LAYOUT_EXPORT
