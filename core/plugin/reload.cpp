/* reload.cpp -- an override's rebuilt hooks taken by the loaded set without a restart
 * (rad_kernel_reload, rad_abi.h; docs/PLUGIN.md §14).
 *
 * The rows the registry and every engine point to are the override's own table, and the override
 * puts the rebuilt hooks into it in place, so nothing here moves a row. What is checked here is
 * what the set alone can tell: that a library can be reloaded at all, and that its rebuild carries
 * device code for the card. What a running engine asked of a row at declare is the engine's to
 * check (Ctx::hooks_check) before reload_finish commits.
 */
#include "registry.h"
#include "../device/device.h"

#include <dlfcn.h>

namespace rad {

namespace {

bool same_hooks(const RadKernelInfo& a, const RadKernelInfo& b) {
    return a.tune_valid == b.tune_valid && a.init == b.init && a.fini == b.fini &&
           a.launch == b.launch && a.scratch == b.scratch && a.layout == b.layout &&
           a.relayout == b.relayout && a.opd_shape == b.opd_shape &&
           a.unrelayout == b.unrelayout && a.replaces == b.replaces;
}

}  /* namespace */

const char* hook_reload_fixed(const RadKernelInfo& served, const RadKernelInfo& rebuilt) {
    /* launch is what a reload is for; scratch is asked again by the engine; tune_valid is asked
     * only while selecting. */
#define HOOK(h) if (served.h != rebuilt.h) return #h;
    HOOK(init) HOOK(fini) HOOK(layout) HOOK(relayout) HOOK(opd_shape) HOOK(unrelayout)
    HOOK(replaces)
#undef HOOK
    return nullptr;
}

int Registry::reload_prepare(HookReload* out, std::string* why) {
    *out = HookReload{};
    std::string names;
    for (const Plugin& p : set_->plugins) {
        if (p.kind != RAD_PLUGIN_KERNEL || !p.handle) continue;
        names += (names.empty() ? "" : ", ") + p.name;
        dlerror();
        auto step = reinterpret_cast<RadKernelReloadFn>(dlsym(p.handle, "rad_kernel_reload"));
        auto at = reinterpret_cast<RadKernelReloadAtFn>(dlsym(p.handle, "rad_kernel_reload_at"));
        if (!step || !at) continue;

        char msg[1024] = "";
        const size_t code_from = dev_code_mark();
        const int st = step(RAD_RELOAD_PREPARE, msg, (int)sizeof msg);
        const size_t code_to = dev_code_mark();
        HookReload::Lib lib{ p.name, p.path, step, st };
        if (st < 0) {
            reload_finish(*out, false);
            *why = p.name + ": " + (msg[0] ? msg : rad_strerror(st));
            return st;
        }
        out->libs.push_back(lib);

        /* As at load (Registry::commit_): a rebuild whose fat binaries hold no code object for
         * the card would leave its device rows unlaunchable. */
        bool device = false;
        for (const KernelRow& r : set_->rows)
            device |= r.plugin == p.name && r.info->domain == RAD_DOMAIN_DEVICE;
        std::string no_code;
        if (device && !set_->no_device_code && code_to > code_from &&
            dev_code_runs(code_from, code_to, set_->code_device, &no_code) == 0) {
            reload_finish(*out, false);
            *why = p.name + ": the rebuilt " + p.path + " has no device code for device " +
                   std::to_string(set_->code_device) + ": " + no_code;
            return RAD_E_UNSUPPORTED;
        }
        for (const KernelRow& r : set_->rows) {
            if (r.plugin != p.name) continue;
            const RadKernelInfo* n = at(r.index);
            if (n && !same_hooks(*n, *r.info)) out->next[r.info] = n;
        }
    }
    if (out->libs.empty()) {
        *why = names.empty()
                   ? std::string("no kernel library is loaded")
                   : "none of the loaded kernel libraries (" + names + ") is an override built by "
                     "rad_add_kernel_override, and only an override's hooks can be reloaded";
        return RAD_E_NOTFOUND;
    }
    return RAD_OK;
}

void Registry::reload_finish(HookReload& r, bool commit) {
    for (HookReload::Lib& l : r.libs) {
        char msg[256] = "";
        l.step(commit ? RAD_RELOAD_COMMIT : RAD_RELOAD_ABORT, msg, (int)sizeof msg);
    }
    r.libs.clear();
    r.next.clear();
}

}  /* namespace rad */
