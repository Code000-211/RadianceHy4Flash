/* kernel_reload_test.cpp -- an override rebuilt under a loaded set, its hooks taken without a
 * reload of the set (Registry::reload_prepare / reload_finish, rad_kernel_reload in
 * cmake/rad_kernel_override.cpp; docs/PLUGIN.md §14).
 *
 * tests/override's library and its overrides, as kernel_override_test.cpp uses them: unit A
 * rebuilt answers 2xx in `<kind>-over` and 3xx in `<kind>-over3`. Each case loads an override from
 * a home of its own and then puts another build at the same path, the way a linker replaces its
 * output, so the file the set loaded is the one that changed.
 *
 * Every rebuilt copy a reload loads stays mapped for the life of the process, and with it the
 * library under it, so the cases share one image of each library: they compare counts before and
 * after, never against a fresh load, and live in a binary of their own.
 *
 * A passing run prints the override's `rad-override:` lines.
 */
#include "rad_test.h"

#include "plugin/registry.h"

#include <dirent.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>

using namespace rad;

namespace {

const char* const kKinds[] = { "plain", "sym" };

std::string kernels(const std::string& kind, const char* suffix = "") {
    return home() + "/testoverride/" + kind + suffix + "/kernels";
}
std::string lib_name(const std::string& kind) { return "radtest_ov" + kind; }

int launch(const KernelRow& row) { return row.info->launch(nullptr, nullptr); }
int launch(const RadKernelInfo* k) { return k->launch(nullptr, nullptr); }

int call(const Plugin* p, const char* name) {
    auto f = reinterpret_cast<int (*)()>(dlsym(p->handle, name));
    return f ? f() : -1;
}

bool exists(const std::string& path) { return access(path.c_str(), F_OK) == 0; }

/* A home of the override's own in a fresh directory, the shadowing file put there as a build
 * would leave it: written beside, then renamed over. */
struct Home {
    std::string dir, kernels_dir, kind;
    explicit Home(const std::string& k) : kind(k) {
        std::string t = radtest::tmpdir() + "/radiance_reload.XXXXXX";
        if (mkdtemp(t.data())) dir = t;
        kernels_dir = dir + "/kernels";
        if (!dir.empty()) mkdir(kernels_dir.c_str(), 0755);
    }
    ~Home() {
        if (dir.empty()) return;
        for (const std::string& f : files()) unlink((kernels_dir + "/" + f).c_str());
        rmdir(kernels_dir.c_str());
        rmdir(dir.c_str());
    }
    std::string path() const { return kernels_dir + "/" + lib_name(kind) + ".so"; }
    /* The file of build `suffix` ("-over", "-over3", "-none", or another kind's) put in place. */
    bool place_build(const std::string& from_kind, const char* suffix) const {
        return place_bytes(read(kernels(from_kind, suffix) + "/" + lib_name(from_kind) + ".so"));
    }
    bool place_bytes(const std::string& bytes) const {
        if (bytes.empty()) return false;
        const std::string tmp = kernels_dir + "/.writing";
        {
            std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
            o.write(bytes.data(), (std::streamsize)bytes.size());
            if (!o) return false;
        }
        return rename(tmp.c_str(), path().c_str()) == 0;
    }
    static std::string read(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    std::vector<std::string> files() const {
        std::vector<std::string> out;
        if (DIR* d = opendir(kernels_dir.c_str())) {
            while (dirent* e = readdir(d))
                if (std::string(e->d_name) != "." && std::string(e->d_name) != "..")
                    out.emplace_back(e->d_name);
            closedir(d);
        }
        return out;
    }
};

/* The override from `h` ahead of the library, as the engine's search path loads it. Skips when
 * the test plugins were not built. */
#define LOAD_OVER(r, h, kind)                                                                     \
    do {                                                                                          \
        if (!exists(kernels(kind, "-over3")))                                                     \
            SKIP("no override test plugins under %s", home().c_str());                            \
        REQUIRE(!(h).dir.empty());                                                                \
        REQUIRE((h).place_build(kind, "-over"));                                                  \
        REQUIRE_EQ((r).load_dirs({ (h).kernels_dir, kernels(kind) }, {}), (int)RAD_OK);           \
        REQUIRE_EQ((r).rows().size(), (size_t)3);                                                 \
    } while (0)

}  /* namespace */

/* The rebuilt unit's hooks go into the rows the set already serves, in place: the same row
 * objects, the same declaration, the new launch and scratch. Prepared, the rebuild is only held
 * beside them. The library's other units keep their calls into the build loaded first. */
TEST(a_rebuilt_override_reloads_its_hooks_into_the_rows_served) {
    for (const std::string kind : kKinds) {
        Home h(kind);
        Registry r;
        LOAD_OVER(r, h, kind);
        const Plugin* p = r.plugin(lib_name(kind));
        REQUIRE(p != nullptr);
        const KernelRow* a = &r.rows()[0];
        const RadKernelInfo* served = a->info;
        CHECK_EQ(launch(*a), 201);
        const int calls = call(p, "ovt_launch_calls");

        REQUIRE(h.place_build(kind, "-over3"));
        HookReload rl;
        std::string why;
        REQUIRE_EQ(r.reload_prepare(&rl, &why), (int)RAD_OK);
        REQUIRE_EQ(rl.libs.size(), (size_t)1);
        CHECK_EQ(rl.libs[0].name, lib_name(kind));
        CHECK_EQ(rl.libs[0].path, h.path());
        CHECK_EQ(rl.libs[0].rows, 1);
        const RadKernelInfo* next = rl.rebuilt(served);
        REQUIRE(next != nullptr);
        CHECK(rl.rebuilt(r.rows()[1].info) == nullptr);
        CHECK(rl.rebuilt(r.rows()[2].info) == nullptr);
        CHECK(hook_reload_fixed(*served, *next) == nullptr);
        CHECK_EQ(launch(next), 301);
        CHECK_EQ(std::string(next->name), std::string("ovt_a"));
        CHECK_EQ(launch(*a), 201);   /* nothing served changes before the commit */
        /* The copy it was read from is gone from the home: a later scan finds only the file. */
        CHECK_EQ(h.files().size(), (size_t)1);

        Registry::reload_finish(rl, true);
        CHECK(rl.libs.empty());
        CHECK(&r.rows()[0] == a);
        CHECK(a->info == served);
        CHECK_EQ(launch(*a), 301);
        CHECK_EQ((int)a->info->scratch(nullptr), 302);
        CHECK_EQ(launch(r.rows()[1]), 103);
        CHECK_EQ(launch(r.rows()[2]), kind == "plain" ? 207 : 107);
        /* The rebuilt unit counts in the library's state, as the first build did: 201, 301 and
         * 301 above. */
        CHECK_EQ(call(p, "ovt_launch_calls"), calls + 3);
        Geometry g;
        g.set_i("M", 8);
        const KernelRow* sel = r.select("ovt_op", g, RAD_DOMAIN_HOST);
        REQUIRE(sel != nullptr);
        CHECK(sel == a);
        CHECK_EQ(launch(*sel), 301);

        /* And back: the build of before, put in place again, is a new file. */
        REQUIRE(h.place_build(kind, "-over"));
        REQUIRE_EQ(r.reload_prepare(&rl, &why), (int)RAD_OK);
        Registry::reload_finish(rl, true);
        CHECK_EQ(launch(*a), 201);
        CHECK_EQ((int)a->info->scratch(nullptr), 202);
    }
}

/* A dropped reload changes nothing served, and the same rebuild can be taken afterwards. */
TEST(an_aborted_reload_leaves_the_rows_as_they_were) {
    for (const std::string kind : kKinds) {
        Home h(kind);
        Registry r;
        LOAD_OVER(r, h, kind);
        REQUIRE(h.place_build(kind, "-over3"));
        HookReload rl;
        std::string why;
        REQUIRE_EQ(r.reload_prepare(&rl, &why), (int)RAD_OK);
        Registry::reload_finish(rl, false);
        CHECK_EQ(launch(r.rows()[0]), 201);
        CHECK_EQ((int)r.rows()[0].info->scratch(nullptr), 202);

        REQUIRE_EQ(r.reload_prepare(&rl, &why), (int)RAD_OK);
        Registry::reload_finish(rl, true);
        CHECK_EQ(launch(r.rows()[0]), 301);
    }
}

/* What a reload refuses, each with its reason, leaving the rows as they were. */
TEST(a_reload_that_cannot_be_done_is_refused_with_its_reason) {
    for (const std::string kind : kKinds) {
        Home h(kind);
        Registry r;
        LOAD_OVER(r, h, kind);
        const std::string other = kind == "plain" ? "sym" : "plain";
        HookReload rl;
        std::string why;
        auto refused = [&](const char* reason, int served = 201) {
            why.clear();
            const int st = r.reload_prepare(&rl, &why);
            CHECK(st < 0);
            if (why.find(reason) == std::string::npos)
                fprintf(stderr, "    expected '%s' in: %s\n", reason, why.c_str());
            CHECK(why.find(reason) != std::string::npos);
            CHECK(why.rfind(lib_name(kind) + ": ", 0) == 0);
            CHECK(rl.libs.empty());
            CHECK(rl.next.empty());
            CHECK_EQ(launch(r.rows()[0]), served);
            CHECK_EQ(h.files().size(), (size_t)1);
        };

        /* The file the rows came from, unchanged. */
        refused("has not changed since its hooks were loaded");
        /* A build of units that define no hook of any row. */
        REQUIRE(h.place_build(kind, "-none"));
        refused("defines no hook of any row");
        /* An override of another library. */
        REQUIRE(h.place_bytes(Home::read(kernels(other, "-over") + "/" + lib_name(other) + ".so")));
        refused("a reload keeps the library under it");
        /* Not a loadable file at all, as a build cut short leaves it. */
        REQUIRE(h.place_bytes("not a shared object\n"));
        refused("does not load");

        /* None of that moved anything: the next good build is taken. */
        REQUIRE(h.place_build(kind, "-over3"));
        REQUIRE_EQ(r.reload_prepare(&rl, &why), (int)RAD_OK);
        Registry::reload_finish(rl, true);
        CHECK_EQ(launch(r.rows()[0]), 301);
        /* Taken once, the same file is unchanged again. */
        refused("has not changed since its hooks were loaded", 301);
    }
}

/* A library that is not an override cannot be reloaded, and the refusal says so. */
TEST(a_set_without_an_override_has_nothing_to_reload) {
    for (const std::string kind : kKinds) {
        if (!exists(kernels(kind))) SKIP("no override test plugins under %s", home().c_str());
        Registry r;
        REQUIRE_EQ(r.load_dir(kernels(kind), {}), (int)RAD_OK);
        HookReload rl;
        std::string why;
        CHECK_EQ(r.reload_prepare(&rl, &why), (int)RAD_E_NOTFOUND);
        CHECK(why.find("none of the loaded kernel libraries (" + lib_name(kind) + ")") !=
              std::string::npos);
        CHECK(rl.libs.empty());

        Registry none;
        CHECK_EQ(none.reload_prepare(&rl, &why), (int)RAD_E_NOTFOUND);
        CHECK_EQ(why, std::string("no kernel library is loaded"));
    }
}

/* The hooks a reload may not change under an engine that resolved the row, named. */
TEST(the_hooks_declare_answered_are_named) {
    RadKernelInfo a{};
    RadKernelInfo b{};
    a.launch = [](const RadArgs*, RadStream) { return 1; };
    b.launch = [](const RadArgs*, RadStream) { return 2; };
    b.scratch = [](const RadArgs*) -> int64_t { return 8; };
    b.tune_valid = a.tune_valid;
    CHECK(hook_reload_fixed(a, b) == nullptr);
    b.init = [](const RadParam*, int, int, int, void** out) { *out = nullptr; return 0; };
    CHECK_EQ(std::string(hook_reload_fixed(a, b)), std::string("init"));
    b.init = nullptr;
    b.opd_shape = [](const RadParam*, int, int, RadOpdDesc*) { return 0; };
    CHECK_EQ(std::string(hook_reload_fixed(a, b)), std::string("opd_shape"));
}

RAD_TEST_MAIN()
