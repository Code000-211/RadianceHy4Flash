/* kernel_override_test.cpp -- a kernel library with one of its units rebuilt
 * (rad_add_kernel_override, cmake/rad_kernel_override.cpp; docs/PLUGIN.md §14).
 *
 * tests/override/ builds a library of three rows over two units, an override of it made of the
 * first unit rebuilt, and an override made of a unit no row names. The rebuilt unit answers 2xx
 * where the library's own answers 1xx, so a hook's answer names the build that ran. Everything
 * goes through the loader, as the engine reaches it: the override shadows the library by carrying
 * its file name in a home ahead of the library's.
 *
 * Each case runs for the library linked two ways (tests/override/CMakeLists.txt): `plain`, whose
 * references to its units the dynamic linker may bind to the override, and `sym`, where only the
 * override's own replacement of the rows' hooks moves them.
 *
 * A passing run prints the override's `rad-override:` lines and the loader's refusal of the one
 * that defines no hook: they are the diagnostics under test.
 */
#include "rad_test.h"

#include "plugin/registry.h"

#include <dlfcn.h>

#include <string>
#include <vector>

using namespace rad;

namespace {

const char* const kKinds[] = { "plain", "sym" };

std::string kernels(const std::string& kind, const char* suffix = "") {
    return home() + "/testoverride/" + kind + suffix + "/kernels";
}
std::string lib_name(const std::string& kind) { return "radtest_ov" + kind; }

std::string errs(const Registry& r) {
    std::string s;
    for (const auto& e : r.load_errors()) { s += e; s += '\n'; }
    return s;
}

/* The row's hooks, called as nothing but this test may call them. */
int launch(const KernelRow& row) { return row.info->launch(nullptr, nullptr); }

/* A function of the library's that is not a row hook, through a plugin's handle. */
int call(const Plugin* p, const char* name) {
    auto f = reinterpret_cast<int (*)()>(dlsym(p->handle, name));
    return f ? f() : -1;
}

Geometry geom(long long M) {
    Geometry g;
    g.set_i("M", M);
    return g;
}

/* The three rows as the library declares them, whichever file serves them. */
void check_rows_are_the_librarys(const Registry& r, const std::string& name) {
    REQUIRE_EQ(r.rows().size(), (size_t)3);
    const char* const names[] = { "ovt_a", "ovt_b", "ovt_c" };
    const int prio[] = { 70, 50, 10 };
    for (int i = 0; i < 3; ++i) {
        const KernelRow& row = r.rows()[(size_t)i];
        CHECK_EQ(std::string(row.info->name), std::string(names[i]));
        CHECK_EQ(std::string(row.info->op), std::string("ovt_op"));
        CHECK_EQ(row.info->priority, prio[i]);
        CHECK_EQ(row.info->domain, (int)RAD_DOMAIN_HOST);
        CHECK_EQ(row.info->n_constraints, i == 0 ? 1 : 0);
        CHECK_EQ(row.plugin, name);
        CHECK_EQ(row.index, i);
        /* The per-row exports, each true for one row only. */
        CHECK_EQ(row.concurrent, i == 0);
        CHECK_EQ(row.chain, i == 1);
        CHECK_EQ(row.chain_fenced, i == 1);
        CHECK_EQ(row.collective, i == 2);
        CHECK(!row.reference);
    }
    CHECK(r.schema("ovt_op") != nullptr);
}

}  /* namespace */

/* The override is the library to the loader -- its name, its rows in its order with its
 * constraints, priorities and per-row flags, so selection is the library's -- and the rebuilt
 * unit's hooks are the ones that run. */
TEST(the_override_serves_the_librarys_rows_with_the_rebuilt_units_hooks) {
    for (const std::string kind : kKinds) {
        const std::string name = lib_name(kind);
        Registry r;
        /* No hierarchy and no name: the override's home comes first, and its file shadows the
         * library's. */
        const int st = r.load_dirs({ kernels(kind, "-over"), kernels(kind) }, {});
        if (st == RAD_E_IO) SKIP("no override test plugins under %s", home().c_str());
        REQUIRE_EQ(st, (int)RAD_OK);
        REQUIRE_EQ(r.plugins().size(), (size_t)1);
        const Plugin* p = r.plugin(name);
        REQUIRE(p != nullptr);
        CHECK_EQ(p->path, kernels(kind, "-over") + "/" + name + ".so");
        CHECK_EQ(p->version, std::string("1.5+override"));
        CHECK_EQ(p->build_target, std::string("host"));
        check_rows_are_the_librarys(r, name);

        /* Unit A is the rebuilt one: both of its row's hooks. Unit B's is the library's. */
        CHECK_EQ(launch(r.rows()[0]), 201);
        CHECK_EQ((int)r.rows()[0].info->scratch(nullptr), 202);
        CHECK_EQ(launch(r.rows()[1]), 103);
        /* The rebuilt launch counts in the library's state, not in a copy of its own: what a
         * unit takes from the rest of the library resolves in the library. */
        CHECK_EQ(call(p, "ovt_launch_calls"), 1);
        /* The library's own open hook ran, once. */
        CHECK_EQ(call(p, "ovt_open_calls"), 1);

        /* A launch of the library's that calls into unit A. The override brought the library in,
         * so a preemptible reference is bound to the rebuilt unit; a library that binds its
         * references itself keeps its own copy. */
        CHECK_EQ(launch(r.rows()[2]), kind == "plain" ? 207 : 107);

        /* Selection is the library's: the constrained row at its priority, and the next row
         * where its constraint fails. */
        const KernelRow* small = r.select("ovt_op", geom(8), RAD_DOMAIN_HOST);
        REQUIRE(small != nullptr);
        CHECK_EQ(std::string(small->info->name), std::string("ovt_a"));
        CHECK_EQ(launch(*small), 201);
        const KernelRow* big = r.select("ovt_op", geom(64), RAD_DOMAIN_HOST);
        REQUIRE(big != nullptr);
        CHECK_EQ(std::string(big->info->name), std::string("ovt_b"));
    }
}

/* Closing the override leaves nothing of it behind: the library loads again on its own, every
 * hook its own, and so does the override after that. */
TEST(the_library_alone_is_unchanged_after_an_override_was_loaded) {
    for (const std::string kind : kKinds) {
        const std::string name = lib_name(kind);
        {
            Registry over;
            const int st = over.load_dirs({ kernels(kind, "-over"), kernels(kind) }, {});
            if (st == RAD_E_IO) SKIP("no override test plugins under %s", home().c_str());
            REQUIRE_EQ(st, (int)RAD_OK);
            CHECK_EQ(launch(over.rows()[0]), 201);
        }
        {
            Registry base;
            REQUIRE_EQ(base.load_dir(kernels(kind), {}), (int)RAD_OK);
            const Plugin* p = base.plugin(name);
            REQUIRE(p != nullptr);
            CHECK_EQ(p->path, kernels(kind) + "/" + name + ".so");
            CHECK_EQ(p->version, std::string("1.5"));
            check_rows_are_the_librarys(base, name);
            CHECK_EQ(launch(base.rows()[0]), 101);
            CHECK_EQ((int)base.rows()[0].info->scratch(nullptr), 102);
            CHECK_EQ(launch(base.rows()[1]), 103);
            CHECK_EQ(launch(base.rows()[2]), 107);
            /* A fresh image: the count is this load's own. */
            CHECK_EQ(call(p, "ovt_launch_calls"), 1);
        }
        {
            Registry again;
            REQUIRE_EQ(again.load_dirs({ kernels(kind, "-over"), kernels(kind) }, {}), (int)RAD_OK);
            CHECK_EQ(launch(again.rows()[0]), 201);
        }
    }
}

/* A process that already holds the library (a tool comparing the two builds) still gets the
 * rebuilt hooks from the override, by the override's own replacement: the library's references
 * were bound when it loaded and the dynamic linker moves none of them. Its calls into the rebuilt
 * unit from other units then stay its own, linked either way. */
TEST(a_library_loaded_first_gets_the_rebuilt_hooks_and_keeps_its_own_cross_unit_calls) {
    for (const std::string kind : kKinds) {
        const std::string name = lib_name(kind);
        Registry base;
        const int st = base.load_dir(kernels(kind), {});
        if (st == RAD_E_IO) SKIP("no override test plugins under %s", home().c_str());
        REQUIRE_EQ(st, (int)RAD_OK);

        Registry over;
        REQUIRE_EQ(over.load_dirs({ kernels(kind, "-over"), kernels(kind) }, {}), (int)RAD_OK);
        check_rows_are_the_librarys(over, name);
        CHECK_EQ(launch(over.rows()[0]), 201);
        CHECK_EQ((int)over.rows()[0].info->scratch(nullptr), 202);
        CHECK_EQ(launch(over.rows()[1]), 103);
        CHECK_EQ(launch(over.rows()[2]), 107);

        /* The library's own registry is untouched: one image, two tables of rows. */
        CHECK_EQ(launch(base.rows()[0]), 101);
        CHECK_EQ((int)base.rows()[0].info->scratch(nullptr), 102);
        CHECK_EQ(launch(base.rows()[2]), 107);
        /* Both launches of row 0 counted in the one library. */
        CHECK_EQ(call(base.plugin(name), "ovt_launch_calls"), 2);
    }
}

/* An override whose units define no hook of any row would serve the library unchanged under
 * another build's name. It is refused at open, and the library it shadows is not loaded in its
 * place. */
TEST(an_override_that_defines_no_hook_of_any_row_is_refused) {
    for (const std::string kind : kKinds) {
        Registry r;
        const int st = r.load_dirs({ kernels(kind, "-none"), kernels(kind) }, {});
        if (st == RAD_E_IO) SKIP("no override test plugins under %s", home().c_str());
        CHECK_EQ(st, (int)RAD_E_INVAL);
        CHECK(errs(r).find("refused to open") != std::string::npos);
        CHECK(r.plugins().empty());
        CHECK(r.rows().empty());
    }
}

RAD_TEST_MAIN()
