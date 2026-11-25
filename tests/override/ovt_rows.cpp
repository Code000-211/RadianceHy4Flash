/* ovt_rows.cpp -- the declaration of tests/override's kernel library: three rows over two units,
 * the way libr4d's row table stands beside its kernel units.
 *
 *   ovt_a   launch and scratch from unit A, the unit an override rebuilds
 *   ovt_b   launch from unit B
 *   ovt_c   a launch of this file's that calls into unit A
 *
 * The per-row exports answer differently for each row, so a library that re-exports the rows has
 * to keep them at their indices. Nothing here computes: the launches return a number that names
 * the build that ran, and the library lives outside every kernels/ the engine scans.
 */
#include "ovt.h"
#include "rad_abi_layout.h"

#include <cstring>

int ovt_launches = 0;

namespace {

int g_open = 0;
int g_close = 0;
/* volatile: the count is read where it is asked for, so rad_kernel_at's question below stays a
 * call. */
volatile int g_rows = 3;

const RadParamSpec   kParams[]   = { { "M", RAD_P_INT, RAD_REQUIRED, 0 } };
const RadOperandSpec kOperands[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };
const RadOpSchema    kSchemas[]  = {
    { "ovt_op", kParams, 1, kOperands, 2, "an op for the override test" },
};

const RadConstraint kSmall[] = { RAD_CLE("M", 16) };

/* A launch outside the units, reaching into one. */
int ovt_launch_c(const RadArgs*, RadStream) { return ovt_helper_a(); }

const RadKernelInfo kRows[] = {
    { .name = "ovt_a", .op = "ovt_op", .family = "test", .computes = "nothing", .shape = "M<=16",
      .dtypes = "any", .domain = RAD_DOMAIN_HOST, .priority = 70,
      .constraints = kSmall, .n_constraints = 1,
      .launch = ovt_launch_a, .scratch = ovt_scratch_a },
    { .name = "ovt_b", .op = "ovt_op", .family = "test", .computes = "nothing", .shape = "any",
      .dtypes = "any", .domain = RAD_DOMAIN_HOST, .priority = 50, .launch = ovt_launch_b },
    { .name = "ovt_c", .op = "ovt_op", .family = "test", .computes = "nothing", .shape = "any",
      .dtypes = "any", .domain = RAD_DOMAIN_HOST, .priority = 10, .launch = ovt_launch_c },
};

const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, OVT_NAME, "1.5", "three rows over two units", "host",
};

}  /* namespace */

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

int rad_plugin_open(void) { ++g_open; return RAD_OK; }
void rad_plugin_close(void) { ++g_close; }
int ovt_open_calls(void) { return g_open; }
int ovt_launch_calls(void) { return ovt_launches; }

int rad_kernel_schema_count(void) { return 1; }
const RadOpSchema* rad_kernel_schema_at(int i) { return i == 0 ? &kSchemas[0] : nullptr; }
/* The exports call one another, as libr4d's do (its rad_kernel_at asks rad_kernel_count, its
 * per-row answers ask rad_kernel_at). Out of line, so the calls are real ones the dynamic linker
 * may bind to an override's exports of the same names: an override that reads the rows through
 * these has to stand being called back while it does. */
__attribute__((noinline)) int rad_kernel_count(void) { return g_rows; }
__attribute__((noinline)) const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < rad_kernel_count() ? &kRows[i] : nullptr;
}

static bool row_is(int i, const char* name) {
    const RadKernelInfo* k = rad_kernel_at(i);
    return k && std::strcmp(k->name, name) == 0;
}
int rad_kernel_concurrent(int i) { return row_is(i, "ovt_a"); }
int rad_kernel_chain(int i) { return row_is(i, "ovt_b") ? 2 : 0; }
int rad_kernel_collective(int i) { return row_is(i, "ovt_c"); }

}  /* extern "C" */

RAD_PLUGIN_ABI_LAYOUT_EXPORT
