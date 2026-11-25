/* rl_rows.cpp -- the declaration of tests/override's library for a running engine
 * (kernel_reload_e2e_test.cpp): one host row for `sample_argmax`, whose launch and scratch come
 * from rl_unit.cpp, the unit its overrides rebuild, and whose init and fini are this file's.
 *
 * It declares no schema: the op's is libref's, which loads behind it. Ahead of libref in the
 * hierarchy, its row takes every greedy pick, so the tokens a request returns name the build of
 * rl_unit.cpp that ran.
 */
#include "rad_abi.h"
#include "rad_abi_layout.h"

extern "C" {

int     rl_argmax(const RadArgs*, RadStream);
int64_t rl_argmax_scratch(const RadArgs*);

/* An instance with nothing in it: what matters is that declare asked for one. */
int rl_argmax_init(const RadParam*, int, int, int, void** out) {
    *out = nullptr;
    return RAD_OK;
}
void rl_argmax_fini(void*) {}

}  /* extern "C" */

namespace {

const RadKernelInfo kRows[] = {
    { .name = "rl_argmax", .op = "sample_argmax", .family = "sample",
      .computes = "a fixed token per row, the build's", .shape = "any", .dtypes = "any",
      .domain = RAD_DOMAIN_HOST, .priority = 10, .init = rl_argmax_init, .fini = rl_argmax_fini,
      .launch = rl_argmax, .scratch = rl_argmax_scratch },
};

const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "radtest_rl", "1.0", "a greedy pick that names its build", "host",
};

}  /* namespace */

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
int rad_kernel_schema_count(void) { return 0; }
const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
int rad_kernel_count(void) { return 1; }
const RadKernelInfo* rad_kernel_at(int i) { return i == 0 ? &kRows[0] : nullptr; }

}  /* extern "C" */

RAD_PLUGIN_ABI_LAYOUT_EXPORT
