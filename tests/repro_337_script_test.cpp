/* repro_337_script_test.cpp -- the #337 harness's own checks, run as one test the build already
 * registers (issue #337).
 *
 * Two of the harness's checks are not C++: scripts/repro-337.sh's `--summarise` read of a run
 * directory a dead server left behind (tests/repro_337_script_test.sh), and the tool-call turn
 * scripts/agentreplay.py drives against a fake server (`--selftest`). Their checks stand where they
 * are -- shell and python are the right tools for both -- and this file is what puts them under
 * ctest without a line in tests/CMakeLists.txt: that file is a locked path (AGENTS.md), while a
 * *_test.cpp is registered by the glob, so adding this file adds a test.
 *
 * Neither check touches a card. The tool a check needs -- sh, python3 -- is a machine capability, so
 * its absence is SKIP. The scripts are checked-in files of this tree, so their absence is
 * ARTIFACT_MISSING: a broken build or a wrong cwd under RAD_TEST_STRICT, which ctest sets. */
#include "rad_test.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;

bool on_path(const char* tool) {
    const char* path = std::getenv("PATH");
    const std::string all = path ? path : "";
    size_t at = 0;
    while (at <= all.size()) {
        size_t c = all.find(':', at);
        if (c == std::string::npos) c = all.size();
        const std::string d = all.substr(at, c - at);
        if (!d.empty() && ::access((d + "/" + tool).c_str(), X_OK) == 0) return true;
        at = c + 1;
    }
    return false;
}

/* Walking up `from`, the first directory that holds scripts/repro-337.sh. */
bool tree_above(fs::path p, std::string* out) {
    std::error_code ec;
    for (int i = 0; i < 6 && !p.empty(); ++i, p = p.parent_path())
        if (fs::exists(p / "scripts" / "repro-337.sh", ec)) {
            *out = p.string();
            return true;
        }
    return false;
}

/* The source tree this binary was built from: a directory holding scripts/repro-337.sh, found by
 * walking up from where the binary lives (build/tests/...) and from where ctest runs it. */
bool source_tree(std::string* out) {
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec).parent_path();
    if (!ec && tree_above(exe, out)) return true;
    return tree_above(fs::current_path(ec), out);
}

/* Run `cmd`, and on failure say what it printed. */
void runs_clean(const std::string& what, const std::string& cmd) {
    FILE* p = ::popen((cmd + " 2>&1").c_str(), "r");
    REQUIRE(p != nullptr);
    std::string out;
    char buf[4096];
    while (size_t n = ::fread(buf, 1, sizeof buf, p)) out.append(buf, n);
    const int rc = ::pclose(p);
    if (rc != 0) fprintf(stderr, "%s exited %d:\n%s\n", what.c_str(), rc, out.c_str());
    CHECK_EQ(rc, 0);
}

}  /* namespace */

/* A run directory a dead server left behind: the fault is counted and printed, a clean run is not,
 * and half-written runs are summarised rather than raised on. */
TEST(the_repro_scripts_summarise_of_a_dead_server) {
    if (!on_path("sh")) SKIP("sh is not on PATH");
    std::string root;
    if (!source_tree(&root)) ARTIFACT_MISSING("scripts/repro-337.sh up from the binary or the cwd");
    NEED_ARTIFACT(root + "/tests/repro_337_script_test.sh");
    runs_clean("tests/repro_337_script_test.sh", "sh '" + root + "/tests/repro_337_script_test.sh'");
}

/* The tool-call turn #337's session is made of, against a fake server. */
TEST(agentreplay_selftest) {
    if (!on_path("python3")) SKIP("python3 is not on PATH");
    std::string root;
    if (!source_tree(&root)) ARTIFACT_MISSING("scripts/repro-337.sh up from the binary or the cwd");
    NEED_ARTIFACT(root + "/scripts/agentreplay.py");
    runs_clean("scripts/agentreplay.py --selftest",
               "python3 '" + root + "/scripts/agentreplay.py' --selftest");
}

RAD_TEST_MAIN()
