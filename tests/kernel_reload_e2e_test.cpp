/* kernel_reload_e2e_test.cpp -- a rebuilt override taken by a running server, with no model reload
 * (POST /admin/kernels/reload, core/fleet/fleet.cpp; docs/PLUGIN.md §14), on the host backend.
 *
 * The radiance binary serves the tiny qwen35 (tests/tiny_model.h) on libref, with tests/override's
 * argmax library ahead of it: its one row takes every greedy pick and gives each row the token its
 * build of rl_unit.cpp was compiled with (tests/override/CMakeLists.txt). The library is served
 * through an override in a home of the test's own, and a rebuild is another build of the override
 * renamed over that file, as a linker leaves its output. A reply's tokens then say which build ran.
 *
 * RAD_E2E_SERVER and RAD_E2E_HOME come from tests/CMakeLists.txt.
 */
#include "decision_e2e.h"

#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace decision_e2e;

namespace {

std::string build_of(const char* tag) {
    return std::string(RAD_E2E_HOME) + "/testoverride/rl-" + tag + "/kernels/radtest_rl.so";
}

bool exists(const std::string& path) { return access(path.c_str(), F_OK) == 0; }

/* Build `tag` of the override put at `path`: written beside it, then renamed over it. */
bool place(const char* tag, const std::string& path) {
    std::ifstream in(build_of(tag), std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty()) return false;
    const std::string tmp = path + ".writing";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), (std::streamsize)bytes.size());
        if (!out) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

json greedy(int n) {
    return json{ { "prompt", std::vector<int32_t>{ 10, 20, 30, 40, 50 } }, { "n_predict", n },
                 { "temperature", 0.0 }, { "ignore_eos", true }, { "return_tokens", true } };
}

std::vector<int32_t> tokens(int port, int n) {
    int status = 0;
    const json r = call(port, "POST", "/completion", greedy(n), &status);
    if (status != 200 || !r.is_object() || !r.contains("tokens")) {
        fprintf(stderr, "    /completion answered %d: %s\n", status, r.dump().c_str());
        return {};
    }
    return r["tokens"].get<std::vector<int32_t>>();
}

std::string show(const std::vector<int32_t>& t) {
    std::string s;
    for (int32_t v : t) s += std::to_string(v) + " ";
    return s;
}

bool all(const std::vector<int32_t>& t, int32_t v, size_t n) {
    if (t.size() != n) return false;
    for (int32_t x : t) if (x != v) return false;
    return true;
}

/* POST /admin/kernels/reload: its status, and the body as text. */
int reload(int port, std::string* body) {
    Call c;
    *body = std::string();
    if (!c.send(port, "POST", "/admin/kernels/reload", "")) return -1;
    return c.read(body);
}

size_t count(const std::string& hay, const std::string& needle) {
    size_t n = 0;
    for (size_t at = 0; (at = hay.find(needle, at)) != std::string::npos; at += needle.size()) ++n;
    return n;
}

}  /* namespace */

/* The rebuild runs from the next step on, under the model loaded once; a reload that would change
 * what declare sized or initialised is refused by name and the server goes on with the build it
 * had. */
TEST(a_running_server_takes_a_rebuilt_override_without_reloading_its_model) {
    if (!exists(build_of("a")) || !exists(build_of("init")))
        SKIP("no reload test plugins under %s", RAD_E2E_HOME);
    tiny::TempDir dir("kreload");
    REQUIRE(dir.ok());
    const std::string model = dir / "qwen35";
    const std::string home = dir / "home";
    REQUIRE(mkdir(model.c_str(), 0755) == 0);
    REQUIRE(mkdir(home.c_str(), 0755) == 0);
    REQUIRE(mkdir((home + "/kernels").c_str(), 0755) == 0);
    REQUIRE(tiny::write_qwen35(model, {}));
    const std::string file = home + "/kernels/radtest_rl.so";
    REQUIRE(place("a", file));

    /* The override's home first: it shadows the library's, which comes next for the override to
     * load as its base. */
    const std::vector<std::string> env = {
        "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=1024", "OMP_WAIT_POLICY=passive",
        "OMP_NUM_THREADS=2",
        "RADIANCE_HOME=" + home + ":" + RAD_E2E_HOME + "/testoverride/rl:" + RAD_E2E_HOME };
    const std::vector<std::string> argv = radtest::config_argv(
        { RAD_E2E_SERVER, "--model", model, "--kernels", "radtest_rl,libref",
          "--debug-accept-reference-kernels", "--host", "127.0.0.1", "--port", "0" });
    Server s;
    REQUIRE(s.start_child(argv, env));

    std::vector<int32_t> t = tokens(s.port, 6);
    if (!all(t, 5, 6)) fprintf(stderr, "    before the reload: %s\n", show(t).c_str());
    CHECK(all(t, 5, 6));

    /* A request in flight while the reload waits for a step boundary: each step runs one build,
     * so its tokens are the first build's and then the second's, never mixed back. */
    std::vector<int32_t> during;
    std::thread long_one([&] { during = tokens(s.port, 96); });

    REQUIRE(place("b", file));
    std::string body;
    int st = reload(s.port, &body);
    if (st != 200) fprintf(stderr, "    reload answered %d: %s\n", st, body.c_str());
    REQUIRE_EQ(st, 200);
    const json j = json::parse(body, nullptr, false);
    REQUIRE(j.is_object() && j.contains("reloaded"));
    REQUIRE_EQ(j["reloaded"].size(), (size_t)1);
    CHECK_EQ(j["reloaded"][0]["library"].get<std::string>(), std::string("radtest_rl"));
    CHECK_EQ(j["reloaded"][0]["file"].get<std::string>(), file);
    CHECK_EQ(j["reloaded"][0]["rows"].get<int>(), 1);
    CHECK_EQ(j["engines"].get<int>(), 1);

    long_one.join();
    CHECK_EQ(during.size(), (size_t)96);
    bool seen_new = false, ordered = true, known = true;
    for (int32_t x : during) {
        known = known && (x == 5 || x == 9);
        if (x == 9) seen_new = true;
        else if (seen_new) ordered = false;
    }
    if (!known || !ordered) fprintf(stderr, "    in flight: %s\n", show(during).c_str());
    CHECK(known);
    CHECK(ordered);

    t = tokens(s.port, 6);
    if (!all(t, 9, 6)) fprintf(stderr, "    after the reload: %s\n", show(t).c_str());
    CHECK(all(t, 9, 6));

    /* Refusals: each names its reason, and the build that ran before still runs. */
    auto refused = [&](const char* reason) {
        const int code = reload(s.port, &body);
        if (code != 409 || body.find(reason) == std::string::npos)
            fprintf(stderr, "    expected 409 naming '%s', got %d: %s\n", reason, code,
                    body.c_str());
        CHECK_EQ(code, 409);
        CHECK(body.find(reason) != std::string::npos);
        const std::vector<int32_t> after = tokens(s.port, 6);
        CHECK(all(after, 9, 6));
    };
    refused("has not changed since its hooks were loaded");
    REQUIRE(place("scratch", file));
    refused("bytes of scratch where the running build asks");
    REQUIRE(place("init", file));
    refused("defines the init hook anew");

    /* And a good build after the refused ones is taken. */
    REQUIRE(place("a", file));
    st = reload(s.port, &body);
    CHECK_EQ(st, 200);
    t = tokens(s.port, 6);
    CHECK(all(t, 5, 6));

    const std::string log = s.output();
    /* The model was loaded once, and the override opened once; the two reloads were taken. */
    CHECK_EQ(count(log, "serves 1 of the 1 rows of"), (size_t)1);
    CHECK_EQ(count(log, " reloaded: 1 of the 1 rows of "), (size_t)2);
    CHECK_EQ(s.stop(), 0);
}

/* A server over the library itself, with no override on its path: the route answers 409 naming
 * why, and the server serves on with the build it loaded. */
TEST(a_server_without_an_override_refuses_a_reload_by_name) {
    if (!exists(std::string(RAD_E2E_HOME) + "/testoverride/rl/kernels/radtest_rl.so"))
        SKIP("no reload test plugins under %s", RAD_E2E_HOME);
    tiny::TempDir dir("kreload0");
    REQUIRE(dir.ok());
    const std::string model = dir / "qwen35";
    REQUIRE(mkdir(model.c_str(), 0755) == 0);
    REQUIRE(tiny::write_qwen35(model, {}));
    const std::vector<std::string> env = {
        "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=1024", "OMP_WAIT_POLICY=passive",
        "OMP_NUM_THREADS=2",
        "RADIANCE_HOME=" + std::string(RAD_E2E_HOME) + "/testoverride/rl:" + RAD_E2E_HOME };
    const std::vector<std::string> argv = radtest::config_argv(
        { RAD_E2E_SERVER, "--model", model, "--kernels", "radtest_rl,libref",
          "--debug-accept-reference-kernels", "--host", "127.0.0.1", "--port", "0" });
    Server s;
    REQUIRE(s.start_child(argv, env));

    CHECK(all(tokens(s.port, 6), 3, 6));
    std::string body;
    const int st = reload(s.port, &body);
    if (st != 409) fprintf(stderr, "    reload answered %d: %s\n", st, body.c_str());
    CHECK_EQ(st, 409);
    CHECK(body.find("is an override built by rad_add_kernel_override") != std::string::npos);
    CHECK(all(tokens(s.port, 6), 3, 6));
    CHECK_EQ(count(s.output(), " reloaded: "), (size_t)0);
    CHECK_EQ(s.stop(), 0);
}

RAD_TEST_MAIN()
