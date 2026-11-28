/* projection_test.cpp -- the per-request switch that feeds the control vector.
 *
 * Two halves, both host-side:
 *
 *   1. the derivation (core/sched/derive.cpp, `projection_on_rows`) that turns the step's requests
 *      into the int32 row mask the projection op reads. Getting this wrong is silent in both
 *      directions: a mask that is never set serves the stock model to a client that asked for the
 *      vector, and one that is always set applies it to every request, including those that
 *      explicitly turned it off. Mixed steps are the case that matters (two conversations in one
 *      step, one projected), and that is what these cases build.
 *
 *   2. the reference operation's arithmetic (libref/ref_hc.cpp, `hc_project`), called through the
 *      plugin the way the engine calls it, checked against the operation the vector's package
 *      documents: h -= scale * (h . v) v per hyper-connection stream, and an untouched row where
 *      the mask is 0.
 *
 * The kernel parity check (libavx against libref) is libavx/avx_cases.cpp, fed to rad-avx-check;
 * libr4d's device kernel has no in-tree parity case yet (the rad-kbench fixture is still missing).
 * This file pins the meaning.
 */
#include "rad_test.h"

#include "rad_core.h"
#include "sched/derive.h"

#include "../arch/common/rad_arch.h"

#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace {

/* A step of `n_seq` sequences of `tokens_each` tokens, each with its own request, and the two
 * arrays a derivation reads. */
struct Step {
    std::vector<rad::Request>  reqs;
    std::vector<rad::StepEntry> entries;
    std::vector<int32_t>       cu;
    rad::StepPlan              plan;

    Step(int n_seq, int tokens_each) {
        reqs.resize((size_t) n_seq);
        entries.resize((size_t) n_seq);
        cu.push_back(0);
        for (int i = 0; i < n_seq; ++i) {
            entries[(size_t) i].req = &reqs[(size_t) i];
            entries[(size_t) i].n_tokens = tokens_each;
            cu.push_back(cu.back() + tokens_each);
        }
        plan.e = entries;
        plan.n_tok = cu.back();
    }
};

int run_derive(const Step& s, int32_t* out, int64_t cap, int64_t* n_out) {
    const int idx = rad::derive_index("projection_on_rows");
    if (idx < 0) return -999;
    rad::DeriveInput in;
    in.plan = &s.plan;
    in.cu_seqlens = s.cu.data();
    in.n_tok = s.plan.n_tok;
    in.n_seq = (int64_t) s.plan.e.size();
    return rad::derive_row(idx)->fn(in, out, cap, n_out);
}

/* libref's kernel table, dlopen'd from the build's radiance home (ctest sets RADIANCE_HOME), the
 * same way ref_test reaches it. */
const RadKernelInfo* libref_kernel(const char* op) {
    static int (*count)(void) = nullptr;
    static const RadKernelInfo* (*at)(int) = nullptr;
    if (!count) {
        std::vector<std::string> paths;
        const std::string home = ::home();
        paths.push_back(home + "/kernels/libref.so");
        paths.push_back("build/radiance_home/kernels/libref.so");
        paths.push_back("./libref.so");
        for (const std::string& p : paths) {
            void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!h) continue;
            count = (int (*)(void)) dlsym(h, "rad_kernel_count");
            at = (const RadKernelInfo* (*)(int)) dlsym(h, "rad_kernel_at");
            if (count && at) break;
            count = nullptr;
            at = nullptr;
        }
        if (!count) return nullptr;
    }
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && std::strcmp(k->op, op) == 0) return k;
    }
    return nullptr;
}

}  // namespace

TEST(the_mask_follows_each_requests_own_choice) {
    Step s(2, 3);
    s.reqs[0].sp.use_control_vector = true;    /* this conversation asked for the vector */
    s.reqs[1].sp.use_control_vector = false;   /* this one did not */

    int32_t mask[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int64_t n = 0;
    CHECK_OK(run_derive(s, mask, 8, &n));
    CHECK_EQ(n, (int64_t) 6);
    CHECK_EQ(mask[0], 1);
    CHECK_EQ(mask[2], 1);
    /* The second sequence's rows: untouched. A mixed step is what a naive "any request in the
     * step asked for it" would get wrong. */
    CHECK_EQ(mask[3], 0);
    CHECK_EQ(mask[5], 0);
}

TEST(the_mask_is_all_zero_when_no_request_asked) {
    Step s(3, 2);
    int32_t mask[8] = { 9, 9, 9, 9, 9, 9, 9, 9 };
    int64_t n = 0;
    CHECK_OK(run_derive(s, mask, 8, &n));
    CHECK_EQ(n, (int64_t) 6);
    for (int i = 0; i < 6; ++i) CHECK_EQ(mask[i], 0);
}

TEST(the_derivation_refuses_a_buffer_too_small) {
    Step s(2, 4);
    int32_t mask[4] = {};
    int64_t n = 0;
    CHECK(run_derive(s, mask, 4, &n) < 0);   /* 8 rows in 4 slots */
}

TEST(the_reference_projection_removes_the_direction_per_stream) {
    /* Directly the operation the package's README states, on a synthetic stream: after one
     * application every stream's component along v is (1 - scale) times what it was, and with a
     * unit direction at the stored scale 1 it is exactly zero, i.e. h -= (h.v) v. */
    const int64_t n = 8, hc = 2, rows = 2;
    std::vector<float> h((size_t) (rows * hc * n));
    std::vector<float> v((size_t) n);
    std::vector<int32_t> on{ 1, 0 };
    for (int64_t i = 0; i < n; ++i) v[(size_t) i] = (i % 2) ? 0.6f : 0.8f;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < hc; ++c)
            for (int64_t i = 0; i < n; ++i)
                h[(size_t) ((r * hc + c) * n + i)] = (float) (0.1 * (r + 1) + 0.05 * (double) c +
                                                             0.01 * (double) i);

    /* The expected result, computed here independently of the kernel. */
    std::vector<float> want = h;
    for (int64_t c = 0; c < hc; ++c) {
        double dot = 0.0;
        for (int64_t i = 0; i < n; ++i) dot += (double) h[(size_t) ((0 * hc + c) * n + i)] * v[(size_t) i];
        for (int64_t i = 0; i < n; ++i)
            want[(size_t) ((0 * hc + c) * n + i)] =
                (float) ((double) h[(size_t) ((0 * hc + c) * n + i)] - dot * (double) v[(size_t) i]);
    }

    /* Call libref's own launch, the way the engine does: through the plugin's kernel table. */
    const RadKernelInfo* k = libref_kernel("hc_project");
    if (!k || !k->launch) {
        ARTIFACT_MISSING("libref.so with an hc_project kernel");
    }

    RadTensor t[3]{};
    t[0].data = h.data(); t[0].dtype = RAD_F32; t[0].rank = 2;
    t[0].shape[0] = rows; t[0].shape[1] = hc * n;
    t[0].stride[0] = hc * n; t[0].stride[1] = 1;
    t[1].data = v.data(); t[1].dtype = RAD_F32; t[1].rank = 1;
    t[1].shape[0] = n; t[1].stride[0] = 1;
    t[2].data = on.data(); t[2].dtype = RAD_I32; t[2].rank = 1;
    t[2].shape[0] = rows; t[2].stride[0] = 1;

    const RadParam ps[] = {
        RAD_RANGE("M", 1, (int64_t) rows),
        RAD_INT("n", n),
        RAD_INT("hc", hc),
        RAD_F64("scale", 1.0),
        RAD_STR("dtype", "f32"),
    };
    RadArgs a{};
    a.t = t; a.n_t = 3;
    a.p = ps; a.n_p = (int)(sizeof(ps) / sizeof(ps[0]));

    const int rc = k->launch(&a, nullptr);
    CHECK_OK(rc);

    for (size_t i = 0; i < h.size(); ++i) CHECK_NEAR(h[i], want[i], 1e-5);

    /* The row with a zero mask is untouched, bit for bit: that is what makes an unprojected
     * request identical to a deployment with no vector loaded. */
    for (int64_t c = 0; c < hc; ++c)
        for (int64_t i = 0; i < n; ++i) {
            const size_t k = (size_t) ((1 * hc + c) * n + i);
            CHECK_EQ(std::memcmp(&h[k], &want[k], sizeof(float)), 0);
        }
}

TEST(the_reference_hc_add_adds_scaled_vector_and_skips_off_rows) {
    const int64_t n = 8, hc = 2, rows = 2;
    const RadKernelInfo* k = libref_kernel("hc_add");
    if (!k || !k->launch) {
        ARTIFACT_MISSING("libref.so with an hc_add kernel");
    }

    for (const bool per_stream : { false, true }) {
        std::vector<float> h((size_t) (rows * hc * n));
        std::vector<float> v((size_t) ((per_stream ? hc : 1) * n));
        std::vector<int32_t> on{ 1, 0 };
        const double scale = per_stream ? -0.5 : 1.5;
        for (int64_t vr = 0; vr < (per_stream ? hc : 1); ++vr)
            for (int64_t i = 0; i < n; ++i)
                v[(size_t) (vr * n + i)] = (float) ((vr + 1) * (0.25 + 0.1 * (double) i));
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t c = 0; c < hc; ++c)
                for (int64_t i = 0; i < n; ++i)
                    h[(size_t) ((r * hc + c) * n + i)] = (float) (0.2 * (r + 1) + 0.05 * c + 0.01 * i);

        std::vector<float> want = h;
        for (int64_t c = 0; c < hc; ++c)
            for (int64_t i = 0; i < n; ++i)
                want[(size_t) (c * n + i)] =
                    (float) ((double) h[(size_t) (c * n + i)] +
                             scale * (double) v[(size_t) ((per_stream ? c : 0) * n + i)]);

        RadTensor t[3]{};
        t[0].data = h.data(); t[0].dtype = RAD_F32; t[0].rank = 2;
        t[0].shape[0] = rows; t[0].shape[1] = hc * n;
        t[0].stride[0] = hc * n; t[0].stride[1] = 1;
        t[1].data = v.data(); t[1].dtype = RAD_F32;
        if (per_stream) {
            t[1].rank = 2; t[1].shape[0] = hc; t[1].shape[1] = n;
            t[1].stride[0] = n; t[1].stride[1] = 1;
        } else {
            t[1].rank = 1; t[1].shape[0] = n; t[1].stride[0] = 1;
        }
        t[2].data = on.data(); t[2].dtype = RAD_I32; t[2].rank = 1;
        t[2].shape[0] = rows; t[2].stride[0] = 1;

        const RadParam ps[] = {
            RAD_RANGE("M", 1, (int64_t) rows),
            RAD_INT("n", n),
            RAD_INT("hc", hc),
            RAD_INT("vsep", per_stream ? n : 0),
            RAD_F64("scale", scale),
            RAD_STR("dtype", "f32"),
        };
        RadArgs a{};
        a.t = t; a.n_t = 3;
        a.p = ps; a.n_p = (int) (sizeof(ps) / sizeof(ps[0]));

        CHECK_OK(k->launch(&a, nullptr));
        for (size_t i = 0; i < h.size(); ++i) CHECK_NEAR(h[i], want[i], 1e-5);
        for (int64_t c = 0; c < hc; ++c)
            for (int64_t i = 0; i < n; ++i) {
                const size_t idx = (size_t) ((1 * hc + c) * n + i);
                CHECK_EQ(std::memcmp(&h[idx], &want[idx], sizeof(float)), 0);
            }
    }
}

/* Think-phase gating interactions: THINK_AFTER=N (N > 0) leaves all prefill tokens unsteered and
 * starts steering only at think decode token N, both with and without THINK_ONLY, and turns off
 * once </think> is emitted. */
TEST(think_phase_gating_think_after_and_prefill_last_only) {
    rad::Request r;
    r.sp.use_control_vector = true;
    r.sp.reasoning.open = 100;
    r.sp.reasoning.close = 101;
    r.sp.reasoning.starts_open = true;
    r.prompt = { 1, 2, 3, 4, 5 };   /* prompt_len = 5 */

    const rad::CvecGate after3{ /*think_only=*/true, /*output_only=*/false, /*prefill_last_only=*/false, /*think_after=*/3 };
    const rad::CvecGate after3_alone{ false, false, false, 3 };
    const rad::CvecGate last_only_think{ true, false, true, 0 };

    /* Prefill (output empty): THINK_AFTER=3 must leave every prefill token unsteered. */
    rad::StepEntry pf{};
    pf.req = &r; pf.ctx_len = 0; pf.n_tokens = 5;
    CHECK(!rad::cvec_entry_has_active_rows(pf, after3));
    CHECK(!rad::cvec_entry_has_active_rows(pf, after3_alone));
    /* PREFILL_LAST_ONLY with THINK_AFTER=0 steers only the final prompt token (pos == 4). */
    CHECK(rad::cvec_entry_has_active_rows(pf, last_only_think));
    for (int64_t p = 0; p < 4; ++p) CHECK(!rad::cvec_token_active(&r, p, last_only_think));
    CHECK(rad::cvec_token_active(&r, 4, last_only_think));

    /* Decode think tokens 1, 2: still below think_after=3. */
    r.output = { 10, 11 };
    rad::StepEntry dec{};
    dec.req = &r; dec.ctx_len = 6; dec.n_tokens = 1;
    CHECK(!rad::cvec_entry_has_active_rows(dec, after3));
    CHECK(!rad::cvec_entry_has_active_rows(dec, after3_alone));

    /* Decode think token 3 (rc.body == 3): steering turns ON. */
    r.output = { 10, 11, 12 };
    dec.ctx_len = 7;
    CHECK(rad::cvec_entry_has_active_rows(dec, after3));
    CHECK(rad::cvec_entry_has_active_rows(dec, after3_alone));
    /* Even if prompt is recomputed after preemption, prompt positions < 5 stay unsteered. */
    CHECK(!rad::cvec_entry_has_active_rows(pf, after3));
    for (int64_t p = 0; p < 5; ++p) CHECK(!rad::cvec_token_active(&r, p, after3));
    CHECK(rad::cvec_token_active(&r, 5, after3));

    /* After </think> (token 101): steering turns OFF. */
    r.output = { 10, 11, 12, 101 };
    dec.ctx_len = 8;
    CHECK(!rad::cvec_entry_has_active_rows(dec, after3));
    CHECK(!rad::cvec_entry_has_active_rows(dec, after3_alone));
    CHECK(!rad::cvec_entry_has_active_rows(dec, last_only_think));
}

/* Ramp gating: THINK_RAMP=4 linearly ramps from think_after=2 to full strength via negative
 * fixed-point flags in [-9999, -1], while hc_add interprets >0 as 1.0x and <0 as (-flag)/10000. */
TEST(think_ramp_gating_and_hc_add_fractional_scale) {
    rad::Request r;
    r.sp.use_control_vector = true;
    r.sp.reasoning.open = 100;
    r.sp.reasoning.close = 101;
    r.sp.reasoning.starts_open = true;
    r.prompt = { 1, 2, 3, 4, 5 };   /* prompt_len = 5 */

    const rad::CvecGate ramp{ /*think_only=*/true, /*output_only=*/false, /*prefill_last_only=*/false,
                              /*think_after=*/2, /*think_ramp=*/4 };

    /* Prefill stays unsteered. */
    for (int64_t p = 0; p < 5; ++p) CHECK_EQ(rad::cvec_row_flag(&r, p, ramp), 0);

    /* Body 0, 1: below think_after=2 -> 0. */
    r.output = { 10 };
    CHECK(!rad::cvec_req_active(&r, ramp));
    CHECK_EQ(rad::cvec_row_flag(&r, 5, ramp), 0);

    /* Body 2..5: linear ramp 1/5, 2/5, 3/5, 4/5 -> -2000, -4000, -6000, -8000. */
    r.output = { 10, 11 };
    CHECK(rad::cvec_req_active(&r, ramp));
    CHECK_EQ(rad::cvec_row_flag(&r, 6, ramp), -2000);
    /* Speculative draft tokens at pos 7, 8, 9, 10 advance along the ramp. */
    CHECK_EQ(rad::cvec_row_flag(&r, 7, ramp), -4000);
    CHECK_EQ(rad::cvec_row_flag(&r, 8, ramp), -6000);
    CHECK_EQ(rad::cvec_row_flag(&r, 9, ramp), -8000);
    CHECK_EQ(rad::cvec_row_flag(&r, 10, ramp), 1);

    /* After </think>: turns off completely. */
    r.output = { 10, 11, 12, 101 };
    CHECK(!rad::cvec_req_active(&r, ramp));
    CHECK_EQ(rad::cvec_row_flag(&r, 8, ramp), 0);

    /* Verify reference hc_add applies fractional ramp on negative flags and 1.0x on flag=1. */
    const RadKernelInfo* k = libref_kernel("hc_add");
    if (!k || !k->launch) ARTIFACT_MISSING("libref.so with an hc_add kernel");
    const int64_t n = 4, hc = 2, rows = 4;
    std::vector<float> h((size_t) (rows * hc * n), 1.0f);
    std::vector<float> v((size_t) n, 2.0f);
    std::vector<int32_t> on{ 1, -5000, -2500, 0 };
    RadTensor t[3]{};
    t[0].data = h.data(); t[0].dtype = RAD_F32; t[0].rank = 2;
    t[0].shape[0] = rows; t[0].shape[1] = hc * n; t[0].stride[0] = hc * n; t[0].stride[1] = 1;
    t[1].data = v.data(); t[1].dtype = RAD_F32; t[1].rank = 1;
    t[1].shape[0] = n; t[1].stride[0] = 1;
    t[2].data = on.data(); t[2].dtype = RAD_I32; t[2].rank = 1;
    t[2].shape[0] = rows; t[2].stride[0] = 1;
    const RadParam ps[] = {
        RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("hc", hc),
        RAD_INT("vsep", 0), RAD_F64("scale", 0.5), RAD_STR("dtype", "f32"),
    };
    RadArgs a{}; a.t = t; a.n_t = 3; a.p = ps; a.n_p = (int) (sizeof(ps) / sizeof(ps[0]));
    CHECK_OK(k->launch(&a, nullptr));
    /* Row 0 (flag=1): 1.0 + 0.5*2.0 = 2.0; Row 1 (-5000): 1.5; Row 2 (-2500): 1.25; Row 3 (0): 1.0 */
    const float expected[4] = { 2.0f, 1.5f, 1.25f, 1.0f };
    for (int64_t rr = 0; rr < rows; ++rr)
        for (int64_t j = 0; j < hc * n; ++j)
            CHECK_NEAR(h[(size_t) (rr * hc * n + j)], expected[rr], 1e-5);
}

/* OUTPUT_ONLY=1: leaves prefill and <think> unsteered, turns ON immediately once </think> closes. */
TEST(output_only_gating_steers_only_post_think_output) {
    rad::Request r;
    r.sp.use_control_vector = true;
    r.sp.reasoning.open = 100;
    r.sp.reasoning.close = 101;
    r.sp.reasoning.starts_open = true;
    r.prompt = { 1, 2, 3, 4, 5 };

    const rad::CvecGate out_only{ /*think_only=*/false, /*output_only=*/true };

    /* Prefill and inside <think>: OFF. */
    rad::StepEntry pf{}; pf.req = &r; pf.ctx_len = 0; pf.n_tokens = 5;
    CHECK(!rad::cvec_entry_has_active_rows(pf, out_only));
    r.output = { 10, 11, 12 };
    rad::StepEntry dec{}; dec.req = &r; dec.ctx_len = 7; dec.n_tokens = 1;
    CHECK(!rad::cvec_entry_has_active_rows(dec, out_only));

    /* Once </think> (101) is emitted: ON for decode, still OFF for prompt recompute. */
    r.output = { 10, 11, 12, 101 };
    dec.ctx_len = 8;
    CHECK(rad::cvec_entry_has_active_rows(dec, out_only));
    CHECK_EQ(rad::cvec_row_flag(&r, 8, out_only), 1);
    CHECK_EQ(rad::cvec_row_flag(&r, 4, out_only), 0);
}

/* CAST mode: verifies phase coding (1=ROUTE, 2=THINK_EARLY, 3=THINK_LATE, 4=OUTPUT) AND
 * reference hc_cast_route + hc_cast_add latching across two concurrent slots with different routes! */
TEST(cast_sequence_latched_routing_and_phase_steering) {
    rad::Request r;
    r.sp.use_control_vector = true;
    r.sp.reasoning.open = 100;
    r.sp.reasoning.close = 101;
    r.sp.reasoning.starts_open = true;
    r.prompt = { 1, 2, 3, 4, 5 };  /* prompt_len = 5 -> last prefill pos = 4 */

    rad::CvecGate cg{};
    cg.cast_mode = true;
    cg.cast_think_after = 2;

    /* Earlier prefill positions stay 0; last prefill pos (4) emits phase 1 (ROUTE). */
    for (int64_t p = 0; p < 4; ++p) CHECK_EQ(rad::cvec_row_flag(&r, p, cg), 0);
    CHECK_EQ(rad::cvec_row_flag(&r, 4, cg), 1);
    /* Decode inside <think>: body=1 -> THINK_EARLY (2); body=2 -> THINK_LATE (3). */
    r.output = { 10 };
    CHECK_EQ(rad::cvec_row_flag(&r, 5, cg), 2);
    r.output = { 10, 11 };
    CHECK_EQ(rad::cvec_row_flag(&r, 6, cg), 3);
    /* After </think>: OUTPUT (4). */
    r.output = { 10, 11, 101 };
    CHECK_EQ(rad::cvec_row_flag(&r, 7, cg), 4);

    /* Test ref_hc_cast_route + ref_hc_cast_add on 2 routes, 2 experts, hc=2, n=4. */
    const RadKernelInfo* kr = libref_kernel("hc_cast_route");
    const RadKernelInfo* ka = libref_kernel("hc_cast_add");
    if (!kr || !ka) ARTIFACT_MISSING("libref.so with hc_cast_route and hc_cast_add");

    const int64_t n = 4, hc = 2, hn = hc * n, n_routes = 2, n_experts = 2, max_slots = 4;
    /* rtbl = [mu(8), W0(8), W1(8), bias(2)]. W0 = +e0, W1 = +e1. */
    std::vector<float> rtbl((size_t)((1 + n_routes) * hn + n_routes), 0.0f);
    rtbl[(size_t)(1 * hn + 0)] = 1.0f;
    rtbl[(size_t)(2 * hn + 1)] = 1.0f;
    std::vector<int32_t> seq_route((size_t)max_slots, -1);

    /* Prefill batch: row 0 (slot 0) aligns with W0 -> route 0; row 1 (slot 2) aligns with W1 -> route 1. */
    std::vector<float> h_pf((size_t)(2 * hn), 0.0f);
    h_pf[0] = 5.0f;
    h_pf[(size_t)(hn + 1)] = 5.0f;
    std::vector<int32_t> on_pf = { (1 << 20) | (0 << 8) | 1, (1 << 20) | (2 << 8) | 1 };

    RadTensor tr[4]{};
    tr[0].data = h_pf.data(); tr[0].dtype = RAD_F32; tr[0].rank = 2;
    tr[0].shape[0] = 2; tr[0].shape[1] = hn; tr[0].stride[0] = hn; tr[0].stride[1] = 1;
    tr[1].data = rtbl.data(); tr[1].dtype = RAD_F32; tr[1].rank = 1;
    tr[1].shape[0] = (int64_t)rtbl.size(); tr[1].stride[0] = 1;
    tr[2].data = on_pf.data(); tr[2].dtype = RAD_I32; tr[2].rank = 1;
    tr[2].shape[0] = 2; tr[2].stride[0] = 1;
    tr[3].data = seq_route.data(); tr[3].dtype = RAD_I32; tr[3].rank = 1;
    tr[3].shape[0] = max_slots; tr[3].stride[0] = 1;
    const RadParam pr[] = {
        RAD_RANGE("M", 1, 2), RAD_INT("n", n), RAD_INT("hc", hc),
        RAD_INT("n_routes", n_routes), RAD_INT("max_slots", max_slots),
        RAD_F64("min_margin", 0.0), RAD_STR("dtype", "f32"),
    };
    RadArgs ar{}; ar.t = tr; ar.n_t = 4; ar.p = pr; ar.n_p = (int)(sizeof(pr)/sizeof(pr[0]));
    CHECK_OK(kr->launch(&ar, nullptr));
    CHECK_EQ(seq_route[0], 0);
    CHECK_EQ(seq_route[2], 1);
    CHECK_EQ(seq_route[1], -1);

    /* Route 0 -> [-1, 0, -1] (expert 0 only in THINK_LATE); Route 1 -> [-1, -1, 1] (expert 1 only in OUTPUT). */
    std::vector<int32_t> rmap = { -1, 0, -1,  -1, -1, 1 };
    /* v: expert 0 adds +10, expert 1 adds +20. */
    std::vector<float> v((size_t)(n_experts * hn), 0.0f);
    for (int64_t j = 0; j < hn; ++j) { v[(size_t)j] = 10.0f; v[(size_t)(hn + j)] = 20.0f; }

    /* Decode batch of 4 rows:
     *   row 0: slot 0, THINK_LATE (3) -> expert 0 (+10)
     *   row 1: slot 0, OUTPUT     (4) -> none     (+0)
     *   row 2: slot 2, THINK_LATE (3) -> none     (+0)
     *   row 3: slot 2, OUTPUT     (4) -> expert 1 (+20) */
    std::vector<float> h_dec((size_t)(4 * hn), 1.0f);
    std::vector<int32_t> on_dec = {
        (1 << 20) | (0 << 8) | 3,
        (1 << 20) | (0 << 8) | 4,
        (1 << 20) | (2 << 8) | 3,
        (1 << 20) | (2 << 8) | 4,
    };
    RadTensor ta[5]{};
    ta[0].data = h_dec.data(); ta[0].dtype = RAD_F32; ta[0].rank = 2;
    ta[0].shape[0] = 4; ta[0].shape[1] = hn; ta[0].stride[0] = hn; ta[0].stride[1] = 1;
    ta[1].data = v.data(); ta[1].dtype = RAD_F32; ta[1].rank = 2;
    ta[1].shape[0] = n_experts * hc; ta[1].shape[1] = n; ta[1].stride[0] = n; ta[1].stride[1] = 1;
    ta[2].data = on_dec.data(); ta[2].dtype = RAD_I32; ta[2].rank = 1;
    ta[2].shape[0] = 4; ta[2].stride[0] = 1;
    ta[3].data = seq_route.data(); ta[3].dtype = RAD_I32; ta[3].rank = 1;
    ta[3].shape[0] = max_slots; ta[3].stride[0] = 1;
    ta[4].data = rmap.data(); ta[4].dtype = RAD_I32; ta[4].rank = 1;
    ta[4].shape[0] = n_routes * 3; ta[4].stride[0] = 1;
    const RadParam pa[] = {
        RAD_RANGE("M", 1, 4), RAD_INT("n", n), RAD_INT("hc", hc),
        RAD_INT("n_routes", n_routes), RAD_INT("n_experts", n_experts),
        RAD_INT("max_slots", max_slots), RAD_F64("scale", 1.0), RAD_STR("dtype", "f32"),
    };
    RadArgs aa{}; aa.t = ta; aa.n_t = 5; aa.p = pa; aa.n_p = (int)(sizeof(pa)/sizeof(pa[0]));
    CHECK_OK(ka->launch(&aa, nullptr));
    const float want[4] = { 11.0f, 1.0f, 1.0f, 21.0f };
    for (int r0 = 0; r0 < 4; ++r0)
        for (int j = 0; j < hn; ++j)
            CHECK_NEAR(h_dec[(size_t)(r0 * hn + j)], want[r0], 1e-5);
}

RAD_TEST_MAIN()
