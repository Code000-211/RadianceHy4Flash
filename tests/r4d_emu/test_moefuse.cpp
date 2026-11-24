/* tests/r4d_emu: GLM's decode fusions of the MoE block (r4d_glm.hip) on the host SIMT emulator,
 * each byte for byte against the separate launches it replaces.
 *
 * router_sigmoid_topk_scatter against router_sigmoid_topk (the emulated kernel), then moe_scatter's
 * stable counting sort over its ids written out here, then the `fill` of the shared gate with 1:
 * random, tied and NaN ranks (a row with fewer finite ranks than top_k selects expert 0 for its
 * empty slots, and the sort takes those as written), slices and protected experts, rows past 64
 * wrapping over the waves, the sort's bound and its refusal past it. Pitch padding, guards past
 * every output and the gate's untouched rows show an overrun.
 *
 * router_gemm_sigmoid_scatter (the block's projection fused in front of that selection) against
 * gemm_nt_f32 then router_sigmoid_topk_scatter: the logits byte for byte as well, at the heights
 * that reach every tile form and wave count of both, with a bf16 and an f32 router weight, tied
 * ranks and a row of NaNs, slices and protected experts, the gate's fill and the inverse, and the
 * refusal past the decode band.
 *
 * moe_scatter_inv and moe_gather_inv: the small sort's kernel with the inverse against itself
 * without, and the inverse against the table the gather builds from `sorted`; then moe_gather's
 * kernel reading that inverse against its self-scan (T <= 512) or the prepass's table (above), at
 * GLM's f32 routing weights and bf16 ones, with the ungated shared arm, a sigmoid-gated one and
 * none, and slices that drop slots. moe_scatter's three-pass form (T > 1024) is not here: its
 * kernels take dynamic LDS through `extern __shared__`, which the emulator cannot give; the card
 * checks it (r4d_selftest --case moe_scatter_inv_i32). */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "r4d_moe_sort.h"
#include "r4d_moe_gather.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int32_t kPoison = -77;
constexpr uint16_t kGatePoison = 0x7e57;

/* moe_scatter's contract: slot i = token * top_k + slot goes to its expert's run in ascending i;
 * an expert outside the slice drops the slot; protected experts are placed last in list order;
 * sorted rows past the live count are -1. */
struct SortModel {
    std::vector<int32_t> sorted, off, cnt;
};
int32_t place_of(int32_t e, int nl, const std::vector<int>& prot) {
    int32_t below = 0;
    for (size_t j = 0; j < prot.size(); ++j) {
        if (prot[j] == e) return nl - (int32_t)prot.size() + (int32_t)j;
        below += prot[j] < e;
    }
    return e - below;
}
SortModel sort_model(const std::vector<int32_t>& ids, int M, int k, int ip, int nl, int base,
                     const std::vector<int>& prot) {
    const int T = M * k;
    std::vector<int32_t> place((size_t)T);
    SortModel s;
    s.cnt.assign((size_t)nl, 0);
    s.off.assign((size_t)nl + 1, 0);
    s.sorted.assign((size_t)T, -1);
    for (int i = 0; i < T; ++i) {
        const int32_t e = ids[(size_t)(i / k) * ip + i % k] - base;
        place[(size_t)i] = e >= 0 && e < nl ? place_of(e, nl, prot) : -1;
        if (place[(size_t)i] >= 0) ++s.cnt[(size_t)place[(size_t)i]];
    }
    for (int e = 0; e < nl; ++e) s.off[(size_t)e + 1] = s.off[(size_t)e] + s.cnt[(size_t)e];
    std::vector<int32_t> seen((size_t)nl, 0);
    for (int i = 0; i < T; ++i)
        if (place[(size_t)i] >= 0) {
            const int32_t e = place[(size_t)i];
            s.sorted[(size_t)(s.off[(size_t)e] + seen[(size_t)e]++)] = i;
        }
    return s;
}

/* The table the gather builds from `sorted` (gather_kernel's self-scan, or its prepass): slot
 * sorted[p] at row p, every other slot -1. */
std::vector<int32_t> inverse_of(const std::vector<int32_t>& sorted) {
    std::vector<int32_t> inv(sorted.size(), -1);
    for (size_t p = 0; p < sorted.size(); ++p)
        if (sorted[p] >= 0) inv[(size_t)sorted[p]] = (int32_t)p;
    return inv;
}

/* moe_scatter's small-form kernel (r4d_moe_sort.h) over contiguous ids, with or without the
 * inverse. */
struct Scatter {
    int rc = RAD_OK;
    std::vector<int32_t> sorted, off, cnt, inv;
};
Scatter scatter_kernel(const std::vector<int32_t>& ids, int nl, int base, const char* protect,
                       bool with_inv) {
    const int T = (int)ids.size();
    Scatter s;
    s.sorted.assign((size_t)T, kPoison);
    s.off.assign((size_t)nl + 1, kPoison);
    s.cnt.assign((size_t)nl, kPoison);
    s.inv.assign((size_t)T, kPoison);
    RadParam pp{};
    pp.key = "protect";
    pp.kind = RAD_P_STR;
    pp.sval = protect;
    RadArgs pa{};
    pa.p = &pp;
    pa.n_p = protect ? 1 : 0;
    r4d_moesort::Protect prot;
    s.rc = r4d_moesort::protect_of(&pa, nl, &prot);
    if (s.rc != RAD_OK) return s;
    if (with_inv)
        R4D_EMIT(nullptr, (r4d_moesort::scatter_small_kernel<true>), dim3(1), dim3(256), 0,
                 ids.data(), s.sorted.data(), s.off.data(), s.cnt.data(), (uint32_t)T, (uint32_t)nl,
                 (int32_t)base, prot, s.inv.data());
    else
        R4D_EMIT(nullptr, (r4d_moesort::scatter_small_kernel<false>), dim3(1), dim3(256), 0,
                 ids.data(), s.sorted.data(), s.off.data(), s.cnt.data(), (uint32_t)T, (uint32_t)nl,
                 (int32_t)base, prot, (int32_t*)nullptr);
    return s;
}

struct RouterShape {
    int M, ne, k, norm;
    int mode;              /* 0 random, 1 tied ranks, 2 NaN rows */
    int nl, base;          /* the slice: nl 0 is the whole router */
    const char* protect;
    bool gate;
};

void router_scatter_cases(std::mt19937& rng, int& cases, int& fails) {
    const RouterShape shapes[] = {
        { 1, 288, 8, 1, 0, 0, 0, nullptr, true },      { 2, 288, 8, 1, 0, 0, 0, nullptr, true },
        { 6, 288, 8, 1, 0, 0, 0, nullptr, true },      { 8, 288, 8, 1, 1, 0, 0, nullptr, true },
        { 9, 288, 8, 1, 0, 0, 0, nullptr, false },     { 16, 288, 8, 1, 2, 0, 0, nullptr, true },
        { 17, 288, 8, 1, 0, 0, 0, nullptr, true },     { 24, 288, 8, 1, 1, 0, 0, nullptr, true },
        { 33, 288, 8, 1, 0, 0, 0, nullptr, true },     { 48, 288, 8, 1, 0, 0, 0, nullptr, true },
        { 48, 288, 8, 1, 2, 0, 0, nullptr, true },     { 65, 288, 8, 1, 1, 0, 0, nullptr, true },
        { 100, 288, 8, 0, 0, 0, 0, nullptr, true },    { 128, 288, 8, 1, 0, 0, 0, nullptr, true },
        { 24, 288, 8, 1, 0, 144, 144, nullptr, true }, { 24, 288, 8, 1, 1, 96, 32, "3,7,90", true },
        { 48, 512, 8, 1, 0, 0, 0, nullptr, true },     { 32, 32, 32, 0, 1, 0, 0, nullptr, true },
        { 8, 512, 16, 1, 0, 0, 0, nullptr, true },     { 9, 512, 16, 1, 1, 0, 0, nullptr, true },
        { 16, 512, 16, 1, 2, 0, 0, nullptr, true },    { 5, 300, 10, 1, 2, 0, 0, "0,299", true },
    };
    std::uniform_int_distribution<int> ud(-1000, 1000);
    for (const RouterShape& sh : shapes) {
        const int M = sh.M, ne = sh.ne, k = sh.k, T = M * k;
        const int nl = sh.nl ? sh.nl : ne, base = sh.base;
        const int lp = ne + 5, bp = 2, ip = k + 3, wp = k + 1, gp = 3;
        const float factor = 2.5f;
        std::vector<float> lg((size_t)M * lp), bias((size_t)ne * bp);
        for (auto& v : lg) v = sh.mode == 1 ? (float)(ud(rng) % 8) * 0.5f : (float)ud(rng) / 200;
        for (auto& v : bias)
            v = sh.mode == 1 ? (float)(ud(rng) % 5) * 0.125f : (float)ud(rng) / 2000;
        if (sh.mode == 2)
            for (int r = 0; r < M; r += 2) {
                /* every other row keeps (r % 5) finite ranks, fewer than top_k */
                for (int e = 0; e < ne; ++e) lg[(size_t)r * lp + e] = NAN;
                for (int j = 0; j < r % 5; ++j)
                    lg[(size_t)r * lp + (size_t)(ud(rng) + 1000) % ne] = 0.25f * (float)j;
            }
        std::vector<int> prot;
        if (sh.protect)
            for (const char* p = sh.protect; *p;) {
                char* end = nullptr;
                prot.push_back((int)std::strtol(p, &end, 10));
                p = *end == ',' ? end + 1 : end;
            }

        RadTensor t[8]{};
        t[0].data = lg.data();
        t[0].dtype = RAD_F32;
        t[0].rank = 2;
        t[0].shape[0] = M;
        t[0].shape[1] = ne;
        t[0].stride[0] = lp;
        t[0].stride[1] = 1;
        t[1].data = bias.data();
        t[1].dtype = RAD_F32;
        t[1].rank = 1;
        t[1].shape[0] = ne;
        t[1].stride[0] = bp;
        t[2].dtype = RAD_I32;
        t[2].rank = 2;
        t[2].shape[0] = M;
        t[2].shape[1] = k;
        t[2].stride[0] = ip;
        t[2].stride[1] = 1;
        t[3].dtype = RAD_F32;
        t[3].rank = 2;
        t[3].shape[0] = M;
        t[3].shape[1] = k;
        t[3].stride[0] = wp;
        t[3].stride[1] = 1;
        t[4].dtype = RAD_I32;
        t[4].rank = 1;
        t[4].shape[0] = T;
        t[4].stride[0] = 1;
        t[5].dtype = RAD_I32;
        t[5].rank = 1;
        t[5].shape[0] = nl + 1;
        t[5].stride[0] = 1;
        t[6].dtype = RAD_I32;
        t[6].rank = 1;
        t[6].shape[0] = nl;
        t[6].stride[0] = 1;
        t[7].dtype = RAD_BF16;
        t[7].rank = 2;
        t[7].shape[0] = M;
        t[7].shape[1] = 1;
        t[7].stride[0] = gp;
        t[7].stride[1] = 1;
        std::vector<RadParam> p(9);
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "n_expert", RAD_P_INT, ne };
        p[2] = { "top_k", RAD_P_INT, k };
        p[3] = { "norm", RAD_P_INT, sh.norm };
        p[4].key = "route_scale";
        p[4].kind = RAD_P_F64;
        p[4].dval = factor;
        p[5].key = "dtype";
        p[5].kind = RAD_P_STR;
        p[5].sval = "f32";
        p[6] = { "n_local", RAD_P_INT, nl };
        p[7] = { "expert_base", RAD_P_INT, base };
        p[8].key = "protect";
        p[8].kind = RAD_P_STR;
        p[8].sval = sh.protect;
        const int n_p = sh.protect ? 9 : 8;

        /* the separate launches: the selection, then the sort and fill modelled */
        std::vector<int32_t> ids_ref((size_t)M * ip, kPoison);
        std::vector<float> w_ref((size_t)M * wp, -7.25f);
        t[2].data = ids_ref.data();
        t[3].data = w_ref.data();
        RadArgs sel{};
        sel.t = t;
        sel.n_t = 4;
        sel.p = p.data();
        sel.n_p = 6;
        ++cases;
        if (r4d_rad_router_sigmoid_topk(&sel, nullptr) != RAD_OK) {
            std::printf("FAIL router_sigmoid_topk launch M=%d ne=%d\n", M, ne);
            ++fails;
            continue;
        }
        const SortModel ref = sort_model(ids_ref, M, k, ip, nl, base, prot);
        /* moe_scatter's own kernel over the ids, contiguous as it reads them, held to the model;
         * with the inverse, which must be the table the gather builds from `sorted`. */
        std::vector<int32_t> flat((size_t)T);
        for (int i = 0; i < T; ++i) flat[(size_t)i] = ids_ref[(size_t)(i / k) * ip + i % k];
        Scatter ks = scatter_kernel(flat, nl, base, sh.protect, true);
        ++cases;
        if (ks.rc != RAD_OK || ks.sorted != ref.sorted || ks.off != ref.off || ks.cnt != ref.cnt ||
            ks.inv != inverse_of(ref.sorted)) {
            std::printf("FAIL moe_scatter kernel against its model at M=%d ne=%d slice %d+%d\n", M,
                        ne, base, nl);
            ++fails;
        }

        for (int with_inv = 0; with_inv < 2; ++with_inv) {
            std::vector<int32_t> ids((size_t)M * ip, kPoison), sorted((size_t)T + 4, kPoison),
                off((size_t)nl + 5, kPoison), cnt((size_t)nl + 4, kPoison),
                inv((size_t)T + 4, kPoison);
            std::vector<float> w((size_t)M * wp, -7.25f);
            std::vector<uint16_t> gate((size_t)(M + 2) * gp, kGatePoison);
            RadTensor tf[9];
            std::memcpy(tf, t, sizeof t);
            tf[2].data = ids.data();
            tf[3].data = w.data();
            tf[4].data = sorted.data();
            tf[5].data = off.data();
            tf[6].data = cnt.data();
            tf[7].data = sh.gate ? gate.data() : nullptr;
            tf[8] = tf[4];
            tf[8].data = inv.data();
            RadArgs a{};
            a.t = tf;
            a.n_t = with_inv ? 9 : 8;
            a.p = p.data();
            a.n_p = n_p;
            ++cases;
            const int rc = r4d_rad_router_sigmoid_topk_scatter(&a, nullptr);
            int bad = 0;
            bad += ids != ids_ref;
            bad += std::memcmp(w.data(), w_ref.data(), w.size() * 4) != 0;
            bad += std::memcmp(sorted.data(), ks.sorted.data(), (size_t)T * 4) != 0;
            bad += std::memcmp(off.data(), ks.off.data(), (size_t)(nl + 1) * 4) != 0;
            bad += std::memcmp(cnt.data(), ks.cnt.data(), (size_t)nl * 4) != 0;
            if (with_inv) bad += std::memcmp(inv.data(), ks.inv.data(), (size_t)T * 4) != 0;
            for (int i = 0; i < 4; ++i)
                bad += sorted[(size_t)T + i] != kPoison || off[(size_t)nl + 1 + i] != kPoison ||
                       cnt[(size_t)nl + i] != kPoison || inv[(size_t)T + i] != kPoison;
            if (!with_inv)
                for (int i = 0; i < T; ++i) bad += inv[(size_t)i] != kPoison;
            for (int r = 0; r < M + 2; ++r)
                for (int c = 0; c < gp; ++c) {
                    const uint16_t want = sh.gate && r < M && c == 0 ? 0x3f80 : kGatePoison;
                    bad += gate[(size_t)r * gp + c] != want;
                }
            if (rc != RAD_OK || bad) {
                std::printf(
                    "FAIL router_sigmoid_topk_scatter at M=%d ne=%d k=%d mode %d slice %d+%d%s%s%s "
                    "(rc %d, %d mismatching outputs)\n",
                    M, ne, k, sh.mode, base, nl, sh.protect ? " protect " : "",
                    sh.protect ? sh.protect : "", with_inv ? " with inverse" : "", rc, bad);
                ++fails;
            }
        }
    }
    /* Past the sort's one workgroup: refused, nothing written. */
    {
        const int M = 129, ne = 288, k = 8;
        std::vector<float> lg((size_t)M * ne, 0.5f), bias((size_t)ne, 0.f),
            w((size_t)M * k, -7.25f);
        std::vector<int32_t> ids((size_t)M * k, kPoison), sorted((size_t)M * k, kPoison),
            off((size_t)ne + 1, kPoison), cnt((size_t)ne, kPoison);
        RadTensor t[7]{};
        auto mat = [](RadTensor& x, void* d, uint32_t dt, int r, int c) {
            x.data = d;
            x.dtype = dt;
            x.rank = 2;
            x.shape[0] = r;
            x.shape[1] = c;
            x.stride[0] = c;
            x.stride[1] = 1;
        };
        auto vec = [](RadTensor& x, void* d, int n) {
            x.data = d;
            x.dtype = RAD_I32;
            x.rank = 1;
            x.shape[0] = n;
            x.stride[0] = 1;
        };
        mat(t[0], lg.data(), RAD_F32, M, ne);
        t[1].data = bias.data();
        t[1].dtype = RAD_F32;
        t[1].rank = 1;
        t[1].shape[0] = ne;
        t[1].stride[0] = 1;
        mat(t[2], ids.data(), RAD_I32, M, k);
        mat(t[3], w.data(), RAD_F32, M, k);
        vec(t[4], sorted.data(), M * k);
        vec(t[5], off.data(), ne + 1);
        vec(t[6], cnt.data(), ne);
        RadParam p[6]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "n_expert", RAD_P_INT, ne };
        p[2] = { "top_k", RAD_P_INT, k };
        p[3] = { "norm", RAD_P_INT, 1 };
        p[4].key = "route_scale";
        p[4].kind = RAD_P_F64;
        p[4].dval = 2.5;
        p[5].key = "dtype";
        p[5].kind = RAD_P_STR;
        p[5].sval = "f32";
        RadArgs a{};
        a.t = t;
        a.n_t = 7;
        a.p = p;
        a.n_p = 6;
        ++cases;
        const int rc = r4d_rad_router_sigmoid_topk_scatter(&a, nullptr);
        if (rc == RAD_OK || ids[0] != kPoison || sorted[0] != kPoison) {
            std::printf("FAIL router_sigmoid_topk_scatter served M*top_k=%d past the sort's bound "
                        "(rc %d)\n",
                        M * k, rc);
            ++fails;
        }
    }
}

struct GatherShape {
    int M, n, nl, base;
    bool f32w; /* GLM's f32 routing weights, else bf16 */
    int shared; /* 0 none, 1 the ungated arm (gate 1, act none), 2 a sigmoid gate */
};

/* moe_gather's kernel (r4d_moe_gather.h) as its launcher runs it, reading the table moe_scatter_inv
 * wrote, against the same kernel building the table itself from `sorted` (T <= 512) or reading the
 * prepass's table (above). Ids are distinct per row from 288 experts, a slice drops slots. */
void gather_inverse_cases(std::mt19937& rng, int& cases, int& fails) {
    const GatherShape shapes[] = {
        { 1, 4096, 288, 0, true, 1 },  { 6, 4096, 288, 0, true, 1 }, { 8, 512, 288, 0, true, 1 },
        { 24, 512, 144, 72, true, 1 }, { 48, 256, 288, 0, true, 1 }, { 64, 256, 288, 0, true, 0 },
        { 100, 256, 96, 0, true, 1 },  { 6, 300, 288, 0, false, 2 }, { 33, 256, 288, 0, false, 0 },
    };
    const int k = 8, ne = 288;
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const GatherShape& sh : shapes) {
        const int M = sh.M, n = sh.n, T = M * k, nl = sh.nl;
        std::vector<int32_t> ids((size_t)T);
        for (int r = 0; r < M; ++r) {
            std::vector<int32_t> pool(ne);
            for (int e = 0; e < ne; ++e) pool[(size_t)e] = e;
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int j = 0; j < k; ++j) ids[(size_t)(r * k + j)] = pool[(size_t)j];
        }
        Scatter plain = scatter_kernel(ids, nl, sh.base, nullptr, false);
        Scatter inv = scatter_kernel(ids, nl, sh.base, nullptr, true);
        ++cases;
        if (plain.rc != RAD_OK || inv.rc != RAD_OK || inv.sorted != plain.sorted ||
            inv.off != plain.off || inv.cnt != plain.cnt || inv.inv != inverse_of(plain.sorted)) {
            std::printf("FAIL moe_scatter with the inverse at M=%d slice %d+%d\n", M, sh.base, nl);
            ++fails;
            continue;
        }
        const int ye_ld = n + 3, ew_ld = k + 2, y_ld = n + 5, sh_ld = n + 1, sg_ld = 2;
        std::vector<uint16_t> ye((size_t)T * ye_ld), shp((size_t)M * sh_ld), sg((size_t)M * sg_ld);
        for (auto& v : ye) v = f2b(nd(rng));
        for (auto& v : shp) v = f2b(nd(rng));
        for (size_t i = 0; i < sg.size(); ++i) sg[i] = sh.shared == 1 ? f2b(1.f) : f2b(nd(rng));
        std::vector<float> ewf((size_t)M * ew_ld);
        std::vector<uint16_t> ewb((size_t)M * ew_ld);
        for (size_t i = 0; i < ewf.size(); ++i) {
            ewf[i] = std::fabs(nd(rng)) * 0.3f;
            ewb[i] = f2b(ewf[i]);
        }
        const void* ew = sh.f32w ? (const void*)ewf.data() : (const void*)ewb.data();
        const uint16_t* shv = sh.shared ? shp.data() : nullptr;
        const uint16_t* sgv = sh.shared ? sg.data() : nullptr;
        const int sig = sh.shared == 2;
        const bool small = T <= (int)r4d_moegather::kGatherSelfT;
        /* the launcher's rule: the self-scan up to 512 slots, the prepass's table above */
        const std::vector<int32_t> table = inverse_of(plain.sorted);
        const unsigned gy = (unsigned)((n + 255) / 256);
        auto gather = [&](const int32_t* tab, const int32_t* sorted, std::vector<uint16_t>& y) {
            y.assign((size_t)M * y_ld, 0x7e57);
            if (sh.f32w)
                R4D_EMIT(nullptr, (r4d_moegather::gather_kernel<256, true>), dim3((unsigned)M, gy),
                         dim3(256), 0, ye.data(), ew, tab, y.data(), (uint32_t)M, (uint32_t)n,
                         (uint32_t)k, (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld, sorted,
                         (uint32_t)T, shv, sgv, (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
            else
                R4D_EMIT(nullptr, (r4d_moegather::gather_kernel<256, false>), dim3((unsigned)M, gy),
                         dim3(256), 0, ye.data(), ew, tab, y.data(), (uint32_t)M, (uint32_t)n,
                         (uint32_t)k, (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld, sorted,
                         (uint32_t)T, shv, sgv, (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
        };
        std::vector<uint16_t> y_old, y_new;
        if (small)
            gather(nullptr, plain.sorted.data(), y_old);
        else
            gather(table.data(), nullptr, y_old);
        gather(inv.inv.data(), nullptr, y_new);
        ++cases;
        if (y_new != y_old) {
            size_t first = 0;
            while (first < y_new.size() && y_new[first] == y_old[first]) ++first;
            std::printf("FAIL moe_gather from the inverse at M=%d n=%d slice %d+%d %s weights, "
                        "shared %d: first difference at %zu\n",
                        M, n, sh.base, nl, sh.f32w ? "f32" : "bf16", sh.shared, first);
            ++fails;
        }
    }
}

/* moe_gather's sixteen-slot form (gather_kernel16, r4d_moe_gather.h): top_k = 16 with eight
 * columns a lane, one sixteen-byte load a slot, against the scalar form's sums -- the same table
 * from the inverse and the same self-scan from `sorted`, at Hy4-flash's decode heights (one row,
 * the c8 verify's 32, and past the self-scan bound), f32 routing weights and bf16, the ungated
 * shared arm, a sigmoid-gated one and none, and a slice that drops slots. */
struct SixteenShape {
    int M, n, nl, base;
    bool f32w; /* Hy4-flash's f32 routing weights, else bf16 */
    int shared; /* 0 none, 1 the ungated arm, 2 a sigmoid gate */
};
void gather_sixteen_cases(std::mt19937& rng, int& cases, int& fails) {
    const SixteenShape shapes[] = {
        { 1, 2560, 288, 0, true, 0 },  { 1, 2560, 288, 0, true, 1 }, { 8, 2560, 288, 0, true, 2 },
        { 32, 2560, 288, 0, true, 1 }, { 32, 3072, 288, 0, false, 0 }, { 40, 2560, 288, 0, true, 1 },
        { 64, 2560, 288, 0, true, 2 }, { 33, 512, 144, 0, true, 1 },  { 48, 256, 96, 96, false, 0 },
        { 2, 4096, 288, 0, true, 0 },
    };
    const int k = 16, ne = 288;
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const SixteenShape& sh : shapes) {
        const int M = sh.M, n = sh.n, T = M * k, nl = sh.nl;
        std::vector<int32_t> ids((size_t)T);
        for (int r = 0; r < M; ++r) {
            std::vector<int32_t> pool(ne);
            for (int e = 0; e < ne; ++e) pool[(size_t)e] = e;
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int j = 0; j < k; ++j) ids[(size_t)(r * k + j)] = pool[(size_t)j];
        }
        Scatter plain = scatter_kernel(ids, nl, sh.base, nullptr, false);
        Scatter inv = scatter_kernel(ids, nl, sh.base, nullptr, true);
        ++cases;
        if (plain.rc != RAD_OK || inv.rc != RAD_OK || inv.inv != inverse_of(plain.sorted)) {
            std::printf("FAIL moe_gather's sixteen-slot form: the inverse at M=%d slice %d+%d\n", M,
                        sh.base, nl);
            ++fails;
            continue;
        }
        /* The form reads a row as one sixteen-byte load: every pitch a multiple of eight and the
         * rows sixteen-byte aligned, which the launcher checks before it takes the form. */
        const int ye_ld = n + 8, ew_ld = k + 2, y_ld = n + 8, sh_ld = n + 8, sg_ld = 2;
        std::vector<uint16_t> ye((size_t)T * ye_ld), shp((size_t)M * sh_ld), sg((size_t)M * sg_ld);
        std::vector<float> ewf((size_t)M * ew_ld);
        std::vector<uint16_t> ewb((size_t)M * ew_ld);
        if (((uintptr_t)ye.data() | (uintptr_t)shp.data() | (uintptr_t)ewf.data()) % 16 != 0) {
            std::printf("FAIL moe_gather's sixteen-slot form: unaligned buffers at M=%d\n", M);
            ++fails;
            continue;
        }
        for (auto& v : ye) v = f2b(nd(rng));
        for (auto& v : shp) v = f2b(nd(rng));
        for (size_t i = 0; i < sg.size(); ++i) sg[i] = sh.shared == 1 ? f2b(1.f) : f2b(nd(rng));
        for (size_t i = 0; i < ewf.size(); ++i) {
            ewf[i] = std::fabs(nd(rng)) * 0.3f;
            ewb[i] = f2b(ewf[i]);
        }
        const void* ew = sh.f32w ? (const void*)ewf.data() : (const void*)ewb.data();
        const uint16_t* shv = sh.shared ? shp.data() : nullptr;
        const uint16_t* sgv = sh.shared ? sg.data() : nullptr;
        const int sig = sh.shared == 2;
        const unsigned gy1 = (unsigned)((n + 255) / 256);
        const int kcols = (int)r4d_moegather::kGatherCols;
        const unsigned gy8 = (unsigned)((n + 256 * kcols - 1) / (256 * kcols));
        auto gather = [&](const int32_t* tab, const int32_t* sorted, bool cols,
                          std::vector<uint16_t>& y) {
            y.assign((size_t)M * y_ld, kGatePoison);
            const dim3 grid((unsigned)M, cols ? gy8 : gy1);
#define R4D_EMU_GATHER(K)                                                                          \
    R4D_EMIT(nullptr, (K), grid, dim3(256), 0, ye.data(), ew, tab, y.data(), (uint32_t)M,          \
             (uint32_t)n, (uint32_t)k, (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld, sorted,   \
             (uint32_t)T, shv, sgv, (uint32_t)sh_ld, (uint32_t)sg_ld, sig)
            if (cols) {
                if (sh.f32w) R4D_EMU_GATHER((r4d_moegather::gather_kernel16<256, true>));
                else R4D_EMU_GATHER((r4d_moegather::gather_kernel16<256, false>));
            } else {
                if (sh.f32w) R4D_EMU_GATHER((r4d_moegather::gather_kernel<256, true>));
                else R4D_EMU_GATHER((r4d_moegather::gather_kernel<256, false>));
            }
#undef R4D_EMU_GATHER
        };
        const std::vector<int32_t> table = inverse_of(plain.sorted);
        std::vector<uint16_t> y_ref, y_inv, y_self_ref, y_self;
        gather(table.data(), nullptr, false, y_ref);
        gather(inv.inv.data(), nullptr, true, y_inv);
        ++cases;
        if (y_inv != y_ref) {
            size_t first = 0;
            while (first < y_inv.size() && y_inv[first] == y_ref[first]) ++first;
            std::printf("FAIL moe_gather's sixteen-slot form from the inverse at M=%d n=%d slice "
                        "%d+%d %s weights, shared %d: first difference at %zu\n",
                        M, n, sh.base, nl, sh.f32w ? "f32" : "bf16", sh.shared, first);
            ++fails;
        }
        if (T <= (int)r4d_moegather::kGatherSelfT) {
            gather(nullptr, plain.sorted.data(), false, y_self_ref);
            gather(nullptr, plain.sorted.data(), true, y_self);
            ++cases;
            if (y_self != y_self_ref) {
                size_t first = 0;
                while (first < y_self.size() && y_self[first] == y_self_ref[first]) ++first;
                std::printf("FAIL moe_gather's sixteen-slot form's self-scan at M=%d n=%d %s "
                            "weights, shared %d: first difference at %zu\n",
                            M, n, sh.f32w ? "f32" : "bf16", sh.shared, first);
                ++fails;
            }
        }
    }
}

/* The router's projection and the selection it feeds in one launch (router_gemm_sigmoid_scatter),
 * byte for byte against the two it replaces: gemm_nt_f32 then router_sigmoid_topk_scatter, which the
 * two blocks above already hold to their own arithmetic. The heights reach every tile form and
 * wave count the launcher picks (two rows by two columns to 16 rows, eight by one above; 8, 16 and
 * 32 waves), at a bf16 and an f32 router weight, with tied ranks and a row of NaNs (a row of x),
 * a slice with protected experts, the shared gate's fill and the inverse. The logits it writes are
 * the unfused GEMM's byte for byte, pitch padding included, and the guards past every output and
 * the gate's untouched rows show an overrun; a height past the decode band is refused. With
 * `hq_group` the launch also fills the latent's rotated codes, byte for byte against the served
 * quantiser over the same activation (had_quant_act_fp8 or, int8, had_quant_act_i8g including
 * IQ8R's 256-wide group), pitch padding and guards included. */
struct FusedShape {
    int M, K, ne, k, norm;
    bool bf16w;
    int mode; /* 0 random, 1 tied ranks, 2 NaN rows */
    int nl, base;
    bool gate, inv;
    const char* protect;
    int hqg = 0;         /* the latent's codes in the same launch: 128 or 256, 0 for none */
    bool hqi8 = false;   /* int8 codes rather than E4M3 */
};

void fused_router_cases(std::mt19937& rng, int& cases, int& fails) {
    const FusedShape shapes[] = {
        { 1, 512, 288, 8, 1, true, 0, 0, 0, true, false, nullptr },
        { 2, 512, 288, 8, 1, true, 0, 0, 0, true, false, nullptr },
        { 6, 1536, 288, 8, 1, true, 1, 0, 0, true, true, nullptr },
        { 8, 512, 288, 8, 1, false, 0, 0, 0, true, true, nullptr },
        { 9, 512, 288, 8, 1, true, 0, 0, 0, false, true, nullptr },
        { 16, 512, 288, 8, 1, true, 2, 0, 0, true, true, nullptr },
        { 17, 512, 288, 8, 1, false, 0, 0, 0, true, false, nullptr },
        { 31, 512, 288, 8, 1, true, 0, 0, 0, true, true, nullptr },
        { 32, 300, 288, 8, 1, true, 1, 0, 0, true, true, nullptr },
        { 33, 512, 288, 8, 1, true, 0, 0, 0, true, false, nullptr },
        { 48, 512, 288, 8, 1, false, 2, 0, 0, true, true, nullptr },
        { 64, 512, 288, 8, 1, true, 0, 0, 0, true, true, nullptr },
        { 24, 512, 288, 8, 1, true, 0, 144, 144, true, true, nullptr },
        { 24, 512, 288, 8, 1, true, 1, 96, 32, true, true, "3,7,90" },
        { 64, 512, 512, 16, 0, true, 0, 0, 0, false, false, nullptr },
        { 9, 512, 512, 16, 1, false, 1, 0, 0, true, true, nullptr },
        { 1, 288, 32, 32, 0, true, 1, 0, 0, true, false, nullptr },
        { 5, 300, 300, 10, 1, true, 2, 0, 0, true, false, "0,299" },
        /* The latent's codes in the same launch: E4M3 and int8 over the 128-wide group, IQ8R's 256,
         * at every tile form, with the gate, the inverse and a slice. */
        { 1, 512, 288, 8, 1, true, 0, 0, 0, true, true, nullptr, 128, false },
        { 6, 1536, 288, 8, 1, true, 1, 0, 0, true, false, nullptr, 128, true },
        { 16, 512, 288, 8, 1, false, 0, 0, 0, true, true, nullptr, 256, true },
        { 24, 512, 288, 8, 1, true, 0, 96, 32, true, true, "3,7,90", 256, true },
        { 48, 512, 288, 8, 1, true, 0, 0, 0, false, true, nullptr, 128, false },
        { 64, 512, 512, 16, 0, true, 0, 0, 0, false, false, nullptr, 128, true },
    };
    constexpr float kLogPoison = -7.25f;
    std::uniform_int_distribution<int> ud(-1000, 1000);
    for (const FusedShape& sh : shapes) {
        const int M = sh.M, K = sh.K, ne = sh.ne, k = sh.k, T = M * k;
        const int nl = sh.nl ? sh.nl : ne, base = sh.base;
        const long xp = K + 4, rwp = K + 2, yp = ne + 5, ip = k + 3, wp = k + 1, gp = 3;
        std::vector<uint16_t> x((size_t)M * xp);
        for (auto& v : x) v = f2b((float)ud(rng) / 256.f);
        if (sh.mode == 2)
            for (int r = 0; r < M; r += 2)
                for (int c = 0; c < K; ++c) x[(size_t)r * xp + c] = 0x7fc0; /* NaN: every rank */
        std::vector<float> wf, bias((size_t)ne * 2);
        std::vector<uint16_t> wb;
        if (sh.bf16w) {
            wb.resize((size_t)ne * rwp);
            for (auto& v : wb) v = f2b((float)ud(rng) / 2048.f);
        } else {
            wf.resize((size_t)ne * rwp);
            for (auto& v : wf) v = (float)ud(rng) / 2048.f;
        }
        for (auto& v : bias)
            v = sh.mode ? (float)(ud(rng) % 5) * 0.125f : (float)ud(rng) / 2000.f;
        std::vector<int> prot;
        if (sh.protect)
            for (const char* q = sh.protect; *q;) {
                char* end = nullptr;
                prot.push_back((int)std::strtol(q, &end, 10));
                q = *end == ',' ? end + 1 : end;
            }

        /* The row's operands: x, weight, bias, logits, ids, weights, sorted, off, cnt, gate?,
         * inverse? */
        RadTensor t[11]{};
        auto mat = [](RadTensor& d, void* p, uint32_t dt, int r, int c, long ld) {
            d.data = p;
            d.dtype = dt;
            d.rank = 2;
            d.shape[0] = r;
            d.shape[1] = c;
            d.stride[0] = ld;
            d.stride[1] = 1;
        };
        auto vec = [](RadTensor& d, void* p, long n) {
            d.data = p;
            d.dtype = RAD_I32;
            d.rank = 1;
            d.shape[0] = n;
            d.stride[0] = 1;
        };
        mat(t[0], x.data(), RAD_BF16, M, K, xp);
        mat(t[1], sh.bf16w ? (void*)wb.data() : (void*)wf.data(),
            sh.bf16w ? (uint32_t)RAD_BF16 : (uint32_t)RAD_F32, ne, K, rwp);
        t[2].data = bias.data();
        t[2].dtype = RAD_F32;
        t[2].rank = 1;
        t[2].shape[0] = ne;
        t[2].stride[0] = 2;
        mat(t[3], nullptr, RAD_F32, M, ne, yp);
        RadTensor& ids = t[4];
        RadTensor& ew = t[5];
        RadTensor& sorted = t[6];
        RadTensor& off = t[7];
        RadTensor& cnt = t[8];
        RadTensor& gate = t[9];
        RadTensor& inv = t[10];
        mat(ids, nullptr, RAD_I32, M, k, ip);
        mat(ew, nullptr, RAD_F32, M, k, wp);
        vec(sorted, nullptr, T);
        vec(off, nullptr, nl + 1);
        vec(cnt, nullptr, nl);
        mat(gate, nullptr, RAD_BF16, M, 1, gp);
        vec(inv, nullptr, T);

        RadParam p[11]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "K", RAD_P_INT, K };
        p[2] = { "n_expert", RAD_P_INT, ne };
        p[3] = { "n_local", RAD_P_INT, nl };
        p[4] = { "top_k", RAD_P_INT, k };
        p[5] = { "weight_bf16", RAD_P_INT, sh.bf16w ? 1 : 0 };
        p[6] = { "norm", RAD_P_INT, sh.norm };
        p[7].key = "route_scale";
        p[7].kind = RAD_P_F64;
        p[7].dval = 2.5;
        p[8].key = "dtype";
        p[8].kind = RAD_P_STR;
        p[8].sval = "f32";
        p[9] = { "expert_base", RAD_P_INT, base };
        p[10].key = "protect";
        p[10].kind = RAD_P_STR;
        p[10].sval = sh.protect;
        const int n_p = sh.protect ? 11 : 10;

        /* The latent's codes, when the row fills them: the served quantiser over the same
         * activation is the reference, pitched and guarded as the fused operands are. */
        const int hqg = sh.hqg, hqp = hqg ? K + 4 : 0, hsp = hqg ? K / hqg : 0;
        const uint32_t hqdt = sh.hqi8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3;
        std::vector<uint8_t> ref_q, q2;
        std::vector<float> ref_s, s2;

        /* The two launches the row replaces: the projection, then the selection over its logits. */
        std::vector<float> ref_lg((size_t)M * yp, kLogPoison), ref_w((size_t)M * wp, -7.25f);
        std::vector<int32_t> ref_ids((size_t)M * ip, kPoison), ref_sorted((size_t)T + 4, kPoison),
            ref_off((size_t)nl + 5, kPoison), ref_cnt((size_t)nl + 4, kPoison),
            ref_inv((size_t)T + 4, kPoison);
        std::vector<uint16_t> ref_gate((size_t)(M + 2) * gp, kGatePoison);
        {
            RadTensor g[3] = { t[0], t[1], t[3] };
            g[2].data = ref_lg.data();
            RadParam gq[5]{};
            gq[0] = { "M", RAD_P_INT, M };
            gq[1] = { "N", RAD_P_INT, ne };
            gq[2] = { "K", RAD_P_INT, K };
            gq[3].key = "dtype";
            gq[3].kind = RAD_P_STR;
            gq[3].sval = "bf16";
            gq[4] = { "weight_bf16", RAD_P_INT, sh.bf16w ? 1 : 0 };
            RadArgs ga{};
            ga.t = g;
            ga.n_t = 3;
            ga.p = gq;
            ga.n_p = 5;
            ++cases;
            const int rc = r4d_rad_gemm_nt_f32(&ga, nullptr);
            if (rc != RAD_OK) {
                std::printf("FAIL gemm_nt_f32 before the fused router at M=%d K=%d ne=%d (rc %d)\n",
                            M, K, ne, rc);
                ++fails;
                continue;
            }
            RadTensor s[9] = { t[3], t[2], t[4], t[5], t[6], t[7], t[8], t[9], t[10] };
            s[0].data = ref_lg.data();
            s[2].data = ref_ids.data();
            s[3].data = ref_w.data();
            s[4].data = ref_sorted.data();
            s[5].data = ref_off.data();
            s[6].data = ref_cnt.data();
            s[7].data = sh.gate ? ref_gate.data() : nullptr;
            s[8].data = ref_inv.data();
            RadArgs sa{};
            sa.t = s;
            sa.p = p;
            sa.n_p = n_p;
            sa.n_t = sh.inv ? 9 : (sh.gate ? 8 : 7);
            const int src = r4d_rad_router_sigmoid_topk_scatter(&sa, nullptr);
            if (src != RAD_OK) {
                std::printf("FAIL router_sigmoid_topk_scatter before the fused router at M=%d (rc "
                            "%d)\n",
                            M, src);
                ++fails;
                continue;
            }
        }
        if (hqg) {
            ref_q.assign((size_t)M * hqp + 8, 0x5A);
            ref_s.assign((size_t)M * hsp + 4, -7.25f);
            RadTensor qt[3] = { t[0], RadTensor{}, RadTensor{} };
            mat(qt[1], ref_q.data(), hqdt, M, K, hqp);
            mat(qt[2], ref_s.data(), RAD_F32, M, hsp, hsp);
            RadParam qp[4]{};
            qp[0] = { "M", RAD_P_INT, M };
            qp[1] = { "n", RAD_P_INT, K };
            qp[2] = { "group", RAD_P_INT, hqg };
            qp[3].key = "dtype";
            qp[3].kind = RAD_P_STR;
            qp[3].sval = "bf16";
            RadArgs qa{};
            qa.t = qt;
            qa.n_t = 3;
            qa.p = qp;
            qa.n_p = 4;
            ++cases;
            const int qrc = sh.hqi8 ? r4d_rad_had_quant_act_i8g(&qa, nullptr)
                                    : r4d_rad_had_quant_act_fp8(&qa, nullptr);
            if (qrc != RAD_OK) {
                std::printf("FAIL had_quant_act_%s before the fused router at M=%d K=%d g=%d "
                            "(rc %d)\n",
                            sh.hqi8 ? "i8g" : "fp8", M, K, hqg, qrc);
                ++fails;
                continue;
            }
        }

        for (int with_inv = 0; with_inv < (sh.inv ? 2 : 1); ++with_inv) {
            std::vector<float> lg((size_t)M * yp, kLogPoison), w((size_t)M * wp, -7.25f);
            std::vector<int32_t> ids2((size_t)M * ip, kPoison), sorted2((size_t)T + 4, kPoison),
                off2((size_t)nl + 5, kPoison), cnt2((size_t)nl + 4, kPoison),
                inv2((size_t)T + 4, kPoison);
            std::vector<uint16_t> g2((size_t)(M + 2) * gp, kGatePoison);
            RadTensor f[13];
            std::memcpy(f, t, sizeof t);
            f[3].data = lg.data();
            f[4].data = ids2.data();
            f[5].data = w.data();
            f[6].data = sorted2.data();
            f[7].data = off2.data();
            f[8].data = cnt2.data();
            f[9].data = sh.gate ? g2.data() : nullptr;
            f[10].data = inv2.data();
            f[11] = RadTensor{};
            f[12] = RadTensor{};
            RadParam p2[13];
            std::memcpy(p2, p, sizeof p);
            int n_p2 = n_p;
            if (hqg) {
                q2.assign(ref_q.size(), 0x5A);
                s2.assign(ref_s.size(), -7.25f);
                mat(f[11], q2.data(), hqdt, M, K, hqp);
                mat(f[12], s2.data(), RAD_F32, M, hsp, hsp);
                if (!with_inv) f[10].data = nullptr;   /* the codes' operands sit past both */
                p2[n_p2++] = { "hq_group", RAD_P_INT, hqg };
                p2[n_p2].key = "hq_dtype";
                p2[n_p2].kind = RAD_P_STR;
                p2[n_p2++].sval = sh.hqi8 ? "i8" : "f8e4m3";
            }
            RadArgs fa{};
            fa.t = f;
            fa.p = p2;
            fa.n_p = n_p2;
            fa.n_t = hqg ? 13 : (with_inv ? 11 : (sh.gate ? 10 : 9));
            ++cases;
            const int rc = r4d_rad_router_gemm_sigmoid_scatter(&fa, nullptr);
            int bad = 0;
            char why[16] = { 0 };
            int nwhy = 0;
            if (std::memcmp(lg.data(), ref_lg.data(), lg.size() * sizeof(float))) {
                ++bad;
                why[nwhy++] = 'l';
            }
            if (std::memcmp(ids2.data(), ref_ids.data(), ids2.size() * sizeof(int32_t))) {
                ++bad;
                why[nwhy++] = 'i';
            }
            if (std::memcmp(w.data(), ref_w.data(), w.size() * sizeof(float))) {
                ++bad;
                why[nwhy++] = 'w';
            }
            if (std::memcmp(sorted2.data(), ref_sorted.data(), sorted2.size() * 4)) {
                ++bad;
                why[nwhy++] = 's';
            }
            if (std::memcmp(off2.data(), ref_off.data(), off2.size() * 4)) {
                ++bad;
                why[nwhy++] = 'o';
            }
            if (std::memcmp(cnt2.data(), ref_cnt.data(), cnt2.size() * 4)) {
                ++bad;
                why[nwhy++] = 'c';
            }
            if (with_inv) {
                if (std::memcmp(inv2.data(), ref_inv.data(), inv2.size() * 4)) {
                    ++bad;
                    why[nwhy++] = 'v';
                }
            } else {
                for (int i = 0; i < T + 4; ++i)
                    if (inv2[(size_t)i] != kPoison) {
                        ++bad;
                        why[nwhy++] = 'V';
                        break;
                    }
            }
            if (hqg && (std::memcmp(q2.data(), ref_q.data(), q2.size()) ||
                        std::memcmp(s2.data(), ref_s.data(), s2.size() * sizeof(float)))) {
                ++bad;
                why[nwhy++] = 'q';
            }
            for (int r = 0; r < M + 2; ++r)
                for (int c = 0; c < gp; ++c) {
                    const uint16_t want = sh.gate && r < M && c == 0 ? 0x3f80 : kGatePoison;
                    if (g2[(size_t)r * gp + c] != want) {
                        ++bad;
                        why[nwhy++] = 'g';
                        r = M + 2;
                        break;
                    }
                }
            why[nwhy] = 0;
            if (rc != RAD_OK || bad) {
                std::printf(
                    "FAIL router_gemm_sigmoid_scatter at M=%d K=%d ne=%d k=%d mode %d slice %d+%d "
                    "%s weights%s%s%s%s (rc %d, %d mismatching outputs: %s)\n",
                    M, K, ne, k, sh.mode, base, nl, sh.bf16w ? "bf16" : "f32",
                    sh.protect ? " protect " : "", sh.protect ? sh.protect : "",
                    sh.gate ? " gate" : "", with_inv ? " with inverse" : "", rc, bad, why);
                ++fails;
            }
        }
    }
    /* Past the decode band: refused, nothing written. */
    {
        const int M = 65, K = 512, ne = 288, k = 8;
        std::vector<uint16_t> x((size_t)M * K, 0x3f80);
        std::vector<float> wh((size_t)ne * K, 0.f), bias((size_t)ne, 0.f),
            lg((size_t)M * ne, kLogPoison), w((size_t)M * k, -7.25f);
        std::vector<int32_t> ids((size_t)M * k, kPoison), sorted((size_t)M * k, kPoison),
            off((size_t)ne + 1, kPoison), cnt((size_t)ne, kPoison);
        RadTensor t[9]{};
        auto mat = [](RadTensor& d, void* p, uint32_t dt, int r, int c) {
            d.data = p;
            d.dtype = dt;
            d.rank = 2;
            d.shape[0] = r;
            d.shape[1] = c;
            d.stride[0] = c;
            d.stride[1] = 1;
        };
        auto vec = [](RadTensor& d, void* p, long n) {
            d.data = p;
            d.dtype = RAD_I32;
            d.rank = 1;
            d.shape[0] = n;
            d.stride[0] = 1;
        };
        mat(t[0], x.data(), RAD_BF16, M, K);
        mat(t[1], wh.data(), RAD_F32, ne, K);
        t[2].data = bias.data();
        t[2].dtype = RAD_F32;
        t[2].rank = 1;
        t[2].shape[0] = ne;
        t[2].stride[0] = 1;
        mat(t[3], lg.data(), RAD_F32, M, ne);
        mat(t[4], ids.data(), RAD_I32, M, k);
        mat(t[5], w.data(), RAD_F32, M, k);
        vec(t[6], sorted.data(), M * k);
        vec(t[7], off.data(), ne + 1);
        vec(t[8], cnt.data(), ne);
        RadParam p[10]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "K", RAD_P_INT, K };
        p[2] = { "n_expert", RAD_P_INT, ne };
        p[3] = { "n_local", RAD_P_INT, ne };
        p[4] = { "top_k", RAD_P_INT, k };
        p[5] = { "weight_bf16", RAD_P_INT, 0 };
        p[6] = { "norm", RAD_P_INT, 1 };
        p[7].key = "route_scale";
        p[7].kind = RAD_P_F64;
        p[7].dval = 2.5;
        p[8].key = "dtype";
        p[8].kind = RAD_P_STR;
        p[8].sval = "f32";
        RadArgs a{};
        a.t = t;
        a.n_t = 9;
        a.p = p;
        a.n_p = 9;
        ++cases;
        const int rc = r4d_rad_router_gemm_sigmoid_scatter(&a, nullptr);
        if (rc == RAD_OK || lg[0] != kLogPoison || ids[0] != kPoison || sorted[0] != kPoison) {
            std::printf("FAIL router_gemm_sigmoid_scatter served M=%d past the decode band (rc "
                        "%d)\n",
                        M, rc);
            ++fails;
        }
    }
}
}   // namespace

int main() {
    std::mt19937 rng(23);
    int fails = 0, cases = 0;
    router_scatter_cases(rng, cases, fails);
    gather_inverse_cases(rng, cases, fails);
    gather_sixteen_cases(rng, cases, fails);
    fused_router_cases(rng, cases, fails);
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
