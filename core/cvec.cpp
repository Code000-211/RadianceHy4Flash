/* cvec.cpp -- loading and validating a GGUF control vector (core/cvec.h). The reader is
 * core/text/gguf.cpp, which maps the file and sizes its tensors. */
#include "cvec.h"

#include "text/gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include "util/json.h"

namespace rad {
namespace {

/* FNV-1a over the table, so two different vectors never share a cache partition. */
uint64_t hash_table(const ControlVector& cv) {
    uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*) p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    mix(cv.dir.data(), cv.dir.size() * sizeof(float));
    mix(cv.scales.data(), cv.scales.size() * sizeof(float));
    /* The layout and mode are part of the identity: the same bytes read as one shared direction,
     * as a per-stream table, or as `add` vs `project` are different operators, and blocks computed
     * under one must never be served to a request steered by another. */
    const int64_t shape[3] = { cv.rank, cv.per_stream, cv.mode };
    mix(shape, sizeof(shape));
    if (cv.is_cast()) {
        mix(cv.cast_router.data(), cv.cast_router.size() * sizeof(float));
        mix(cv.cast_rmap.data(), cv.cast_rmap.size() * sizeof(int32_t));
        const int64_t cmeta[4] = { cv.cast_router_layer, cv.cast_n_routes, cv.cast_n_experts, cv.cast_think_after };
        mix(cmeta, sizeof(cmeta));
        mix(&cv.cast_min_margin, sizeof(cv.cast_min_margin));
    }
    return h ? h : 1ull;
}

}  /* namespace */

std::string control_vector_sidecar_path(const std::string& model_path) {
    const size_t slash = model_path.find_last_of("/\\");
    const size_t dot   = model_path.find_last_of('.');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return (has_ext ? model_path.substr(0, dot) : model_path) + ".control-vector.gguf";
}

namespace {

bool file_present(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    return f.tellg() > 0;
}

}  /* namespace */

int control_vector_discover(const std::string& model_path, const std::string& arch,
                            const std::string& radiance_home, std::string* found,
                            std::string* tried) {
    if (!found) return RAD_E_INVAL;
    found->clear();
    if (tried) tried->clear();

    std::vector<std::string> cand;
    const size_t slash = model_path.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? std::string(".") : model_path.substr(0, slash);
    if (!model_path.empty()) {
        cand.push_back(control_vector_sidecar_path(model_path));
        /* One file for a directory of containers: each quant of the family shares the directions,
         * since the file names the architecture and its width, not the quantisation. */
        cand.push_back(dir + "/control-vector.gguf");
    }
    if (!arch.empty() && !radiance_home.empty()) {
        size_t at = 0;
        while (at <= radiance_home.size()) {
            const size_t colon = radiance_home.find(':', at);
            const std::string home = radiance_home.substr(
                at, colon == std::string::npos ? std::string::npos : colon - at);
            if (!home.empty())
                cand.push_back(home + "/vectors/" + arch + "-control-vector.gguf");
            if (colon == std::string::npos) break;
            at = colon + 1;
        }
    }

    for (const std::string& p : cand) {
        if (tried) {
            if (!tried->empty()) *tried += ", ";
            *tried += p;
        }
        if (file_present(p)) { *found = p; return RAD_OK; }
    }
    return RAD_OK;
}


/* The package's own JSON beside the vector (`<stem>.json`), the only place the operation's
 * parameters are stated: `{"mode": "project", "scale": 1.0, "layer_range": [4, 44]}` for the
 * published Flash-Next vector, or `"mode": "add"` for additive steering. Absent, the defaults
 * apply (`project`, scale 1, full layer range). A file that is not JSON or names an unknown mode
 * is refused. */
namespace {

struct PackageJson {
    int64_t mode  = RAD_CVEC_MODE_PROJECT;
    int     first = 0, last = -1;   /* -1: not stated */
    float   scale = 1.0f;
    /* CAST metadata (required when mode == RAD_CVEC_MODE_CAST). */
    int64_t              cast_router_layer = 0;
    int64_t              cast_n_routes     = 0;
    int64_t              cast_n_experts    = 0;
    int64_t              cast_think_after  = 24;
    float                cast_min_margin   = 0.0f;
    std::vector<float>   cast_bias;
    std::vector<int32_t> cast_rmap;
    std::vector<std::string> cast_route_names;
};

int read_package_json(const std::string& gguf_path, PackageJson* out, std::string* why) {
    *out = PackageJson{};
    std::string stem = gguf_path;
    const size_t dot = stem.find_last_of('.');
    const size_t slash = stem.find_last_of("/\\");
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
        stem = stem.substr(0, dot);
    const std::string path = stem + ".json";
    std::ifstream f(path);
    if (!f) return RAD_OK;
    try {
        const Json j = Json::parse(f);
        if (const auto m = j.find("mode"); m != j.end()) {
            if (!m->is_string()) {
                *why = "control vector " + path + ": 'mode' must be a string ('project', 'add', or 'cast')";
                return RAD_E_INVAL;
            }
            const std::string mode_s = m->get<std::string>();
            if (mode_s == "project") {
                out->mode = RAD_CVEC_MODE_PROJECT;
            } else if (mode_s == "add") {
                out->mode = RAD_CVEC_MODE_ADD;
            } else if (mode_s == "cast") {
                out->mode = RAD_CVEC_MODE_CAST;
            } else {
                *why = "control vector " + path + ": mode '" + mode_s +
                       "'; only 'project', 'add', and 'cast' are implemented";
                return RAD_E_UNSUPPORTED;
            }
        }
        if (const auto r = j.find("layer_range"); r != j.end() && r->is_array() && r->size() == 2 &&
                                                  (*r)[0].is_number_integer() &&
                                                  (*r)[1].is_number_integer()) {
            out->first = (*r)[0].get<int>();
            out->last  = (*r)[1].get<int>();
        }
        if (const auto s = j.find("scale"); s != j.end() && s->is_number())
            out->scale = (float) s->get<double>();
        if (out->mode == RAD_CVEC_MODE_CAST) {
            const auto cj = j.find("cast");
            if (cj == j.end() || !cj->is_object()) {
                *why = "control vector " + path + ": mode 'cast' requires a 'cast' object in JSON";
                return RAD_E_INVAL;
            }
            out->cast_router_layer = cj->value("router_layer", (int64_t)0);
            out->cast_n_routes     = cj->value("n_routes", (int64_t)0);
            out->cast_n_experts    = cj->value("n_experts", (int64_t)0);
            out->cast_think_after  = cj->value("think_after", (int64_t)24);
            out->cast_min_margin   = (float) cj->value("min_margin", 0.0);
            if (const auto b = cj->find("router_bias"); b != cj->end() && b->is_array()) {
                for (const auto& v : *b) out->cast_bias.push_back((float) v.get<double>());
            }
            if (const auto rm = cj->find("route_map"); rm != cj->end() && rm->is_array()) {
                for (const auto& row : *rm) {
                    if (!row.is_array() || row.size() != 3) {
                        *why = "control vector " + path + ": each cast.route_map entry must be [think_early, think_late, output]";
                        return RAD_E_INVAL;
                    }
                    for (int q = 0; q < 3; ++q) out->cast_rmap.push_back(row[q].get<int32_t>());
                }
            }
            if (const auto rn = cj->find("route_names"); rn != cj->end() && rn->is_array()) {
                for (const auto& nm : *rn)
                    if (nm.is_string()) out->cast_route_names.push_back(nm.get<std::string>());
            }
        }
    } catch (const std::exception& e) {
        *why = "control vector " + path + ": not readable JSON (" + e.what() + ")";
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

}  /* namespace */

int control_vector_load(const std::string& path, const std::string& arch, int64_t n_embd,
                        int64_t n_model_layers, ControlVector* out, std::string* err) {
    const auto fail = [err](const std::string& why) {
        if (err) *err = why;
        return RAD_E_INVAL;
    };
    if (!out) return RAD_E_INVAL;
    *out = ControlVector{};

    GgufFile f;
    const int rc = f.open(path.c_str());
    if (rc != RAD_OK)
        return fail("control vector " + path + ": cannot read it as a GGUF container");

    std::string a;
    if (!f.get_str("general.architecture", &a) || a != "controlvector")
        return fail("control vector " + path + ": general.architecture is '" + a +
                    "', not 'controlvector' -- this is not a control vector");

    /* The hint is optional in llama.cpp's generator, but when a file states one it must be this
     * model's architecture: a direction is a basis vector inferred from that architecture's
     * residual stream, and nothing else. */
    std::string hint;
    if (f.get_str("controlvector.model_hint", &hint) && !hint.empty() && !arch.empty() &&
        hint != arch)
        return fail("control vector " + path + ": the vector is for architecture '" + hint +
                    "', this container is '" + arch + "' -- a direction from another model's "
                    "residual basis would move hidden states along a meaningless axis");

    /* The file's directions: direction.<l>, one or more rows, F32, n_embd wide. In `cast` mode,
     * the GGUF also carries `router.mu` ([n_embd, 4]) and `router.w` ([n_embd, 4 * n_routes]),
     * validated below after the package JSON is read. */
    int64_t n_dir = 0, dir_first = -1, dir_last = -1;
    int64_t file_rank = 0;
    for (const GgufTensor& t : f.tensors()) {
        if (t.name == "router.mu" || t.name == "router.w") continue;
        if (t.name.rfind("direction.", 0) != 0)
            return fail("control vector " + path + ": tensor '" + t.name +
                        "' is not a direction.<layer> entry");
        int64_t layer = 0;
        try {
            layer = std::stoll(t.name.substr(10));
        } catch (...) {
            return fail("control vector " + path + ": tensor name '" + t.name +
                        "' does not end in a layer number");
        }
        /* 1 .. n_layers-1: the table is indexed by layer (row 0 is unused, the format carrying no
         * direction for it), so direction.<n_layers> would write one row past it. */
        if (layer < 1 || layer >= n_model_layers)
            return fail("control vector " + path + ": direction." + std::to_string(layer) +
                        " is outside this model's " + std::to_string(n_model_layers) + " layers");
        int64_t k = 1;
        if (t.n_dims == 1) {
            k = 1;
        } else if (t.n_dims == 2 && t.ne[1] >= 1) {
            k = t.ne[1];
        } else {
            return fail("control vector " + path + ": " + t.name + " has rank " +
                        std::to_string(t.n_dims) + "; a direction is one row, or [n_embd, k] for "
                        "k directions");
        }
        if (t.ne[0] != n_embd)
            return fail("control vector " + path + ": " + t.name + " is " +
                        std::to_string(t.ne[0]) + " wide; this model's residual streams are " +
                        std::to_string(n_embd));
        if (gguf_to_rad_dtype(t.ggml_type) != RAD_F32)
            return fail("control vector " + path + ": " + t.name +
                        " is not F32 -- only an f32 direction is read");
        if (k > kControlVectorMaxRank)
            return fail("control vector " + path + ": " + t.name + " carries " +
                        std::to_string(k) + " directions; at most " +
                        std::to_string(kControlVectorMaxRank) +
                        " are applied, one per declared hook");
        if (file_rank == 0) file_rank = k;
        else if (k != file_rank)
            return fail("control vector " + path + ": " + t.name + " carries " +
                        std::to_string(k) + " directions and another layer carries " +
                        std::to_string(file_rank) + "; every layer must carry the same count");
        if (dir_first < 0 || layer < dir_first) dir_first = layer;
        if (layer > dir_last)                  dir_last  = layer;
        ++n_dir;
    }
    if (n_dir == 0) return fail("control vector " + path + ": the file holds no directions");

    /* The package's parameters: its JSON, else every direction the file carries at scale 1, as
     * llama.cpp applies a control vector given no layer range. */
    PackageJson pj;
    std::string pj_why;
    if (const int prc = read_package_json(path, &pj, &pj_why); prc != RAD_OK) return fail(pj_why);
    int first = pj.first, last = pj.last;
    float scale = pj.scale;
    const int64_t mode = pj.mode;
    if (last < 0) { first = (int) dir_first; last = (int) dir_last; }
    if (first < 1 || last < first)
        return fail("control vector " + path + ": the layer range " + std::to_string(first) + ":" +
                    std::to_string(last) + " is empty");
    if (last >= n_model_layers)
        return fail("control vector " + path + ": the layer range " + std::to_string(first) + ":" +
                    std::to_string(last) + " runs past this model's " +
                    std::to_string(n_model_layers) + " layers");
    if (mode == RAD_CVEC_MODE_ADD || mode == RAD_CVEC_MODE_CAST) {
        if (scale == 0.0f || !std::isfinite(scale))
            return fail("control vector " + path + ": its scale is not a finite non-zero number");
    } else {
        if (!(scale > 0.0f) || !std::isfinite(scale))
            return fail("control vector " + path + ": its scale is not a positive number");
    }

    out->n_embd = n_embd;
    /* Indexed by layer: rows 0..n_layers-1. Row 0 stays zero, the format carrying no direction
     * for layer 0, and every row a model can name exists. */
    out->n_layers = n_model_layers;
    out->n_dirs = n_dir;
    out->rank = file_rank;
    /* The package says whether its rows are per-stream. A per-stream file carries
     * `rank = streams * groups` rows per layer; the engine steers group k of stream s with
     * its own vector. Confirmed against the row count so a mislabelled file is refused. */
    int64_t per_stream_kv = 0;
    const bool per_stream = f.get_i64("controlvector.per_stream", &per_stream_kv) && per_stream_kv;
    if (per_stream && (file_rank % 4) != 0)
        return fail("control vector " + path + ": it is marked per_stream but carries " +
                    std::to_string(file_rank) + " rows per layer, which is not a multiple of 4 "
                    "hyper-connection streams");
    out->per_stream = per_stream ? 1 : 0;
    out->mode = mode;
    out->first = first;
    out->last = last;
    out->user_scale = scale;
    out->path = path;
    out->dir.assign((size_t) (out->n_layers * file_rank * n_embd), 0.0f);
    out->scales.assign((size_t) out->n_layers, 0.0f);

    if (mode == RAD_CVEC_MODE_CAST) {
        if (!per_stream)
            return fail("control vector " + path + ": cast mode requires controlvector.per_stream=1");
        const int64_t n_routes  = pj.cast_n_routes;
        const int64_t n_experts = pj.cast_n_experts;
        const int64_t rlayer    = pj.cast_router_layer;
        if (n_routes < 1 || n_routes > 8 || n_experts < 1 || n_experts > 8)
            return fail("control vector " + path + ": cast n_routes and n_experts must be in 1..8");
        if (rlayer < 1 || rlayer >= n_model_layers)
            return fail("control vector " + path + ": cast router_layer out of range");
        if (file_rank != 4 * n_experts)
            return fail("control vector " + path + ": cast rank must equal 4 * n_experts");
        if ((int64_t) pj.cast_bias.size() != n_routes || (int64_t) pj.cast_rmap.size() != n_routes * 3)
            return fail("control vector " + path + ": cast router_bias / route_map size mismatch");

        const GgufTensor* t_mu = f.tensor("router.mu");
        const GgufTensor* t_w  = f.tensor("router.w");
        if (!t_mu || !t_w)
            return fail("control vector " + path + ": cast mode requires 'router.mu' and 'router.w' tensors");
        if (t_mu->n_dims != 2 || t_mu->ne[0] != n_embd || t_mu->ne[1] != 4 ||
            gguf_to_rad_dtype(t_mu->ggml_type) != RAD_F32)
            return fail("control vector " + path + ": router.mu must be F32 [n_embd, 4]");
        if (t_w->n_dims != 2 || t_w->ne[0] != n_embd || t_w->ne[1] != 4 * n_routes ||
            gguf_to_rad_dtype(t_w->ggml_type) != RAD_F32)
            return fail("control vector " + path + ": router.w must be F32 [n_embd, 4*n_routes]");
        const float* mu_p = (const float*) f.tensor_data(*t_mu);
        const float* w_p  = (const float*) f.tensor_data(*t_w);
        if (!mu_p || !w_p)
            return fail("control vector " + path + ": router tensors lie outside the file");

        const int64_t hn = 4 * n_embd;
        out->cast_router.resize((size_t) ((1 + n_routes) * hn + n_routes));
        for (int64_t i = 0; i < hn; ++i) {
            if (!std::isfinite(mu_p[i])) return fail("control vector " + path + ": non-finite router.mu");
            out->cast_router[(size_t) i] = mu_p[i];
        }
        for (int64_t i = 0; i < n_routes * hn; ++i) {
            if (!std::isfinite(w_p[i])) return fail("control vector " + path + ": non-finite router.w");
            out->cast_router[(size_t) (hn + i)] = w_p[i];
        }
        for (int64_t r = 0; r < n_routes; ++r) {
            if (!std::isfinite(pj.cast_bias[(size_t) r]))
                return fail("control vector " + path + ": non-finite router_bias");
            out->cast_router[(size_t) ((1 + n_routes) * hn + r)] = pj.cast_bias[(size_t) r];
        }
        out->cast_rmap         = pj.cast_rmap;
        out->cast_router_layer = rlayer;
        out->cast_n_routes     = n_routes;
        out->cast_n_experts    = n_experts;
        out->cast_think_after  = pj.cast_think_after;
        out->cast_min_margin   = pj.cast_min_margin;
        out->cast_route_names  = pj.cast_route_names;
    } else {
        if (f.tensor("router.mu") || f.tensor("router.w"))
            return fail("control vector " + path + ": router tensors require mode 'cast'");
    }

    std::vector<uint8_t> has_dir((size_t) out->n_layers, 0);
    double worst = 0.0, worst_dot = 0.0;
    for (int64_t l = 1; l < n_model_layers; ++l) {
        const GgufTensor* t = f.tensor("direction." + std::to_string(l));
        if (!t) continue;
        const float* d = (const float*) f.tensor_data(*t);
        if (!d) return fail("control vector " + path + ": " + t->name + " lies outside the file");
        has_dir[(size_t) l] = 1;

        if (mode == RAD_CVEC_MODE_ADD || mode == RAD_CVEC_MODE_CAST) {
            /* Additive control vectors keep their raw per-layer / per-stream F32 magnitudes
             * unnormalised; zero rows are legal (no-op on that stream/layer), only NaN/Inf is
             * rejected. */
            for (int64_t k = 0; k < file_rank; ++k) {
                const float* src = d + (size_t) k * (size_t) n_embd;
                float* dst = &out->dir[(size_t) ((l * file_rank + k) * n_embd)];
                for (int64_t i = 0; i < n_embd; ++i) {
                    if (!std::isfinite(src[i]))
                        return fail("control vector " + path + ": " + t->name + " direction " +
                                    std::to_string(k) + " has a non-finite element");
                    dst[i] = src[i];
                }
            }
            out->scales[(size_t) l] = (l >= first && l <= last) ? scale : 0.0f;
            continue;
        }

        double first_norm = 1.0;
        for (int64_t k = 0; k < file_rank; ++k) {
            const float* src = d + (size_t) k * (size_t) n_embd;
            double nn = 0.0;
            for (int64_t i = 0; i < n_embd; ++i) nn += (double) src[i] * (double) src[i];
            const double norm = std::sqrt(nn);
            if (!(norm > 0.0) || !std::isfinite(norm))
                return fail("control vector " + path + ": " + t->name + " direction " +
                            std::to_string(k) + " has no direction");
            if (k == 0) first_norm = norm;

            float* dst = &out->dir[(size_t) ((l * file_rank + k) * n_embd)];
            const float inv = (float) (1.0 / norm);
            for (int64_t i = 0; i < n_embd; ++i) dst[i] = src[i] * inv;
            if (l >= first && l <= last) worst = std::max(worst, std::fabs(norm - 1.0));
            /* A basis is applied as k chained projections. Rows that are not orthogonal do not
             * remove the k-space the package was derived for: the second projection takes the part
             * of the stream along v_2 that v_1 left, and removes it whole. */
            /* A per-stream package's rows are groups of one direction per stream, and only the
             * rows of one stream are chained on one stream: row k meets rows k-4, k-8, ... */
            const int64_t step = per_stream ? 4 : 1;
            for (int64_t j = k - step; j >= 0; j -= step) {
                const float* b = &out->dir[(size_t) ((l * file_rank + j) * n_embd)];
                double dot = 0.0;
                for (int64_t i = 0; i < n_embd; ++i) dot += (double) dst[i] * (double) b[i];
                if (l >= first && l <= last) worst_dot = std::max(worst_dot, std::fabs(dot));
            }
        }
        /* One strength per layer. A single direction keeps the original `norm * scale` (the file's
         * own magnitude is the strength); an orthonormal rank-k basis removes each component whole,
         * so its strength is the package scale and the rows must be normal (checked below). */
        out->scales[(size_t) l] =
            (l >= first && l <= last)
                ? (file_rank == 1 ? (float) (first_norm * scale) : scale)
                : 0.0f;

        /* A layer inside the range with no direction would silently be unsteered. */
    }
    for (int64_t l = first; l <= last; ++l)
        if (!has_dir[(size_t) l] || out->scales[(size_t) l] == 0.0f)
            return fail("control vector " + path + ": the range " + std::to_string(first) + ":" +
                        std::to_string(last) + " includes layer " + std::to_string(l) +
                        ", which has no direction in the file");

    if (mode == RAD_CVEC_MODE_PROJECT && worst > 1e-2)
        RAD_WARN("control vector: %s: the directions are not unit-normalised (worst |norm - 1| "
                 "= %.4f); each is divided by its own norm, so the operation is unchanged",
                 path.c_str(), worst);
    if (mode == RAD_CVEC_MODE_PROJECT && worst_dot > 1e-2)
        RAD_WARN("control vector: %s: the rows of a layer are not orthogonal (worst |v_i.v_j| = "
                 "%.4f); the chained projections then remove more than the intended %lld-space",
                 path.c_str(), worst_dot, (long long) (per_stream ? file_rank / 4 : file_rank));
    return RAD_OK;
}

namespace {
struct CastDevLayout {
    int64_t off_router    = 0;
    int64_t off_rmap      = 0;
    int64_t off_seq_route = 0;
    int64_t total_bytes   = 0;
};

int64_t align64(int64_t n) { return (n + 63) & ~int64_t(63); }

CastDevLayout cast_dev_layout(const ControlVector& cv) {
    CastDevLayout l{};
    const int64_t dir_bytes = (int64_t) cv.dir.size() * 4;
    if (!cv.is_cast()) {
        l.total_bytes = dir_bytes;
        return l;
    }
    l.off_router    = align64(dir_bytes);
    l.off_rmap      = align64(l.off_router + (int64_t) cv.cast_router.size() * 4);
    l.off_seq_route = align64(l.off_rmap + (int64_t) cv.cast_rmap.size() * 4);
    l.total_bytes   = align64(l.off_seq_route + kCastMaxSlots * 4);
    return l;
}
}  /* namespace */

RadControlVector control_vector_view(const ControlVector& cv, const float* dir_device) {
    RadControlVector v{};
    v.dir = dir_device;
    v.scales = cv.scales.data();
    v.n_embd = cv.n_embd;
    v.n_layers = cv.n_layers;
    v.rank = cv.rank > 0 ? cv.rank : 1;
    v.per_stream = cv.per_stream ? 1 : 0;
    v.mode = cv.mode;
    if (cv.is_cast() && dir_device) {
        const CastDevLayout lay = cast_dev_layout(cv);
        const char* base = (const char*) dir_device;
        v.cast_router       = (const float*)   (base + lay.off_router);
        v.cast_rmap         = (const int32_t*) (base + lay.off_rmap);
        v.cast_seq_route    = (int32_t*)       (const_cast<char*>(base) + lay.off_seq_route);
        v.cast_router_layer = cv.cast_router_layer;
        v.cast_n_routes     = cv.cast_n_routes;
        v.cast_n_experts    = cv.cast_n_experts;
        v.cast_think_after  = cv.cast_think_after;
        v.cast_max_slots    = kCastMaxSlots;
        v.cast_min_margin   = cv.cast_min_margin;
    }
    return v;
}

const float* control_vector_upload(const ControlVector& cv, int rank) {
    if (!cv.loaded()) return nullptr;
    const CastDevLayout lay = cast_dev_layout(cv);
    const int64_t bytes = lay.total_bytes;
    void* d = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    if (!d) {
        RAD_ERR("control vector: rank %d: could not allocate %.2f MiB on the card for the "
                "directions", rank, (double) bytes / (1024.0 * 1024.0));
        return nullptr;
    }
    RadStream s = nullptr;
    if (rad_stream_create(&s, 0) < 0) {
        rad_dev_free(d, RAD_MEM_DEVICE);
        return nullptr;
    }
    int rc = RAD_OK;
    if (!cv.is_cast()) {
        rc = rad_memcpy_async(d, cv.dir.data(), bytes, s);
    } else {
        std::vector<uint8_t> staging((size_t) bytes, 0);
        std::memcpy(staging.data(), cv.dir.data(), cv.dir.size() * sizeof(float));
        std::memcpy(staging.data() + lay.off_router, cv.cast_router.data(),
                    cv.cast_router.size() * sizeof(float));
        std::memcpy(staging.data() + lay.off_rmap, cv.cast_rmap.data(),
                    cv.cast_rmap.size() * sizeof(int32_t));
        std::vector<int32_t> init_slots((size_t) kCastMaxSlots, -1);
        std::memcpy(staging.data() + lay.off_seq_route, init_slots.data(),
                    init_slots.size() * sizeof(int32_t));
        rc = rad_memcpy_async(d, staging.data(), bytes, s);
    }
    rad_stream_sync(s);
    rad_stream_destroy(s);
    if (rc < 0) {
        RAD_ERR("control vector: rank %d: uploading the directions failed: %s", rank,
                rad_dev_last_error());
        rad_dev_free(d, RAD_MEM_DEVICE);
        return nullptr;
    }
    return (const float*) d;
}

uint64_t control_vector_salt(const ControlVector& cv) {
    if (!cv.loaded()) return 0;
    return hash_table(cv) | 1ull;
}

}  /* namespace rad */
