#include "runtime/kvmove.h"

#include "rad_internal.h"

#include <algorithm>

namespace rad {

int64_t issue_block_copies(const KVManager& kv, const std::vector<BlockCopy>& copies,
                           RadStream s) {
    struct Run { int32_t g, src, dst, n; };
    std::vector<BlockCopy> m(copies);
    std::sort(m.begin(), m.end(), [](const BlockCopy& a, const BlockCopy& b) {
        return a.group != b.group ? a.group < b.group : a.src_block < b.src_block;
    });
    std::vector<Run> runs;
    for (const BlockCopy& c : m) {
        if (c.group < 0 || c.src_block < 0 || c.dst_block < 0) continue;
        if (!runs.empty()) {
            Run& u = runs.back();
            if (u.g == c.group && c.src_block == u.src + u.n && c.dst_block == u.dst + u.n) {
                ++u.n;
                continue;
            }
        }
        runs.push_back(Run{ c.group, c.src_block, c.dst_block, 1 });
    }
    int64_t issued = 0;
    for (const Run& u : runs) {
        const KVGroupPlan* p = kv.plan(u.g);
        if (!p || !p->paged() || p->zero || p->streamed() || !p->base) continue;
        const int64_t frag  = p->layer_bytes_per_block();
        const int64_t pitch = p->layer_stride();
        const int64_t nl    = p->n_layers > 0 ? p->n_layers : 1;
        if (frag <= 0 || (int64_t)u.src + u.n > p->pool_blocks() ||
            (int64_t)u.dst + u.n > p->pool_blocks())
            return RAD_E_INVAL;
        char* base = static_cast<char*>(p->base);
        const int st = rad_memcpy_2d_async(base + (int64_t)u.dst * frag, pitch,
                                           base + (int64_t)u.src * frag, pitch,
                                           frag * u.n, nl, s);
        if (st < 0) return st;
        ++issued;
    }
    return issued;
}

}  /* namespace rad */
