/* kvmove.h -- the device side of a KV loan compaction (M-15b).
 *
 * core/mem decides which blocks move (PrefixCache::plan_compaction, KVManager::
 * relocate_cached_block) and names each move as a BlockCopy; this issues the bytes, as TierExec
 * does for the tiers. core/mem has no stream (spec §5.4).
 *
 * A paged group is layer-major: layer l owns `[base + l * layer_stride, + layer_stride)` and a
 * block in it is one fragment of `layer_bytes_per_block()` at `block * fragment`. A block is
 * therefore `n_layers` fragments at the layer stride, which is one 2-D copy, and a run of blocks
 * whose sources and destinations are both consecutive is one 2-D copy `n * fragment` wide. */
#pragma once
#include "mem/kv.h"
#include "rad_core.h"

#include <cstdint>
#include <vector>

namespace rad {

/* Enqueues every copy on `s`, in this rank's pool, after coalescing them into runs; returns how
 * many 2-D copies it issued, or a negative status from the first that failed. The copies must
 * not overlap: every destination free when it was taken and every source held, as
 * relocate_cached_block guarantees. A group with no bytes on this rank (rad_kv_group_zero), a
 * streamed group and a group that is not paged move nothing here; their ids still moved. */
int64_t issue_block_copies(const KVManager& kv, const std::vector<BlockCopy>& copies, RadStream s);

}  /* namespace rad */
