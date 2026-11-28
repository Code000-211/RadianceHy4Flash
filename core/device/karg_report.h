/* karg_report.h -- what the fault report says about the launch it names (issue #337).
 *
 * A queue's read index stands on a packet until the packet completes, so the packet it points at
 * is the launch that faulted and that launch's argument slot still holds its arguments. The report
 * used to scan the slot for pointer-shaped words: it printed the leftovers of earlier launches in
 * the same slot as "argument +N" (a slot is 1 KiB, a segment is at most 768 bytes, and only the
 * segment is written), it could not say anything at all about a word it judged not to be a pointer
 * (so the three sizes and the chained-launch record, the two things a reader needs, were the three
 * lines it dropped), and it never printed the chain record at RAD_CHAIN_KARG_OFF, which is the only
 * metadata a chained kernel acts on. Three readers of issue #337's own dump decoded the same
 * arguments into three different shapes because of it.
 *
 * Here every line is one parameter of the launch, in the code object's own order, taken from the
 * metadata (KernelArg: offset, size, value_kind, name): an address is printed with the location the
 * dump already knows how to find, and a value is printed as the number it is. The chain record,
 * when the slot carries one, comes last with its target, its counter and where that counter is.
 *
 * Card-free: it is bytes and metadata, and the caller supplies the address lookup, so
 * tests/karg_report_test.cpp covers the lines without a GPU. */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "aql.h"

namespace rad {
namespace aql {

/* One launch's argument slot. `kernarg_size` is the launch's own segment length: its parameters
 * end there, and the bytes after it in the slot belong to whichever launch used the slot before. */
struct KargSlot {
    const unsigned char*          base = nullptr;
    uint32_t                      bytes = 0;        /* readable bytes at `base` (one slot) */
    uint32_t                      kernarg_size = 0; /* this launch's segment */
    const std::vector<KernelArg>* args = nullptr;   /* the code object's layout, in order */
};

/* The lines for that launch: one per parameter of the segment, then the chain record's line when
 * the slot holds a valid record. `where` locates an address (the dump's own lookup: a pageable
 * chunk, a live range, a pinned piece, "in no range ..."). Empty when there is no layout to print,
 * in which case the caller prints the slot's pointer-shaped words as it did before. */
std::vector<std::string> karg_lines(const KargSlot& slot,
                                    const std::function<std::string(uintptr_t)>& where);

}  /* namespace aql */
}  /* namespace rad */
