/* karg_report.h -- what the fault report says about a launch (issue #337).
 *
 * A packet's argument slot holds its launch's arguments until the engine reuses the slot, so the
 * report can print them exactly. Which packet to print is a separate question and aql_inflight.h
 * answers it: the read index is not a completion mark, so the packet it points at is not
 * necessarily the launch that faulted. The report
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

/* The live range an address lies in, as the caller knows them: a record number that is equal for
 * two addresses in one range, and the range's bounds. `rec` 0 means no range holds the address. */
struct KargRange {
    uint64_t  rec = 0;
    uintptr_t lo = 0;
    uintptr_t hi = 0;
};

/* One launch's argument slot. `kernarg_size` is the launch's own segment length: its parameters
 * end there, and the bytes after it in the slot belong to whichever launch used the slot before. */
struct KargSlot {
    const unsigned char*          base = nullptr;
    uint32_t                      bytes = 0;        /* readable bytes at `base` (one slot) */
    uint32_t                      kernarg_size = 0; /* this launch's segment */
    bool                          chained = false;  /* the code object reads the record at +992 */
    const std::vector<KernelArg>* args = nullptr;   /* the code object's layout, in order */

    /* The fault address this dump is about, and the live range it can have run off (the caller
     * numbers the ranges it knows; 0 means none). fault_ran_off is the test: the address lies in
     * that range, or no more than the range's own length past its end. With `range_of`, a parameter
     * that points into that same range gets the distance named on its line -- that parameter is the
     * launch the fault address points at, which the ring's order can only guess at (issue #337).
     * Zero and empty for a stall dump, which has no fault address. */
    uintptr_t                     fault = 0;
    uint64_t                      fault_rec = 0;
    std::function<KargRange(uintptr_t)> range_of;
};

/* The lines for that launch: one per parameter of the segment, then the record's line (this launch
 * waited on it, it was not chained, or no record is valid). `where` locates an address (the dump's
 * own lookup: a pageable chunk, a live range, a pinned piece, "in no range ..."). Empty when there
 * is no layout to print, in which case the caller prints the slot's pointer-shaped words as it did
 * before. */
std::vector<std::string> karg_lines(const KargSlot& slot,
                                    const std::function<std::string(uintptr_t)>& where);

}  /* namespace aql */
}  /* namespace rad */
