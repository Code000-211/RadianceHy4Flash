/* karg_report.cpp -- the fault report's argument lines (karg_report.h, issue #337). */
#include "karg_report.h"

#include <cinttypes>
#include <cstring>

#include "rad_abi.h"
#include "../rad_internal.h"

namespace rad {
namespace aql {
namespace {

uint64_t load(const unsigned char* p, uint32_t n) {
    uint64_t v = 0;
    std::memcpy(&v, p, n < 8 ? n : 8);
    return v;
}

/* Whether a parameter's bytes are an address or a number. The metadata reader decides it once, from
 * the value_kind, into `KernelArg::address` (aql_code.cpp: kind_is_address); the report reads that
 * flag instead of spelling the kinds a second time, where the two lists could drift apart. A number
 * shown as "in no range this process holds" is the line this report exists to remove, so a kind
 * nobody named is printed as the number it may be. */
bool is_address(const KernelArg& a) {
    return a.address;
}

}  /* namespace */

std::vector<std::string> karg_lines(const KargSlot& slot,
                                    const std::function<std::string(uintptr_t)>& where) {
    std::vector<std::string> out;
    if (!slot.base || !slot.args || slot.args->empty() || slot.bytes == 0) return out;

    for (size_t i = 0; i < slot.args->size(); ++i) {
        const KernelArg& a = (*slot.args)[i];
        if ((uint64_t)a.offset + a.size > slot.bytes) {
            out.push_back(fmt("the slot is %u bytes and parameter %zu runs to +%u, past its end",
                              slot.bytes, i, a.offset + a.size));
            break;
        }
        if ((uint64_t)a.offset + a.size > slot.kernarg_size) {
            /* The launch's segment ends before this parameter: what is here is another launch's. */
            out.push_back(fmt("this launch's segment ends at +%u, before parameter %zu at +%u; "
                              "nothing past it belongs to this launch",
                              slot.kernarg_size, i, a.offset));
            break;
        }
        std::string head;
        if (a.hidden != Hidden::None) {
            /* The metadata's own spelling of an implicit argument (`hidden_block_count_x`): the
             * reader keeps it in `KernelArg::kind` and `hidden_kind` maps it to the enum, so
             * reading it here is what keeps the report and the refusal message from drifting
             * into two spellings of one slot. */
            head = fmt("%s (%u bytes at +%u)", a.kind.empty() ? "hidden" : a.kind.c_str(), a.size,
                       a.offset);
        } else {
            const std::string name = a.name.empty() ? fmt("%zu", i) : a.name;
            head = fmt("argument %s (%s, %u bytes at +%u)", name.c_str(),
                       a.kind.empty() ? "by_value" : a.kind.c_str(), a.size, a.offset);
        }
        if (a.hidden == Hidden::Pad || a.size == 0) {
            out.push_back(head);
        } else if (is_address(a)) {
            const uint64_t v = load(slot.base + a.offset, a.size);
            if (v == 0) {
                out.push_back(head + " = 0 (null)");
            } else {
                std::string line = head + fmt(" = 0x%" PRIx64 ": ", v) +
                                   (where ? where((uintptr_t)v) : std::string("not located"));
                /* Does this parameter point into the range the fault address is nearest to? Then
                 * this launch is the one the fault address points at: the read ran off the end of
                 * a buffer this launch reads, or into one it wrote. That is worth a line of its
                 * own, because the ring's order cannot say which launch faulted (#337). */
                if (slot.fault && slot.range_of && slot.fault_rec) {
                    const KargRange r = slot.range_of((uintptr_t)v);
                    if (r.rec && r.rec == slot.fault_rec) {
                        const uintptr_t f = slot.fault;
                        line += f < r.lo ? fmt("; the fault address is %s before the start of this"
                                               " same range",
                                               humanb((int64_t)(r.lo - f)).c_str())
                              : f >= r.hi ? fmt("; the fault address is %s past the end of this"
                                                " same range",
                                                humanb((int64_t)(f - r.hi)).c_str())
                                          : "; the fault address is inside this same range";
                    }
                }
                out.push_back(line);
            }
        } else {
            out.push_back(head + fmt(" = %" PRIu64, load(slot.base + a.offset, a.size)));
        }
    }

    /* The chained-launch record: the kernel reads it at RAD_CHAIN_KARG_OFF and waits for that
     * counter if and only if the record's magic and check hold (libr4d/r4d_chain.h). Only the
     * chained form of a kernel reads it, and the backend writes the record for exactly those
     * launches (`if (k->chained) chain_record(...)`), so that is the only case in which the bytes
     * there are this launch's. A plain kernel's slot still holds whatever record an earlier
     * chained launch in it left: saying the launch waited would be a claim about a launch that
     * never read the slot at all. */
    if ((uint64_t)RAD_CHAIN_KARG_OFF + RAD_CHAIN_REC_BYTES <= slot.bytes) {
        if (!slot.chained) {
            out.push_back(fmt("chain record at +%d: not read, this launch is the plain form of its "
                              "kernel; what stands there belongs to an earlier launch in this slot",
                              (int)RAD_CHAIN_KARG_OFF));
            return out;
        }
        const unsigned char* p = slot.base + RAD_CHAIN_KARG_OFF;
        const uint64_t magic = load(p, 8), target = load(p + 8, 8);
        const uint64_t ctr = load(p + 16, 8), check = load(p + 24, 8);
        const bool ok = magic == RAD_CHAIN_MAGIC &&
                        check == (magic ^ target ^ ctr ^ RAD_CHAIN_CHECK);
        if (ok) {
            out.push_back(fmt("chain record at +%d: target %" PRIu64 ", counter 0x%" PRIx64 ": %s",
                              (int)RAD_CHAIN_KARG_OFF, target, ctr,
                              where ? where((uintptr_t)ctr).c_str() : "not located") +
                          "; the magic and check hold, so this launch waited for that counter");
        } else {
            out.push_back(fmt("chain record at +%d: no valid record", (int)RAD_CHAIN_KARG_OFF) +
                          ", so this launch did not wait for another kernel");
        }
    }
    return out;
}

}  /* namespace aql */
}  /* namespace rad */
