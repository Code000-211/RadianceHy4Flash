/* karg_report.cpp -- the fault report's argument lines (karg_report.h, issue #337). */
#include "karg_report.h"

#include <cinttypes>
#include <cstring>

#include "rad_abi.h"
#include "../rad_internal.h"

namespace rad {
namespace aql {
namespace {

const char* hidden_name(Hidden h) {
    switch (h) {
    case Hidden::None:          return nullptr;
    case Hidden::Pad:           return "hidden padding";
    case Hidden::BlockCountX:   return "hidden block_count_x";
    case Hidden::BlockCountY:   return "hidden block_count_y";
    case Hidden::BlockCountZ:   return "hidden block_count_z";
    case Hidden::GroupSizeX:    return "hidden group_size_x";
    case Hidden::GroupSizeY:    return "hidden group_size_y";
    case Hidden::GroupSizeZ:    return "hidden group_size_z";
    case Hidden::RemainderX:    return "hidden remainder_x";
    case Hidden::RemainderY:    return "hidden remainder_y";
    case Hidden::RemainderZ:    return "hidden remainder_z";
    case Hidden::GlobalOffsetX: return "hidden global_offset_x";
    case Hidden::GlobalOffsetY: return "hidden global_offset_y";
    case Hidden::GlobalOffsetZ: return "hidden global_offset_z";
    case Hidden::GridDims:      return "hidden grid_dims";
    case Hidden::DynamicLds:    return "hidden dynamic_lds";
    case Hidden::Unsupported:   return "hidden unsupported";
    }
    return "hidden";
}

uint64_t load(const unsigned char* p, uint32_t n) {
    uint64_t v = 0;
    std::memcpy(&v, p, n < 8 ? n : 8);
    return v;
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
        const char* hname = hidden_name(a.hidden);
        std::string head;
        if (hname) {
            head = fmt("%s (%u bytes at +%u)", hname, a.size, a.offset);
        } else {
            const std::string name = a.name.empty() ? fmt("%zu", i) : a.name;
            head = fmt("argument %s (%s, %u bytes at +%u)", name.c_str(),
                       a.kind.empty() ? "by_value" : a.kind.c_str(), a.size, a.offset);
        }
        if (a.hidden == Hidden::Pad || a.size == 0) {
            out.push_back(head);
        } else if (a.size >= 8) {
            const uint64_t v = load(slot.base + a.offset, a.size);
            if (v == 0)
                out.push_back(head + " = 0 (null)");
            else
                out.push_back(head + fmt(" = 0x%" PRIx64 ": ", v) +
                              (where ? where((uintptr_t)v) : std::string("not located")));
        } else {
            out.push_back(head + fmt(" = %" PRIu64, load(slot.base + a.offset, a.size)));
        }
    }

    /* The chained-launch record: the kernel reads it at RAD_CHAIN_KARG_OFF and waits for that
     * counter if and only if the record's magic and check hold (libr4d/r4d_chain.h). Printing it,
     * with the counter's location, tells a chained launch that waited from an unchained one. */
    if ((uint64_t)RAD_CHAIN_KARG_OFF + RAD_CHAIN_REC_BYTES <= slot.bytes) {
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
