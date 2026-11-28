/* karg_report_test.cpp -- the fault report's argument lines (core/device/karg_report.h, #337).
 *
 * The report a faulting queue prints is the only evidence anyone gets for a kernel that died on a
 * card, so it must say what the launch's arguments were. Issue #337's dump showed why it did not:
 * it scanned 768 bytes of a 1 KiB slot for pointer-shaped words, so it printed the leftovers of
 * earlier launches as "argument +328", it could not say anything about a 32-bit size (three of the
 * dump's eleven lines were pairs of small integers read as addresses), and it never printed the
 * chained-launch record the kernel itself acts on. The layout here is that kernel's own
 * (gemm_i8_kpar_kernel<4,256,1,r4dchain>: five pointers, then rows, cols, M, a_ld, sa_ld, then the
 * hidden arguments), with the values #337's dump decoded to. */
#include "rad_test.h"

#include "device/karg_report.h"
#include "rad_abi.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::aql;

namespace {

/* The kernel whose arguments issue #337's dump holds: (const uint8_t* wq, const uint16_t* ws,
 * const int8_t* a, const float* ascale, uint16_t* c, uint32_t rows, cols, M, a_ld, sa_ld). */
std::vector<KernelArg> i8_kpar_layout() {
    std::vector<KernelArg> v;
    const char* ptrs[5] = {"wq", "ws", "a", "ascale", "c"};
    for (int i = 0; i < 5; ++i) {
        KernelArg a;
        a.offset = (uint32_t)(8 * i);
        a.size = 8;
        a.kind = "global_buffer";
        a.address = aql::kind_is_address(a.kind);   /* what code_object_kernels fills */
        a.name = ptrs[i];
        v.push_back(a);
    }
    const char* vals[5] = {"rows", "cols", "M", "a_ld", "sa_ld"};
    for (int i = 0; i < 5; ++i) {
        KernelArg a;
        a.offset = (uint32_t)(40 + 4 * i);
        a.size = 4;
        a.kind = "by_value";
        a.address = aql::kind_is_address(a.kind);
        a.name = vals[i];
        v.push_back(a);
    }
    /* An implicit argument carries the metadata's own spelling in `kind` (`.value_kind`) and the
     * enum `hidden_kind` reads it to (aql_code.cpp); the report prints `kind`, so one name for one
     * slot stands in the report and in the refusal message. */
    const Hidden hid[8] = {Hidden::BlockCountX, Hidden::BlockCountY, Hidden::BlockCountZ,
                           Hidden::GroupSizeX,  Hidden::GroupSizeY,  Hidden::GroupSizeZ,
                           Hidden::GridDims,    Hidden::DynamicLds};
    const char* hkind[8] = {"hidden_block_count_x", "hidden_block_count_y", "hidden_block_count_z",
                            "hidden_group_size_x",  "hidden_group_size_y",  "hidden_group_size_z",
                            "hidden_grid_dims",     "hidden_dynamic_lds_size"};
    for (int i = 0; i < 8; ++i) {
        KernelArg a;
        a.offset = (uint32_t)(64 + 4 * i);
        a.size = 4;
        a.kind = hkind[i];
        a.hidden = hid[i];
        a.address = aql::kind_is_address(a.kind);
        v.push_back(a);
    }
    return v;
}

void put32(std::vector<unsigned char>& s, uint32_t off, uint32_t v) {
    std::memcpy(s.data() + off, &v, 4);
}
void put64(std::vector<unsigned char>& s, uint32_t off, uint64_t v) {
    std::memcpy(s.data() + off, &v, 8);
}

/* A slot as the engine leaves it: the launch's segment written, then the bytes after it as the
 * launch before it left them (0xa5 here, so a line that prints them is visible). */
std::vector<unsigned char> hy4_slot() {
    std::vector<unsigned char> s(1024, 0xa5);
    const uint64_t wq = 0x7cc1d6c36000ull, ws = 0x7cc1d7836000ull, a = 0x7ca036d84300ull;
    const uint64_t ascale = 0x7ca037295800ull, c = 0x7ca0371ca300ull;
    put64(s, 0, wq);
    put64(s, 8, ws);
    put64(s, 16, a);
    put64(s, 24, ascale);
    put64(s, 32, c);
    put32(s, 40, 3072);  /* rows: N */
    put32(s, 44, 4096);  /* cols: K */
    put32(s, 48, 6);     /* M */
    put32(s, 52, 4096);  /* a_ld */
    put32(s, 56, 16);    /* sa_ld */
    put32(s, 64, 192);   /* block_count_x: N/16 */
    put32(s, 68, 1);
    put32(s, 72, 1);
    put32(s, 76, 128);   /* group_size_x: KW*kWave */
    put32(s, 80, 1);
    put32(s, 84, 1);
    put32(s, 88, 3);
    put32(s, 92, 0);
    return s;
}

const uint64_t kCtr = 0x7cb34f58d200ull;

/* The counter the record names lies in a host-mapped range here; everything else is said not to
 * be in a range this process holds, which is what #337's dump said of its fault address. */
std::string fake_where(uintptr_t a) {
    if (a == kCtr) return "inside live #94 host-mapped 10.46 GiB at 0x7cb23b400000 +4.31 GiB";
    if (a == 0x7cc1d6c36000ull || a == 0x7cc1d7836000ull)
        return "in pageable of owner 1 card 0, 24.37 GiB long, +23.16 GiB, chunk 370: sealed=1";
    if (a == 0x7ca036d84300ull || a == 0x7ca037295800ull || a == 0x7ca0371ca300ull)
        return "in pageable of owner 1 card 0, 2.24 GiB long, +39.52 MiB, chunk 0: sealed=0";
    return "in no range this process holds or recently freed";
}

const std::string* find_line(const std::vector<std::string>& lines, const char* needle) {
    for (const std::string& l : lines)
        if (l.find(needle) != std::string::npos) return &l;
    return nullptr;
}

std::vector<std::string> lines_of(const std::vector<unsigned char>& slot, uint32_t kernarg_size,
                                  const std::vector<KernelArg>& layout, bool chained = true) {
    return karg_lines(KargSlot{slot.data(), (uint32_t)slot.size(), kernarg_size, chained, &layout},
                      fake_where);
}

/* The record a chained launch leaves at RAD_CHAIN_KARG_OFF (libr4d/r4d_chain.h). */
void put_record(std::vector<unsigned char>& s, uint64_t target, uint64_t ctr) {
    put64(s, RAD_CHAIN_KARG_OFF, RAD_CHAIN_MAGIC);
    put64(s, RAD_CHAIN_KARG_OFF + 8, target);
    put64(s, RAD_CHAIN_KARG_OFF + 16, ctr);
    put64(s, RAD_CHAIN_KARG_OFF + 24, RAD_CHAIN_MAGIC ^ target ^ ctr ^ RAD_CHAIN_CHECK);
}

/* #337's dump, one line per parameter: the operands by name and where they lie, the sizes as the
 * numbers they are, and the hidden arguments the launch was given. */
TEST(arguments_are_named_and_typed) {
    const std::vector<KernelArg> layout = i8_kpar_layout();
    const std::vector<std::string> lines = lines_of(hy4_slot(), 96, layout);

    /* 5 pointers + 5 sizes + 8 hidden arguments, and the chain record's line: this launch was
     * issued without one. */
    CHECK_EQ(lines.size(), 19u);

    CHECK(find_line(lines, "argument wq (global_buffer, 8 bytes at +0) = 0x7cc1d6c36000: in "
                           "pageable of owner 1 card 0") != nullptr);
    CHECK(find_line(lines, "argument c (global_buffer, 8 bytes at +32) = 0x7ca0371ca300") !=
          nullptr);
    CHECK(find_line(lines, "argument rows (by_value, 4 bytes at +40) = 3072") != nullptr);
    CHECK(find_line(lines, "argument cols (by_value, 4 bytes at +44) = 4096") != nullptr);
    CHECK(find_line(lines, "argument M (by_value, 4 bytes at +48) = 6") != nullptr);
    CHECK(find_line(lines, "argument a_ld (by_value, 4 bytes at +52) = 4096") != nullptr);
    CHECK(find_line(lines, "argument sa_ld (by_value, 4 bytes at +56) = 16") != nullptr);
    CHECK(find_line(lines, "hidden_block_count_x (4 bytes at +64) = 192") != nullptr);
    CHECK(find_line(lines, "hidden_group_size_x (4 bytes at +76) = 128") != nullptr);
    CHECK(find_line(lines, "hidden_dynamic_lds_size (4 bytes at +92) = 0") != nullptr);

    /* Not one line is the slot's leftovers: the segment ends at +96 and the rest is 0xa5. */
    for (const std::string& l : lines) {
        CHECK(l.find("a5a5a5a5") == std::string::npos);
        CHECK(l.find("+328") == std::string::npos);
    }
}

/* A 32-bit word is a value, never "in no range": that is what made three lines of #337's dump read
 * as addresses that no allocation held. */
TEST(counts_are_not_addresses) {
    const std::vector<KernelArg> layout = i8_kpar_layout();
    const std::vector<std::string> lines = lines_of(hy4_slot(), 96, layout);
    for (const std::string& l : lines) {
        if (l.rfind("argument rows", 0) == 0 || l.rfind("argument cols", 0) == 0)
            CHECK(l.find("in no range") == std::string::npos);
    }
    CHECK(find_line(lines, "3072") != nullptr);
    CHECK(find_line(lines, "4096") != nullptr);
}

/* The chained launch's record: printing it with the counter's location says whether this launch
 * waited for another kernel, which is invisible in the arguments themselves. */
TEST(chain_record_is_printed) {
    const std::vector<KernelArg> layout = i8_kpar_layout();
    std::vector<unsigned char> slot = hy4_slot();
    put_record(slot, 4711, kCtr);
    const std::vector<std::string> lines = lines_of(slot, 96, layout);
    CHECK_EQ(lines.size(), 19u);
    const std::string* rec = find_line(lines, "chain record at +");
    REQUIRE(rec != nullptr);
    CHECK(rec->find("target 4711") != std::string::npos);
    CHECK(rec->find("counter 0x7cb34f58d200: inside live #94 host-mapped 10.46 GiB") !=
          std::string::npos);
    CHECK(rec->find("the magic and check hold") != std::string::npos);

    /* A record whose check no longer matches is one the kernel would not act on. */
    std::vector<unsigned char> bad = hy4_slot();
    put_record(bad, 4711, kCtr);
    put64(bad, RAD_CHAIN_KARG_OFF + 24, RAD_CHAIN_CHECK);
    const std::vector<std::string> bad_lines = lines_of(bad, 96, layout);
    const std::string* bad_rec = find_line(bad_lines, "chain record at +");
    REQUIRE(bad_rec != nullptr);
    CHECK(bad_rec->find("no valid record") != std::string::npos);
    CHECK(bad_rec->find("did not wait") != std::string::npos);
}

/* The record is a chained launch's, and only a chained launch's: the backend writes it in
 * `if (k->chained) chain_record(...)` and the plain form of a kernel never reads the slot. A slot a
 * plain launch reused still holds the record an earlier chained launch left there, and reading it
 * as this launch's would say "this launch waited for that counter" about a kernel that does not
 * look at the slot at all. */
TEST(chain_record_is_not_claimed_for_a_plain_launch) {
    const std::vector<KernelArg> layout = i8_kpar_layout();
    std::vector<unsigned char> slot = hy4_slot();
    put_record(slot, 4711, kCtr);  /* a valid record from an earlier launch in this slot */
    const std::vector<std::string> lines = lines_of(slot, 96, layout, false);
    const std::string* rec = find_line(lines, "chain record at +");
    REQUIRE(rec != nullptr);
    CHECK(rec->find("not read") != std::string::npos);
    CHECK(rec->find("waited") == std::string::npos);
    CHECK(rec->find("target 4711") == std::string::npos);
    CHECK(rec->find("belongs to an earlier launch") != std::string::npos);
}

/* A 64-bit parameter is a value unless the metadata says it is an address: a `uint64_t` and a
 * `uint64_t *` are both eight bytes, and `hidden_global_offset_x` is an eight-byte value. The old
 * dump decided by size, so every one of them printed as "in no range this process holds" -- the
 * integer-read-as-an-address line issue #337's own dump was chased through. */
TEST(an_eight_byte_value_is_not_an_address) {
    std::vector<KernelArg> layout;
    KernelArg n;
    n.offset = 0; n.size = 8; n.kind = "by_value"; n.name = "n";
    n.address = aql::kind_is_address(n.kind);
    layout.push_back(n);
    KernelArg go;
    go.offset = 8; go.size = 8; go.kind = "hidden_global_offset_x";
    go.hidden = Hidden::GlobalOffsetX;
    go.address = aql::kind_is_address(go.kind);
    layout.push_back(go);
    KernelArg p;
    p.offset = 16; p.size = 8; p.kind = "global_buffer"; p.name = "p";
    p.address = aql::kind_is_address(p.kind);
    layout.push_back(p);
    KernelArg k;
    k.offset = 24; k.size = 4; k.kind = "by_value"; k.name = "k";
    k.address = aql::kind_is_address(k.kind);
    layout.push_back(k);

    /* And the table itself: the pointer kinds, and nothing else. */
    CHECK(aql::kind_is_address("global_buffer"));
    CHECK(aql::kind_is_address("dynamic_shared_pointer"));
    CHECK(!aql::kind_is_address("by_value"));
    CHECK(!aql::kind_is_address("hidden_global_offset_x"));
    CHECK(!aql::kind_is_address("ext_image"));
    CHECK(!aql::kind_is_address(""));

    std::vector<unsigned char> slot(1024, 0xa5);
    put64(slot, 0, 0x7ca037295800ull);  /* a count that happens to look like an address */
    put64(slot, 8, 4096);
    put64(slot, 16, 0x7ca000001000ull);
    put32(slot, 24, 7);

    int asked = 0;
    const std::vector<std::string> lines = karg_lines(
        KargSlot{slot.data(), (uint32_t)slot.size(), 32, false, &layout},
        [&asked](uintptr_t) {
            ++asked;
            return std::string("in no range this process holds or recently freed");
        });
    const std::string want = "argument n (by_value, 8 bytes at +0) = " +
                             std::to_string(0x7ca037295800ull);
    CHECK(find_line(lines, want.c_str()) != nullptr);
    CHECK(find_line(lines, "hidden_global_offset_x (8 bytes at +8) = 4096") != nullptr);
    CHECK(find_line(lines, "argument k (by_value, 4 bytes at +24) = 7") != nullptr);
    CHECK(find_line(lines, "argument p (global_buffer, 8 bytes at +16) = 0x7ca000001000: in no "
                           "range this process holds") != nullptr);
    /* The address lookup is asked about the address and about nothing else. */
    CHECK_EQ(asked, 1);
}

/* A segment shorter than the metadata's parameters: what follows is another launch's, and the
 * report says so instead of printing it as arguments. */
TEST(past_the_segment_is_not_printed) {
    const std::vector<KernelArg> layout = i8_kpar_layout();
    const std::vector<std::string> lines = lines_of(hy4_slot(), 60, layout);
    CHECK(find_line(lines, "argument sa_ld") != nullptr);
    /* The needle is the spelling the report prints (`kind`, the metadata's value_kind), so this
     * check fails if a hidden slot past the segment is printed after all. */
    CHECK(find_line(lines, "hidden_block_count_x") == nullptr);
    CHECK(find_line(lines, "hidden_") == nullptr);
    CHECK(find_line(lines, "this launch's segment ends at +60") != nullptr);
}

/* The slot's end, when the metadata runs past the slot: a report bug is reported, not crashed. */
TEST(past_the_slot_is_reported) {
    const std::vector<KernelArg> layout = i8_kpar_layout();
    std::vector<unsigned char> slot = hy4_slot();
    slot.resize(48);  /* room for the five pointers and two sizes */
    const std::vector<std::string> lines = lines_of(slot, 96, layout);
    CHECK(find_line(lines, "the slot is 48 bytes") != nullptr);
    CHECK(find_line(lines, "argument rows") != nullptr);
    CHECK(find_line(lines, "argument M") == nullptr);
}

/* The parameter that points into the same live range the fault address is nearest to says so, and
 * no other line does: that parameter names the launch the fault address points at, which is the
 * question the ring's order cannot answer (#337). */
TEST(the_parameter_nearest_the_fault_address_is_named) {
    const std::vector<KernelArg> layout = i8_kpar_layout();
    std::vector<unsigned char> slot = hy4_slot();

    /* The activation arena of #337's report: c (+32) is inside it, the fault address just past its
     * end, and the two weight operands are in another range. */
    const uintptr_t c = 0x7ca0371ca300ull;
    const uintptr_t arena_lo = 0x7ca036d84000ull, arena_hi = 0x7ca038000000ull;
    auto ranges = [&](uintptr_t a) {
        KargRange r;
        if (a >= arena_lo && a < arena_hi) { r.rec = 12; r.lo = arena_lo; r.hi = arena_hi; }
        else if (a >= 0x7cc1d6c36000ull && a < 0x7cc1d7837000ull) {
            r.rec = 7; r.lo = 0x7cc1d6c36000ull; r.hi = 0x7cc1d7837000ull;
        }
        return r;
    };
    KargSlot ks{slot.data(), (uint32_t)slot.size(), 96, true, &layout};
    ks.fault = arena_hi + 0x800;
    ks.fault_rec = 12;
    ks.range_of = ranges;
    const std::vector<std::string> lines = karg_lines(ks, fake_where);
    const std::string* hit = find_line(lines, "argument c ");
    CHECK(hit != nullptr);
    CHECK(hit && hit->find("the fault address is 2.00 KiB past the end of this same range") !=
                       std::string::npos);
    /* Every activation operand of the launch lies in that one arena, so all three say it, and the
     * weight operands in another range do not. */
    int said = 0;
    for (const std::string& l : lines)
        if (l.find("this same range") != std::string::npos) ++said;
    CHECK_EQ(said, 3);
    const std::string* wq = find_line(lines, "argument wq ");
    CHECK(wq && wq->find("this same range") == std::string::npos);

    /* The fault address before the range's start, and inside it, are said as such. */
    ks.fault = arena_lo - 0x10;
    CHECK(find_line(karg_lines(ks, fake_where), "16 B before the start of this same range") !=
          nullptr);
    ks.fault = c;
    CHECK(find_line(karg_lines(ks, fake_where), "the fault address is inside this same range") !=
          nullptr);

    /* A stall dump (no fault address) says nothing about one, and neither does a fault address
     * whose nearest range the caller does not know. */
    ks.fault = 0;
    ks.fault_rec = 12;
    for (const std::string& l : karg_lines(ks, fake_where))
        CHECK(l.find("this same range") == std::string::npos);
    ks.fault = arena_hi + 0x800;
    ks.fault_rec = 0;
    for (const std::string& l : karg_lines(ks, fake_where))
        CHECK(l.find("this same range") == std::string::npos);
}

/* No layout (a kernel object the backend did not read metadata for): nothing to say, and the
 * caller keeps its old dump. */
TEST(no_layout_means_no_lines) {
    std::vector<unsigned char> slot = hy4_slot();
    CHECK(karg_lines(KargSlot{slot.data(), (uint32_t)slot.size(), 96, true, nullptr}, fake_where)
              .empty());
    const std::vector<KernelArg> empty;
    CHECK(karg_lines(KargSlot{slot.data(), (uint32_t)slot.size(), 96, true, &empty}, fake_where)
              .empty());
    CHECK(karg_lines(KargSlot{nullptr, 0, 96, true, &empty}, fake_where).empty());
}

}  /* namespace */

RAD_TEST_MAIN()
