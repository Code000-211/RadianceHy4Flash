/* pchash.cpp -- pc_checksum and pc_key_of (pcstore.h), in a unit of their own: the scheduler's
 * global suffix cache uses them, and linking all of pcstore.o into the scheduler's tests would
 * bring the real device layer over their weak fakes. */
#include "mem/pcstore.h"

#include "util/simd.h"

#include <chrono>

namespace rad {

namespace {
inline uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
}  // namespace

uint64_t pc_checksum(const void* data, int64_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t lane[8] = { 0xc2b2ae3d27d4eb4full, 0x9e3779b185ebca87ull, 0x165667b19e3779f9ull ^ (uint64_t)n,
                         0x85ebca77c2b2ae63ull, 0x27d4eb2f165667c5ull, 0x94d049bb133111ebull,
                         0x9e3779b97f4a7c15ull ^ (uint64_t)n, 0xff51afd7ed558ccdull };
    const size_t stripes = n > 0 ? (size_t)n / 64 : 0;
    if (stripes) simd::stripe_acc_u64x8(lane, p, stripes);
    uint64_t a = 0x6a09e667f3bcc908ull ^ (uint64_t)n;
    for (int j = 0; j < 8; ++j) a = mix64(a ^ lane[j]) + (uint64_t)j;
    for (size_t i = stripes * 64; i < (size_t)n; ++i) a = mix64(a ^ p[i]);
    return mix64(a);
}

BlockHash pc_key_of(const std::string& text) {
    BlockHash h;
    uint64_t a = 0x243f6a8885a308d3ull, b = 0x13198a2e03707344ull;
    const uint64_t s = pc_checksum(text.data(), (int64_t)text.size());
    for (unsigned char c : text) { a = mix64(a ^ c); b = mix64(b + c * 0x9e3779b97f4a7c15ull); }
    h.hi = mix64(a ^ s);
    h.lo = mix64(b + s);
    if (!h.valid()) h.lo = 1;
    return h;
}

int64_t pc_unix_ms() {
    return (int64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

}  /* namespace rad */
