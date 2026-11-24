// k47_copy2d_probe.cpp — K-47 verification: byte-identity + device timing for the
// aligned 16-byte strided copy (rad_aql_blit_copy2d -> aql_copy2d16 / r4dchain_copy2d16).
// Standalone: links rad_core (the aql device backend, blit ISA embedded). No model, no plugins.
// Build (against a build dir B and worktree W):
//   c++ -DRAD_HAVE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -I$W/core -I$W/abi -I$B/rad_include \
//       -O2 -std=gnu++20 -fPIE -c k47_copy2d_probe.cpp -o probe.o
//   c++ probe.o -o k47_probe $B/core/librad_core.a $B/core/librad_json.a $B/core/librad_simd.a \
//       $B/vendor/librad_vendor_*.a /usr/lib/x86_64-linux-gnu/libamdhip64.so \
//       /usr/lib/x86_64-linux-gnu/libhsa-runtime64.so -lpthread -ldl
#include "rad_device.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

extern "C" void rad_dev_chain_infer(int);  // dispatch.cpp: owner 0 (a tool) chain answer

static int g_fail = 0;
#define CK(x) do{ if((x)!=RAD_OK){ fprintf(stderr,"FAIL %s:%d %s: %s\n",__FILE__,__LINE__,#x,rad_dev_last_error()); return 1;} }while(0)

struct Case { const char* name; uint64_t width, height, spitch, dpitch; };

static int run_case(const Case& c, int chained, int iters) {
    rad_dev_chain_infer(chained ? 1 : 0);
    const uint64_t sbytes = c.spitch * c.height, dbytes = c.dpitch * c.height;
    void* dsrc = rad_dev_alloc(sbytes, RAD_MEM_DEVICE);
    void* ddst = rad_dev_alloc(dbytes, RAD_MEM_DEVICE);
    if (!dsrc || !ddst) { fprintf(stderr,"alloc fail: %s\n", rad_dev_last_error()); return 1; }
    std::vector<unsigned char> hsrc(sbytes), hdst(dbytes), href(dbytes);
    for (uint64_t r = 0; r < c.height; r++)
        for (uint64_t b = 0; b < c.spitch; b++)
            hsrc[r*c.spitch+b] = (unsigned char)((r*131 + b*17 + 7) & 0xff);
    std::memset(href.data(), 0xA5, dbytes);                 // poison padding
    for (uint64_t r = 0; r < c.height; r++)
        std::memcpy(href.data()+r*c.dpitch, hsrc.data()+r*c.spitch, c.width);
    RadStream s; CK(rad_stream_create(&s, 0));
    CK(rad_memcpy_async(dsrc, hsrc.data(), sbytes, s));
    CK(rad_memset_async(ddst, 0xA5, dbytes, s));
    CK(rad_stream_sync(s));
    // byte-identity: one copy, read back, compare full dst (incl. untouched padding)
    CK(rad_memcpy_2d_async(ddst, c.dpitch, dsrc, c.spitch, c.width, c.height, s));
    CK(rad_stream_sync(s));
    CK(rad_memcpy_async(hdst.data(), ddst, dbytes, s));
    CK(rad_stream_sync(s));
    const int bi = (std::memcmp(hdst.data(), href.data(), dbytes) == 0);
    if (!bi) {
        g_fail++;
        fprintf(stderr,"BI FAIL %s chained=%d\n", c.name, chained);
        for (uint64_t i = 0; i < dbytes; i++)
            if (hdst[i] != href[i]) { fprintf(stderr,"  first diff @%llu got %02x want %02x\n",
                (unsigned long long)i, hdst[i], href[i]); break; }
    }
    // timing: warmup, then iters launches between two stream events
    for (int i = 0; i < 10; i++) rad_memcpy_2d_async(ddst,c.dpitch,dsrc,c.spitch,c.width,c.height,s);
    CK(rad_stream_sync(s));
    RadEvent e0, e1; CK(rad_event_create(&e0)); CK(rad_event_create(&e1));
    CK(rad_event_record(e0, s));
    for (int i = 0; i < iters; i++) rad_memcpy_2d_async(ddst,c.dpitch,dsrc,c.spitch,c.width,c.height,s);
    CK(rad_event_record(e1, s));
    CK(rad_stream_sync(s));
    float ms = 0; CK(rad_event_elapsed_ms(e0, e1, &ms));
    const double us = (double)ms * 1000.0 / iters;
    const double gbps = (double)c.width * c.height / (us * 1e-6) / 1e9;
    printf("%-20s chain=%d  %9.2f us/launch  %8.1f GB/s  BI=%s\n",
           c.name, chained, us, gbps, bi ? "OK" : "FAIL");
    rad_event_destroy(e0); rad_event_destroy(e1);
    rad_stream_destroy(s);
    rad_dev_free(dsrc, RAD_MEM_DEVICE); rad_dev_free(ddst, RAD_MEM_DEVICE);
    return bi ? 0 : 1;
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 200;
    const int n = rad_dev_count();
    if (n < 1) { fprintf(stderr,"no device\n"); return 1; }
    CK(rad_dev_set(0));
    RadDeviceProps p; rad_dev_props(0, &p);
    printf("device: %s  iters=%d\n", p.name, iters);
    const Case cases[] = {
        {"narrow128x32768",   128,  32768,   128,   128},  // w16=8: the indexer-row defect
        {"narrow128x4096pad", 128,   4096,   256,   256},  // w16=8, padded pitch
        {"med256x2048",       256,   2048,   256,   256},  // w16=16
        {"wide4096x512",     4096,    512,  4096,  4096},  // w16=256 (block count unchanged)
        {"wide8192x256",     8192,    256,  8192,  8192},  // w16=512
        {"tall128x65535",     128,  65535,   128,   128},  // old by-cap boundary
        {"huge128x100000",    128, 100000,   128,   128},  // total>2048*256: grid-stride wrap
    };
    int rc = 0;
    for (const auto& c : cases) { rc |= run_case(c, 0, iters); rc |= run_case(c, 1, iters); }
    rad_dev_chain_infer(0);
    if (g_fail) printf("RESULT: %d BI FAILURES\n", g_fail);
    else        printf("RESULT: ALL BYTE-IDENTICAL\n");
    return g_fail ? 1 : 0;
}
