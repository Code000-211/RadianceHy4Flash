# PCIe host->device bandwidth on the inference host: where the TP2 bandwidth goes (raw lines)

Question (from the M-20 DMA bench, `measurements/M-20.md` §4): a 0.58 MiB `down` copy over the xprefetch SDMA path takes
25.7 us on one card but ~57 us when both TP2 cards copy at once, and long lists give only ~18 GB/s per card, although
each card has its own Gen4 x16 root port (40:01.1 / c0:01.1) and the host has 8 DDR4 channels. Where does it go?

**Answer: host memory, not PCIe, not the IOD, not the SDMA engine.** Two cards reading host pages that sit on the
*same* memory region share that region's DRAM bandwidth: ~33-37 GB/s together over SDMA from THP-backed buffers like
the engine's pool (16.5-18.5 per card, the
M-20 "18 GB/s"), ~41-48 GB/s with a zero-copy read kernel. When card 0 reads a chunk from one memory region and card 1
a chunk from a *different* one, both run at full link speed at the same time: **46-50 GB/s aggregate (23-28 per
card)**. So the root ports, the IOD and the fabric carry two full Gen4 links at once; the limit is the DRAM channels
behind the pages. The CPU and both cards draw on one shared budget of ~52-56 GB/s in the region most allocations
land in (CPU read alone ~41-50, CPU + both cards together ~52-56). This fits the owner's DIMM layout
(4 channels 1 x 32 GB, 4 channels 2 x 8 GB; EDAC confirms it, below): the memory is not one uniform 8-way
interleave, and in the bulk region the ceiling is far below 8 x DDR4.

Setup: host `inference`, EPYC 7402P (1 socket, NPS1 as seen: 1 NUMA node), 2 x R9700 gfx1201, kernel 7.0.2-6-pve,
`iommu=pt` (both GPU IOMMU groups `type=identity`: no IOMMU translation), THP `madvise`, shmem THP `never`, hugetlb
pool 0. Everything ran in the LXC container under `flock /tmp/radiance-lanes/gpu.lock gpu2 ...` (cards 0,1; no
radiance server running, checked with `ps` every session); power cap read back `power1_cap 210000000 uW` on both cards
at the start and end of every session; root ports sampled at 16.0 GT/s during the runs (2.5 GT/s only while idle).
Compiler: plain `/opt/rocm/bin/hipcc` (HIP 7.16.26385, AMD clang 24.0.0git), `-O3 -std=c++20 --offload-arch=gfx1201`,
not radiance's repeatable compiler. Sources and session scripts: `measurements/pcie-h2d-tools/`.

**Caveat, host load:** the board automation (Moebius test suites, rustc/clippy) kept the host at load 20-99 the whole
time (up to 38 cores busy, ~80% sys); I waited ~20 min for a quiet window and got none. The GPU lock keeps servers off
the cards, not other CPU/memory traffic. Absolute CPU STREAM numbers and part of the DMA numbers are depressed by
this; comparisons inside one session (same minute) hold. A re-measure on a quiet host (and `dmidecode -t 17` on the
Proxmox host for the configured memory speed) is still owed.

## What the container can and cannot see

- EDAC (`/sys/devices/system/edac/mc/mc0`, `F17h_M31h`, 196608 MB): ranks on all 8 channels. Channels 1-4: two
  16 GiB ranks each (csrow 2+3) = one dual-rank 32 GB DIMM, reported `Unbuffered-DDR4`; channels 0,5,6,7: two 8 GiB
  ranks on csrow 0 and csrow 2 = two 8 GB DIMMs, reported `Registered-DDR4`. Matches "4 x (1 x 32 GB) + 4 x (2 x 8 GB)".
  The mixed RDIMM/UDIMM type EDAC reports is odd (EPYC does not mix them); worth a look on the host with dmidecode.
- Board `GIGABYTE MZ32-AR0`, BIOS `R27` (`/sys/devices/virtual/dmi/id`).
- **Not visible:** configured memory speed (dmidecode: `/dev/mem` and `/sys/firmware/dmi/entries/*/raw` permission
  denied; EDAC has no speed), physical addresses (`/proc/self/pagemap` returns PFN 0 without CAP_SYS_ADMIN,
  `/proc/iomem` is zeroed), the BIOS NPS / interleave settings, hugetlb pool sizing. So the requested low-vs-high
  *physical range* comparison could not be done by address; it was done by bandwidth instead (region probe below),
  and where the engine's pinned pool or the Engram table land physically cannot be stated from inside the container.

## Host memory: region probe (the decisive test)

`region_probe 80 8 3`: maps 80 x 1 GiB THP chunks (every chunk 1024/1024 MiB huge-page backed), then per chunk
CPU read (8 threads over the 4 CCDs) and both cards at once reading their own 256 MiB of the chunk (pinned the engine's
way, `hsa_amd_memory_lock_to_pool` fine-grained), SDMA engine 0 and the zero-copy kernel. Format `chunk:cpu/sdma2/kern2`
GB/s (medians of 3). Session 4 (load 54-62):
```
0:51.0/36.1/42.1 1:33.5/29.0/34.5 2:31.0/31.0/34.6 3:43.8/34.4/45.2 4:49.2/36.4/48.0 5:45.9/35.2/42.9 6:44.0/36.3/41.7 7:46.2/36.0/45.5 
8:49.2/35.6/47.2 9:47.6/36.4/44.4 10:53.7/34.6/47.5 11:46.5/37.1/43.6 12:51.6/35.0/44.9 13:48.3/35.7/44.9 14:49.0/36.2/43.0 15:51.3/35.8/44.2 
16:53.3/35.4/46.5 17:44.6/33.6/47.5 18:49.8/36.3/44.7 19:48.6/34.9/44.6 20:46.6/37.0/43.2 21:50.8/35.7/45.9 22:52.1/36.9/43.8 23:46.0/36.9/43.0 
24:51.5/35.3/44.9 25:52.4/35.0/43.6 26:50.8/35.7/44.9 27:44.0/32.5/39.5 28:48.1/32.6/37.9 29:48.6/34.2/41.1 30:48.4/33.0/38.3 31:47.6/34.1/38.0 
32:44.0/33.4/42.0 33:51.8/34.8/41.3 34:44.6/32.5/40.3 35:44.2/32.7/37.9 36:49.6/35.2/39.7 37:47.4/32.1/42.2 38:45.9/35.3/39.2 39:44.7/33.6/39.5 
40:48.9/35.5/42.0 41:46.6/34.4/39.2 42:45.0/34.3/39.4 43:47.0/34.3/39.5 44:47.3/33.6/39.3 45:45.0/35.5/40.0 46:44.4/33.7/38.7 47:44.2/32.7/38.9 
48:44.2/32.9/38.7 49:44.2/34.5/39.0 50:47.5/34.4/39.6 51:47.6/33.5/39.5 52:46.1/33.9/38.7 53:44.2/34.3/38.3 54:45.0/33.7/39.3 55:49.0/34.0/39.5 
56:45.7/34.4/39.6 57:43.9/33.6/39.7 58:42.6/32.1/38.1 59:46.6/30.5/36.0 60:39.4/30.2/37.1 61:40.3/29.1/38.5 62:36.7/28.4/37.7 63:34.4/28.2/37.4 
64:19.1/27.5/35.1 65:37.4/30.0/41.0 66:42.4/28.2/36.9 67:38.1/31.2/39.8 68:43.5/29.4/40.1 69:39.9/26.7/40.7 70:42.5/32.2/41.4 71:46.1/32.5/42.1 
72:41.1/28.4/41.1 73:41.3/31.7/41.3 74:27.9/33.4/32.8 75:25.5/27.5/27.1 76:26.9/28.8/27.0 77:26.9/29.2/27.8 78:26.9/25.6/27.0 79:47.1/44.1/44.7 
```
Classes (reproducible over two passes in session 3, whose raw scan is in the session log, not kept here, and in session 2 the same pattern at the same chunk indices):
bulk chunks CPU 42-53, 2-card SDMA 31-37, 2-card kernel 37-48; a "slow" group (here 74-78, session 3: 73-77) CPU
25-30, 2-card ~27-33 for both paths; some chunks straddle (79 here: 2-card 44; session 3 chunk 78: 2-card SDMA 49.0 /
kernel 48.8-52.5 and chunk 79: 11-15 everything). The slow group comes from the last ~6 GiB the probe allocated, i.e.
what the kernel hands out once the bulk is used; its ratio to the bulk (~0.55-0.6 on CPU) fits a 4-way vs 8-way region.

Pairs (card 0 reads chunk a, card 1 chunk b; 256 MiB each, 5 reps, median), same session as the scan above:
```
PAIR card0<-chunk 10 card1<-none  0 | sdma-e0 agg  21.0 (card0  21.0 card1   0.0) | kern agg  26.4 (card0  26.4 card1   0.0)
PAIR card0<-chunk 40 card1<-none  0 | sdma-e0 agg  21.9 (card0  21.9 card1   0.0) | kern agg  27.6 (card0  27.6 card1   0.0)
PAIR card0<-chunk 75 card1<-none  0 | sdma-e0 agg  23.1 (card0  23.1 card1   0.0) | kern agg  19.8 (card0  19.8 card1   0.0)
PAIR card0<-chunk 78 card1<-none  0 | sdma-e0 agg  23.3 (card0  23.3 card1   0.0) | kern agg  21.0 (card0  21.0 card1   0.0)
PAIR card0<-chunk 79 card1<-none  0 | sdma-e0 agg  22.1 (card0  22.1 card1   0.0) | kern agg  20.8 (card0  20.8 card1   0.0)
PAIR card0<-chunk 10 card1<-chunk 10 | sdma-e0 agg  26.2 (card0  13.3 card1  13.1) | kern agg  33.1 (card0  16.8 card1  16.8)
PAIR card0<-chunk 78 card1<-chunk 78 | sdma-e0 agg  22.1 (card0  11.2 card1  11.3) | kern agg  21.0 (card0  10.5 card1  10.5)
PAIR card0<-chunk 79 card1<-chunk 79 | sdma-e0 agg  38.1 (card0  19.1 card1  19.5) | kern agg  37.5 (card0  18.7 card1  24.0)
PAIR card0<-chunk 10 card1<-chunk 11 | sdma-e0 agg  33.2 (card0  16.6 card1  16.8) | kern agg  41.1 (card0  22.2 card1  21.9)
PAIR card0<-chunk 40 card1<-chunk 41 | sdma-e0 agg  34.0 (card0  17.0 card1  17.5) | kern agg  41.1 (card0  20.6 card1  20.6)
PAIR card0<-chunk 75 card1<-chunk 76 | sdma-e0 agg  29.3 (card0  14.7 card1  14.7) | kern agg  28.2 (card0  14.2 card1  14.1)
PAIR card0<-chunk 10 card1<-chunk 40 | sdma-e0 agg  31.2 (card0  16.5 card1  16.8) | kern agg  44.4 (card0  22.2 card1  22.5)
PAIR card0<-chunk 10 card1<-chunk 75 | sdma-e0 agg  46.4 (card0  23.2 card1  25.2) | kern agg  49.6 (card0  27.8 card1  24.8)
PAIR card0<-chunk 40 card1<-chunk 75 | sdma-e0 agg  47.6 (card0  23.9 card1  25.6) | kern agg  50.5 (card0  28.0 card1  25.3)
PAIR card0<-chunk 10 card1<-chunk 79 | sdma-e0 agg  46.0 (card0  23.0 card1  25.1) | kern agg  49.4 (card0  27.8 card1  24.7)
PAIR card0<-chunk 75 card1<-chunk 79 | sdma-e0 agg  29.6 (card0  14.9 card1  14.8) | kern agg  28.3 (card0  14.2 card1  14.1)
PAIR card0<-chunk 78 card1<-chunk 10 | sdma-e0 agg  45.8 (card0  24.4 card1  22.9) | kern agg  50.1 (card0  25.0 card1  27.9)
```
- bulk+bulk (10:11, 40:41, 10:40; 10:10 = both halves of one chunk): 26-34 SDMA / 33-44 kernel aggregate, 13-17 per card: the TP2 deficit.
- bulk+slow (10:75, 40:75, 10:79, 78:10): **46-48 SDMA / 49-50 kernel aggregate, each card 23-28 = full link**.
- slow+slow (75:76, 75:79, 78:78): 22-30.
- one card alone from a bulk chunk 21-22 SDMA / 26-28 kernel; from a slow chunk 22-23 SDMA but only ~20 kernel.

So the two root ports and the IOD deliver ~50 GB/s of DMA reads (two full links) as soon as the two cards' pages are
on different channel sets. Conclusion for the owner's question: **this is a memory-layout / interleave effect, not a
Rome IOD or fabric limit for DMA from two root ports.** One more point: even the bulk region gives CPU + both cards
only ~52-56 GB/s together (next section), which is about half of what an 8-way DDR4-2666..3200 interleave should
sustain; whether that is the memory speed the 2-DIMM channels force, a coarser interleave than assumed, or the
co-tenant load cannot be separated from inside the container.

## CPU-side STREAM and the shared budget

`cpu_stream 2 5` (threads spread over CCDs first; read = sum, copy = memcpy, copy counts read+write), session 2 start
(load ~90):
```
CPUSTREAM read threads= 1 GiB=2  GB/s med   14.1 min   13.9 max   14.3
CPUSTREAM copy threads= 1 GiB=2  GB/s med   24.2 min   24.0 max   24.7
CPUSTREAM read threads= 2 GiB=2  GB/s med   25.5 min   24.8 max   26.0
CPUSTREAM copy threads= 2 GiB=2  GB/s med   38.1 min   37.8 max   38.5
CPUSTREAM read threads= 4 GiB=2  GB/s med   39.6 min   38.2 max   40.4
CPUSTREAM copy threads= 4 GiB=2  GB/s med   46.0 min   44.7 max   47.0
CPUSTREAM read threads= 8 GiB=2  GB/s med   48.8 min   47.8 max   49.6
CPUSTREAM copy threads= 8 GiB=2  GB/s med   49.7 min   48.5 max   50.1
CPUSTREAM read threads=12 GiB=2  GB/s med   50.8 min   48.6 max   51.1
CPUSTREAM copy threads=12 GiB=2  GB/s med   51.2 min   50.1 max   51.7
CPUSTREAM read threads=16 GiB=2  GB/s med   50.4 min   49.6 max   51.1
CPUSTREAM copy threads=16 GiB=2  GB/s med   52.0 min   51.4 max   52.4
CPUSTREAM read threads=24 GiB=2  GB/s med   52.5 min   51.8 max   53.3
CPUSTREAM copy threads=24 GiB=2  GB/s med   52.0 min   51.9 max   52.4
CPUSTREAM read threads=48 GiB=2  GB/s med   51.4 min   48.9 max   52.1
CPUSTREAM copy threads=48 GiB=2  GB/s med   52.0 min   49.7 max   52.1
```
Read saturates at ~50-57 GB/s from 8 threads on (session 1, load ~20: 52-57) — a memory ceiling, not a thread limit
(single thread 14 GB/s). 8 x DDR4-3200 would be 204.8 GB/s theoretical, ~140 GB/s STREAM on Rome; even 8 x DDR4-2400
would give ~100.

Shared budget (session 3, engine-style THP buffers, both cards 256 MiB copies, beside an 8-thread CPU read that alone
runs 40.5 GB/s; during the overlap the CPU read fell to 27.6-29.2):
```
[overlap] H2D devs=01 host=engine path=e0 case=big | card0 GB/s med 12.34 min 7.94 max 15.68 | card1 GB/s med 11.81 min 7.69 max 15.17 | aggregate med 23.63 min 15.39 max 30.34
[overlap] H2D devs=01 host=engine path=kern case=big | card0 GB/s med 13.99 min 7.59 max 18.33 | card1 GB/s med 13.86 min 7.49 max 18.54 | aggregate med 27.72 min 14.98 max 36.66
H2D devs=01 host=engine path=e0 case=big | card0 GB/s med 14.91 min 14.38 max 15.26 | card1 GB/s med 14.86 min 14.52 max 15.21 | aggregate med 29.72 min 28.76 max 30.43
H2D devs=01 host=engine path=kern case=big | card0 GB/s med 17.53 min 17.12 max 17.68 | card1 GB/s med 17.87 min 17.45 max 18.01 | aggregate med 35.06 min 34.23 max 35.36
CPUSTREAM read threads= 8 GiB=4  GB/s med   40.5 min   39.5 max   41.3
```
CPU 28 + DMA 24-28 = 52-56 GB/s during the overlap, the same total as either alone at its best: one budget for the CPU
and both cards. (In session 1/2 a 16-thread CPU read halved one card's DMA to 10-16 GB/s.) The engine's host kernels
(the Engram gather, CPU-side MoE parts) therefore also take bandwidth away from the expert prefetch and vice versa.

## Matrix: host buffer x path x size, one card vs both (session 2, load ~90, 7 reps, GB/s median [min-max])

Paths: `hip1` hipMemcpyAsync one stream, `hip2` two streams, `e0`/`e1`/`e01` `hsa_amd_memory_async_copy_on_engine`
on SDMA engine 0 / 1 / alternating (both report free, mask 0x3), `kern` zero-copy streaming-read kernel (one launch per
copy). Cases: `big` 1 x 256 MiB, `l058` 400 x 608256 B, `l143` 160 x 1499546 B (scattered sources in a 512 MiB pool
per card). Host kinds: `hipmalloc` hipHostMalloc (4 KiB pages: hipHostMalloc memory is shmem, THP `never`),
`hipmalloc_nc` +NonCoherent, `reg4k` mmap+MADV_NOHUGEPAGE+hipHostRegister, `regthp` 2 MiB aligned+MADV_HUGEPAGE+
hipHostRegister (512/512 MiB THP), `hugetlb` unavailable (`mmap(MAP_HUGETLB)`: ENOMEM, pool 0, not settable in the
container), `engine` = `core/device/aql.cpp` `pin_pieces()`: private anon, 2 MiB aligned, MADV_HUGEPAGE + DONTFORK,
4-thread prefault, `hsa_amd_memory_lock_to_pool(host fine-grained, all cards)` — HIP does not know these ranges, so
`hip1/hip2` on `engine` fall back to HIP's staged pageable copy (the 12-13 GB/s `l058` rows; irrelevant to the engine,
which uses its own SDMA queue). Both cards' buffers are allocated back to back, so in one process both usually land in
the same region.

| host | path | case | card0 alone | card1 alone | both: card0 / card1 | both: aggregate [min-max] |
|---|---|---|---|---|---|---|
| engine | e0 | big | 24.8 [24.6-24.9] | 24.3 [24.1-24.4] | 18.3 / 18.0 | 35.93 [35.36-36.45] |
| engine | e0 | l058 | 20.4 [20.3-20.4] | 20.7 [20.5-20.8] | 16.5 / 16.9 | 32.97 [32.71-33.41] |
| engine | e0 | l143 | 22.6 [21.8-22.7] | 23.2 [22.3-23.3] | 17.0 / 17.3 | 33.98 [33.83-34.31] |
| engine | e01 | big | 24.8 [24.7-24.9] | 24.3 [24.2-24.4] | 18.6 / 18.3 | 36.58 [36.42-36.65] |
| engine | e01 | l058 | 25.4 [25.3-25.6] | 25.8 [25.7-25.8] | 19.5 / 19.7 | 38.93 [37.60-39.47] |
| engine | e01 | l143 | 26.0 [25.9-26.1] | 26.1 [26.0-26.2] | 18.6 / 19.0 | 37.31 [37.00-37.92] |
| engine | e1 | big | 24.9 [24.7-24.9] | 24.3 [24.3-24.5] | 18.6 / 18.2 | 36.44 [35.79-36.81] |
| engine | e1 | l058 | 20.4 [20.3-20.4] | 20.7 [20.5-20.8] | 16.2 / 16.6 | 32.45 [31.98-32.83] |
| engine | e1 | l143 | 22.6 [21.8-22.7] | 23.1 [22.3-23.1] | 17.0 / 17.3 | 33.96 [33.34-34.23] |
| engine | hip1 | big | 24.6 [24.4-24.6] | 24.2 [24.0-24.3] | 18.6 / 18.3 | 36.61 [36.46-36.75] |
| engine | hip1 | l058 | 12.8 [12.7-12.9] | 13.1 [12.9-13.1] | 12.8 / 12.9 | 25.53 [25.13-25.67] |
| engine | hip1 | l143 | 19.3 [16.7-19.4] | 19.3 [17.4-19.6] | 15.7 / 15.8 | 31.40 [31.02-31.99] |
| engine | hip2 | big | 24.5 [24.3-24.6] | 23.9 [23.7-24.0] | 18.4 / 18.0 | 36.05 [35.87-36.30] |
| engine | hip2 | l058 | 13.3 [13.0-13.4] | 13.5 [13.3-13.6] | 12.9 / 12.5 | 25.01 [23.66-25.88] |
| engine | hip2 | l143 | 19.6 [17.7-19.7] | 19.9 [18.3-20.0] | 15.8 / 15.8 | 31.55 [31.02-32.11] |
| engine | kern | big | 28.3 [28.3-28.3] | 28.3 [28.2-28.3] | 24.9 / 25.2 | 49.84 [49.68-49.93] |
| engine | kern | l058 | 19.9 [19.3-20.1] | 20.6 [20.1-20.9] | 16.2 / 16.7 | 32.39 [32.25-32.85] |
| engine | kern | l143 | 21.8 [21.3-22.2] | 22.4 [20.9-22.6] | 17.2 / 17.5 | 34.33 [34.15-34.75] |
| hipmalloc | e0 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.5 / 22.5 | 44.92 [44.60-45.15] |
| hipmalloc | e0 | l058 | 22.8 [22.0-22.8] | 22.8 [22.0-22.8] | 20.2 / 20.2 | 40.41 [40.29-40.60] |
| hipmalloc | e0 | l143 | 25.8 [25.8-25.8] | 25.8 [25.6-25.8] | 21.5 / 21.5 | 42.85 [42.58-43.17] |
| hipmalloc | e01 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.7 / 22.6 | 45.09 [44.83-45.17] |
| hipmalloc | e01 | l058 | 27.5 [27.4-27.5] | 27.5 [27.4-27.5] | 22.8 / 22.7 | 45.38 [45.13-45.45] |
| hipmalloc | e01 | l143 | 27.9 [27.8-27.9] | 27.9 [27.8-27.9] | 23.1 / 23.0 | 45.98 [45.80-46.18] |
| hipmalloc | e1 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.5 / 22.4 | 44.82 [44.72-44.94] |
| hipmalloc | e1 | l058 | 22.8 [22.0-22.8] | 22.8 [22.1-22.8] | 20.4 / 20.3 | 40.65 [40.46-40.86] |
| hipmalloc | e1 | l143 | 25.8 [25.8-25.8] | 25.8 [25.6-25.8] | 21.5 / 21.6 | 43.05 [42.49-43.27] |
| hipmalloc | hip1 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.4 / 22.2 | 44.49 [43.77-44.85] |
| hipmalloc | hip1 | l058 | 22.7 [21.9-22.7] | 22.7 [22.0-22.7] | 20.1 / 20.1 | 40.23 [39.77-40.34] |
| hipmalloc | hip1 | l143 | 25.7 [25.6-25.7] | 25.7 [25.5-25.7] | 20.9 / 20.9 | 41.83 [41.37-42.11] |
| hipmalloc | hip2 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.6 / 22.5 | 45.03 [44.47-45.20] |
| hipmalloc | hip2 | l058 | 27.6 [27.4-27.7] | 27.6 [27.0-27.9] | 22.6 / 21.8 | 43.54 [42.01-44.74] |
| hipmalloc | hip2 | l143 | 27.8 [27.8-27.8] | 27.8 [27.8-27.8] | 22.9 / 22.6 | 45.29 [43.57-46.02] |
| hipmalloc | kern | big | 28.3 [28.3-28.3] | 28.3 [28.3-28.3] | 24.3 / 24.1 | 48.30 [47.99-48.52] |
| hipmalloc | kern | l058 | 23.4 [23.4-23.4] | 23.4 [23.2-23.5] | 22.1 / 22.0 | 44.07 [43.07-44.42] |
| hipmalloc | kern | l143 | 26.1 [26.0-26.1] | 26.1 [26.0-26.1] | 22.7 / 22.9 | 45.43 [43.44-46.16] |
| hipmalloc_nc | e0 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.5 / 22.4 | 44.70 [44.34-45.09] |
| hipmalloc_nc | e0 | l058 | 22.8 [22.0-22.8] | 22.8 [22.0-22.8] | 20.2 / 20.2 | 40.39 [39.95-40.63] |
| hipmalloc_nc | e0 | l143 | 25.8 [25.8-25.8] | 25.8 [25.6-25.8] | 21.3 / 21.3 | 42.58 [41.45-42.87] |
| hipmalloc_nc | e01 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.4 / 22.3 | 44.57 [44.31-44.85] |
| hipmalloc_nc | e01 | l058 | 27.5 [27.4-27.5] | 27.5 [27.4-27.5] | 22.6 / 22.4 | 44.88 [44.48-45.10] |
| hipmalloc_nc | e01 | l143 | 27.9 [27.8-27.9] | 27.8 [27.8-27.9] | 22.8 / 22.7 | 45.45 [45.02-45.66] |
| hipmalloc_nc | e1 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.6 / 22.5 | 44.99 [44.81-45.31] |
| hipmalloc_nc | e1 | l058 | 22.8 [22.0-22.8] | 22.8 [22.0-22.8] | 20.2 / 20.2 | 40.39 [39.98-40.51] |
| hipmalloc_nc | e1 | l143 | 25.8 [25.8-25.8] | 25.8 [25.6-25.8] | 21.6 / 21.4 | 42.90 [42.73-43.18] |
| hipmalloc_nc | hip1 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.5 / 22.4 | 44.71 [44.19-44.96] |
| hipmalloc_nc | hip1 | l058 | 22.7 [21.9-22.7] | 22.7 [21.9-22.7] | 19.9 / 19.9 | 39.71 [39.24-40.21] |
| hipmalloc_nc | hip1 | l143 | 25.7 [25.7-25.7] | 25.7 [25.5-25.7] | 21.4 / 21.4 | 42.64 [42.42-42.87] |
| hipmalloc_nc | hip2 | big | 28.4 [28.4-28.4] | 28.4 [28.2-28.4] | 22.6 / 22.4 | 44.84 [44.74-45.06] |
| hipmalloc_nc | hip2 | l058 | 27.6 [27.5-27.8] | 27.6 [27.4-27.9] | 22.6 / 22.5 | 44.99 [42.74-45.25] |
| hipmalloc_nc | hip2 | l143 | 27.8 [27.8-27.8] | 27.8 [27.8-27.8] | 22.9 / 22.8 | 44.97 [44.64-45.61] |
| hipmalloc_nc | kern | big | 28.3 [28.3-28.3] | 28.3 [28.3-28.3] | 24.3 / 24.0 | 47.93 [47.62-48.35] |
| hipmalloc_nc | kern | l058 | 23.4 [23.4-23.4] | 23.4 [23.2-23.5] | 21.8 / 21.8 | 43.40 [41.90-43.85] |
| hipmalloc_nc | kern | l143 | 26.1 [26.0-26.1] | 26.1 [26.0-26.1] | 22.7 / 22.7 | 45.27 [44.74-46.26] |
| reg4k | e0 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.4 / 22.3 | 44.61 [44.28-45.05] |
| reg4k | e0 | l058 | 22.8 [22.0-22.8] | 22.8 [22.0-22.8] | 20.3 / 20.3 | 40.53 [40.17-40.72] |
| reg4k | e0 | l143 | 25.8 [25.8-25.8] | 25.8 [25.6-25.8] | 21.5 / 21.4 | 42.85 [42.48-43.08] |
| reg4k | e01 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.5 / 22.4 | 44.74 [44.67-44.90] |
| reg4k | e01 | l058 | 27.5 [27.4-27.5] | 27.5 [27.4-27.5] | 22.6 / 22.6 | 45.23 [45.00-45.37] |
| reg4k | e01 | l143 | 27.9 [27.8-27.9] | 27.9 [27.9-27.9] | 22.9 / 22.9 | 45.79 [45.14-46.19] |
| reg4k | e1 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.3 / 22.2 | 44.44 [43.33-44.68] |
| reg4k | e1 | l058 | 22.8 [22.0-22.8] | 22.8 [22.0-22.8] | 20.2 / 20.2 | 40.44 [40.21-40.63] |
| reg4k | e1 | l143 | 25.8 [25.8-25.8] | 25.8 [25.6-25.8] | 21.7 / 21.6 | 43.18 [42.31-43.37] |
| reg4k | hip1 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.5 / 22.4 | 44.77 [44.25-44.99] |
| reg4k | hip1 | l058 | 22.7 [21.9-22.7] | 22.7 [22.0-22.7] | 19.9 / 20.0 | 39.85 [38.95-40.31] |
| reg4k | hip1 | l143 | 25.7 [25.6-25.7] | 25.7 [25.5-25.7] | 21.1 / 21.1 | 42.24 [41.80-42.55] |
| reg4k | hip2 | big | 28.4 [28.4-28.4] | 28.4 [28.4-28.4] | 22.6 / 22.4 | 44.90 [44.51-45.01] |
| reg4k | hip2 | l058 | 27.5 [27.2-27.6] | 27.6 [27.6-27.8] | 22.6 / 22.5 | 44.97 [44.83-45.16] |
| reg4k | hip2 | l143 | 27.8 [27.8-27.8] | 27.8 [27.8-27.8] | 22.9 / 22.9 | 45.72 [45.41-45.80] |
| reg4k | kern | big | 28.3 [28.3-28.3] | 28.3 [28.3-28.3] | 24.7 / 24.5 | 48.96 [48.73-49.09] |
| reg4k | kern | l058 | 23.4 [23.4-23.5] | 23.5 [23.2-23.5] | 21.8 / 21.8 | 43.56 [43.37-43.83] |
| reg4k | kern | l143 | 26.1 [26.1-26.1] | 26.1 [26.0-26.1] | 23.4 / 23.3 | 46.58 [46.22-47.04] |
| regthp | e0 | big | 25.5 [25.4-25.6] | 24.6 [24.6-24.7] | 19.2 / 18.6 | 37.26 [37.08-37.28] |
| regthp | e0 | l058 | 20.3 [20.2-20.3] | 20.4 [20.1-20.5] | 16.4 / 16.4 | 32.70 [32.02-32.79] |
| regthp | e0 | l143 | 23.3 [23.2-23.4] | 22.5 [22.3-22.7] | 17.6 / 17.3 | 34.54 [34.01-34.67] |
| regthp | e01 | big | 25.4 [25.4-25.6] | 24.5 [24.4-24.5] | 18.9 / 18.4 | 36.79 [35.94-37.13] |
| regthp | e01 | l058 | 24.9 [24.6-25.0] | 25.4 [25.1-25.6] | 18.5 / 18.6 | 37.06 [36.54-37.40] |
| regthp | e01 | l143 | 26.4 [26.3-26.4] | 26.2 [25.9-26.3] | 19.8 / 19.3 | 38.64 [38.09-38.98] |
| regthp | e1 | big | 25.6 [25.4-25.6] | 24.4 [24.4-24.5] | 18.8 / 18.3 | 36.54 [36.26-37.03] |
| regthp | e1 | l058 | 20.0 [19.8-20.2] | 20.2 [19.9-20.3] | 16.1 / 16.1 | 32.13 [31.80-32.29] |
| regthp | e1 | l143 | 23.1 [22.3-23.2] | 22.5 [21.6-22.6] | 17.6 / 17.3 | 34.55 [34.16-34.83] |
| regthp | hip1 | big | 25.5 [25.4-25.6] | 24.7 [24.6-24.7] | 19.1 / 18.5 | 37.06 [36.83-37.41] |
| regthp | hip1 | l058 | 19.9 [19.9-20.1] | 20.2 [20.0-20.2] | 16.0 / 16.1 | 32.06 [31.39-32.61] |
| regthp | hip1 | l143 | 23.1 [22.3-23.2] | 22.3 [21.2-22.5] | 17.5 / 17.2 | 34.43 [33.91-34.75] |
| regthp | hip2 | big | 25.6 [25.5-25.6] | 24.6 [24.5-24.7] | 19.0 / 18.4 | 36.89 [36.70-37.19] |
| regthp | hip2 | l058 | 25.0 [24.7-25.1] | 25.2 [25.2-25.4] | 18.6 / 18.9 | 37.14 [36.64-37.99] |
| regthp | hip2 | l143 | 26.2 [26.2-26.3] | 25.9 [25.8-26.0] | 19.3 / 19.3 | 38.53 [38.03-38.90] |
| regthp | kern | big | 28.3 [28.3-28.3] | 28.3 [28.3-28.3] | 24.1 / 23.8 | 47.65 [47.36-47.87] |
| regthp | kern | l058 | 19.8 [19.6-20.0] | 20.1 [19.9-20.5] | 15.8 / 15.9 | 31.27 [31.06-32.02] |
| regthp | kern | l143 | 22.1 [21.5-22.3] | 21.5 [21.0-21.9] | 17.8 / 17.5 | 35.05 [34.32-35.47] |

Reading the matrix:
- **One card is link-bound**: the kernel reads 28.3 GB/s from every buffer kind (Gen4 x16 reached); SDMA reads 28.4
  from 4 KiB-page buffers but only 24.5-25.5 from THP-backed ones (`regthp`, `engine`) in the same session.
- **Both cards**: THP-backed buffers 16.5-19.7 per card, **33-37 GB/s aggregate over SDMA** (the M-20 18 GB/s), kernel
  47-50; 4 KiB-page buffers (`hipmalloc`, `hipmalloc_nc`, `reg4k`) **40-45 GB/s over SDMA**, kernel 48. In session 1
  (load ~20, same binary) all kinds gave 31-39 over SDMA. The kernel reads more than SDMA under memory contention: the
  SDMA engine keeps fewer reads in flight, so it suffers more from a loaded or narrow channel set.
- **Page size acts through placement, not translation.** With `iommu=pt` (identity domains) there is no IOMMU
  translation. A 2 MiB THP page is physically contiguous and sits in one memory region; a 512 MiB pool of 4 KiB pages
  is scattered over whatever the kernel has free, i.e. over both regions. That spreads the reads over more channels,
  which helps the latency-bound SDMA engine (one card 28.4 vs 24.8; TP2 44.5-45 vs 36) and makes no difference to the
  kernel. Same in the two-process run (raw below): 4 KiB-page buffers 40-45 aggregate, THP 33-37. Session 1 had one
  `regthp` pair at 10-15 GB/s even for one card; the session-2 repeat (raw below) ran `regthp` at 25.0 / 35.3: that
  pair had landed on a slow region. So the effect depends on what memory is free at allocation time.
- **Copy size / descriptor overhead** (ROCr's SDMA queue, one engine, `engine` buffers): one card 0.58 MiB 20.4 GB/s vs 1.43 MiB 22.6 vs
  256 MiB 24.8, i.e. ~5 us fixed per copy; alternating two engines hides it (`e01` 0.58 MiB 25.4-25.8). Under TP2 the
  difference shrinks to 33.0 / 34.0 / 35.9 aggregate (`e0`) and `e01` 38.9 for 0.58 MiB. The engine's own queue
  (`rad_xp_list`) already reaches 24.5-25.4 GB/s on one card for lists of >= 16 downs (M-20 tool below), so the
  per-copy cost there is small.
- hipMemcpyAsync on two streams behaves like `e01` (HIP spreads streams over the two engines): `hipmalloc` 0.58 MiB
  22.7 -> 27.6 one card, 40.2 -> 43.5 TP2.

## M-20 tool re-run (`/root/hy4/wt/m20/build/tests/m20_xpbench_test m20_`, same lock, session 2)

The engine's real path (`rad_xp_list`, own SDMA queue, pinned pool from `rad_dev_alloc`, both pools allocated back to
back). M-20 had 25.7 us (n=1) and 57.2 / 55.4 us two cards; now:
```
M20XP one-card down    bytes=608256 n= 1  us p10/p50/p90 =     24.9     32.2     41.3  per-copy p50   32.2 us   18.9 GB/s
M20XP one-card down    bytes=608256 n= 2  us p10/p50/p90 =     46.2     54.1     70.9  per-copy p50   27.0 us   22.5 GB/s
M20XP one-card down    bytes=608256 n= 4  us p10/p50/p90 =     96.0    118.8    151.9  per-copy p50   29.7 us   20.5 GB/s
M20XP one-card down    bytes=608256 n= 8  us p10/p50/p90 =    188.6    211.4    265.3  per-copy p50   26.4 us   23.0 GB/s
M20XP one-card down    bytes=608256 n=16  us p10/p50/p90 =    372.2    397.9    432.5  per-copy p50   24.9 us   24.5 GB/s
M20XP one-card down    bytes=608256 n=32  us p10/p50/p90 =    746.6    782.4    825.1  per-copy p50   24.5 us   24.9 GB/s
M20XP two-cards down    bytes=608256 n= 1  card0 p50     59.1 us ( 10.3 GB/s) p90     84.8 | card1 p50     62.9 us (  9.7 GB/s) p90    115.0
M20XP two-cards down    bytes=608256 n= 2  card0 p50    110.0 us ( 11.1 GB/s) p90    144.9 | card1 p50    108.4 us ( 11.2 GB/s) p90    153.6
M20XP two-cards down    bytes=608256 n= 4  card0 p50    200.9 us ( 12.1 GB/s) p90    243.6 | card1 p50    201.6 us ( 12.1 GB/s) p90    255.5
M20XP two-cards down    bytes=608256 n= 8  card0 p50    294.2 us ( 16.5 GB/s) p90    372.6 | card1 p50    297.5 us ( 16.4 GB/s) p90    399.6
M20XP two-cards down    bytes=608256 n=16  card0 p50    545.8 us ( 17.8 GB/s) p90    637.6 | card1 p50    549.5 us ( 17.7 GB/s) p90    637.7
M20XP two-cards down    bytes=608256 n=32  card0 p50   1089.6 us ( 17.9 GB/s) p90   1185.5 | card1 p50   1091.4 us ( 17.8 GB/s) p90   1202.2
M20XP one-card 48MiB  us p50 2000.7  25.2 GB/s
M20XP one-card 48MiB  us p50 1998.8  25.2 GB/s
M20XP roundtrip n=1 down: device word seen by host p50 28.8 us (p90 55.2); copies done p50 74.2 us (p90 99.6) after the memset was issued
M20XP roundtrip n=4 down: device word seen by host p50 29.4 us (p90 62.3); copies done p50 174.3 us (p90 233.9) after the memset was issued
M20XP roundtrip n=8 down: device word seen by host p50 29.5 us (p90 55.8); copies done p50 297.6 us (p90 346.1) after the memset was issued
```
(session 1, load ~20: one-card n=1 31.7 us, n=32 24.0 us/copy = 25.4 GB/s; two cards n=1 57.7 / 56.1 us, n=32
1037 / 1058 us = 18.8 / 18.4 GB/s; round trip: device word seen 28.4 us, copies done 64.0 us.) Same picture as M-20:
TP2 long lists 17.8-18.8 GB/s per card = the 33-37 GB/s same-region budget above. The n=1 two-card time (56-63 us) is
more than one 0.58 MiB copy at the shared rate (~33 us): ~24 us of start-up cost when both engines start at once
(fence/doorbell/first-read latency under contention) that this bench does not resolve further. Host handover
(L0 = copies done - device word seen - one copy) stays ~10 us (64.0 - 28.4 - 25.7).

## Conclusion: what limits two cards at once

| suspect | verdict | evidence |
|---|---|---|
| host memory (DRAM channels behind the pages) | **the limit** | same-region pair 31-37 SDMA / 33-48 kernel (THP buffers); different-region pair 46-50 (full link each); CPU+DMA share one ~52-56 GB/s budget |
| PCIe / root ports / Rome IOD fabric | not the limit | two cards reach 46-50 GB/s together when their pages are on different channels; one card 28.3 GB/s |
| page size (4 KiB vs THP), IOMMU | via placement only | iommu=pt identity domains; 4 KiB pools scatter over regions: SDMA one card 28.4 vs 24.8, TP2 40-45 vs 33-37; kernel unchanged |
| copy size / descriptors | small | ROCr queue ~5 us/copy (0.58 MiB 20.4 vs 24.8 GB/s, one card); engine's queue 24.5-25.4 GB/s for lists; TP2: 33.0 vs 35.9 |
| single SDMA engine | small | one card: none for big copies, hides the per-copy gap for small ones; TP2: +18% for 0.58 MiB on ROCr's queue (33.0 -> 38.9), +2% big |
| host-thread handover | latency, not bandwidth | ~10 us a list (L0), plus the device->host observation |

## Recommendations (expected gain from these numbers)

1. **Memory hardware (largest gain).** The goal is that both cards (and the CPU) read from a uniformly interleaved,
   fast memory.
   - 8 x 32 GB (1 DIMM per channel, 8-way, DDR4-3200 RDIMM): 204.8 GB/s theoretical, ~130-150 GB/s STREAM on Rome;
     both cards at link speed with room for the CPU. Expected: TP2 0.58 MiB copy ~24-26 us instead of ~33 us in a
     list (per card 23-25 GB/s, as the different-region pairs show now: +35-45% over 16.5-18.5), long lists
     ~46-50 GB/s aggregate; CPU-side host kernels stop starving the prefetch. 256 GB total.
   - 16 x 16 GB (2 DIMMs per channel, all equal): uniform 8-way, but 2DPC usually runs DDR4-2666/2933 on this class of
     board (170-188 GB/s theoretical, ~110-130 STREAM): the same DMA gain (needs ~52 GB/s + CPU), a bit less CPU
     headroom. 256 GB.
   - Keep the current sticks: the asymmetric population cannot give one uniform interleave; at best the first 8 x 16 GB
     are 8-way and the rest 4-way, possibly at a lower common speed. The measured ceiling in the bulk region (~55 GB/s)
     suggests the current setup is far below what 8 channels should give — verify the BIOS speed first (dmidecode on
     the host), it may be the cheapest fix (e.g. a forced low speed).
2. **Placement, if the sticks stay.** Not controllable from the engine: within one NUMA node Linux chooses physical
   pages freely, user space cannot ask for a physical range, and the container cannot even read the PFNs. Options at
   host level: (a) BIOS NPS2/NPS4 (if the board allows it with this population) makes channel groups NUMA nodes; the
   engine's `prefer_numa_node()` already puts each card's pinned memory on the card's node, which would give each card
   its own channels — expected the different-region result, 46-50 GB/s aggregate; (b) `memmap=` on the Proxmox kernel
   command line to keep the 4-way top range out of general use (costs ~64 GB; it does not raise the bulk ceiling, and
   the probe shows the "slow" region is exactly what lets the second card run at full speed, so this is not
   recommended on its own). TP2 shards are per card, so a per-card node split fits the engine; a table both ranks read
   (the Engram gather runs on the host, "every rank gathers every row") would then cross nodes.
3. **Engine, cheap and independent of hardware:**
   - Two SDMA queues for the expert prefetch under TP2: +18% aggregate for 0.58 MiB lists on ROCr's queues
     (33.0 -> 38.9 GB/s), ~+2% for big copies; on the engine's own queue expect less (its per-copy gap is already
     small), and `aql.cpp` records a 40% slowdown of a VRAM-bound kernel stream with a second engine. Worth an A/B
     only for TP2 decode; not a fix for the 18 GB/s.
   - Larger / coalesced copies: gate_up+down as one unit, adjacent experts merged: 0.58 -> 1.43 MiB gives +3% under
     TP2 (33.0 -> 34.0), single-copy 256 MiB +9% (35.9): small. The engine's lists are already >= 16 copies at decode.
   - Device-side doorbell (the device writes the SDMA ring's write pointer / doorbell itself instead of the host thread
     handing the list over): saves the ~10 us L0 per list (~30% of a 0.58 MiB TP2 copy); the 28 us until the host sees
     a device word (incl. the memset dispatch) is the upper bound of what removing the host round trip can save per
     list.
   - Host pool page size: the engine's THP pool (`pin_pieces`) costs the SDMA path here, because 2 MiB pages put a
     card's reads on one region: one card 24.8 vs 28.4 GB/s (-13%), TP2 36 vs 44.5-45 (-20%) against a 4 KiB-page
     pool in the same session. A 4 KiB-page (or deliberately interleaved) pool for the expert prefetch source is worth
     an A/B on this host: expected +15-25% prefetch bandwidth under TP2 until the memory is rebalanced. Counter-weight:
     `aql.cpp` records 10-21% slower prefill from a 4 KiB pool where an IOMMU translates (not the case here, iommu=pt),
     and the gain is placement luck, not guaranteed. hugetlbfs is not available in the container (pool 0).
   - Do not move the prefetch to a zero-copy kernel: it reads 28.3 GB/s alone and 47-48 under TP2 (more than SDMA's
     36), but it occupies CUs next to the GEMMs (M-20: copy kernels 2.3 GB/s beside a GEMM stream), and one launch a
     0.58 MiB copy gives only 20 GB/s.
   - Avoid heavy CPU-side memory traffic (host gathers, prefaulting, page-cache churn) while the prefetch streams:
     a 16-thread CPU read halved one card's DMA here.

## Raw lines

Two processes (one card each, synced per rep; rank 0 line, aggregate from both ranks' CLOCK_MONOTONIC stamps), session 2:
```
H2D proc0/2 host=engine path=e0 case=big | card0 med 17.94 min 16.56 max 18.39 | aggregate med 35.88 min 33.12 max 36.78
H2D proc0/2 host=engine path=e0 case=l058 | card0 med 16.61 min 15.78 max 16.76 | aggregate med 33.23 min 31.56 max 33.52
H2D proc0/2 host=engine path=e01 case=big | card0 med 18.43 min 16.69 max 18.58 | aggregate med 36.86 min 33.02 max 37.16
H2D proc0/2 host=engine path=e01 case=l058 | card0 med 19.03 min 18.68 max 19.27 | aggregate med 38.07 min 37.37 max 38.54
H2D proc0/2 host=engine path=kern case=big | card0 med 24.78 min 24.55 max 24.83 | aggregate med 49.56 min 49.10 max 49.66
H2D proc0/2 host=engine path=kern case=l058 | card0 med 17.02 min 16.84 max 17.13 | aggregate med 34.04 min 33.68 max 34.27
H2D proc0/2 host=hipmalloc path=hip1 case=big | card0 med 22.57 min 22.49 max 22.62 | aggregate med 44.83 min 44.69 max 45.09
H2D proc0/2 host=hipmalloc path=hip1 case=l058 | card0 med 20.24 min 20.09 max 20.29 | aggregate med 40.34 min 40.12 max 40.51
H2D proc0/2 host=hipmalloc path=e0 case=big | card0 med 22.52 min 22.41 max 22.59 | aggregate med 44.81 min 44.57 max 44.96
H2D proc0/2 host=hipmalloc path=e0 case=l058 | card0 med 20.40 min 20.35 max 20.44 | aggregate med 40.73 min 40.63 max 40.82
H2D proc0/2 host=hipmalloc path=e01 case=big | card0 med 22.66 min 22.52 max 22.76 | aggregate med 45.07 min 44.90 max 45.23
H2D proc0/2 host=hipmalloc path=e01 case=l058 | card0 med 22.59 min 22.44 max 22.75 | aggregate med 44.91 min 44.63 max 45.31
H2D proc0/2 host=hipmalloc path=kern case=big | card0 med 24.42 min 24.09 max 24.58 | aggregate med 48.38 min 47.75 max 48.66
H2D proc0/2 host=hipmalloc path=kern case=l058 | card0 med 22.07 min 21.87 max 22.14 | aggregate med 44.09 min 43.64 max 44.23
H2D proc0/2 host=reg4k path=hip1 case=big | card0 med 22.63 min 22.42 max 22.74 | aggregate med 44.93 min 44.55 max 45.20
H2D proc0/2 host=reg4k path=hip1 case=l058 | card0 med 20.14 min 20.05 max 20.32 | aggregate med 40.28 min 39.68 max 40.58
H2D proc0/2 host=reg4k path=e0 case=big | card0 med 22.57 min 22.44 max 22.72 | aggregate med 44.80 min 44.55 max 45.05
H2D proc0/2 host=reg4k path=e0 case=l058 | card0 med 20.18 min 20.13 max 20.34 | aggregate med 40.36 min 40.22 max 40.61
H2D proc0/2 host=reg4k path=e01 case=big | card0 med 22.54 min 22.36 max 22.65 | aggregate med 44.74 min 44.48 max 45.04
H2D proc0/2 host=reg4k path=e01 case=l058 | card0 med 22.64 min 22.47 max 22.76 | aggregate med 44.99 min 44.65 max 45.35
H2D proc0/2 host=reg4k path=kern case=big | card0 med 24.75 min 24.64 max 24.86 | aggregate med 48.91 min 48.69 max 49.08
H2D proc0/2 host=reg4k path=kern case=l058 | card0 med 21.79 min 21.54 max 22.03 | aggregate med 43.52 min 43.03 max 44.06
```

`regthp` repeat (order check: regthp, engine, regthp in one process; `big` rows), session 2:
```
H2D devs=0 host=regthp path=hip1 case=big | card0 med 25.01 min 24.96 max 25.10
H2D devs=01 host=regthp path=hip1 case=big | card0 med 17.53 min 17.21 max 18.20 | card1 med 17.26 min 17.14 max 17.93 | aggregate med 34.52 min 34.28 max 35.87
H2D devs=0 host=regthp path=e0 case=big | card0 med 25.00 min 24.92 max 25.15
H2D devs=01 host=regthp path=e0 case=big | card0 med 17.91 min 17.67 max 18.14 | card1 med 17.65 min 17.36 max 18.00 | aggregate med 35.30 min 34.73 max 35.82
H2D devs=0 host=regthp path=kern case=big | card0 med 28.25 min 28.21 max 28.27
H2D devs=01 host=regthp path=kern case=big | card0 med 24.10 min 24.02 max 24.26 | card1 med 24.06 min 23.94 max 24.13 | aggregate med 48.12 min 47.88 max 48.25
H2D devs=0 host=engine path=hip1 case=big | card0 med 24.28 min 24.07 max 24.54
H2D devs=01 host=engine path=hip1 case=big | card0 med 17.46 min 17.17 max 17.51 | card1 med 17.63 min 17.36 max 17.68 | aggregate med 34.91 min 34.35 max 35.02
H2D devs=0 host=engine path=e0 case=big | card0 med 24.51 min 24.29 max 24.59
H2D devs=01 host=engine path=e0 case=big | card0 med 17.58 min 17.41 max 17.72 | card1 med 17.70 min 17.62 max 17.89 | aggregate med 35.16 min 34.82 max 35.44
H2D devs=0 host=engine path=kern case=big | card0 med 28.32 min 28.31 max 28.32
H2D devs=01 host=engine path=kern case=big | card0 med 23.40 min 23.28 max 23.54 | card1 med 23.58 min 23.42 max 23.77 | aggregate med 46.81 min 46.57 max 47.08
H2D devs=0 host=regthp path=hip1 case=big | card0 med 24.71 min 24.59 max 24.79
H2D devs=01 host=regthp path=hip1 case=big | card0 med 17.85 min 17.72 max 18.03 | card1 med 17.79 min 17.66 max 17.97 | aggregate med 35.59 min 35.31 max 35.95
H2D devs=0 host=regthp path=e0 case=big | card0 med 24.98 min 24.90 max 25.02
H2D devs=01 host=regthp path=e0 case=big | card0 med 17.93 min 17.66 max 18.02 | card1 med 17.86 min 17.61 max 17.98 | aggregate med 35.73 min 35.21 max 35.95
H2D devs=0 host=regthp path=kern case=big | card0 med 28.31 min 28.26 max 28.32
H2D devs=01 host=regthp path=kern case=big | card0 med 23.92 min 23.91 max 24.02 | card1 med 23.81 min 23.79 max 23.87 | aggregate med 47.61 min 47.59 max 47.74
```
