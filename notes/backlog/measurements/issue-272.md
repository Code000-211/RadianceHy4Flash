# Issue #272: all cards stop together under host-memory pressure (KFD queue eviction)

A bug, not a backlog item. The bar is: the stop is there before and gone after, under the same pressure; no
regression at c16 without pressure; the dump line shows the counters. Measured by adamczhang's agent on RED,
2026-11-27.

## Commits, machine

- **before** = `e3834473` (the merge base). **after** = `d7b61527`, the code commit measured. The runs used that tree
  built before committing; only docs changed after the build.
- **The before arm is the after build with `RLIMIT_MEMLOCK` 8 MiB** (`prlimit --memlock=8388608:8388608`).
  - The new `mlock2` is then refused and the range is left exactly as the base leaves it. The base pin path is
    `pin_pieces` with no lock, and that is the only path these ranges take.
  - So the two arms differ only in the lock.
  - The before arm also has the new dump line and the new report line, which is what the bar asks to see.
  - The after arm runs with memlock unlimited.
- **Machine:**
  - 4 × AMD Radeon AI PRO R9700 (gfx1201), TP4 on HIP 0,1,2,3 (PCI 03, 06, 09, 20:00.0), power cap 270 W each;
  - Linux 7.0.0-44-generic, in-tree amdgpu, `cwsr_enable` 1, ROCm core-10.1;
  - 121 GiB RAM, 8 GiB swap file (3.5 GiB already in use at the start), THP `madvise`;
  - `vm.swappiness` 60, `compact_unevictable_allowed` 1, `compaction_proactiveness` 20.
- **The memlock limit:**
  - The login limit is 8 MiB.
  - For these runs Adam raised it to unlimited with `sudo prlimit` on the agent session's processes only. That lasts
    until those processes exit; nothing system-wide changed.
  - The 8 MiB arms lower it again per server.
- **Config:** `hy4-flash-iqr-v4-u4.rad`, `tp: 4`, `ngram-placement: ram` (the table copied into pinned memory,
  4 × 14.75 GiB plus 4 × 315 MiB), `max-num-seqs: 16`, `prefix-cache-mib: 16384`, `checkpoint-slots: 128`,
  `max-model-len: 524288`, DFly2 drafting on.
  - Startup report: `pinned by this process 88.52 GiB`. The process RSS is 95.3 GiB.
- **Load:** BetterBench 0.6.1, c16, 48 requests a round, back to back for the whole arm (`bb-c16only.json`, sampling
  0.9 / 0.95 / 20).
- **Sampling:** a sampler every 5 s read:
  - every card's `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms`. The KFD lists all five cards; the
    fifth, HIP 4, is unused;
  - `pswpin` / `pswpout`;
  - the server's `VmSwap` / `VmLck` / `VmPin` / `VmRSS`;
  - `MemAvailable` / `SwapFree`.

  The kernel log was read with `journalctl -k`; `dmesg` is not readable here.
- **Link:** all four cards at 32 GT/s x16, AER 0/0/0 before and after every arm.

## 1. The stop under real swap: there before, gone after

The balloon is `squeeze2.py`, an mlocked anonymous allocation the kernel cannot reclaim, so the rest of the system
has to swap.
- **Before:** it grows 256 MiB a second until the server's own `VmSwap` > 0 and `pswpout` rises. That point is
  17.00 GiB, with MemAvailable 5.2 GiB. It then holds for 12 min, adding 256 MiB a minute, to 19.75 GiB.
- **After:** the same profile: grown to 17.00 GiB, then the same 12 min of steps.
- **Stop rules:** swap over 90 %, the OOM killer, or the server exiting. None of them fired.

| | E: before (memlock 8 MiB) | F: after (memlock unlimited) |
|---|---|---|
| `VmLck` at start | 8 MiB | 88.53 GiB (92836652 kB) |
| `VmPin` | 0 | 0 |
| server `VmSwap` at the end of the hold | 1.45 GiB | 0.69 GiB (its unlocked heap) |
| `pswpin` over the hold (pages) | 4973994 − 2783548 = 2,190,446 (8.4 GiB) | 5218416 − 5214167 = 4,249 (17 MiB) |
| `evicted_ms` growth, card 0, from the squeeze to the end | 168 → 44505 (+44.3 s) | 879 → 948 (+69 ms) |
| "no queue advance for 5 s" | 1, at 14:59:3x | 0 |
| fatal / exit | none (recovered) | none |
| slowest c16 round, aggregate tok/s | 331.3 | 701.4 |

Raw lines from the before arm: time, `evicted_ms` for gpu_ids 2277 23334 29311 30180 64672, `pswpin`, `pswpout`,
`/proc/<pid>/status`.
```
14:56:40 260 260 260 255 259  2789554 4291548 VmLck:=8192 VmPin:=0 VmRSS:=95007436 VmSwap:=779768
14:57:10 1146 1145 1158 1137 1150  2834873 4466109 VmLck:=8192 VmPin:=0 VmRSS:=94489120 VmSwap:=1311908
14:57:55 2352 2351 2366 2344 2358  2907824 4539377 VmLck:=8192 VmPin:=0 VmRSS:=94497364 VmSwap:=1311316
14:58:40 8181 8180 8197 8174 8188  3342763 5061397 VmLck:=8192 VmPin:=0 VmRSS:=94205792 VmSwap:=1615504
14:58:55 13039 13035 13060 13035 13048  3584104 5237013 VmLck:=8192 VmPin:=0 VmRSS:=94481404 VmSwap:=1342332
14:59:26 23804 23799 23828 23801 23813  4369187 6135754 VmLck:=8192 VmPin:=0 VmRSS:=93964616 VmSwap:=1862152
14:59:41 44505 44499 44532 44504 44515  5018768 6665516 VmLck:=8192 VmPin:=0 VmRSS:=94487580 VmSwap:=1338984
15:02:41 44505 44499 44532 44504 44515  5020874 6665516 VmLck:=8192 VmPin:=0 VmRSS:=94533352 VmSwap:=1337240
```
The before arm's dump, abridged to the stuck queues and the new line:
```
E device: a synchronisation has seen no queue advance for 5 s; the queues:
E   device 0 queue 0x7d91fe4fc000: written 43428963, read 43428447 -> kernel ..iqr_m16_kernel..
E   device 1 queue 0x7d91fe4ec000: written 43268202, read 43267703 -> kernel ..ar_ihc_junction_4rank_kernel..
E   device 2 queue 0x7d91fe4dc000: written 43268199, read 43267700 -> kernel ..ar_ihc_junction_4rank_kernel..
E   device 3 queue 0x7d91fe4cc000: written 43268199, read 43267700 -> kernel ..ar_ihc_junction_4rank_kernel..
E   driver queue eviction (KFD evicted_ms, in all / since the last dump): card 0 23799, card 1 23804, card 2 23813, card 3 23828; VmSwap 1.34 GiB, VmLck 8.00 MiB
E device: nothing has advanced in the 5 s since: the GPU is not running the packet named above -- ...
```
The after arm's raw lines:
```
15:07:41 REACHED size held 17.00 GiB | server VmSwap 0 kB VmRSS 95747816 kB | swap 39% | MemAvail 5173 MiB | pswpin 5214167 pswpout 6665516
15:08:58 879 877 882 878 880  5216786 6665516 VmLck:=92836652 VmPin:=0 VmRSS:=95768500 VmSwap:=0
15:15:59 893 891 896 892 894  5217420 6746103 VmLck:=92836652 VmPin:=0 VmRSS:=95747412 VmSwap:=138016
15:18:34 939 937 942 938 940  5218229 6842354 VmLck:=92836652 VmPin:=0 VmRSS:=95470240 VmSwap:=460944
15:19:43 hold done held 19.75 GiB | server VmSwap 726044 kB VmRSS 95220908 kB | swap 51% | MemAvail 3551 MiB | pswpin 5218416 pswpout 6912924
15:22:45 948 946 952 948 950  5219185 6912924 VmLck:=92836652 VmPin:=0 VmRSS:=95266200 VmSwap:=725372
```
The kernel log had no amdgpu, kfd, svm, hung-task or OOM lines in either arm.

**What the before arm shows.** The four cards are evicted together, the counts rising within a few ms of each other.
The rise comes when the server's own pages swap and the swap-in climbs. The stop that resulted was named as an
eviction by the new line. This is the issue's signature.

**What the after arm shows.** Nothing close, under the same balloon: the pinned ranges stay resident, and only the
server's unlocked heap swaps (0.7 GiB).

**The helper's room.** Locking left the balloon the same room: 17.00 GiB reached at MemAvailable 5.2 GiB in both arms.

## 2. Reclaim forced on the server alone

The server ran in its own user scope, and `echo "512M swappiness=200" > memory.reclaim` was written to that scope
four times, 90 s apart. This is anon reclaim of that process only. Load was the same.

| step | before (C): `evicted_ms` card 0, then `VmSwap` | after (D): `evicted_ms` card 0, then `VmSwap` |
|---|---|---|
| start | 464, 0 | 0, 0 |
| 1 | 475, 508 MiB | 11, 489 MiB |
| 2 | 535, 963 MiB | 111, 940 MiB |
| 3 | 2248, 1.01 GiB | 208, 1.33 GiB |
| 4 | 4439, 1.01 GiB | 825, 1.54 GiB |
| growth | **+3975 ms** | **+825 ms** |

- **Raw lines:** `out/issue-272/repro2.log` (the PM's tree).
- **Neither arm** reached a 5 s stop.
- **The after arm still evicts some.** Its swapped pages are the process's unlocked heap, so the cards also map
  something there that reclaim may move: a driver buffer object, or a short-lived registration. The lock removes most
  of it; it does not remove all of it.

## 3. Global pressure within an 8 GiB floor

Arm A, before only. A plain balloon held MemAvailable at 10 GiB for 12 min, then at 8.5 GiB for 13 min.
- The server's `VmSwap` stayed 0: the kernel took page cache and other processes' pages instead.
- `evicted_ms` grew 4244 → 4494 (+250 ms, small steps on all cards together, with no swap). That points to page
  migration, not to swap.
- No stop.

So pressure that leaves 8 GiB available does not reach the pinned pages on this box; §1 is the pressure that does.

## 4. No regression without pressure

c16 aggregate tok/s of BetterBench rounds before any pressure, per arm (medians):

| arm | rounds | median | mean |
|---|---|---|---|
| A before | 21 | 717.1 | 719.4 |
| C before | 11 | 720.3 | 718.7 |
| E before | 7 | 724.8 | 726.3 |
| B after | 4 | 729.6 | 723.1 |
| D after | 11 | 722.4 | 721.7 |
| F after | 9 | 721.8 | 722.3 |

- Before 717–725, after 722–730: inside noise.
- Startup to ready: 29–32 s in every arm.
- Greedy sanity "17*23": `391` in arms A–D. E and F's log line was cut before the reply text.
- The lock adds no visible start time; the ranges were prefaulted already.

## Verdicts

- **The stop under real swap:** there before (one 5 s stop, +44.3 s of eviction); gone after (+69 ms). **Holds.**
- **The dump line:** printed with all four cards and VmSwap / VmLck in the before arm's real stop. **Holds.**
- **The soft-fail path on the GPU:**
  - one `not locked against swap` warning;
  - the report's `pinned by this process 88.52 GiB (88.51 GiB not locked against swap)`;
  - the `pinned-unlocked` warning.

  The start goes on. **Holds.**
- **c16 without pressure:** inside noise. **Holds.**
- **Not shown:** a fatal (60 s) stop; that needs more swap than this box has. Also not shown: whether the lock
  covers all of the eviction. §2 says not quite.
