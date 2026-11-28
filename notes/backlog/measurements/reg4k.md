# reg4k — the pinned host pool on 4 KiB pages, TP2, one binary, two arms

Issue #310 ("Host pool on 4 KiB pages instead of THP (TP2 SDMA throughput A/B)"). The lever comes from
`notes/backlog/measurements/pcie-h2d.md`: `reg4k` (mmap + `MADV_NOHUGEPAGE` + lock) copied at
28.4-43.2 GB/s where `regthp` copied at 22.3-28.4 with both TP2 cards copying at once, i.e. +20-25 %
of SDMA H2D throughput against THP-backed buffers. This asks what that is worth inside the engine.

- **Before**: `ac66d3a0` (the merge base).
- **After**: `ae9401c7` ("Let the host pool be pinned on 4 KiB pages on request"), the switch only.
- **Measured by**: Moebius, run 2772 (`run-2772`), lane `iks-A`.
- **GPUs**: 2 x Radeon AI PRO R9700 (gfx1201, 32 GB), TP2, on `iks` (4 such cards, 48 cores, 112 GB).
- **Power cap**: 210 W a card, read back by `lane_begin` from
  `/sys/class/drm/card7/device/hwmon/hwmon11/power1_cap` and `card8/.../hwmon12/power1_cap`.
- **ROCm / driver**: HIP compiler AMD clang 24.0.0git (`d6f6cb69`, ROCm 10.1's clang over the
  VOPD-repeatable libLLVM, `-DRAD_HIP_VOPD_REPEATABLE=ON`), HIP runtime ROCm 7.14 (TheRock nightly,
  gfx120X-all), host compiler the SDK devel clang 23, `$RAD_CONFIGURE` then
  `-DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201`; `amdgpu` of the kernel 7.0.14-30-pve (Proxmox 13; the
  module carries no version string).
- **Container**: none — the host's own toolchain, as the lanes of this repository run.
- **ngram-placement**: `ram` (`/dev/shm/radiance-ngram`); lane `iks-A` = port 8042, host pool
  16000 MiB, prefix cache 8192 MiB.
- **Configs**: `scripts/backlog/coupled03.yaml` (DFly2, `spechub` loaded, tp 2, `expert_tiered`,
  batch 2048, 8 seqs) and `scripts/backlog/nospec.yaml` (the same without the drafter); phases
  `c1,c8,greedy`. `RADIANCE_AQL_STAMP=1` on every run, so each run's pass kinds come from AQL stamps
  (`scripts/backlog/agg.py`).

## Why one binary serves both arms

The switch is default-off and its only effect is the advice on the host pool's own mapping, so the
unset arm of a build that carries it *is* the merge base's code path: same rows, same flags, same
selection, same arithmetic. Both arms therefore come from one binary and no build-to-build difference
(compiler, flags, layout, kernel choice) can sit in the numbers. `RADIANCE_HOST_POOL_PAGES=4k` on the
other arm makes `pin_pieces` `MADV_NOHUGEPAGE` the pool range instead of `MADV_HUGEPAGE`; the Engram
table, the landing zone and every other pinned range keep the advice they had.

## The two arms really differ, and what it costs to set up

Four boots an arm (two a session) with a temporary driver of this run (not kept), which boots the
lane's config and reads `/proc/<pid>/smaps` of the pool's own mapping (it is the mapping nearest the lane's 16000 MiB budget), boot = process start to `/health`:

| arm | boots (ms) | pool mapping | AnonHugePages |
|---|---|---|---|
| unset (THP) | 64431, 64368, 64393, 64412 | 8708 MiB, rss 8708 MiB, each boot | 8708.0 MiB = 100.0 % of size, each |
| `RADIANCE_HOST_POOL_PAGES=4k` | 67587, 67612, 67538, 67572 | 8708 MiB, rss 8708 MiB, each boot | 0.0 MiB = 0.0 % of size, each |

So the arm is what it says (100 % huge against none) and it costs 3.2 s of bring-up (64.40 -> 67.58 s
mean) for the same 8708 MiB, which is the piece-by-piece lock over 4 KiB pages.

## The rows

Six interleaved runs an arm a config: three in the first session, three in the second, the arms
alternating per rep (thp, 4k / 4k, thp / thp, 4k). Mean with the run's own [min-max] behind it; the
pass wall is the decode pass kind's median wall from the AQL stamps (the kind that holds the phase's
time: 78 records at c1 and c8 under DFly2, 255 / 253 under nospec), median over the runs that carry
stamps. `tok/s` is the probe's aggregate for the phase, `ms/step` the engine's own c1 step.

| config | phase | THP tok/s | 4 KiB tok/s | THP ms/step | 4 KiB ms/step | THP pass wall | 4 KiB pass wall |
|---|---|---|---|---|---|---|---|
| DFly2 | c1 | 138.2 [134.0-140.2] | 137.5 [133.4-139.7] | 22.870 [22.684-23.060] | 22.848 [22.752-22.977] | 3861 us [3843-3879] | 3816 us [3814-3851] |
| DFly2 | c8 | 392.6 [379.0-403.8] | 398.1 [376.9-418.0] | — | — | 5039 us [5039] | 4998 us [4995-5027] |
| nospec | c1 | 74.8 [74.7-74.9] | 74.6 [74.5-74.7] | 13.144 [13.121-13.161] | 13.166 [13.149-13.181] | 12966 us [12950-12979] | 13000 us [12983-13003] |
| nospec | c8 | 316.2 [315.1-317.5] | 316.6 [315.8-317.3] | — | — | 24072 us [24015-24074] | 24058 us [24024-24138] |

TTFT: c1 80 ms against 80 ms (DFly2) and 72 ms against 72 ms (nospec); the worst first token of c8
329 against 330 ms (DFly2) and 283 against 284 ms (nospec).

Read against the bar (§2):

1. **Output identical**: the greedy phase's texts are the same under both arms — 4 texts a config,
   one sha256 set each (`0075e702dfdf…` for DFly2, `0699709c4be2…` for nospec) across all 24 runs.
2. **The step measurably faster in ms**: no. The decode pass wall is 3861 → 3816 us at c1 under DFly2
   (-1.2 %, and the arm's own spread is 35 us), 5039 → 4998 at c8 (-0.8 %), and under nospec, where
   the pass is the same work every step, 12966 → 13000 us at c1 (+0.26 %) and 24072 → 24058 us at c8
   (-0.06 %). The c1 step is 22.870 → 22.848 ms (DFly2) and 13.144 → 13.166 ms (nospec).
3. **End to end not worse beyond noise**: c8 under DFly2 is +1.4 % (392.6 → 398.1 tok/s), which is
   inside the arm's own 379.0-418.0 spread and is not the pass: it is speculative acceptance
   (tok/step 20.28-23.54 over these runs), and nospec, whose tok/step is pinned at 1.00 / 8.03, moves
   +0.1 % at c8 and -0.2 % at c1. TTFT is the same to the millisecond.

So the 20-25 % of SDMA H2D throughput the `pcie-h2d` bench sees against THP does not reach the served
engine at TP2: the pool's pages are not what a c1 or c8 step waits on. Bring-up pays +3.2 s for it.
**Bar not met**; the switch is reverted in the same branch and the item is closed
(`notes/backlog/closed/reg4k.md`).

## How to repeat it

    scripts/backlog/lane.sh iks-A 'scripts/backlog/run.sh <tag> scripts/backlog/nospec.yaml c1,c8,greedy \
        RADIANCE_HOST_POOL_PAGES=4k RADIANCE_AQL_STAMP=1'

and the same without the variable for the other arm, the arms alternating run by run. The pool's page
state is `/proc/<pid>/smaps` of the mapping it allocates: `RADIANCE_HOST_POOL_PAGES` is set around
`host_.init` in `core/mem/pools.cpp` and read by `pin_pieces` in `core/device/aql.cpp`.

## Raw lines

Read back from the two sessions' own output by a temporary reader of this run (not kept). Session 2
carries the stamps
(`pass …` lines); session 1's stamps died with the iks tree that held them, so it carries the probe
rows only — its e2e numbers are in the table above.

### session 2, 2026-11-28 05:44Z-06:14Z, iks-A

```
reg4k-4k-cpv-1 | c1 tps=139.5 ms_step=22.795 tok/step=3.33 ttft_ms=[80] | c8 tps=391.0 ms_step=None tok/step=20.28 ttft_ms=[325, 325, 78, 325, 323, 323, 323, 323] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
    pass c1: 78 passes x 271 dispatches, wall median 3851 us
    pass c1: 76 passes x 1125 dispatches, wall median 17481 us
    pass c1: 1 passes x 1615 dispatches, wall median 71639 us
    pass c1: 1 passes x 1169 dispatches, wall median 18585 us
    pass c8: 82 passes x 266 dispatches, wall median 5027 us
    pass c8: 51 passes x 1302 dispatches, wall median 46796 us
    pass c8: 21 passes x 271 dispatches, wall median 3782 us
    pass c8: 5 passes x 1125 dispatches, wall median 20452 us
reg4k-4k-cpv-2 | c1 tps=139.7 ms_step=22.752 tok/step=3.33 ttft_ms=[79] | c8 tps=418.0 ms_step=None tok/step=23.01 ttft_ms=[328, 77, 327, 327, 327, 327, 326, 326] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
    pass c1: 78 passes x 271 dispatches, wall median 3814 us
    pass c1: 76 passes x 1125 dispatches, wall median 17528 us
    pass c1: 1 passes x 1615 dispatches, wall median 71438 us
    pass c1: 1 passes x 1169 dispatches, wall median 18365 us
    pass c8: 82 passes x 266 dispatches, wall median 4998 us
    pass c8: 47 passes x 1302 dispatches, wall median 46162 us
    pass c8: 9 passes x 271 dispatches, wall median 3982 us
    pass c8: 3 passes x 1418 dispatches, wall median 48090 us
reg4k-4k-cpv-3 | c1 tps=139.6 ms_step=22.776 tok/step=3.33 ttft_ms=[80] | c8 tps=417.0 ms_step=None tok/step=23.54 ttft_ms=[330, 77, 329, 328, 328, 328, 328, 327] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
    pass c1: 78 passes x 271 dispatches, wall median 3816 us
    pass c1: 76 passes x 1125 dispatches, wall median 17527 us
    pass c1: 1 passes x 1615 dispatches, wall median 71336 us
    pass c1: 1 passes x 1169 dispatches, wall median 18292 us
    pass c8: 82 passes x 266 dispatches, wall median 4995 us
    pass c8: 49 passes x 1302 dispatches, wall median 46513 us
    pass c8: 7 passes x 271 dispatches, wall median 4248 us
    pass c8: 3 passes x 1410 dispatches, wall median 43629 us
reg4k-4k-nsv-1 | c1 tps=74.7 ms_step=13.149 tok/step=1.00 ttft_ms=[73] | c8 tps=317.0 ms_step=None tok/step=8.03 ttft_ms=[72, 271, 270, 271, 270, 270, 270, 269] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
    pass c1: 255 passes x 1040 dispatches, wall median 12983 us
    pass c1: 1 passes x 1594 dispatches, wall median 69803 us
    pass c8: 253 passes x 1124 dispatches, wall median 24058 us
    pass c8: 1 passes x 1608 dispatches, wall median 67677 us
    pass c8: 1 passes x 1921 dispatches, wall median 195679 us
    pass c8: 1 passes x 1128 dispatches, wall median 21530 us
reg4k-4k-nsv-2 | c1 tps=74.6 ms_step=13.173 tok/step=1.00 ttft_ms=[72] | c8 tps=315.8 ms_step=None tok/step=8.03 ttft_ms=[274, 74, 273, 273, 273, 273, 272, 272] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
    pass c1: 255 passes x 1040 dispatches, wall median 13000 us
    pass c1: 1 passes x 1592 dispatches, wall median 69509 us
    pass c8: 253 passes x 1124 dispatches, wall median 24138 us
    pass c8: 1 passes x 1608 dispatches, wall median 69915 us
    pass c8: 1 passes x 1921 dispatches, wall median 195044 us
    pass c8: 1 passes x 1128 dispatches, wall median 22465 us
reg4k-4k-nsv-3 | c1 tps=74.5 ms_step=13.181 tok/step=1.00 ttft_ms=[72] | c8 tps=316.4 ms_step=None tok/step=8.03 ttft_ms=[284, 93, 93, 283, 282, 281, 281, 282] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
    pass c1: 255 passes x 1040 dispatches, wall median 13003 us
    pass c1: 1 passes x 1592 dispatches, wall median 68972 us
    pass c8: 253 passes x 1124 dispatches, wall median 24024 us
    pass c8: 1 passes x 1613 dispatches, wall median 88727 us
    pass c8: 1 passes x 1918 dispatches, wall median 185782 us
    pass c8: 1 passes x 1128 dispatches, wall median 21667 us
reg4k-thp-cpv-1 | c1 tps=134.0 ms_step=22.876 tok/step=3.20 ttft_ms=[79] | c8 tps=395.9 ms_step=None tok/step=20.69 ttft_ms=[328, 77, 327, 326, 326, 326, 325, 325] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
    pass c1: 81 passes x 271 dispatches, wall median 3852 us
    pass c1: 79 passes x 1125 dispatches, wall median 17200 us
    pass c1: 1 passes x 1615 dispatches, wall median 71097 us
    pass c1: 1 passes x 1169 dispatches, wall median 18307 us
    pass c8: 78 passes x 266 dispatches, wall median 5039 us
    pass c8: 46 passes x 1302 dispatches, wall median 46225 us
    pass c8: 23 passes x 271 dispatches, wall median 3926 us
    pass c8: 2 passes x 1450 dispatches, wall median 47820 us
reg4k-thp-cpv-2 | c1 tps=140.0 ms_step=22.723 tok/step=3.33 ttft_ms=[79] | c8 tps=399.0 ms_step=None tok/step=21.11 ttft_ms=[329, 78, 327, 328, 327, 327, 327, 327] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
    pass c1: 78 passes x 271 dispatches, wall median 3843 us
    pass c1: 76 passes x 1125 dispatches, wall median 17506 us
    pass c1: 1 passes x 1615 dispatches, wall median 70901 us
    pass c1: 1 passes x 1169 dispatches, wall median 18362 us
    pass c8: 83 passes x 266 dispatches, wall median 4968 us
    pass c8: 48 passes x 1302 dispatches, wall median 45784 us
    pass c8: 16 passes x 271 dispatches, wall median 4141 us
    pass c8: 4 passes x 1418 dispatches, wall median 44530 us
reg4k-thp-cpv-3 | c1 tps=137.9 ms_step=23.06 tok/step=3.33 ttft_ms=[81] | c8 tps=403.8 ms_step=None tok/step=22.02 ttft_ms=[75, 326, 325, 325, 325, 324, 324, 323] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
    pass c1: 78 passes x 271 dispatches, wall median 3879 us
    pass c1: 76 passes x 1125 dispatches, wall median 17507 us
    pass c1: 1 passes x 1615 dispatches, wall median 71476 us
    pass c1: 1 passes x 1169 dispatches, wall median 18241 us
    pass c8: 82 passes x 266 dispatches, wall median 5004 us
    pass c8: 46 passes x 1302 dispatches, wall median 45722 us
    pass c8: 13 passes x 271 dispatches, wall median 4279 us
    pass c8: 4 passes x 1260 dispatches, wall median 32747 us
reg4k-thp-nsv-1 | c1 tps=74.7 ms_step=13.151 tok/step=1.00 ttft_ms=[72] | c8 tps=316.6 ms_step=None tok/step=8.03 ttft_ms=[272, 74, 271, 270, 270, 268, 269, 269] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
    pass c1: 255 passes x 1040 dispatches, wall median 12966 us
    pass c1: 1 passes x 1592 dispatches, wall median 68622 us
    pass c8: 253 passes x 1124 dispatches, wall median 24072 us
    pass c8: 1 passes x 1606 dispatches, wall median 70106 us
    pass c8: 1 passes x 1921 dispatches, wall median 192862 us
    pass c8: 1 passes x 1128 dispatches, wall median 22397 us
reg4k-thp-nsv-2 | c1 tps=74.8 ms_step=13.134 tok/step=1.00 ttft_ms=[72] | c8 tps=317.5 ms_step=None tok/step=8.03 ttft_ms=[272, 73, 270, 270, 270, 270, 269, 269] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
    pass c1: 255 passes x 1040 dispatches, wall median 12950 us
    pass c1: 1 passes x 1592 dispatches, wall median 68683 us
    pass c8: 253 passes x 1124 dispatches, wall median 24015 us
    pass c8: 1 passes x 1606 dispatches, wall median 69160 us
    pass c8: 1 passes x 1921 dispatches, wall median 192838 us
    pass c8: 1 passes x 1128 dispatches, wall median 21445 us
reg4k-thp-nsv-3 | c1 tps=74.8 ms_step=13.148 tok/step=1.00 ttft_ms=[72] | c8 tps=316.2 ms_step=None tok/step=8.03 ttft_ms=[272, 74, 271, 269, 270, 270, 269, 269] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
    pass c1: 255 passes x 1040 dispatches, wall median 12979 us
    pass c1: 1 passes x 1592 dispatches, wall median 68547 us
    pass c8: 253 passes x 1124 dispatches, wall median 24074 us
    pass c8: 1 passes x 1606 dispatches, wall median 69538 us
    pass c8: 1 passes x 1921 dispatches, wall median 192629 us
    pass c8: 1 passes x 1128 dispatches, wall median 22502 us
```

### session 1, 2026-11-28 05:18Z-05:42Z, iks-A

```
reg4k-4k-cpv-1 | c1 tps=133.5 ms_step=22.974 tok/step=3.20 ttft_ms=[80] | c8 tps=376.9 ms_step=None tok/step=19.32 ttft_ms=[74, 325, 325, 325, 325, 324, 324, 324] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
reg4k-4k-cpv-2 | c1 tps=133.4 ms_step=22.977 tok/step=3.20 ttft_ms=[80] | c8 tps=393.9 ms_step=None tok/step=20.90 ttft_ms=[74, 323, 323, 322, 322, 322, 321, 320] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
reg4k-4k-cpv-3 | c1 tps=139.4 ms_step=22.814 tok/step=3.33 ttft_ms=[79] | c8 tps=392.1 ms_step=None tok/step=20.08 ttft_ms=[326, 326, 79, 325, 325, 325, 324, 323] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
reg4k-4k-nsv-1 | c1 tps=74.6 ms_step=13.176 tok/step=1.00 ttft_ms=[72] | c8 tps=316.1 ms_step=None tok/step=8.03 ttft_ms=[72, 272, 271, 270, 271, 271, 270, 270] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
reg4k-4k-nsv-2 | c1 tps=74.7 ms_step=13.158 tok/step=1.00 ttft_ms=[72] | c8 tps=317.3 ms_step=None tok/step=8.03 ttft_ms=[71, 272, 271, 270, 270, 270, 270, 270] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
reg4k-4k-nsv-3 | c1 tps=74.7 ms_step=13.162 tok/step=1.00 ttft_ms=[72] | c8 tps=316.8 ms_step=None tok/step=8.03 ttft_ms=[284, 283, 92, 91, 283, 282, 282, 282] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
reg4k-thp-cpv-1 | c1 tps=140.2 ms_step=22.684 tok/step=3.33 ttft_ms=[79] | c8 tps=393.5 ms_step=None tok/step=20.48 ttft_ms=[75, 324, 324, 323, 323, 323, 323, 323] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
reg4k-thp-cpv-2 | c1 tps=138.6 ms_step=22.955 tok/step=3.33 ttft_ms=[80] | c8 tps=384.7 ms_step=None tok/step=20.08 ttft_ms=[76, 325, 325, 325, 324, 323, 323, 323] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
reg4k-thp-cpv-3 | c1 tps=138.8 ms_step=22.924 tok/step=3.33 ttft_ms=[79] | c8 tps=379.0 ms_step=None tok/step=20.28 ttft_ms=[75, 325, 325, 325, 324, 324, 324, 323] | greedy 5b7a3975defa/256c f55fe02071d8/256c 0075e702dfdf/103c 9ae27a62e99d/64c
reg4k-thp-nsv-1 | c1 tps=74.9 ms_step=13.121 tok/step=1.00 ttft_ms=[71] | c8 tps=316.7 ms_step=None tok/step=8.03 ttft_ms=[271, 73, 270, 269, 269, 269, 269, 267] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
reg4k-thp-nsv-2 | c1 tps=74.7 ms_step=13.161 tok/step=1.00 ttft_ms=[72] | c8 tps=315.1 ms_step=None tok/step=8.03 ttft_ms=[283, 283, 91, 91, 282, 280, 281, 281] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
reg4k-thp-nsv-3 | c1 tps=74.7 ms_step=13.149 tok/step=1.00 ttft_ms=[72] | c8 tps=315.5 ms_step=None tok/step=8.03 ttft_ms=[88, 282, 281, 86, 280, 279, 279, 279] | greedy d6da4fb33eed/256c 0699709c4be2/256c e7a23c5d9a2f/102c cad756fe870b/64c
```
