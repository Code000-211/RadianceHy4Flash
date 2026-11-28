# Issue #296 (K-08 follow-up): where the `moe_iqr` / `moe_down_dec` gap at verify heights lives

Measurement only; no code change at the measured commit. Machine `RED-LINUX` (the local four-R9700 box), measured
2026-11-26 by adamczhang's agent. Every number here is from this machine in this session; the only quoted figures are
the issue's own (§2.3 of `optimisation-backlog.md`) and K-31's (`closed/K-31.md`), each marked as quoted.

## 0. Lane record

| | |
|---|---|
| commit | `8faf751e` (upstream default branch at measurement time) for every served row and every "head" kernel row. Before = after: nothing in the tree changes |
| bench build | `8faf751e` + the bench patch (below), built separately; used for the down row, the pooled and alias modes and the two ablation pins. The bench rows are now commit `34d6088d`, the parent of this note; the two pins stay a local patch |
| GPUs | AMD Radeon AI PRO R9700 (`gfx1201`). Kernel rows: one card, HIP 0 = PCI `0000:03:00.0`. Served: TP2 on HIP 0, 1 = `03:00.0`, `06:00.0` (both Gen5 x16). The box's fifth device (`21:00.0`, Gen4) was never visible to any run |
| power cap | 270 W, `power1_cap` read back before every served run and before the sweeps (`cap 270000000` on every card) |
| ROCm / driver | ROCm core 10.1.0 (`/opt/rocm/core-10.1`), HIP 7.16.26385, AMD clang 24.0.0git; kernel `7.0.0-44-generic` |
| container | `none (bare metal)` |
| model | keptn `hy4-flash-iqr-v4-u3.rad` (138,093,449,672 B); `ngram-placement: ram` |
| served config | `scripts/backlog/coupled03.yaml` as committed at `8faf751e` (DFly2, coupled, 5 proposals, `placement: expert_tiered`), two keys changed: the model path and `host-pool-mib: 23000` (39000 does not fit beside the engram table in this box's 121 GiB) |
| tune state | none. `r4d_selftest` launches the rows directly at their rules (it reads no tune cache), the served engine found no rad-tune cache under its `RADIANCE_HOME`, and no tune file was written (checked after the runs) |
| link | AER and link of the box's card 4 logged before and after every run: `fatal=0 nonfatal=0`, root link 32 GT/s x4, card link 32 GT/s x16, in every line |

The bench patch (`r4d_selftest.cpp`, plus K-31's two test pins ported to today's `r4d_moe_iqr.hip`). The first three
items and the "GB/s real" column are committed in `34d6088d`, so `--perfmoeiqr` at this note's commit runs them. The pins
are not committed:
- `--perfmoeiqr` with `R4D_PERF_IQR_SHAPE=hy4` now also times **Hy4's TP2 routed down** on the gate_up row's own routings,
  through the row the engine serves it with: `moe_gemm_w4a8i8`, `w4nl64a8i8h`, N 1536, K 768, two E4M3 scales per 128 of K.
  In the decode band that row is `moe_down_dec_kernel`. Every expert has its own random codes behind a pointer table,
  and the routings are taken in turn, as for gate_up.
- `R4D_PERF_IQR_POOL=n`: about one expert in n, fixed by a hash of its index, is read across the link from
  fine-grained pinned host memory (the pre-W2 pooled read), for both rows.
- `R4D_PERF_IQR_ALIAS_N=16`: expert j reads expert j % 16's bytes, which puts 28 MB a row in the card's cache. This
  gives the kernel's time with no HBM traffic, without K-31's single-hot-copy artefact.
- (local patch, not committed) K-31's `r4d_moe_iqr_pin_tpw` (1/2/4) and `r4d_moe_iqr_pin_dot_rows` (2/3/4), rebased over #315's tight grid and W2.
  The decode body takes DR as a template argument; the ablation launches are unchained and non-W2 only.
- Each line also prints "GB/s real": the routings' experts as stored (gate_up 885,248 code-plane + 6,400 scale bytes =
  891,648; down 589,824 + 18,432 = 608,256), plus the activation and its scales, plus the output. All are means over
  the 8 routings.

Gate_up rows from the head binary and the bench build read identical bytes (equal output fnv at every height). The
head binary is 1.8-2.5 % slower at T >= 320 (instantiation layout), so gate_up figures below are the head binary's.
The down kernel's code is the head's (`r4d_moe.hip` is untouched by the patch).

The committed rows give the bench build's output to the bit: at M = 8 the gate_up and down fnv are the same as the
bench build's for resident, pool 16 and alias 16 (`7ea0044bc8563148`, `9a3ed5f1776338b3`, alias down `b3fac9726b7fd37e`).

Commands (card 0, 3 reps interleaved, the row order reversed on rep 2). The first line runs at this note's commit; the
`TPW` and `DR` lines also need the local pin patch (the committed selftest does not read those two variables):

    R4D_PERF_IQR_SHAPE=hy4 R4D_PERF_IQR_B=3 R4D_PERF_IQR_PINS=0 [R4D_PERF_IQR_POOL=64|16] [R4D_PERF_IQR_ALIAS_N=16] \
      r4d_selftest libr4d.so --perfmoeiqr M 20 512 16 0 8          # T = 16 M, M = 1 8 20 30 40 48, 8 routings in turn
    ... R4D_PERF_IQR_PROJ=0 R4D_PERF_IQR_TPW=1,2,4 ...              # TPW pinned in turn (4,2,1 on rep 2)
    ... R4D_PERF_IQR_PROJ=0 R4D_PERF_IQR_PINS=0 R4D_PERF_IQR_DR=3|4 ...
    # served, 6 servers in turn (on off off on on off), each: radiance --config coupled03-local.yaml, probe.py c8, stop
    HIP_VISIBLE_DEVICES=0,1 RADIANCE_AQL_STAMP=1 RADIANCE_LOG_STEPS=1 [RADIANCE_W2=0] radiance --config ...
    python3 notes/backlog/measurements/W2-tools/w2stamps.py <run>.csv 1314 --on   # W2 on: the c8 verify tape
    python3 notes/backlog/measurements/W2-tools/w2stamps.py <run>.csv 1191        # W2 off: the c8 verify tape
    python3 notes/backlog/measurements/W2-tools/w2stamps.py <run>.csv 1137        # either: the small verify tape

The interleaving is A/B/B/A order by row within a height: three reps, with rows in forward order on reps 1 and 3 and
reversed on rep 2. Served arms run `on off off on on off`.

## 1. Answer

**At `8faf751e` the gap is 3.3 ms a pass, not ~8 ms, and it lives in `moe_iqr`'s own compute.** Today's c8 DFly2
verify pass runs at median T = 512 (400..736). With W2 on (the default) gate_up / down is **1.81** against a byte
ratio of 1.46: gate_up is 80.5 us a layer above byte scaling, 3.3 ms a pass. With W2 off it is 1.91 and 4.2 ms.

At T = 512 the synthetic, resident, cold kernel pair has the same excess, 77 us a layer.
- The resident first launches carry ~3.0 ms of the served 3.3 ms (~91 %).
- The W2 tail carries ~0.3 ms (~9 %): gate_up's tail waits for the copies.
- With W2 off, another ~0.9 ms comes from the link reads: gate_up is the first kernel to touch a miss.

The kernel deficit is not bytes. Gate_up's time with its weights in cache is still **87 %** of its cold time at
T = 480..768; down drops to **57 %**. So down is HBM-bound, and u3 gate_up is bound by its trellis decode and per-row
work, with no more than ~13 % to gain from memory.

The excess grows with rows per expert. Per MB read it goes from 0.07 us at T = 128 to 0.37 at T = 768: gate_up's real
GB/s falls from 558 to 489 while down's rises from 580 to 597.

The issue's other candidates measure as costs, not gains:
- **block count and the `off[]` scan:** TPW 2/4 is **+2.5..15 % slower** at every T >= 128, on identical bytes.
- **`dot_run` vs the WMMA crossover:** dot rows 3/4 are **+0.7..5.5 % slower**.
- **pooled-read scheduling inside the kernel:** a link-read expert costs down *more* than gate_up (27 against 14 us an
  expert at 1 in 16), so it is not gate_up's anomaly.

The rest of the issue's ~8 ms (2.27 at ~40 rows, §2.3, another lane) is not present at today's head. K-08, #315's
tight grid and W2 have merged since, and today's c8 verify is lower (32 tokens). This lane cannot split that history.

**Next kernel step:** the decode-ALU per weight of the u3 computed trellis in `moe_iqr` (`Dec<F>` in `dot_run` /
`frag()`) and its per-row cost. Grid, TPW, dot-rows and HBM levers are exhausted at this shape.
- A 2-row run on the WMMA tile (dot rows 1) is the one untested crossover.
- F-02 (routed down to u3 trellis) is estimated from bytes. This result says it should be checked against the
  trellis decode floor (gate_up at u3 has no HBM headroom).

## 2. Kernel sweep: Hy4 TP2 u3, per height (cold, resident, 8 routings in turn)

Medians of 3 reps, us a launch. Gate_up is `8faf751e`'s binary; down is the bench row (the head's `moe_down_dec_kernel`).
MB = bytes a launch really reads (the routings' experts as stored, activation, output). "Excess" = gate_up - (byte
ratio x down), and "x41" is that excess over 41 routed layers.

| T | gate_up us | down us | time ratio | MB gu / dn | byte ratio | GB/s gu / dn | excess us | x41 ms |
|---|---|---|---|---|---|---|---|---|
| 16 | 33.6 | 17.0 | **1.98** | 14.3 / 9.8 | 1.459 | 426 / 576 | +8.8 | 0.36 |
| 128 | 183.1 | 120.4 | **1.52** | 102.1 / 69.8 | 1.463 | 558 / 580 | +7.0 | 0.29 |
| 320 | 407.3 | 253.7 | **1.61** | 213.8 / 146.4 | 1.460 | 525 / 577 | +36.8 | 1.51 |
| 480 | 545.6 | 325.7 | **1.68** | 279.6 / 191.6 | 1.459 | 512 / 588 | +70.3 | 2.88 |
| 640 | 661.5 | 382.4 | **1.73** | 331.5 / 227.2 | 1.459 | 501 / 594 | +103.6 | 4.25 |
| 768 | 729.2 | 409.8 | **1.78** | 356.9 / 244.7 | 1.459 | 489 / 597 | +131.5 | 5.39 |

The real byte ratio is 1.46 at every height: an expert is 891,648 B against 608,256 B (1.466), and both kernels read
the same union of experts. The issue's 1.52 was a dense estimate. The card's peak is 640 GB/s.

At one token (T = 16) both grids are small and latency-bound, so the 1.98 there is launch shape, worth 0.36 ms a pass.
Uniform routing gives one 16-row tile per expert at every height (longest run 6 rows), so no expert is decoded twice.
Workgroups at T = 640, from the launch rules:
- gate_up: (T/2 + 64 + 1) x N/64 = 385 x 24 = 9,240 blocks of 4 waves;
- down: (min(T, 512) + T/16 + 1) x N/256 = 553 x 6 = 3,318 blocks of 4 waves.

TPW 2 halves gate_up's count and is slower (§4).

### 2.1 The no-HBM floor (`R4D_PERF_IQR_ALIAS_N=16`), bench build, medians of 3

| T | gate_up cold | gate_up floor | floor / cold | down cold | down floor | floor / cold |
|---|---|---|---|---|---|---|
| 16 | 33.6 | 30.0 | 0.89 | 17.1 | 15.5 | 0.91 |
| 128 | 182.8 | 146.4 | 0.80 | 119.9 | 74.1 | 0.62 |
| 320 | 397.6 | 331.4 | 0.83 | 251.1 | 144.7 | 0.58 |
| 480 | 531.5 | 463.3 | 0.87 | 323.5 | 185.7 | 0.57 |
| 640 | 647.6 | 560.5 | 0.87 | 382.6 | 219.0 | 0.57 |
| 768 | 712.0 | 609.0 | 0.86 | 408.6 | 236.7 | 0.58 |

At T = 640 gate_up's floor (560.5 us) is already above byte-scaled down (1.459 x 382.6 = 558 us). The floor is cache
reads (the 28 MB sit in the card's last-level cache), not zero cost; down's 0.57 shows how much a memory-bound kernel
gains there.

A two-point fit of gate_up's floor (T = 128 and 768): ~1.05 us per MB plus ~0.30 us per routed row. That predicts
543 us at T = 640 (measured 560): about a third of gate_up's time at verify heights scales with rows, the rest with
bytes decoded.

The run-map pin (`R4D_PERF_IQR_PINS=4,0`, head binary) does not engage in the decode band: it gives the rule's time and
bytes at every T (raw lines §6.2).

### 2.2 Pooled reads (fine-grained host, `R4D_PERF_IQR_POOL`), bench build, medians of 3

"used / pooled" = experts a routing reads / of them across the link. Deltas are against pool 0 of the same build.

| T | used / pooled p64 | gu p64 | dn p64 | used / pooled p16 | gu p16 | dn p16 | gu us/pooled (p16) | dn us/pooled (p16) |
|---|---|---|---|---|---|---|---|---|
| 16 | 16.0 / 0.2 | 34.8 (+0.9) | 21.4 (+4.4) | 16.0 / 1.8 | 60.7 (+26.8) | 52.9 (+35.9) | 14.9 | 19.9 |
| 128 | 114.0 / 1.4 | 184.1 (+1.0) | 122.1 (+1.7) | 114.0 / 7.8 | 251.0 (+67.9) | 289.6 (+169.2) | 8.7 | 21.7 |
| 320 | 238.6 / 3.9 | 401.2 (+3.6) | 254.1 (+0.4) | 238.6 / 17.5 | 655.4 (+257.8) | 705.2 (+451.5) | 14.7 | 25.8 |
| 480 | 311.9 / 5.4 | 534.0 (+0.9) | 330.7 (+5.0) | 311.9 / 22.2 | 866.9 (+333.8) | 908.6 (+582.9) | 15.0 | 26.3 |
| 640 | 369.5 / 6.5 | 649.0 (-0.3) | 382.9 (+0.5) | 369.5 / 25.2 | 997.9 (+348.6) | 1066.3 (+683.9) | 13.8 | 27.1 |
| 768 | 397.5 / 6.8 | 715.2 (-1.5) | 408.5 (-1.3) | 397.5 / 25.5 | 1004.9 (+288.2) | 1103.5 (+693.7) | 11.3 | 27.2 |


At 1 in 64 (about the served miss rate: ~3.7 misses a layer among ~240 experts used, ~1.5 %) the link reads hide behind the resident work in both kernels.
At 1 in 16 both become link-bound, and down pays more per pooled expert than gate_up (27 against 14 us at T = 640).
Over the whole launch that is at least 14 GB/s of link reads for down (14.9 MB in 1,066 us) and 22.5 GB/s for
gate_up (22.5 MB in 998 us). The in-kernel pooled read is therefore not
gate_up's anomaly. In serving with W2 off, gate_up absorbs more miss time only because it reads a miss first (§3).

## 3. Served: TP2 u3 DFly2 c8, AQL stamps, W2 on (default) and off

Per run: `w2stamps.py`'s per-layer medians (us) on the c8 verify tape: 1,314 records with W2 on (two launches a
kernel a layer, the first resident, the second the misses' tail), 1,191 with W2 off. Then the small verify tape
(1,137 records, c1-c3 steps, T 32..~240). "gu - 1.459 dn" sums both launches. The c8 steps' heights come from
`RADIANCE_LOG_STEPS`. The W2 counters cover the whole server run.

| run | c8 passes | gu first | gu tail | dn first | dn tail | gu/dn | gu - 1.459 dn (us a layer) | x41 (ms a pass) | pass wall p50 ms | small: gu / dn (ratio) | c8 steps, T p50 (min..max) | c8 agg tok/s, ms/step, tok/step | W2 misses r0,r1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1-on | 48 | 376.1 | 38.0 | 209.1 | 20.9 | 1.800 | 78.5 | 3.22 | 44.42 | 80.9 / 50.8 (1.59) | 71, 512 (400..704) | 376.081, None, 19.883 | [13529, 13764] |
| r2-off | 47 | 431.2 | 0.0 | 225.6 | 0.0 | 1.911 | 102.0 | 4.18 | 46.66 | 80.1 / 50.2 (1.60) | 70, 512 (416..704) | 374.254, None, 20.898 | [] |
| r3-off | 49 | 428.4 | 0.0 | 224.0 | 0.0 | 1.912 | 101.6 | 4.16 | 47.01 | 81.2 / 50.7 (1.60) | 71, 512 (416..688) | 380.454, None, 21.558 | [] |
| r4-on | 41 | 394.1 | 39.6 | 218.0 | 21.6 | 1.810 | 84.1 | 3.45 | 45.74 | 92.6 / 54.9 (1.69) | 68, 512 (448..720) | 365.501, None, 18.963 | [14251, 14722] |
| r5-on | 49 | 378.1 | 38.0 | 208.9 | 21.1 | 1.809 | 80.5 | 3.30 | 45.18 | 79.4 / 50.2 (1.58) | 69, 512 (416..736) | 401.487, None, 22.261 | [13492, 13713] |
| r6-off | 51 | 418.6 | 0.0 | 218.0 | 0.0 | 1.920 | 100.5 | 4.12 | 44.84 | 79.6 / 50.0 (1.59) | 73, 512 (400..640) | 377.478, None, 21.333 | [] |
on: median gu 416.1  dn 230.0  ratio 1.809  excess 80.5 us a layer = 3.30 ms a pass; wall 45.18; small ratio 1.59
off: median gu 428.4  dn 224.0  ratio 1.912  excess 101.6 us a layer = 4.16 ms a pass; wall 46.66; small ratio 1.60

The c8 verify heights are the same in every run (T = 16 x n_tok, median 512, 400..736; 68-73 steps a run), so the
arms compare at equal height. Splitting the W2-on 80.5 us a layer:
- the resident first launches: 378.1 - 1.459 x 209.1 = 73 us (3.0 ms a pass);
- the tails: 38.0 - 1.459 x 21.1 = 7 us (0.3 ms).

W2 off adds ~21 us a layer (0.9 ms) of link reads to gate_up's excess. The synthetic pair at T = 512 (interpolated,
§2) gives 77 us, so the served resident excess is the kernel's, and no serving-side effect is left over.

At the small tape the ratio is 1.59; the issue quoted 1.61 at one row. End to end (probe `c8`): 365-401 tok/s
aggregate in both arms, run to run.

## 4. Ablations (gate_up, us a launch, medians of 3; % against the bench build's rule on the same card)

Re-run at `8faf751e` because #315 (tight grid) and W2 changed `r4d_moe_iqr.hip` after K-31's base `0b100d6d`.
TPW pins are byte-identical to the rule at every T (fnv `== first`). Dot rows 3/4 move bytes from T = 128 up (ORD:
the dot path's sums) and are level at T = 16 (no run of 3+ rows).

| T | rule | tpw1 | tpw2 | tpw4 | dr3 | dr4 | bytes |
|---|---|---|---|---|---|---|---|
| 16 | 33.9 | 33.5 (-1.2 %) | 33.4 (-1.5 %) | 33.8 (-0.3 %) | 33.6 (-0.9 %) | 34.0 (+0.3 %) | A=B Y; tpw =rule; dr3 =rule, dr4 =rule |
| 128 | 183.1 | 183.0 (-0.1 %) | 210.6 (+15.0 %) | 202.7 (+10.7 %) | 182.3 (-0.4 %) | 184.6 (+0.8 %) | A=B Y; tpw =rule; dr3 moved, dr4 moved |
| 320 | 397.6 | 398.3 (+0.2 %) | 425.1 (+6.9 %) | 431.6 (+8.6 %) | 400.4 (+0.7 %) | 412.5 (+3.7 %) | A=B Y; tpw =rule; dr3 moved, dr4 moved |
| 480 | 533.1 | 532.3 (-0.2 %) | 562.0 (+5.4 %) | 566.8 (+6.3 %) | 541.6 (+1.6 %) | 557.9 (+4.7 %) | A=B Y; tpw =rule; dr3 moved, dr4 moved |
| 640 | 649.3 | 649.9 (+0.1 %) | 670.7 (+3.3 %) | 675.2 (+4.0 %) | 662.1 (+2.0 %) | 682.5 (+5.1 %) | A=B Y; tpw =rule; dr3 moved, dr4 moved |
| 768 | 716.7 | 715.7 (-0.1 %) | 734.3 (+2.5 %) | 738.4 (+3.0 %) | 734.6 (+2.5 %) | 756.3 (+5.5 %) | A=B Y; tpw =rule; dr3 moved, dr4 moved |


Both arms are slower at every verify height, as K-31 found before #315 (quoted, `closed/K-31.md`: TPW 2/4 +0.4..23 %,
dot rows 3/4 level to +4 %). TPW 2's loss shrinks with T (+15 % at T = 128, +2.5 % at 768; at T = 640 +3.3 % now against K-31's quoted +6.2 %)
but never reverses. Neither is the gap: the block count and the `off[]` scan are not the cost, and no threshold crosses
over in 128..768.

## 5. Overlap

Open PR #317 (K-54) changes `r4d_moe.hip`'s prefill run-map down (`moe_gemm_q_kernel`, `dmt == 4` column tiles a
block). It does not touch `moe_down_dec_kernel`, its body, or the decode-band rule `r4d_moe_down_dec_g`. The down
kernel measured here is the same with or without it; this file measures today's head, not the PR.

## 6. Raw lines

Full logs (each line carries min..max of the five timed stretches, TFLOP/s and GB/s of codes) and the stamp CSVs
are kept off-tree. Below is one line a row: us a launch for reps 1 2 3, GB/s real (rep 1), and the fnv prefix of
the output (the same fnv across arms = the same bytes).

### 6.1 Kernel sweep (`A` = `8faf751e`; `p0`/`p64`/`p16` = pool; `tpw` rows list tpw1/2/4; `dr3`/`dr4`)

```
T16   A    gu       32.6 33.8 33.6                   fnv 7bc23338
T16   dr3  gu       33.1 33.6 34.0       432.6 GB/s  fnv 7bc23338
T16   dr4  gu       33.5 34.0 34.2       428.0 GB/s  fnv 7bc23338
T16   p0   dn       16.6 17.0 17.2       591.6 GB/s  fnv 7627d3db
T16   p0   gu       33.2 33.9 34.1       430.8 GB/s  fnv 7bc23338
T16   p16  dn       52.8 52.9 53.7       185.4 GB/s  fnv 7627d3db
T16   p16  gu       60.6 60.7 60.7       236.2 GB/s  fnv 7bc23338
T16   p64  dn       21.4 21.4 21.5       457.4 GB/s  fnv 7627d3db
T16   p64  gu       34.8 34.8 35.2       410.9 GB/s  fnv 7bc23338
T16   tpw  gu tpw1  33.1 33.5 33.7       432.2 GB/s  fnv 7bc23338
T16   tpw  gu tpw2  33.4 33.4 33.7       428.6 GB/s  fnv 7bc23338
T16   tpw  gu tpw4  33.0 33.8 33.8       434.4 GB/s  fnv 7bc23338
T128  A    gu       181.1 183.1 183.7                fnv 7ea0044b
T128  dr3  gu       181.3 182.3 183.6    562.7 GB/s  fnv b6538f5f
T128  dr4  gu       182.6 184.6 185.5    559.0 GB/s  fnv b6538f5f
T128  p0   dn       120.2 120.4 120.6    581.1 GB/s  fnv 9a3ed5f1
T128  p0   gu       181.3 183.1 183.8    562.8 GB/s  fnv 7ea0044b
T128  p16  dn       289.6 291.8 288.8    241.2 GB/s  fnv 9a3ed5f1
T128  p16  gu       250.1 251.0 251.5    408.0 GB/s  fnv 7ea0044b
T128  p64  dn       121.4 122.1 122.3    575.4 GB/s  fnv 9a3ed5f1
T128  p64  gu       182.4 184.2 184.1    559.5 GB/s  fnv 7ea0044b
T128  tpw  gu tpw1  181.5 183.0 184.1    562.4 GB/s  fnv 7ea0044b
T128  tpw  gu tpw2  209.3 210.6 211.7    487.5 GB/s  fnv 7ea0044b
T128  tpw  gu tpw4  202.2 202.7 203.5    504.7 GB/s  fnv 7ea0044b
T320  A    gu       402.7 407.3 410.1                fnv e8db637e
T320  dr3  gu       398.2 400.4 405.1    536.9 GB/s  fnv 5aaf3f3d
T320  dr4  gu       408.4 412.5 415.5    523.5 GB/s  fnv 5aaf3f3d
T320  p0   dn       251.4 254.0 253.7    582.2 GB/s  fnv c37b4417
T320  p0   gu       394.6 397.6 401.9    541.7 GB/s  fnv e8db637e
T320  p16  dn       705.2 697.5 706.6    207.6 GB/s  fnv c37b4417
T320  p16  gu       655.0 655.4 655.7    326.4 GB/s  fnv e8db637e
T320  p64  dn       251.6 254.3 254.1    581.8 GB/s  fnv c37b4417
T320  p64  gu       397.5 401.2 403.9    537.8 GB/s  fnv e8db637e
T320  tpw  gu tpw1  395.7 398.3 401.4    540.3 GB/s  fnv e8db637e
T320  tpw  gu tpw2  422.3 425.1 426.0    506.3 GB/s  fnv e8db637e
T320  tpw  gu tpw4  428.5 431.6 434.3    499.0 GB/s  fnv e8db637e
T480  A    gu       537.8 545.6 548.3                fnv ebc89207
T480  dr3  gu       534.0 541.6 544.8    523.6 GB/s  fnv c7ed9b5c
T480  dr4  gu       551.7 557.9 561.5    506.8 GB/s  fnv f618a431
T480  p0   dn       322.2 325.7 326.5    594.6 GB/s  fnv 3570539a
T480  p0   gu       525.2 533.1 537.3    532.4 GB/s  fnv ebc89207
T480  p16  dn       907.5 920.8 908.6    211.1 GB/s  fnv 3570539a
T480  p16  gu       866.4 867.2 866.9    322.7 GB/s  fnv ebc89207
T480  p64  dn       327.2 330.7 331.8    585.4 GB/s  fnv 3570539a
T480  p64  gu       528.9 534.0 537.3    528.7 GB/s  fnv ebc89207
T480  tpw  gu tpw1  525.6 532.3 536.2    532.0 GB/s  fnv ebc89207
T480  tpw  gu tpw2  555.1 562.0 563.9    503.7 GB/s  fnv ebc89207
T480  tpw  gu tpw4  563.4 566.8 568.4    496.3 GB/s  fnv ebc89207
T640  A    gu       653.1 661.5 663.7                fnv 2e225852
T640  dr3  gu       657.2 662.1 664.5    504.4 GB/s  fnv 9345b418
T640  dr4  gu       672.2 682.5 684.0    493.2 GB/s  fnv 0b6c79ba
T640  p0   dn       382.5 382.4 382.0    594.1 GB/s  fnv f5f6a176
T640  p0   gu       644.7 649.3 653.4    514.2 GB/s  fnv 2e225852
T640  p16  dn       1063.2 1068.1 1066.3 213.7 GB/s  fnv f5f6a176
T640  p16  gu       998.7 997.9 997.1    331.9 GB/s  fnv 2e225852
T640  p64  dn       382.1 386.9 382.9    594.7 GB/s  fnv f5f6a176
T640  p64  gu       642.0 649.0 651.3    516.3 GB/s  fnv 2e225852
T640  tpw  gu tpw1  644.6 649.9 652.4    514.2 GB/s  fnv 2e225852
T640  tpw  gu tpw2  665.6 670.7 675.2    498.0 GB/s  fnv 2e225852
T640  tpw  gu tpw4  669.3 675.2 677.2    495.3 GB/s  fnv 2e225852
T768  A    gu       717.9 729.2 730.4                fnv 39dc76a8
T768  dr3  gu       730.8 742.4 734.6    488.3 GB/s  fnv 8888f7bf
T768  dr4  gu       752.1 756.3 757.6    474.5 GB/s  fnv c6b53c99
T768  p0   dn       408.9 409.8 409.8    598.5 GB/s  fnv 4bc887d9
T768  p0   gu       708.5 716.8 716.7    503.7 GB/s  fnv 39dc76a8
T768  p16  dn       1118.6 1103.5 1097.3 218.8 GB/s  fnv 4bc887d9
T768  p16  gu       1005.1 1004.9 1004.3 355.1 GB/s  fnv 39dc76a8
T768  p64  dn       408.5 413.0 408.3    599.1 GB/s  fnv 4bc887d9
T768  p64  gu       704.8 715.2 715.6    506.3 GB/s  fnv 39dc76a8
T768  tpw  gu tpw1  708.4 715.7 716.9    503.8 GB/s  fnv 39dc76a8
T768  tpw  gu tpw2  727.3 736.3 734.3    490.7 GB/s  fnv 39dc76a8
T768  tpw  gu tpw4  730.7 738.4 745.2    488.4 GB/s  fnv 39dc76a8
```

### 6.2 No-HBM floor and the run-map pin (`al16` = alias 16; `map` = head binary, `map4` = PINS 4, `map0` = rule)

```
T16   al16 dn       15.5 15.6 15.4       630.0 GB/s  fnv fc9639d9
T16   al16 gu       29.5 30.0 30.0       485.8 GB/s  fnv 7bc23338
T16   map  gu map0  32.9 33.1 33.3                   fnv 7bc23338
T16   map  gu map4  32.9 33.6 33.4                   fnv 7bc23338
T16   p0   dn       17.1 17.1 17.3       571.5 GB/s  fnv 7627d3db
T16   p0   gu       33.7 33.6 33.6       425.1 GB/s  fnv 7bc23338
T128  al16 dn       74.1 74.1 73.1       943.1 GB/s  fnv b3fac972
T128  al16 gu       145.9 146.4 149.0    699.3 GB/s  fnv 7ea0044b
T128  map  gu map0  182.4 182.7 182.9                fnv 7ea0044b
T128  map  gu map4  182.3 182.6 183.5                fnv 7ea0044b
T128  p0   dn       119.9 119.7 121.1    582.3 GB/s  fnv 9a3ed5f1
T128  p0   gu       181.9 182.8 183.0    561.2 GB/s  fnv 7ea0044b
T320  al16 dn       143.6 144.7 145.8    1019.1 GB/s fnv 5df1271b
T320  al16 gu       339.4 331.4 330.5    629.8 GB/s  fnv e8db637e
T320  map  gu map0  403.2 406.4 407.3                fnv e8db637e
T320  map  gu map4  403.7 406.6 407.5                fnv e8db637e
T320  p0   dn       249.8 251.4 251.1    586.0 GB/s  fnv c37b4417
T320  p0   gu       395.9 397.6 397.6    540.0 GB/s  fnv e8db637e
T480  al16 dn       182.6 185.7 187.7    1049.2 GB/s fnv 0d435ea1
T480  al16 gu       458.2 463.3 465.8    610.3 GB/s  fnv ebc89207
T480  map  gu map0  539.1 543.6 545.4                fnv ebc89207
T480  map  gu map4  539.2 542.7 546.6                fnv ebc89207
T480  p0   dn       322.9 323.6 323.5    593.2 GB/s  fnv 3570539a
T480  p0   gu       527.7 531.5 532.5    529.8 GB/s  fnv ebc89207
T640  al16 dn       216.6 219.0 220.4    1049.0 GB/s fnv 073270e8
T640  al16 gu       554.8 560.7 560.5    597.5 GB/s  fnv 2e225852
T640  map  gu map0  652.5 658.9 661.8                fnv 2e225852
T640  map  gu map4  653.6 659.3 661.9                fnv 2e225852
T640  p0   dn       382.6 380.4 382.6    593.8 GB/s  fnv f5f6a176
T640  p0   gu       645.0 647.6 648.7    514.0 GB/s  fnv 2e225852
T768  al16 dn       233.4 236.7 237.1    1048.6 GB/s fnv bf0fe19e
T768  al16 gu       613.8 606.4 609.0    581.4 GB/s  fnv 39dc76a8
T768  map  gu map0  718.1 721.9 725.6                fnv 39dc76a8
T768  map  gu map4  721.7 720.0 725.9                fnv 39dc76a8
T768  p0   dn       408.4 408.6 409.7    599.3 GB/s  fnv 4bc887d9
T768  p0   gu       708.8 712.2 712.0    503.5 GB/s  fnv 39dc76a8
```

### 6.3 Served runs

```
#### r1-on
runs/r1-on.csv: 48 passes of 1314 records; ordered dispatches a pass: median 184
gu first 376.1 380.3 15.57
dn first 209.1 211.0 8.63
gu tail 38.0 46.6 1.60
dn tail 20.9 25.9 0.87
pass wall 44.42 mean 46.93 ms
runs/r1-on.csv: 28 passes of 1137 records; ordered dispatches a pass: median 148
gu first 80.9 91.2 3.09
dn first 50.8 55.5 1.93
c8 verify steps by n_tok (T = 16 n_tok): 25:1 26:2 27:1 28:1 29:6 30:5 31:2 32:45 34:4 35:2 38:1 44:1
probe c8: {'wall': 5.446, 'gen': 2048, 'agg_tps': 376.081, 'sum_req_dec_tps': 448.264, 'decode_steps': 103.0, 'tok_per_step': 19.883}
rank 0: W2: 3690 routed layers planned, 13529 misses, 11945 units posted, 10660 read from the ring and 1285 in place by the tail (89.2% landed); 3220 lists, 33 posts drop
rank 1: W2: 3690 routed layers planned, 13764 misses, 12094 units posted, 9643 read from the ring and 2451 in place by the tail (79.7% landed); 3228 lists, 31 posts dropp
4 cap 270000000;1 post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;1 pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;
#### r2-off
runs/r2-off.csv: 47 passes of 1191 records; ordered dispatches a pass: median 178
gu first 431.2 432.4 17.46
dn first 225.6 228.8 9.30
pass wall 46.66 mean 48.37 ms
runs/r2-off.csv: 27 passes of 1137 records; ordered dispatches a pass: median 147
gu first 80.1 89.7 3.08
dn first 50.2 54.7 1.92
c8 verify steps by n_tok (T = 16 n_tok): 26:1 28:1 29:1 30:2 31:6 32:41 34:5 35:4 36:1 39:1 40:1 41:2 42:3 44:1
probe c8: {'wall': 5.472, 'gen': 2048, 'agg_tps': 374.254, 'sum_req_dec_tps': 445.054, 'decode_steps': 98.0, 'tok_per_step': 20.898}
4 cap 270000000;1 post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;1 pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;
#### r3-off
runs/r3-off.csv: 49 passes of 1191 records; ordered dispatches a pass: median 178
gu first 428.4 427.9 17.56
dn first 224.0 226.7 9.32
pass wall 47.01 mean 48.76 ms
runs/r3-off.csv: 28 passes of 1137 records; ordered dispatches a pass: median 147
gu first 81.2 90.6 3.07
dn first 50.7 55.2 1.92
c8 verify steps by n_tok (T = 16 n_tok): 26:2 28:1 29:3 30:1 31:2 32:45 34:3 35:7 36:1 39:1 40:1 41:1 42:2 43:1
probe c8: {'wall': 5.383, 'gen': 2048, 'agg_tps': 380.454, 'sum_req_dec_tps': 441.915, 'decode_steps': 95.0, 'tok_per_step': 21.558}
4 cap 270000000;1 post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;1 pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;
#### r4-on
runs/r4-on.csv: 41 passes of 1314 records; ordered dispatches a pass: median 180
gu first 394.1 398.0 16.31
dn first 218.0 218.1 8.95
gu tail 39.6 49.6 1.65
dn tail 21.6 27.8 0.88
pass wall 45.74 mean 49.05 ms
runs/r4-on.csv: 40 passes of 1137 records; ordered dispatches a pass: median 140
gu first 92.6 94.0 4.01
dn first 54.9 57.1 2.40
c8 verify steps by n_tok (T = 16 n_tok): 28:2 29:2 30:4 31:2 32:38 34:2 35:4 36:1 37:1 38:1 39:3 41:3 42:1 43:3 45:1
probe c8: {'wall': 5.603, 'gen': 2048, 'agg_tps': 365.501, 'sum_req_dec_tps': 449.304, 'decode_steps': 108.0, 'tok_per_step': 18.963}
rank 0: W2: 3526 routed layers planned, 14251 misses, 12269 units posted, 10619 read from the ring and 1650 in place by the tail (86.6% landed); 3042 lists, 80 posts drop
rank 1: W2: 3526 routed layers planned, 14722 misses, 12458 units posted, 9512 read from the ring and 2946 in place by the tail (76.4% landed); 3055 lists, 79 posts dropp
4 cap 270000000;1 post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;1 pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;
#### r5-on
runs/r5-on.csv: 49 passes of 1314 records; ordered dispatches a pass: median 182
gu first 378.1 382.4 15.73
dn first 208.9 210.4 8.66
gu tail 38.0 45.2 1.63
dn tail 21.1 25.2 0.88
pass wall 45.18 mean 46.81 ms
runs/r5-on.csv: 27 passes of 1137 records; ordered dispatches a pass: median 149
gu first 79.4 90.5 3.08
dn first 50.2 55.1 1.92
c8 verify steps by n_tok (T = 16 n_tok): 26:1 28:1 30:2 31:4 32:42 34:6 35:6 39:1 40:1 41:2 42:2 46:1
probe c8: {'wall': 5.101, 'gen': 2048, 'agg_tps': 401.487, 'sum_req_dec_tps': 471.583, 'decode_steps': 92.0, 'tok_per_step': 22.261}
rank 0: W2: 3444 routed layers planned, 13492 misses, 11823 units posted, 10590 read from the ring and 1233 in place by the tail (89.6% landed); 3063 lists, 25 posts drop
rank 1: W2: 3444 routed layers planned, 13713 misses, 11937 units posted, 9717 read from the ring and 2220 in place by the tail (81.4% landed); 3073 lists, 25 posts dropp
4 cap 270000000;1 post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;1 pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;
#### r6-off
runs/r6-off.csv: 51 passes of 1191 records; ordered dispatches a pass: median 178
gu first 418.6 420.1 17.20
dn first 218.0 222.0 9.07
pass wall 44.84 mean 47.47 ms
runs/r6-off.csv: 27 passes of 1137 records; ordered dispatches a pass: median 147
gu first 79.6 89.9 3.05
dn first 50.0 54.7 1.92
c8 verify steps by n_tok (T = 16 n_tok): 25:1 26:2 27:1 28:3 29:1 30:2 31:3 32:47 33:1 34:3 35:4 36:1 38:1 39:2 40:1
probe c8: {'wall': 5.425, 'gen': 2048, 'agg_tps': 377.478, 'sum_req_dec_tps': 431.756, 'decode_steps': 96.0, 'tok_per_step': 21.333}
4 cap 270000000;1 post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;1 pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=765 fatal=0 nonfatal=0;
```
