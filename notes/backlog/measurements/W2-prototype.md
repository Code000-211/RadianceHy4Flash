# W2 prototype (issue #309, part of #307 / M-20): measurements

Prototype of "Weg 2 with a per-layer budget" (`notes/backlog/design/M-20.md` §5-§7), behind `RADIANCE_W2`
(default off when measured below; on by default since the merge, see the dated line). Code: `abi/rad_w2.h`, `libr4d/r4d_w2.h`, `core/runtime/w2.{h,cpp}`,
`arch/common/rad_block_moe_fp8.h`. Tools: `W2-tools/` (`w2ab.py`, `w2stamps.py`, `w2fit.py`, the rad-kbench patch).

**2026-11-26: merged, default on since `ba10ee5e9`** (owner decision: nobody serves without the drafter, so the
DFly2 c8 verify pass gain below is taken and the no-spec cost of the ring reservation, -0.1..-0.4 %, is accepted).
`RADIANCE_W2=0` turns it off and gives the base launches. Follow-ups stay on #307 (late-unit causes, the post off
the critical path, the ring reservation's cost); `W2-proposals.md` has the experiments of `w2-explore`, which are
not merged.

**Verdict (before the merge): not merged.** The switch-off build is the base (same launches, same text, same speed). With the switch
on, greedy text is byte-identical and the routed block of a DFly2 c8 verify pass is 1.0-1.1 ms shorter (-3.2 %),
the pass 0.7 ms (-1.5 %, 6 interleaved runs an arm), but that is a quarter of what the design modelled (-5 ms), tok/s
at c8 does not resolve it (run-to-run +-5 %), and no-spec c1 / c8 are 0.25 % / 0.3 % slower with the switch on.
§5 has why the model was off and what the full card needs.

## 0. First: weights in uncached device memory (the step-0 microbench)

**`moe_down_dec` is 17-26 % slower a launch when its weights sit in `rad_dev_alloc_uncached` memory** (M = 1..32,
cold weights; level at M = 48). `moe_iqr` could not be timed by rad-kbench on this container (its layout hook refuses
the u3 gate_up table at the tool's TP1 declare: "cannot lay out the table 'w' from trellis:codes=u3 ... container
format error"), so the gate_up number is missing.

iks, one R9700 (`~/moebius-gpu/bin/gpu1`, 210 W), tree = base `0b100d6d2` + `W2-tools/kbench-rot-uncached.patch`
(`RAD_KBENCH_ROT_UNCACHED=1` puts the timer's rotated weight copies into uncached device memory), built with
`~/moebius-gpu/configure.sh`. `rad-kbench -m hy4-flash-iqr-v4-u3.rad --op moe_gemm_q --max-tok 48 --max-cases 2
--timing-only --bench-flush 4096 -v`, six sessions interleaved `n u n u n u`; row `moe_gemm_w4a8i8` (the kernel is
`moe_down_dec_kernel`), N 1536, K 1536, 512 experts, top 16, every launch reading its four weight tables from one of
8 copies spanning 4 GiB. us a launch, two cases a session:

| M | normal VRAM (n1 / n2 / n3) | uncached (u1 / u2 / u3) | uncached / normal (medians) |
|---|---|---|---|
| 1 | 43.0 43.1 / 42.9 42.9 / 43.0 43.1 | 53.5 53.6 / 53.6 53.7 / 53.5 53.5 | 1.247 |
| 2 | 79.0 79.2 / 79.2 79.3 / 79.3 79.1 | 97.4 99.7 / 97.0 95.8 / 98.6 96.4 | 1.227 |
| 4 | 159.4 160.6 / 159.2 159.3 / 158.7 159.6 | 186.1 193.1 / 187.0 189.2 / 192.7 192.0 | 1.196 |
| 8 | 329.0 329.1 / 329.7 327.5 / 330.9 327.1 | 410.1 401.7 / 409.4 383.8 / 390.0 392.9 | 1.208 |
| 16 | 635.0 634.5 / 637.5 636.4 / 633.0 635.2 | 732.8 735.9 / 750.7 736.8 / 764.3 756.4 | 1.171 |
| 32 | 909.1 1041.0 / 1048.5 1052.3 / 911.7 1042.8 | 1358.8 1377.3 / 1317.6 1243.7 / 1313.9 1248.5 | 1.263 |
| 48 | 1294.0 1300.2 / 1124.3 1293.1 / 1296.4 1298.4 | 1576.7 1186.6 / 1658.2 1231.7 / 1198.0 1279.6 | 0.969 |

So the ring is **ordinary device memory with a system-scope acquire** on each layer's first tail launch
(`rad_dev_acquire_next`, kept with the dispatch in a recorded pass), as the issue says for a "yes";
`RADIANCE_W2_RING=uncached` selects the uncached ring. In the engine the two are level within the runs' spread
(§4): only the tail's two to three copied experts read the ring, and the acquire has a price of its own.

## 1. Setup

Local lane (inference host): 2 x AMD Radeon AI PRO R9700 (gfx1201), TP2, `cap 210 W` read back on both cards by
`lane.sh` before every run, kernel 7.0.2-6-pve, ROCm 10.1, u3 container (`hy4-flash-iqr-v4-u3.rad`),
`ngram-placement: ram`, host pool 39000 MiB, prefix cache 16384 MiB. Both trees built per CARDS.md §1.1
(`/opt/rocm-repeatable` HIP compiler, `RAD_HIP_VOPD_REPEATABLE=ON`, `-DCMAKE_MODULE_LINKER_FLAGS=-Wl,--build-id`).

- **base** = `0b100d6d2` (merge base, default branch), tree `/root/hy4/wt/w2-base`;
- **off** = `9f1190839` (the prototype), no switch; **on** = the same with `RADIANCE_W2=1` (ring: ordinary device
  memory + acquire, the default); **unc** = `RADIANCE_W2=1 RADIANCE_W2_RING=uncached`.

Every run is one server through `scripts/backlog/lane.sh local 'scripts/backlog/run.sh <tag> <cfg> <phases> ...'`
with `RAD_SRC=/root/hy4/wt/w2-base` (one corpus for every arm), all `server-exit=0`:

| rounds | arms, in this order | config, phases | env |
|---|---|---|---|
| `k1` `k2` `k3` | base, off, on | `coupled03.yaml` `greedy,c1,c8` | `RADIANCE_AQL_STAMP=1 RADIANCE_LOG_STEPS=1` |
| `n1` `n2` `n3` | base, off, on | `nospec.yaml` `greedy,c1,c8` | - |
| `r1` `r2` `r3` | off, on, unc | `coupled03.yaml` `greedy,c8` | `RADIANCE_AQL_STAMP=1` |

The rounds ran back to back, `k1 k2 n2 k3 n3 r1 r2 r3` (`n1` a quarter of an hour before `k1`), 14:47-15:35 UTC on
2026-11-26.

What a c8 verify pass is here: 8 sequences, **32 rows most often** (`n_tok` 27..45 by the load-aware depth rule; one
run's histogram: 29 x 32, 14 x 34, 5 x 42, 4 x 41), top-k 16, so 512 sorted rows; hit 0.987, 111-118 missing units a
step, **2.9 misses a routed layer** (10,448 misses over 3,649 planned layers), 41 routed layers a pass.

## 2. Switch off against the base (no regression, the same launches)

Greedy (`probe.py greedy`, temperature 0): `5b7a3975 / f55fe020 / 0075e702 / 8f60e420` (sha256 heads; 256 / 256 / 103 /
64 tokens) in every `coupled03.yaml` run of every arm, 18 runs; `d6da4fb3 / 0699709c / e7a23c5d / 8f60e420` in every
`nospec.yaml` run, 9 runs. Byte-identical between base, off, on and unc.

`w2ab.py --seq 1191 <base k1,k2,k3> <off k1,k2,k3>`: `119 and 126 passes of 1191 records; distinct (kernel, chained)
sequences: 1 and 1, 1 in both; the same set: True` (kernel names without template arguments: `moe_iqr_kernel`'s
symbol gained a defaulted template parameter, its code did not change). 1,179 dispatches a pass in both, 176-180
ordered.

| | base (k1 / k2 / k3) | off | on |
|---|---|---|---|
| DFly2 c1, tok/s | 117.90 / 119.43 / 119.08 | 119.75 / 119.12 / 118.99 | 118.77 / 119.00 / 119.63 |
| DFly2 c1, ms a step | 26.376 / 25.724 / 26.116 | 25.968 / 25.791 / 26.134 | 26.184 / 26.131 / 25.996 |
| DFly2 c8, tok/s aggregate | 400.72 / 381.72 / 395.01 | 385.34 / 388.76 / 393.77 | 385.63 / 410.91 / 409.15 |
| ... its decode steps, tok a step | 89, 23.01 / 100, 20.48 / 96, 21.33 | 94, 21.79 / 96, 21.33 / 91, 22.51 | 94, 21.79 / 85, 24.09 / 89, 23.01 |
| no-spec c1, tok/s (n1 / n2 / n3) | 75.99 / 75.93 / 76.10 | 75.94 / 75.98 / 75.93 | 75.79 / 75.71 / 75.76 |
| no-spec c1, ms a step | 13.160 / 13.170 / 13.141 | 13.168 / 13.161 / 13.169 | 13.194 / 13.208 / 13.200 |
| no-spec c8, tok/s aggregate | 321.81 / 321.72 / 323.11 | 320.57 / 322.23 / 320.55 | 320.25 / 320.14 / 321.34 |

Off against base: level everywhere (DFly2 c1 medians 119.12 / 119.08, no-spec c1 75.94 / 75.99, c8 320.57 / 321.81,
all inside the three runs' spread). The DFly2 c8 tok/s of one run follows its trajectory (which request joins
which step; 85 to 100 decode steps for the same 2,048 tokens), so the arms cannot be read from it: the two runs that
did decode the same trajectory, `k1-off` and `k1-on` (94 steps, 21.79 tok a step), are 385.34 and 385.63 tok/s.

**On, where the form is not taken** (a DFly2 c1 verify pass is 6 rows and a no-spec pass 1 or 8, under the 16-row
threshold, so every launch is the base's): DFly2 c1 level; **no-spec c1 is 0.03 ms a step slower in all three rounds
(13.20 against 13.17, -0.25 % tok/s) and no-spec c8 0.1-0.5 % lower**. The switch still reserves the ring (34.4 MiB
a card, ~24 expert units of ~19,400) and readies the card's second DMA queue; which of the two it is was not
separated.

## 3. Switch on: the DFly2 c8 verify pass (AQL stamps)

Per routed layer (`w2stamps.py`, p50 over every layer of every stamped pass of the kind; us):

| | off k1 / k2 / k3 | on k1 / k2 / k3 |
|---|---|---|
| `moe_iqr` (first launch) | 530.3 / 540.8 / 537.5 | 452.9 / 460.9 / 452.6 |
| `moe_down_dec` (first launch) | 245.7 / 250.0 / 248.4 | 233.2 / 236.6 / 232.7 |
| router end -> resident down end (the copies' window) | 801.1 / 816.0 / 810.6 | 732.0 / 739.6 / 724.2 |
| tail `moe_iqr` (stamp incl. its wait behind the down) | | 39.5 / 39.6 / 39.9 (mean 51.5 / 51.7 / 51.1) |
| tail `moe_down_dec` | | 23.2 / 23.1 / 23.3 (mean 27.4 / 27.0 / 27.7) |
| tail, gate_up start -> down end | | 66.9 / 66.6 / 67.0 (mean 83.2 / 83.0 / 83.0) |
| **routed span: router end -> last routed down's end** | **801.1 / 816.0 / 810.6** (mean 816.3 / 830.3 / 835.9) | **795.9 / 796.5 / 782.8** (mean 806.1 / 814.6 / 799.0) |
| ordered dispatches a pass (median) | 176 / 180 / 178 of 1,179 | 174 / 174 / 175 of 1,302 |

So the resident launches lose the misses' stall: gate_up -78..-85 us a layer, down -12..-16 us; the tail gives 67 us
of it back at the median and 83 us on average (a late unit is read in place inside the tail). The three tail
launches a layer are chained and not ordered: **no ordered dispatch is added** (174-175 against 176-180).

Passes compared (`w2fit.py`): a pass's wall moves with its rows and the cards' clock (sd ~6 % inside a run), so each
pass is divided by a yardstick from inside it, the exclusive time of its 376 trunk linears (`iqr_m16_kernel`,
`gemm_i8_kpar_kernel`; no expert read, nothing W2 changes; 5.02-5.15 ms a pass in every arm). Raw lines:

```
base   k1-base.stamps.csv   40 passes  wall  49.96 ms  routed span  34.18 ms  yardstick 5.081 ms  wall/y  9.833  span/y  6.728
base   k2-base.stamps.csv   39 passes  wall  48.92 ms  routed span  33.28 ms  yardstick 5.075 ms  wall/y  9.639  span/y  6.557
base   k3-base.stamps.csv   40 passes  wall  49.60 ms  routed span  33.83 ms  yardstick 5.088 ms  wall/y  9.748  span/y  6.650
off    k1-off.stamps.csv    47 passes  wall  49.11 ms  routed span  33.47 ms  yardstick 5.054 ms  wall/y  9.717  span/y  6.622
off    k2-off.stamps.csv    43 passes  wall  49.90 ms  routed span  34.04 ms  yardstick 5.107 ms  wall/y  9.771  span/y  6.666
off    k3-off.stamps.csv    36 passes  wall  50.01 ms  routed span  34.27 ms  yardstick 5.092 ms  wall/y  9.820  span/y  6.730
on     k1-on.stamps.csv     45 passes  wall  49.26 ms  routed span  33.05 ms  yardstick 5.114 ms  wall/y  9.633  span/y  6.463
on     k2-on.stamps.csv     48 passes  wall  49.82 ms  routed span  33.40 ms  yardstick 5.148 ms  wall/y  9.679  span/y  6.488
on     k3-on.stamps.csv     38 passes  wall  49.03 ms  routed span  32.76 ms  yardstick 5.119 ms  wall/y  9.578  span/y  6.399
base   3 runs  wall/y  9.740 (sd 0.097)  span/y  6.645 (sd 0.086)  plain wall  49.49 ms  span  33.77 ms
off    3 runs  wall/y  9.769 (sd 0.052)  span/y  6.673 (sd 0.054)  plain wall  49.67 ms  span  33.93 ms  vs base: wall +0.30 % = +0.15 ms a pass, span +0.42 % = +0.14 ms a pass
on     3 runs  wall/y  9.630 (sd 0.050)  span/y  6.450 (sd 0.046)  plain wall  49.37 ms  span  33.07 ms  vs base: wall -1.13 % = -0.56 ms a pass, span -2.93 % = -0.99 ms a pass
off    r1-off.stamps.csv    43 passes  wall  49.58 ms  routed span  33.88 ms  yardstick 5.061 ms  wall/y  9.797  span/y  6.695
off    r2-off.stamps.csv    40 passes  wall  48.13 ms  routed span  32.73 ms  yardstick 5.016 ms  wall/y  9.595  span/y  6.524
off    r3-off.stamps.csv    47 passes  wall  49.30 ms  routed span  33.63 ms  yardstick 5.063 ms  wall/y  9.738  span/y  6.643
on     r1-on.stamps.csv     43 passes  wall  48.10 ms  routed span  32.28 ms  yardstick 5.041 ms  wall/y  9.543  span/y  6.403
on     r2-on.stamps.csv     44 passes  wall  48.58 ms  routed span  32.67 ms  yardstick 5.044 ms  wall/y  9.631  span/y  6.476
on     r3-on.stamps.csv     45 passes  wall  48.24 ms  routed span  32.29 ms  yardstick 5.082 ms  wall/y  9.493  span/y  6.355
unc    r1-unc.stamps.csv    40 passes  wall  49.10 ms  routed span  33.11 ms  yardstick 5.067 ms  wall/y  9.690  span/y  6.533
unc    r2-unc.stamps.csv    41 passes  wall  47.93 ms  routed span  31.97 ms  yardstick 5.080 ms  wall/y  9.435  span/y  6.293
unc    r3-unc.stamps.csv    46 passes  wall  47.40 ms  routed span  31.65 ms  yardstick 5.043 ms  wall/y  9.398  span/y  6.276
off    6 runs (k, r)  wall/y  9.740 (sd 0.080)  span/y  6.647 (sd 0.071)  plain wall  49.34 ms  span  33.67 ms
on     6 runs (k, r)  wall/y  9.593 (sd 0.068)  span/y  6.431 (sd 0.053)  plain wall  48.84 ms  span  32.74 ms  vs off: wall -1.51 % = -0.74 ms a pass, span -3.25 % = -1.09 ms a pass
unc    3 runs (r)     wall/y  9.508 (sd 0.159)  span/y  6.367 (sd 0.144)  plain wall  48.14 ms  span  32.24 ms  vs off: wall -2.38 % = -1.17 ms a pass, span -4.20 % = -1.41 ms a pass
```

- **Routed span a verify pass: 33.67 -> 32.74 ms, -3.25 % = -1.09 ms** (on against off, 6 runs each, every on run
  below every off run of its round). `moe_iqr` + `moe_down_dec` summed over every launch of a pass (stamp
  durations, `w2ab.py`): 32.66 -> 31.52 ms, -1.14 ms (-3.5 %); `moe_iqr` 22.13 -> 20.85 ms, `moe_down_dec`
  10.53 -> 10.67 ms (its first launch -0.5 ms, its tail +0.65 ms).
- **Verify pass: -1.51 % = -0.74 ms** of 49.3 ms (difference of the normalised means 0.147 with a standard error of
  0.043: 3.4 sigma); against the base in the `k` rounds -1.13 %. A c8 step is this pass plus the drafter's (6 ms), so
  about -1.3 % a step.
- **tok/s at c8: not resolved.** Medians 395.0 (base) / 388.8 (off) / 409.2 (on) in the `k` rounds and 374.9 (off) /
  380.9 (on) / 386.0 (unc) in the `r` rounds, with single runs between 367 and 411 whatever the arm.

Copies (`server.log` at exit, rank 0; rank 1 within 0.5 %):

```
k1-on  W2: units landed / posted by list length: 1:647/687 2:1311/1398 3:1516/1653 4:1450/1584 5:1466/1620 6:1438/1614 7:840/945 8:238/296; 1 misses past the tail's 16 left to the first pass
k1-on  W2: 3649 routed layers planned, 10448 misses, 9797 units posted, 8906 read from the ring and 891 in place by the tail (90.9% landed); 3097 lists, 1 posts dropped
k2-on  W2: 3321 routed layers planned, 10264 misses, 9606 units posted, 8662 read from the ring and 944 in place by the tail (90.2% landed); 2852 lists, 6 posts dropped
k3-on  W2: 3321 routed layers planned, 10085 misses, 9458 units posted, 8624 read from the ring and 834 in place by the tail (91.2% landed); 2866 lists, 3 posts dropped
```

94 % of the misses are posted (the budget and the 16-expert tail bound leave 6 %), 90-91 % of the posted units are on
the card when the tail's gate_up reads them: 94 % of one-unit lists, 89-92 % at 3-6 units, 80-89 % at 7-8.

## 4. The ring: ordinary device memory + acquire against uncached (in the engine)

`r` rounds above: on (cached ring, a system-scope acquire a layer) -1.59 % a pass against off, unc -2.08 %; pooled with
the `k` rounds -1.51 % (6 runs) and -2.38 % (3 runs). The difference between the two rings (0.085 in the normalised
wall, standard error 0.10) is not resolved. Tail kernels at p50, unc r1 / r2 / r3: gate_up 44.5 / 40.1 / 38.5 us, down
24.3 / 23.3 / 22.8 us, against 39.5-39.9 and 23.1-23.3 us with the cached ring: 1-2 us a layer at most, as the
microbench's +20 % of a two-to-three-expert launch would give. Greedy text is identical with both. The cached ring
stays the default because the issue says so for a measured loss; nothing here says it is the faster one.

## 5. What was learned (why -0.7 ms and not -5 ms)

1. **The stall W2 can take back is smaller than modelled, and almost all of it is gate_up's.** With the misses out of
   the first launches, `moe_iqr` drops 530-541 -> 453-461 us and `moe_down_dec` only 246-250 -> 233-237 us a layer:
   ~80 us + ~13 us at 2.9 misses, i.e. ~28 us + ~5 us a miss and 3.8 ms a pass, where the design had 21.7 + 31.5 us a
   miss and 6.5 ms (its DFly2 c8 down slope came from the headroom fit, not from stamps). The window is 725-740 us
   (800-816 us without W2), not 620.
2. **A tail costs 50-65 us of critical path, not 12.** Resident down end to tail down end: p50 49 us, mean 61 us
   (exploratory run of the same kernels): gate_up 22, quantiser 6, down 21. The pass-height rule (same M, same kernel
   instance) gives a miss the verify pass's geometry, one wave walking a tile's whole K, which is built for
   throughput at 512 rows and takes ~20 us for a single expert however few there are. A layer without a miss
   still pays ~20 us for its three launches (10 % of the layers; "no tape jump" costs 0.08 ms a pass, the working
   tails 2.5).
3. **Early-exit launches are not free at this grid.** With the tail on the first launch's grid (every block of
   12,700 runs the offset search before it finds nothing) the two tail GEMMs took 130 + 46 us a layer and the pass was
   4 ms slower than without W2. The tail's grid is therefore bounded to the tiles 16 misses can make
   (`rad_w2_tail_tiles`: 16 + T / 16 against 529 + 1 at T = 512..768) and its search numbers the misses' tiles alone;
   a layer with more misses leaves the rest to the first pass (0-63 of ~10,500 misses a run). Geometry per expert,
   and so every sum, is unchanged: r4d_selftest holds first launch + tail to the one launch's bytes.
4. **The plan must not fence.** Tags and release fences behind the mailbox stores (the prefetch's way) made the
   router's launch 10 us longer a layer; a sum over the post's words that the host thread checks costs nothing on
   the card.
5. **The done word belongs on the card and should count units.** Read from pinned host memory it cost the tail a
   link round trip; one word a list threw a whole list away for one late unit (79-85 % landed). It is now uncached
   device memory behind the ring, written by the list's fence behind every unit (`RAD_XP_DONE_EACH`): 90-91 %.
6. **Uncached weights are slower for the GEMM (§0), the acquire is not free either (§4)**: with two or three experts
   a layer in the ring neither matters.
7. Bit identity held everywhere it was looked for: 27 greedy runs here and 14 exploratory ones before them, and the selftest's split launches against the
   plain launch byte for byte (ring, in place, a partly landed list, a moved unit).

## 6. Merge bar (issue #309)

| clause | result |
|---|---|
| switch off = byte-identical, no regression | holds (§2) |
| switch on: greedy byte-identical | holds (§2) |
| `moe_iqr` + `moe_down_dec` stall a DFly2 c8 verify pass measurably lower (AQL stamps) | holds: routed span -1.09 ms (-3.25 %), §3 |
| DFly2 c8 tok/s / ms a step better than noise | the pass is -0.74 ms (-1.5 %, 3.4 sigma over 6 + 6 runs); tok/s is inside its run-to-run spread |
| no ordered-dispatch regression | holds: 174-175 ordered of 1,302 against 176-180 of 1,179 |
| nospec / c1 not worse with the switch on | DFly2 c1 level; **no-spec c1 -0.25 %, no-spec c8 -0.1..-0.5 %** |

Not merged: the gain is real but a quarter of the modelled one and below what a served tok/s shows, and the switch
costs no-spec a quarter of a percent while idle. The branch `w2-prototype` keeps the code, its tests and this note.

## 7. For the full card

- **Overlap the tail with the resident down.** The copies have landed long before the resident gate_up ends
  (a three-unit list ~260 us into a 450 us launch); a tail gate_up + quantiser running beside the resident down
  (233 us) leaves only the tail's down (~21 us) on the critical path. It needs the tail's rows of `egu` / `eff` in
  buffers of their own (today the tail quantiser rewrites `eff`, which the resident down reads, so the tape orders
  it behind) or a row-masked quantiser, and it changes the issue's "all misses after the resident pass".
  The tails would then cost about 1 ms a pass instead of 2.9, against the 3.8 ms of stall they remove.
- **A real skip for layers without a miss** and for passes whose layers mostly have one: one miss saves ~33 us and
  its tail costs ~50.
- **The idle cost at no-spec** (ring reservation or the second DMA queue) wants separating before the switch can be
  on anywhere by default.
- The budget by distinct routed experts lands 90 %; lists of seven and eight are the late ones, so the window line
  could be 10 % shorter, or the tail could wait a bounded time for the last unit.
- `moe_iqr` on uncached memory is still unmeasured (rad-kbench cannot lay the u3 gate_up table out).

## 8. Tests

Card-less: `w2_test` (budget from the whole-unit list times, window, switch and knobs, post sum, tail grid bound),
`runtime_test` `w2_*` (off until opened, a layer's binding and epoch), `hyv4flash_arch_test` `w2_*` (switch off:
the issues it always made; on: router operand, `w2` / `w2_tail` operands, the tail's order and pass height),
`r4d_emu_moefuse` (the plan in the router's launch against a host model over three budgets, spill past 16 misses,
the block predicates): 330 cases, 0 failures; `ops_doc_check` and the `kbench*` fixture replays pass with the two
optional operands. On a card: `device_test a_dma_list_can_count_its_ranges_into_the_done_word`, `r4d_selftest
--case moe_gemm_iqr` (224 ok, 0 FAIL; 116 W2 lines) and `--case moe_gemm_w4a8i8` (734 ok, 0 FAIL; 65 W2 lines).
`ctest -LE 'gpu|oot|kbench|perf'` on the full build: 216 of 226 pass; the ten that do not (`convert_test`,
`images_test`, `videos_test`, `text_test`, `tokenizer_test`, `recipe_test`, `quant_test`, `engine_e2e_test`,
`gbnf_test_shard2`, `device_test` over its CPU budget beside seven others) miss files of this checkout or FFmpeg and
touch nothing of this change.
