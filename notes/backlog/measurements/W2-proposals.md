# W2 (issue #309, M-20 / #307): where the prototype's gain went, and what to build next

Follow-up to `W2-prototype.md` (branch `w2-prototype`, `9f1190839` + `79c2d362e`). Branch **`w2-explore`** (off
`w2-prototype`) holds the experiments of this note behind environment knobs; nothing here is for merge. Tools:
`W2-tools/w2budget.py` (per-layer time budget from AQL stamps), `W2-tools/w2arms.py` (arms compared at equal miss
counts). All timings: local lane (2 x R9700, TP2, 210 W read back by `lane.sh`, u3 container, `coupled03.yaml` /
`nospec.yaml`), 2026-11-26 16:15-@@END@@ UTC, built per CARDS.md §1.1 (`/opt/rocm-repeatable` HIP compiler, VOPD
repeatable), one corpus for every run (`RAD_SRC` = a checkout of the base `0b100d6d2`).

## 0. Result in one page

**Where the modelled -5 ms went** (DFly2 c8 verify pass, 41 routed layers, ms a pass, §2): the resident launches
do get 3.8-5.3 ms shorter (the design's stall is real at this miss rate), but the tail gives 2.4-2.7 ms
back, the plan and its post cost 0.9-1.4 ms (router +8 us a layer, and the ordered gate_up and down launches start
9-17 us later whenever a list was posted), and what is left is -1.2 ms (sd 0.8 over 4 runs; the prototype's
6 + 6 runs: -1.1 ms routed span, -0.74 ms wall). Three things the prototype's note did not have:

1. **A miss served alone behind the resident launches costs 2-3 times what it cost inside them.** With the
   budget at zero (every miss read in place by the tail) the tail alone is 15.2 ms a pass: 181 misses at 84 us
   of critical path each, where each cost ~35 us of stall inside the launches. So every miss the tail does not
   find in the ring loses ~50 us instead of gaining 35, and the 9-10 % late units plus the 6-15 % of misses that
   were never posted are 0.6-0.9 ms of the tail's 2.4-2.7 ms. **The landing rate is the lever, not the tail's
   geometry** - and the obvious repairs do not work (waiting for the done word, faster copies, keeping the
   unposted misses in the first launch: all measured, all slower or level, §5.2).
2. **Late units are not slow copies.** One-unit lists are late 6 % of the time, eight-unit lists 12-20 %; 4 KiB
   pool pages (faster SDMA) and waiting 400 polls for the done word both leave the rate at 91.5 %. A late list is
   one that did not start in time (it yields the link to a stream's hold, or its post was not seen) or whose unit
   was moved by the heat engine's promotion meanwhile; which, is not yet counted.
3. **The run-tile grid is 60 % empty blocks and they are not free**, W2 or not: 512 more empty blocks a row
   (+12,288 a launch) make the pass 2.6 ms longer (sd 1.0, 3 runs; 1,024 more: 5.7 ms), all of it in gate_up,
   so the ~7,100 empty blocks every c8 gate_up launch carries today cost about 1.5 ms a pass (5.5 ns a block). This is the same effect that made the prototype's full-grid tail
   cost 130 + 46 us, and it is the tail's 20 us floor.

**The no-spec regression** (§4) is not the ring and not the idle DMA queue as such: @@NOSPEC@@

**"One launch, misses last" does not work** (§5.1; four layouts built, all byte-identical): the posted experts'
tiles at the end of the one gate_up launch run at ~500 us after the router instead of ~730 us (the tail), so fewer
units have landed (78-89 % against 91 %) and every late one is read in place at the launch's very end, while the
list's remaining copies hold the link. Measured: +0.4..+0.8 ms against W2 off, where the tail form is -1.2 ms.
Overlapping the tail with the resident down (the prototype's first suggestion) is the same trade and is ranked
down accordingly.

**Ranked proposals** (gain = ms a DFly2 c8 verify pass of ~49.8 ms; "meas." = measured here, "est." = estimated
from the measured parts; c1 = DFly2 c1 and no-spec, where the form is not taken unless said):

| # | proposal | c8 gain | c1 / no-spec | cost | bit-exact | evidence |
|---|---|---|---|---|---|---|
| 1 | **Tight run-tile grid** for `moe_iqr` / `moe_down_dec` (grid sized for the usual tile count, a block wraps to tile `x + G` when there are more); independent of W2 | **~1.5 ms est.** from the pad slope (meas.), W2 on or off | ~0 at c1 (grids of 17-97 blocks a row); shrinks W2's tail floor (20 us a layer) too | S-M: two kernels' search loop | yes: the same tile, the same block code | §5.3 |
| 2 | **Fix the VRAM budget's queue measurement** (flips 35.4 -> 175.4 MiB a queue in 1 of ~9 starts, with or without W2; takes 307-338 of 12,823 expert slots a card) | removes a 2.4 % residency loss that costs ~25 % more misses in the runs it hits (~+2 ms a c8 pass there) | the same slots at every shape | S | n/a | §4 |
| 3 | **W2: count the late units by cause and remove the cause** (list not started: yield to a link hold, hand-over; unit moved by a promotion; post dropped) | late + unposted misses are 0.6-0.9 ms of the tail's 2.4-2.7 ms (est. from the tail's distribution and the `k0` exposure, both meas.); @@NY@@ | none | S (counters), then S-M | yes (same bytes, another address) | §5.2 |
| 4 | **W2: take the post off the critical path** (plan's mask in the router, the mailbox stores in an unordered launch beside gate_up, or a device doorbell) | 0.9-1.4 ms meas. as overhead (router +0.33, gate_up start +0.36..0.70, down start +0.25..0.39); recoverable part est. 0.6-1.0 (the plan's 3.4 us a layer stays) | makes the form affordable at c1 (today +0.8 ms a c1 step if forced, meas.) | M | yes | §2, §5.4 |
| 5 | W2: gate_up-first lists (every unit's 0.85 MiB gate_up part before any down part) | landing 77.6 -> 90.4 % where the misses run early (inline form); in the tail form only run together with the wait (no gain resolved) | none | S (built: `RADIANCE_W2_GU_FIRST`) | yes | §5.2 |
| 6 | W2: overlap tail gate_up + quantiser with the resident down | est. 1.1 ms saved (the tail's gate_up + quantiser, 27 us a layer), est. 0.5-1.5 ms lost to reads that come ~230 us earlier and land less: **net ~0, not before 3** | none | M-L (row-masked quantiser or own buffers) | yes | §5.1 |
| 7 | One launch, posted tiles last ("inline") | **+0.4..+0.8 ms slower** meas. | +0.8 ms at c1 meas. | built (4 layouts) | yes (greedy identical with the form at every pass) | §5.1 |
| 8 | Bounded wait for a late unit (`RADIANCE_W2_SPIN`), and unposted misses kept in the first launch (`RADIANCE_W2_INPLACE_FIRST`) | **slower**: wait +1.0..1.4 ms at 400 polls, +34 ms at 3000 (a late list is not about to land); in-place first +3.0..3.3 ms (the reads fight the copies: 82 % landed, 198 posts dropped) | - | built | yes | §5.2 |
| 9 | Host pool on 4 KiB pages with W2 (`reg4k-pool`) | landing 91.5 % against 89.6-91.0 %: no change; pass level | - | built (switch) | yes | §5.2 |
| 10 | Skip empty / one-miss tails | 0.08 ms (empty tails) + ~0.07 ms (one-miss lists): est., a recorded pass cannot drop the launches anyway; 1 covers the floor | - | - | yes | §3 |
| 11 | Smaller-M / latency geometry for the tail GEMMs | est. <= 0.4 ms | - | M | **no**: another K split is another f32 summation order (§6) | code |
| 12 | Per-layer adaptive cap instead of 16 / other budget line | est. < 0.1 ms: 16-35 of ~14,000 misses spill; every list length lands above the 68 % break-even | - | S | yes | §3 |
| 13 | Prefetch layer L+1's misses during layer L | est. ~0: L+1's routing is not known at L, the drafter does not run the trunk's routers, and repeat misses are what the heat engine already promotes | - | L | yes | §6 |

**Recommended next card:** 1 (tight grid) as its own item, because it pays with W2 off, needs no switch and is
byte-identical; with it, 2 (budget measurement) as a bug. For W2 itself the next step is 3 (instrument the late
units by cause, then remove the cause): the feature's ceiling at DFly2 c8 is the 3.8-5.3 ms the resident
launches lose, and the measured path to it goes through the landing rate and the plan's cost, not through
moving the tail earlier.

## 1. Setup, arms, method

Trees: `/root/hy4/wt/w2` and (for the last knobs) `/root/hy4/wt/w2r-b`, both `w2-explore`; `runs/local/<tag>/`.
Every run is one server through `scripts/backlog/lane.sh local 'scripts/backlog/run.sh <tag> coupled03.yaml c8
RADIANCE_AQL_STAMP=1 ...'` (`e` rounds: `greedy,c8` with `RADIANCE_LOG_STEPS=1`; no-spec: `nospec.yaml c1,c8`),
all `server-exit=0`, arms interleaved. The verify pass is the 1191-record pass (1314 with the tail form).

Arms (all `RADIANCE_W2=1` except `off`):

| arm | environment | what |
|---|---|---|
| `off` | - | the prototype binary, switch off |
| `on` | - | the prototype's form: resident launches, tail, ring of ordinary memory + acquire |
| `unc` | `RADIANCE_W2_RING=uncached` | the same with the uncached ring |
| `k0` | `RADIANCE_W2_KMAX=0` | the plan runs, nothing is posted: every miss read in place by the tail |
| `inl` / `inl2g` / `inl3g` / `inl4g` | `RADIANCE_W2_FORM=inline` / `inline2` / `inline3` / `inline4`, uncached ring, `g` = `RADIANCE_W2_GU_FIRST=891648` | one gate_up and one down launch, no tail (§5.1) |
| `ons`, `ongs`, `inl4gs` | `RADIANCE_W2_SPIN=400` (`s2`: 3000) | a block re-reads the done word that often before it reads in place |
| `onf`, `onfg` | `RADIANCE_W2_INPLACE_FIRST=1` | a miss that is not posted stays in the first launch |
| `onny` | `RADIANCE_W2_NO_YIELD=1` | the list does not yield the link to a stream's hold |
| `on4k` | `RADIANCE_HOST_POOL_PAGES=4k` (cherry-pick of `reg4k-pool`) | the pinned pool on 4 KiB pages |
| `pad`, `pad1k` | `RADIANCE_X_GRID_PAD=512` / `1024`, W2 off | that many more empty blocks a row in both routed GEMM launches |
| no-spec | `RADIANCE_W2_AB=reserve` / `queue` / `noreserve` | reserve the ring and open nothing / open the DMA queue and reserve nothing / W2 out of the headroom |

**Two things about comparing c8 runs** that cost this work two wrong turns:

- A sampled c8 run's trajectory sets its misses a step (105-181 here; `c8` alone has more than `greedy,c8`,
  0.982 against 0.985 hit), and the routed block follows them: over the `off` runs
  `routed block = 28.22 + 0.0574 x misses a step` (residual sd 0.46 ms, 7 runs)` (ms a pass, normalised). An arm is therefore read as its distance from that line at
  its own miss count (`w2arms.py`); a single run is good to about +-0.5 ms, so only differences above ~1 ms
  are resolved by one or two runs, and every "vs off" below is of that kind unless a sd is given.
- 1 start in ~9 the engine's VRAM budget measures a hardware queue at 175.4 MiB instead of 35.4 MiB (§4) and the
  card holds 307-338 fewer expert slots; such a run has ~25 % more misses (hit 0.978). `h2-off`, `d-k0`,
  `d-inl2k0`, `l1-inl4gs`, `q1-inl4g`, `q2-on4k`, `v1-off`, `s1-on`, `t1-off` are such runs; `w2arms.py` leaves
  them out (`k0`'s tail and first launches are still quoted, with its misses).

## 2. The budget of a DFly2 c8 verify pass

`w2budget.py`: every pass divided by its yardstick (the exclusive time of its 376 trunk linears) and scaled by
5.05 ms; ms a pass, sd over the runs in brackets. Two groups, because `greedy,c8` runs (`e`) have fewer misses
(115-130 a step) than `c8`-only runs (138-154): `off` / `on` of the `e` rounds, then `off` (`h1 j1 j2 k1 k2`),
`on` (`k1 k2`), `unc` (`h1 h2`), `k0` (`d-k0`, a 181-miss run) and `pad` (`d d2 d4`, W2 off).

| segment | off (e) | on (e) | off | on | unc | k0 | pad |
|---|---|---|---|---|---|---|---|
| router | 1.25 (0.01) | 1.59 (0.01) | 1.24 (0.01) | 1.56 (0.02) | 1.55 (0.01) | 1.38 | 1.23 (0.01) |
| router end -> gate_up start | 0.39 (0.02) | 0.75 (0.05) | 0.41 (0.01) | 1.11 (0.10) | 1.07 (0.05) | 0.47 | 0.39 (0.02) |
| gate_up (first launch) | 21.96 (0.44) | 18.80 (0.27) | 22.65 (0.22) | 18.79 (0.10) | 18.87 (0.24) | 18.23 | 25.64 (0.43) |
| gate_up end -> down start | 0.64 (0.02) | 0.88 (0.04) | 0.64 (0.01) | 1.03 (0.03) | 1.01 (0.05) | 0.69 | 0.63 (0.02) |
| down (first launch) | 10.25 (0.12) | 9.65 (0.15) | 10.92 (0.24) | 9.46 (0.11) | 9.45 (0.07) | 9.64 | 10.77 (0.28) |
| tail (first down end -> last down end) | 0 | 2.44 (0.08) | 0 | 2.73 (0.07) | 2.95 (0.12) | 15.19 | 0 |
| last down end -> gather end | 0.50 | 0.49 | 0.49 | 0.48 | 0.48 | 0.50 | 0.49 |
| rest of the layer (attention, trunk, all-reduce) | 13.00 (0.01) | 13.04 (0.04) | 12.98 (0.04) | 12.97 (0.20) | 12.95 (0.03) | 12.99 | 13.04 (0.03) |
| **routed block** (router start -> gather end) | 34.98 (0.58) | 34.61 (0.60) | 36.35 (0.46) | 35.15 (0.03) | 35.38 (0.54) | 46.10 | 39.16 (0.73) |
| **pass wall** | 48.75 (0.59) | 48.41 (0.64) | 50.16 (0.46) | 48.96 (0.27) | 49.17 (0.50) | 59.91 | 53.03 (0.72) |

Reading it, `on` against `off` in the `e` rounds / in the `c8`-only rounds (a layer is 1/41 of a pass; `router end ->
gate_up start` and `gate_up` belong together because an unordered chained launch's start stamp is its queue
time, likewise the next pair):

- **router +0.34 / +0.32 ms** (+8 us a layer): the plan. With nothing posted (`k0`) it is +0.13 ms: 3.4 us of
  plan, so ~5 us are the stores into the pinned mailbox.
- **router end -> gate_up end: -2.8 / -3.2 ms** (gate_up -3.15 / -3.86, its start +0.36 / +0.70). An ordered
  gate_up launch starts 10 us after its last predecessor's end without W2 (p50; 1,244 layers of `e1-off`) and
  24 us after it with W2 (`e1-on`: 8-40 us, and one layer in seven at the 8-15 us of `off`); with nothing posted
  the delay is gone (`k0` +0.08 ms). So it is the post or the copy list that holds the next ordered dispatch,
  not the kernel; 70 % of the layers' gate_up launches are ordered. It grows with the lists (more misses in the
  `c8`-only rounds).
- **gate_up end -> resident down end: -0.35 / -1.07 ms** (down -0.60 / -1.46, its start +0.25 / +0.39).
- **tail +2.44 / +2.73 ms** (2.95 with the uncached ring): p50 48 us, mean 58-61 us a layer. Its parts from the
  stamps of `e1-on` (1,845 layers): resident down end -> tail gate_up end 21.2 us (mean 30.0), the quantiser
  5.6 us, tail down 20.7 us (22.3). 10 % of the layers have an empty tail at 17-20 us (the three launches'
  floor); p25 41, p75 62, p90 74, **p99 321 us**, 4 % of the layers above 100 us. So: floor 0.8 ms, the misses'
  GEMMs from the ring ~1.0 ms, and the mean's excess over that, 0.6-0.9 ms, is slow tails (late or unposted
  misses read across the link with nothing to hide behind).
- Net: -0.4 / -1.2 ms here (`e2-on` is the one `on` run of nine at the line; §5 has the arms at equal misses).
  **First launches -3.8 / -5.3 ms, later starts +0.6 / +1.1, plan +0.3, tail +2.4 / +2.7.**
- The rest of the pass (13.0 ms) does not move. `pad`: §5.3.

## 3. The prototype's explanations, checked

| claim (`W2-prototype.md` §5) | check | verdict |
|---|---|---|
| the recoverable stall is ~3.8 ms a pass, ~28 us gate_up + ~5 us down a miss | first launches -3.15 and -0.60 ms at 124 misses a pass (25 + 5 us a miss), -3.86 and -1.46 ms at 150 (26 + 10 us) | holds: 3.8 ms at the prototype's miss rate, 5.3 ms at 150 misses; the later starts (§2) take 0.6-1.1 ms of it back |
| a tail costs 50-65 us of critical path, an empty one ~20 | p50 48, mean 58-61, empty 17-20 | holds; but a third of the mean is slow tails, not geometry (§2, §5.2) |
| early-exit launches are not free (130 + 46 us on the full grid) | the pad run: +12,288 empty blocks cost a gate_up launch 65-75 us | holds, and it applies to every launch of the grid, not only the tail (§5.3) |
| "a sum over the post's words ... costs nothing on the card" | router +8 us a layer (5 of them the stores), ordered gate_up start +14 us and down start +6..10 us a layer with a post: 0.9-1.4 ms a pass | **does not hold**: the plan and its post take back a quarter of what the first launches gain |
| done word on the card, 90-91 % landed | 89.6-92.1 % over 9 tail-form runs | holds |
| the two rings are level | `on` -1.20 (sd 0.83, n 4), `unc` -0.97 (n 2); `unc`'s tail is 0.2-0.5 ms longer | level within the spread; the cached ring stays the default |
| "greedy text byte-identical with the switch on" | `probe.py greedy` is four requests one after the other: a DFly2 c1 verify pass has 6 rows, under `RADIANCE_W2_MIN_ROWS` = 16, so **no pass of those runs took the form** (the stats line of such a run plans 0 layers) | the clause was vacuous. Rerun here with `RADIANCE_W2_MIN_ROWS=1`: 10,332 routed layers planned, 7,171 units posted, `5b7a3975 / f55fe020 / 0075e702 / 8f60e420` = the base's (`g-tail1`). The tail form **is** byte-identical |
| tail capped at 16 misses a layer | 16-50 of 13,400-14,100 misses spill a run | harmless |

Two more from the stamps and logs:

- **The unposted misses are in the tail too.** The budget posts 85-88 % of a `c8`-only run's misses (94 % in the
  prototype's runs with fewer misses); the rest are marked `RAD_W2_INPLACE` and read across the link by the tail,
  where §5.2 puts each at ~76 us instead of ~35 us.
- **Late is per list, not per byte**: late share by list length 1..8 over the tail-form runs: 6 / 7 / 8 / 9 /
  9 / 11 / 11 / 12-20 %.

## 4. The no-spec regression (0.25 % at c1 with the switch on and the form never taken)

@@NOSPEC_SECTION@@

## 5. Experiments

@@ARMS@@

### 5.1 One launch, the posted experts' tiles last ("inline")

The idea: no tail launches at all. The plan and the copies as before; the one gate_up launch and the one down
launch run every expert, and the kernel renumbers its blocks so that the posted experts' tiles are dispatched
last and read the ring. Per (tile, column group) the same block code runs once, so the bytes are the one
launch's by construction; greedy with the form forced at every pass (`RADIANCE_W2_MIN_ROWS=1`, 10,332-10,455
routed layers planned a run) gives the base's four sha256 heads for `inline`, `inline2`, `inline3` and `inline4`
with gate_up-first lists (`g-inl2`, `g-inl2g`, `g-inl3g`, `g-inl4g`).

- `inline`: blocks in dispatch order = the other experts' tiles, then the posted ones', then every empty block.
- `inline2`: the launch's rows keep their order; the posted tiles take the last empty blocks of the last rows.
- `inline3`: no renumbering, a posted expert's block reads the ring if its unit has landed by then (it has not:
  the launch is dispatched x fastest, so every expert's first column group runs in the first 4 % of it).
- `inline4`: the launch's own search and numbering untouched (the first launch's cost), a posted expert's blocks
  return where they stand, and the last empty blocks find the posted tiles from an eight-entry list the plan wrote.

Results in the table above: **none is faster than W2 off**, `inline3` (+2.1 ms) shows what a copy beside in-place
reads of the same units costs (the design's §7.1, now on the engine). For `inline4` the gate_up launch's end
(router end -> gate_up end, us a layer, one run each): `off` p50 559 / p90 633, the tail form's resident launch
460 / 525, `inline4` + gate_up-first **537 / 929** (p99 1,271). The posted tiles start ~500 us after the
router; 11-13 % of the units are not there, are read in place at the launch's very end, and that end is the
layer's critical path. A cached ring cannot be used with this form as built: the system-scope acquire on the
ordered gate_up launch makes the pass differ from its recording (`draft pass ... issued differently from its
recording`, `g-inl1`), so these runs use the uncached ring.

### 5.2 Late and unposted misses

- **`k0`** (budget 0: plan, no post, every miss in the tail in place): first launches at their resident-only
  times (gate_up 18.7, down 10.3 ms), **tail 15.2 ms** (370 us a layer for 4.4 misses), pass +7.4 ms against
  the miss line. 181 misses a pass: 84 us of tail each, against the 35 us each cost inside the launches.
- **Spin** (`ons`, `ongs`, `inl4gs`): waiting for the done word does not raise the landing count (91.5 % with
  400 polls) and lengthens the tail to 4.4-4.9 ms; 3,000 polls: +34 ms. The late lists are not in flight.
- **4 KiB pool** (`on4k`): 91.5 % landed, pass -1.0 ms (the tail form's own figure). No gain from faster copies.
- **gate_up-first lists**: inline form 77.6 % (`h1-inl`, `h2-inl`) -> 90.4 % (`i1-inlg`) landed, lists of 5-8 units
  72-74 % -> 85-91 %. Unit layout from `RADIANCE_W2_DEBUG`: gate_up codes +0, gate_up scales +885,248, down
  codes +891,648, down scales +1,481,472 of a 1.43 MiB unit, so the split is 891,648.
- **`RADIANCE_W2_INPLACE_FIRST`** and **`RADIANCE_W2_NO_YIELD`**: @@INPF@@

### 5.3 Empty blocks of the run-tile grid

The decode grid of `moe_iqr` is `((T + 15 min(T, 512)) / 16 + 1) x N / 64` blocks: 513 x 24 = 12,312 at the
verify pass's T = 512 sorted rows, of which the ~215 run tiles a layer has use 42 %; `moe_down_dec` has
`min(T, 512) + (T + 15) / 16 + 1` = 545 a row. Every block runs the offset search (512 loads, a wave scan, two
LDS barriers) before it knows it is empty. `RADIANCE_X_GRID_PAD=512` adds 512 such blocks a row to both launches,
W2 off:

| arm | runs | routed block against the `off` line | gate_up launches | down launches |
|---|---|---|---|---|
| `pad` (+512 a row) | 3 | +2.58 ms (sd 1.01): +1.81, +3.73, +2.21 | 25.64 ms against 22.65 (+73 us a layer) | 10.77 against 10.92 |
| `pad1k` (+1,024 a row) | 1 | +5.70 ms | 28.4 ms | 12.0 ms |

Linear: 5.4-5.7 ns an empty block at this grid's width. The down launch does not show it at 512 (its rows are
few: `N / (G x 64)` column groups). Today's c8 gate_up grid has ~296 empty blocks a row x 24 rows x 41 layers =
291,000 a pass: **~1.5-1.6 ms a pass**, estimated from the slope (the slope below today's count is not measured).
A fix that keeps every sum: size the grid along x for the usual tile count (the plan cannot tell the host, so a
fixed `distinct + T / 16` estimate from the previous pass of the same shape, or simply half of today's bound) and
let block `x` also run tiles `x + G`, `x + 2 G`, ... when the launch has more tiles than `G`: the same tile is
computed by the same code, only by another block. The tail launches of W2 (49 blocks a row for <= 16 misses) get
the same treatment and lose most of their 20 us floor.

### 5.4 The form at c1 and at no-spec c8 (forced with `RADIANCE_W2_MIN_ROWS=1`)

One run each, DFly2 c1 (`coupled03.yaml c1`, a verify pass of 6 rows): `t1-off` 22.29 ms a step (140.4 tok/s),
the tail form `t1-tail` 23.21 ms (+0.92), `inline4` + gate_up-first `t1-inl4g` 23.10 ms (+0.81); 72-76 % of the
posted units landed (the greedy runs at c1: tail form 78.9 %, `inline` 49.9 %, with gate_up-first 78.4-79.3 %).
So at c1 the form costs ~4 % today, as the design expected ("off at c1"): 41 layers x (8 us of plan + 14 us of
later start + a 20 us tail floor) is 1.7 ms against 1.9 ms of stall, of which 3/4 is reached. The 16-row threshold
is right until proposals 1 and 4 have taken the floor and the post's cost; then c1 DFly2 is worth a rerun
(est. -0.3..-0.5 ms a step of 22 ms at best). No-spec c8 with the form forced was not run.

## 6. The other ideas, from the code

- **Smaller-M geometry for the tail.** `launch_form` (`r4d_moe_iqr.hip`) picks the K split by the pass's rows:
  four waves a tile's K under 20 routed rows (`kOneWaveMinRows`), one wave walking all of K above. Each wave
  accumulates `fb * xsr[m][kb >> ash]`-scaled f32 partials per 128-K block and the block reduces the waves'
  partials, so another split is another association of the same f32 terms: not byte-identical, and CARDS.md §2
  would want a bound. The gain would be the tail's two GEMM latencies (21 + 21 us less their floor), <= 0.4 ms.
  Proposal 1 takes the floor without touching a sum.
- **Prefetch for layer L+1.** The plan needs L+1's router output, which needs layer L's output. The existing
  next-layer predictor (`rad_expert_prefetch_step`) is TP1-only and predicts from the previous layer's routing;
  misses are by construction the experts the heat ranking did not expect, and a unit that misses twice is
  promoted by the mover (7,300-7,600 promotions a no-spec c1 run). DFly2's draft tells the tokens, not the
  trunk's routing. Not assessed further.
- **Copy granularity.** A gate_up-only copy (0.85 of 1.43 MiB) is what gate_up-first lists give without a second
  placement class; the down part's stall is 8 us a miss.
- **Budget line.** Break-even for posting a unit: it pays when it lands with p > (76 - 35) / (76 - 35 + 35 + 25)
  ~ 0.4 (a landed unit saves ~35 us and costs ~10 us of tail GEMM; a late one costs ~41 us more than leaving it
  in the first launch): every list length is far above. The budget is not what limits W2.

## 7. Raw lines

@@RAW@@
