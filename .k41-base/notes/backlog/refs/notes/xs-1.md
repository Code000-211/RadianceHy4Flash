# WP-X1 (expert streaming: publish sooner, pace the copies) — notes

Branch `hy4-xstream`; measured on `hy4-perf4` (hy4-perf + hy4-xstream), real container, TP2,
210 W, Engram in RAM, `work/xs1/`. Arms: **B** = old order (RADIANCE_XS_EARLY_PUBLISH=0,
RADIANCE_XS_PACE=0), **A** = X1 a only, **P** = X1 a + b (PACE=-1). Each arm a fresh server.

## What was built

- (a) `Mover::publish_landed` (retire without sending held copies) before every non-draft pass in
  `Engine::rank_loop`. Default on (`RADIANCE_XS_EARLY_PUBLISH=0` = old order).
- (b) Pacing (`MoverConfig::pace_units`, `RADIANCE_XS_PACE`: -1 spread, N a layer, 0 off): the
  step's promotion copies are held on the host (`pace_collect` around issue/balance in
  tiering_dispatch, not behind a staged pass) and sent from the next pass's routed layers' marks
  (`Ctx::xp_mark` → pace hook; now recorded into the tape where the TP1 prefetch is off), each
  share behind a ride on the compute stream, on the mover's stream/DMA lane (cache handling as
  before). Demotions and every other move are never held; a held record's event is never queried;
  leftovers go out unpaced at the next dispatch; reclaim/drop_slab/destructor flush them too.
  **Default: -1 without a drafter, 0 with one** (see below).

## Exactness

Greedy text + top-5 logprobs, c1 (160 tokens) and c8 (8 distinct prompts sent together, 96 tokens
each), compared token for token (`cmp.py`): **identical** for d0-B vs B2, A, P, P2, P-st, and
d5-B vs B2, A, P, P2 (all 9 sequences each). The 8-client check is reproducible run to run.

## Throughput (tok/s, sum of per-request decode rates; misses = missed units a step, rank 0)

No speculation:

| phase | B (3 runs) | A | P (3 runs) | P vs B |
|---|---|---|---|---|
| c1 | 60.1 / 60.1 / 59.8 | 58.6 | 60.2 / 58.7 / 59.9 | ±0 |
| c4 distinct | 148.0 / 150.2 / 149.4 | 149.1 | 144.1 / 151.5 / 153.0 | ±0 (noise ±3 %) |
| c4 one prompt | 166.3 / 166.1 / 166.0 | 165.9 | 168.0 / 168.1 / 167.4 | +1.0 % |
| c8 distinct | 203.6 / 203.3 / 204.8 | 206.7 | 209.6 / 209.4 / 208.9 | **+2.8 %** |
| c8 one prompt | 290.8 / 292.2 / 290.8 | 289.6 | 288.8 / 289.1 / 288.6 | −0.7 % |
| c8 distinct (other prompts) | 195.1 / 195.4 / 196.0 | 195.8 | 198.4 / 201.0 / 199.4 | **+2.0 %** |
| misses a step, c8 distinct | 193 / 195 / 191 | 188 | 189 / 191 / 191 | −1 % |

DFly2 (num-speculative-tokens 5, adaptive window):

| phase | B (2 runs) | A | P (2 runs) |
|---|---|---|---|
| c1 | 105.0 / 101.0 | 101.0 | 101.8 / 99.9 |
| c4 distinct | 192.9 / 185.7 | 189.3 | 194.2 / 192.8 |
| c8 distinct | 242.6 / 244.5 | 246.8 | 238.3 / 244.7 |
| c8 one prompt | 359.1 / 357.7 | 358.4 | 308.1 / 350.8 |
| c8 distinct (other) | 222.3 / 232.3 | 234.0 | 206.9 / 216.3 |
| misses a step, c8 distinct (other) | 658 / 604 | 604 | 765 / 749 |

With DFly2 the held copies wait for the next verify pass's marks (the drafter's passes carry
none), so a promotion lands a whole step later and the misses rise: pacing stays off there.

## Stamps (one pass in 24 on card 0, no spec; `stlay.py`, `kdiff.py`)

| phase | arm | pass wall | routed gate_up+down | layers 1–5 (us) | median 6–41 | excess of 1–5 |
|---|---|---|---|---|---|---|
| c8 distinct | B | 39.85 ms | 21.08 ms | 952 789 880 677 733 | 415 | 1.96 ms |
| c8 distinct | P | 40.15 ms | 19.16 ms | 760 582 463 375 347 | 416 | 0.45 ms |
| c8 distinct-2 | B | 39.38 ms | 20.56 ms | 955 1001 932 896 488 | 387 | 2.34 ms |
| c8 distinct-2 | P | 40.17 ms | 19.87 ms | 759 624 345 424 442 | 426 | 0.47 ms |
| c8 one prompt | B / P | 27.98 / 28.59 | 10.18 / 10.17 | | 220 / 223 | 0.44 / 0.12 ms |

The §1.4 reading is confirmed: the first-layers excess is the copies (2.0–2.3 → 0.45 ms). But
the time moves elsewhere: per kernel (c8 distinct, B → P) the routed GEMMs lose 1.9 ms while the
shared arm right behind the mark (gemm_i8_kpar<2> +0.8 ms, gated_quant +0.5 ms, the dispatch the
ride completes on and its neighbours, now beside the copies) and the all-reduce junctions
(+0.4 ms, link shared with the copies) gain 1.8 ms; the probe's step falls by ~1 ms
(39.3 → 38.2 ms). Next: ride on a later dispatch (behind the shared arm) or pace by bytes so a
share fits the attention stretch, and keep copies out of the junctions' window.

## X1 (a) is nearly inert

`early_published` a step: 0.00–0.05 at 4–8 clients without speculation, 1–3 with DFly2, ≤1.1 at
one client — of 20–47 promotions a step. The host plays pass s+1 while the card still runs pass s
(lead gate 2 dispatches), and the copies issued behind pass s start only when pass s ends, so
none has landed when pass s+1 is played. The replay bounds the lever anyway: lat 2 → 1 removes
2–3 % of the misses (notes/xs-0.md). Kept (correct, free, helps one client and DFly2 a little);
the real version is X3 (publish on the device inside the pass).

## Against the design's estimate

Design X1: 8 clients 196 → 214–222 no spec (misses −10…−20 % plus 2.2 ms). Measured: 203–205 →
209 (+2–3 %), misses −1 %, the 2.2 ms excess gone from the first layers but mostly reappearing in
the kernels beside the copies. DFly2 estimate 188 → 195–200: no gain (pacing off, A within noise).
The replay explains the gap: latency misses are a small share (oracle: at most 18 % / 29 %).

## Files

core/engine.cpp, core/engine_bringup.cpp, core/place/mover.{h,cpp}, core/runtime/{ctx.h,ctx.cpp,
xprefetch.cpp}, core/server/{liveview.h,dashboard.cpp}, tests/place_test.cpp (2 new cases).
