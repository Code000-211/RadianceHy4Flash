# Hy4-flash at TP2 when the routed experts do not fit: making decode scale with clients

Design note. Nothing here is implemented. Numbers marked **M** are measured (existing runs in
`work/perf/`, or this note's probe in `work/xstream/`, real container
`hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad`, 2 x R9700, TP2, 210 W, branch `hy4-perf` @ 6cb3da9e);
everything else is an **estimate** and says what it is derived from. Sources read: `core/place/*`
(planner, heat, mover, stager, xstats), `core/runtime/{xprefetch,stager,residency,ctx}`,
`core/engine.cpp` (`tiering_dispatch`), `core/engine_bringup.cpp`, `core/device/aql.cpp` (DMA
queues, stamps), `arch/common/rad_block_moe_fp8.h`, `arch/hyv4flash/hyv4flash_moe.h`, GUIDE §4.3 /
§4.5, MOE-W4, OPS (`moe_gemm_q`), the friend's repository (`research/friend/radiance-hy4flash`:
plugin, DESIGN, PERF-STATE, PERF-OPTIONS, README, deploy yamls) and `notes/friend-compare.md`.

Probe data of this note: `work/xstream/` — `probe.py` (phases of 256 greedy tokens: 1 / 4 / 8
clients, distinct prompts and one prompt for all clients; `/stats` deltas per phase), `plain.json`
/ `plain.out` (the run), `stamp.json` / `stamp.out` / `stamps.csv` + `stagg.py` (the same run with
every dispatch of one pass in 24 timed on card 0; `stagg.json` is the per-kernel table),
`perfmoe2.out` (the routed GEMM with pooled experts), `d0-hp76.yaml` (prepared, not run).

## 0. Result in one page

1. **The scaling loss is the link, and it is measured.** A non-resident expert is read in place
   over PCIe by the routed GEMMs, on the step's critical path. Per rank and decode step
   (no speculation, **M**): 1 client 13–47 missed experts (23–86 MB), 4 clients 100 (183 MB),
   8 clients 201–216 (366–395 MB). Stamped passes of eight phases (§1.3, every dispatch timed on
   the card) fit

   `pass ≈ (10.6 ms + 0.95 ms x rows) + (3.4 ms + 1.0 us x routed experts + 66 us x missed experts)`

   to within 1.5 ms: the first bracket is everything but the two routed GEMMs, the second is the
   two routed GEMMs. A missed expert is 1.82 MB a rank and 66 us is 27.6 GB/s: the link's rate
   (`heat.h`: 28.5 GB/s up). At 8 clients the misses are 13–14 ms of a 40–42 ms step, at 4 clients
   6.7 of 27, at one warm client 0.9 of 16.5. With DFly2 a verify pass has six rows a sequence
   and the misses are larger still: 158 / 515 / 961 missed experts a step at 1 / 4 / 8 clients
   (**M**, `res/fn2-d5.json`), i.e. 10 of 36–39 ms, 34 of 80 ms and 63 of 133 ms of link time.
   That is why DFly2 gives +70 % at one client and +7 % at eight.
2. **The bound.** With every miss gone: 8 clients 196 → about 295 tok/s without speculation
   and 171–188 → about 325–350 with DFly2; 4 clients 131–149 → 175–200 and 156 → about 270.
   No lever here can do better; the rest of the step is rows (0.95 ms each: the trunk,
   `kernels/perf.md`) and the routed GEMMs' VRAM reads.
3. **The friend's plugin has no better MoE path.** Both plugins declare the core's `MoeFP8` with
   the same configuration (top 16 of 512, `sigmoid_router`, `router_fuse`, `no_reduce`, every
   expert sliced by width), run the same libr4d kernels, the same placement, heat engine, mover
   and pool sizes. His 1-client figure equals ours (57.7 vs 56–64 tok/s). His 4- and 8-client
   figures (196 / 361) are what our fit gives **when the eight rows route the same experts**:
   16.5 ms + 7 x 0.95 ms = 23 ms, 345 tok/s at 210 W (his cards run at 330 W). Our probe with one prompt on all eight clients already
   reaches 288–290 tok/s at 210 W although our eight texts diverge after a while (68 misses a
   step instead of 210). Hypothesis H0, to be confirmed in WP-X0: his `bench_throughput` (not in
   his repository) sends one prompt to every client and his rows stay bit-identical up to 32
   rows, so the union of routed experts does not grow with the client count. If so the "gap" is a
   property of the benchmark, and on distinct prompts his plugin would stream as ours does.
4. **Where the misses come from.** The cards hold 71 % of the experts (14.9K of 20,992 units a
   rank, all of the elastic 25.4 GiB while the KV cache lends). The heat engine holds the hit
   rate at 0.95 with 46 promotions a step at 8 clients, but (a) a promotion becomes visible three
   to four steps after the first miss (§2.4), (b) 81–90 % of promotions are readmits of units
   demoted before, (c) each budgeted victim is copied back to the host (54 GiB d2h against 95 GiB
   h2d in the probe run), (d) the step's swaps are copied under the in-place reads of the pass's
   first five routed layers, which take twice the others' time (2.2 ms a pass at 8 clients, §1.4),
   and (e) nothing prefetches at `tp: 2` (the core's next-layer prefetch is `tp: 1` only). At least 45 % of the misses belong to units that are promoted right after
   (46 promotions a step, each after two or more misses): these are latency misses, removable
   without any prediction.
5. **Levers, in the order to build them** (all move bytes only; outputs are bit-identical by
   construction, §4):

   | # | lever | where | est. misses removed | 8 clients, no spec / DFly2 (now 196 / 188) | 4 clients (now 126–149 / 156) |
   |---|---|---|---|---|---|
   | X0 | routing trace, heat replay tool, heat knobs, H0 check | core (diagnostics), tools | — (decides X1–X4) | — | — |
   | X1 | (a) publish a landed copy before the next pass is played, not behind it (one step sooner); (b) pace the mover's copies between the routed GEMMs: today they collide with the in-place reads of the pass's first five layers (**M**: +2.2 ms a step at 8 clients) | core `engine.cpp`, `place/mover`, `device/aql` | 10–20 % (no spec), 3–6 % (DFly2), plus the 2.2 ms | 214–222 / 195–200 | 146–151 / 159–162 |
   | X2 | admission for many streams: admit on the first miss into copy-free slots (2Q probation), a move budget that follows the pass width, presence credit, replay-chosen decay; host mirror where RAM allows | core `place/heat`, `mover`, planner | to 20–40 % | 222–239 / 212–230 | 150–159 / 172–186 |
   | X3 | early publish on the device: a copy that lands mid-pass serves the layers still ahead (generalised `xp_step`), copies on the second DMA engine | core `runtime/xprefetch`, `device/aql` | to 30–50 % | 230–249 / 221–243 | 154–164 / 178–195 |
   | X4 | router-ahead prefetch at TP2 for first-time misses (next layer's router on this layer's input, on the compute stream, k copies by row count) | core + `MoeFP8` + plugin wiring | 10–20 % of the rest, only if the tap study passes | 235–264 / 227–265 | 156–168 / 181–207 |
   | X5 | link-aware speculation window; shared arm beside the routed GEMMs | core sched, plugin | — | +1–2 % / +5–10 % | +1 % / +3–5 % |
   | X6 | options for the friend: smaller experts (IQR), `kv-cache-dtype: i8`; KV host tier | format, config | capacity +10 points | +10–20 % | +5–10 % |

   The tok/s columns are cumulative down to X4 (bounds: 295 / 325–350 and 175–200 / 270) and come
   from the fit of point 1 with the stated share of misses removed (§5 derives each share; 4
   clients without speculation from 140 tok/s, the middle of the runs' 126–149); X0's replay
   replaces every one of them by a number before code is written. At one client the levers give
   little without speculation (a warm stream misses 13 units a step) and an estimated +10–20 %
   with DFly2 (158 misses a step).
6. **Rejected** (§6): host-CPU MoE for cold experts (needs about 54 G weights/s a rank to match
   the link, a Zen 2 host has less than half; and the libavx kernel is not bit-identical to
   libr4d's), copy-then-compute inside the layer (in-place reads already run at the link's rate),
   a second compute stream for the predictor at TP2 (the core measured +11 us per ordered
   dispatch), `expert-pin-top` (measured no change by the friend; X2 covers what it would do).

## 1. Measurements

### 1.1 What the existing runs say (`work/perf/res/*.json`, `summ.py`, **M**)

| run | 1 client | 4 | 8 | tokens a step (1 client) | hit 1 / 4 / 8 |
|---|---|---|---|---|---|
| `fn2-d0` (no speculation) | 56.9 | 133.4 | 178.6 | 1.00 | 0.955 / 0.951 / 0.943 |
| `fn2-d5` (DFly2, window adaptive) | 98.1 | 152.1 | 170.6 | 3.56 | 0.936 / 0.921 / 0.896 |
| friend's README, his plugin, 330 W | 57.7 / 109.8 | 196.2 / 341.3 | 360.6 / 425–603 | | |

Per rank and step, from the `/stats` deltas of the same files (`routed`, `routed_res` are rank
0's and count each routed expert of a layer once a step; `link.stream_bytes` is summed over the
ranks, 1.82 MB per missed unit, i.e. each missed expert crosses once a step — "union once" is
already how the grouped GEMM works):

| phase | steps | ms a step | routed units | missed units | streamed MB | promotions |
|---|---|---|---|---|---|---|
| no spec, 1 client | 768 | 17.6 | 702 | 32 | 58 | 6.6 |
| no spec, 4 | 515 | 28.3 | 2332 | 114 | 207 | 28 |
| no spec, 8 | 515 | 42.3 | 4019 | 229 | 417 | 47 |
| DFly2, 1 client | 220 | 39 | 2461 | 158 | 289 | 40 |
| DFly2, 4 | 168 | 80 | 6527 | 515 | 939 | 60 |
| DFly2, 8 | 180 | 133 | 9264 | 961 | 1753 | 53 |

A unit is one expert of one layer on one rank (gate_up 1.18 MB + down 0.59 MB + scales; 41 routed
layers x 512 experts = 20,992 units, 38.3 GB a rank). A decode row routes 16 x 41 = 656 units. With
DFly2 at 8 clients a step routes 44 % of all experts and the promotions sit at the move budget
(64 a step): the heat engine is rate-limited exactly where the traffic is highest.

### 1.2 The probe (`work/xstream/plain.out`, **M**, no speculation, one server, phases in this order)

| phase | rows | tok/s | ms a step | hit | routed units | missed units | streamed MB a rank | promotions | distinct texts |
|---|---|---|---|---|---|---|---|---|---|
| c1 | 1 | 53.2 | 18.8 | 0.933 | 697 | 47 | 86 | 11.6 | 1 |
| c8, eight prompts | 8 | 196.3 | 40.7 | 0.950 | 4071 | 203 | 370 | 46.4 | 8 |
| c8, one prompt | 8 | 287.7 | 27.8 | 0.971 | 2365 | 68 | 123 | 21.2 | 8 |
| c4, four prompts | 4 | 126.3 | 31.7 | 0.958 | 2381 | 100 | 183 | 24.6 | 4 |
| c4, one prompt | 4 | 165.4 | 24.2 | 0.966 | 1990 | 67 | 122 | 18.6 | 4 |
| c8, eight other prompts | 8 | 193.5 | 41.4 | 0.947 | 4051 | 216 | 395 | 46.1 | 8 |
| c8, one other prompt | 8 | 290.0 | 27.6 | 0.971 | 2255 | 65 | 118 | 24.2 | 8 |
| c1, same prompt as the phase before | 1 | 64.0 | 15.6 | 0.982 | 707 | 13 | 23 | 2.2 | 1 |

Readings:
- **Cost of a miss.** The two 1-client phases differ only in misses (and the promotions that
  follow them): 3.2 ms for 34 units. The 8-client pairs: 13.3 ms for 140 units and 1650 more
  routed units. §1.3 separates the terms: 66 us a missed unit, 1.0 us a routed one.
- **Cost of a row** with the union held: 0.95 ms (§1.3). The friend's 17.3 → 22.2 ms from 1 to 8
  clients is 0.7 ms a row at 330 W: the same slope.
- **One prompt for all clients still diverges with us**: eight different texts (the prefills run
  in different batch shapes; prefill rows may differ in the last place from each other, E4 of
  `kernels/perf.md`). The streams keep routing alike for a while, hence 2.3K routed units instead
  of 4.1K. Not a defect of this note's subject, but it is why our "one prompt" figure (288) is
  below his (361): his rows stay identical, the union stays at one client's.
- The c4 phase with four prompts took 31.7 ms a step in this run and 26.8 ms in the stamped run
  with the same misses (126 against 149 tok/s; `work/perf` has 131–133): the 4-client figure
  scatters by 15 % between runs, the others by 1–2 %. Unexplained; X0 repeats it.

### 1.3 The routed GEMM alone (`work/xstream/perfmoe2.out`, **M**)

`r4d_selftest --perfmoe 48 20 512 16 P` (libr4d's routed w4 GEMM at Flash-Next's TP2 widths, a
48-row pass at top 16: 768 routed rows over 402 experts; P % of the experts read out of pinned
host memory through the pointer table, as the engine's are):

| | every expert in VRAM | every expert in the pinned pool | ratio |
|---|---|---|---|
| gate_up | 470 us | 9,700 us | 20.6 |
| down | 250 us | 4,850 us | 19.4 |

The launch is link-bound from the first pooled expert on: 402 experts of about 1 MB (gate_up) in
9.7 ms is the link's rate (the tool's own "GB/s of weight" overcounts by the same factor in both
columns). So (a) there is nothing to gain from copying a routed expert first and computing from
VRAM inside the same layer — the in-place read already moves the bytes as fast as the DMA engine
would — and (b) a pooled byte costs twenty times a resident one, which is the 66 us against 1 us
of §1.4.

### 1.4 Stamped passes (`work/xstream/stamps.csv`, `stagg.py`, **M**)

A second run of the probe with `RADIANCE_AQL_STAMP=24` (one pass in 24 on card 0, ten passes a
phase; tok/s within 1 % of the unstamped run except the 4-client phase above). Per pass, ms:

| phase | rows | wall | routed gate_up (41 launches) | routed down (41) | both | everything else | missed units | routed units |
|---|---|---|---|---|---|---|---|---|
| c1, cold stream | 1 | 19.7 | 4.51 | 2.69 | 7.20 | 12.5 | 47 | 697 |
| c1, warm | 1 | 16.5 | 3.41 | 1.52 | 4.94 | 11.5 | 13 | 707 |
| c4, one prompt | 4 | 24.4 | 6.27 | 2.98 | 9.25 | 15.2 | 68 | 1990 |
| c4, four prompts | 4 | 26.6 | 7.20 | 3.66 | 10.86 | 15.7 | 102 | 2390 |
| c8, one prompt | 8 | 27.7–28.3 | 6.3–6.5 | 3.4–3.5 | 9.7–10.1 | 18.0–18.3 | 64–66 | 2254–2365 |
| c8, eight prompts | 8 | 40.1–41.7 | 13.9–14.7 | 7.3–7.8 | 21.1–22.5 | 19.0–19.1 | 201–216 | 4083–4085 |

- "Everything else" depends on the rows only (11.5–12.5, 15.2–15.7, 18.0–19.1 ms): 0.95 ms a row.
  It is the trunk: 334 int8 GEMM launches, 208–249 activation quantisers, 84 junctions, the
  attention rows, the head (0.6 → 1.0 ms).
- The routed GEMMs fit `3.4 ms + 1.0 us x routed + 66 us x missed` (c1 pair: 2.26 ms for 34
  units; the eight-prompt phases are predicted to 0.5–0.9 ms, the four-prompt phase is 1.6 ms
  under). At 8 clients they are 53 % of the pass and two thirds of that is the link.
- **Per layer** (gate_up + down, us; `plain.server.log` gives the residents a layer):

  | | layers 1–5 | layers 6–41, median (range) |
  |---|---|---|
  | c8, eight prompts | 792–943 | 453 (374–603) |
  | c8, one prompt | 218–410 | 240 (175–318) |
  | c1, warm | 111–160 | 117 (97–160) |

  The first five routed layers take twice the others' time although they hold the most resident
  units (411–475 of 512; the slowest later layers hold 224–309). The time does not follow the
  layer's pooled share (correlation −0.54 over the 41 layers). What the first layers share is
  their place in the pass: `tiering_dispatch` issues the step's promotions and demotions right
  behind the pass, 46 x 1.82 MB up and about 53 MB down a rank, 3 ms of link, and the card runs
  them while it is in the pass's first milliseconds, under the in-place reads of layers 1–5. The
  excess is 2.2 ms a pass at 8 clients (0.3 ms with one prompt, 21 promotions; nothing at one warm
  client). Copies and in-place reads share the upstream; a copy that runs between the routed GEMMs
  costs nothing, one that runs under them costs its own link time on the critical path.

### 1.5 Not measured here

- **A host pool that holds every expert** (`work/xstream/d0-hp76.yaml`, `host-pool-mib: 76000`;
  `run2.sh hp76 CFG=d0-hp76.yaml`): prepared, queued behind other jobs on the GPU lock for half an
  hour and withdrawn. It answers whether demotions become free and `starved of a pool slot`
  disappears when the pool mirrors the slab (this host has the RAM, the friend's has not). It
  pins 76 GB beside the 59 GiB Engram copy: run it with nothing else on the box. X0 item.
- **The friend's plugin under this probe** (H0), **DFly2 under this probe** (only `work/perf`'s
  bench has DFly2 figures), **long contexts** (the KV loan recalled), **330 W**.

## 2. How it works today

### 2.1 Budget (startup receipt, **M**)

Per card 31.86 GiB: 4.36 GiB static weights, 1.40 GiB arena, 0.6 GiB queues and driver, 25.39 GiB
elastic = experts' floor 17.20 GiB (what the 19.73 GiB-a-rank host pool cannot hold) + KV cache
7.33 GiB (8 x 65,536 tokens, the most it can address) + prefill staging 0.88 GiB. While no request
holds them the KV cache lends 7.10 GiB (4176 slots) and the arena 2.06 GiB (1210 slots at steps of
at most 256 tokens) to the expert slab: after warm-up 14.6–15.0K units are in VRAM, 6.0–6.4K in
the pinned pool. Units in lent slots keep their host copy (a drop is an address change); units in
the budgeted 17.2 GiB do not (a demotion copies 1.82 MB back to the host and needs a free pool
slot: 4.4–11K `starved of a pool slot` a run).

Under real load the loan shrinks: every 64K tokens of live or cached context take 0.9 GiB = 500
units a rank back. The short-prompt benchmark is the best case for residency.

### 2.2 The pieces (file anchors in `wt/perf`)

- **Planner** (`core/place/planner.{h,cpp}`): units, slab classes, the expert floor
  (`expert_vram_floor`, planner.cpp:101), `--expert-profile` / `--expert-pin-top` (planner.cpp:1130;
  pinned units are never demotion candidates). 32 shadow slots, up to 160 turnover slots kept free
  out of the loan (`RAD_PLACE_TURNOVER_SLOTS`), following demand (`turnover_adaptive`,
  engine_bringup.cpp:3891).
- **In-place reads** (`mover.h` header, `Site::DeviceZeroCopy`): the routed GEMMs take their
  experts through weight tables (`RAD_WTAB`); a pooled expert's entry points into pinned host
  memory and the kernel reads it over the link. `moe_gemm_q` groups the sorted rows by expert, so
  a layer reads each routed expert once a step whatever the row count.
- **Heat engine** (`core/place/heat.{h,cpp}`): per unit LFU with decay 0.95 a step (half-life 14
  steps), credit = rows that chose the expert (capped at 512 a histogram), swap when
  `hot >= cold * 1.25 + 1.5`, at most 64 swaps a step and class. A unit routed by one row once has
  heat 1.0 and is not admitted; twice in a row 1.95 and is. `HeatParams{}` is constructed with its
  defaults at engine_bringup.cpp:3903: no key and no environment variable reaches it.
- **`tiering_dispatch`** (engine.cpp:999), once a step *behind* the pass it just issued:
  `mover.begin_dispatch` (retire: publish the promotions whose copies landed, free quarantined
  slots), credit the histograms that have arrived (the previous step's: the card copies each
  layer's counts back asynchronously), `heat.step` → swaps, `mover.issue` (the copies go out on
  the mover's stream / DMA engine at once), `mover.balance` (fill free lent slots, hottest
  first).
- **Recorded passes** (`ctx.cpp:1287 play_pass`): a decode pass is a tape of 1533 records played
  in stretches, with 328 table-refresh points and 41 routing reports between them (**M**, server
  log). A refresh applies pending residency changes to a layer's weight tables; the host plays the
  whole tape in about a millisecond, far ahead of the card.
- **Prefill stager** (`core/place/stager.{h,cpp}`, `Mover::stage_units`): for steps above 1024
  tokens it copies each layer's pooled experts into an 882 MiB ring up to three layers ahead, by
  DMA, and publishes the copies for that layer's ops. A staged pass is issued live, not played.
  Not armed at decode.
- **Next-layer prefetch** (`core/runtime/xprefetch.{h,cpp,hip}`, `rad_expert_prefetch*` in
  `abi/rad_runtime.h:1027`, called from `MoeFP8::pass`): at layer l the next routed layer's router
  is applied to l's router input on a stream of its own, `xp_plan` writes up to two copies to a
  mailbox in pinned memory, a host thread hands them to the card's *second* DMA engine after a
  mark behind l's routed down GEMM, and `xp_step` (compute stream, before l+1's GEMMs) redirects
  the table entries by compare-and-swap where the copy has landed. Replay-compatible (`kTapeXp*`).
  **Off at `tp >= 2`** (engine_bringup.cpp:5138–5145): with its stream beside the pass every
  ordered dispatch started 11 us later (Flash-Next TP2: 20.5 against 10.1 ms a step). It also
  takes a bf16 router and at most 16 rows; Hy4's router is f32 with a correction bias, and the
  plugin does not chain `xp_next`.
- **Disk tier's routing-aware stager** (`Mover::stage_file_units`): waits for a layer's counts,
  then reads only the routed on-disk experts. One wait on the card per routed layer; the pattern
  for "exact, in the layer" staging, see §6.

### 2.3 What the exit report says (**M**, `plain.server.log`, rank 0 = rank 1)

`56109 promotions, 50979 demotions; 17658 distinct units promoted, 45431 readmits (81 % of
promotions were of something already demoted)`; `95.35 GiB h2d, 54.37 GiB d2h`; `223 starved of a
slab slot, 4513 of a pool slot`; resident units a layer 224–475 of 512 (mean 363, sd 78). The long
run of `work/perf/d0ram` has 90 % readmits. The policy keeps re-buying the same units.

### 2.4 Promotion latency today

A unit first missed in pass s:
1. pass s runs; its counts land when the pass ends;
2. pass s+1 is issued; behind it `tiering_dispatch(s+1)` credits H_s. Heat 1.0: not admitted
   unless two rows chose it;
3. missed again in s+1; after pass s+2 is issued, `tiering_dispatch(s+2)` credits H_{s+1}, heat
   1.95, the swap is issued, the copy runs beside pass s+2;
4. `tiering_dispatch(s+3)` retires and publishes it — after pass s+3 was played. Pass s+4 is the
   first to read it from VRAM.

Four misses (s … s+3) for a unit that recurs every step, three when two rows chose it at once
(admitted at step 2). Only the first is unavoidable without prediction. A promoted single-row unit
was missed in s and s+1 for certain and in s+2 and s+3 with the probability p that it is routed
again: 2 + 2p misses a promotion. With 46 promotions a step at 8 clients that is 92 (p = 0) to 184
(p = 1) of the 203 misses a step: at least 45 % of the misses are of units the engine is in the
middle of admitting. At DFly2 with 8 clients the promotions sit at the budget (53–60 a step
against 961 misses), so there the admission rate, not its latency, is the limit. X0's replay gives
the real shares.

## 3. The friend's plugin: what differs for this subject

- MoE: nothing. `hyv4flash.cpp:1975–2010` is our `Ffn::probe` field for field (`mc.rows =
  min(T, 2048)` is the core's default). Same shared-expert order (after the routed arm).
- Placement: his `deploy/example-tp2-dfly2.yaml` is our `d5.yaml` (39,000 MiB pool, 65,536
  context, 8 sequences, 2048 batched tokens). His notes list "decode expert prefetch at tp >= 2"
  and `expert-profile` / `expert-pin-top` as considered and rejected ("already measured, no
  change", PERF-OPTIONS).
- Rows: his attention forms are row-stable up to 32 rows and every prefill row runs absorbed, so
  identical requests presumably stay identical (not verified here); ours diverge after prefill
  (§1.2).
- DFly2 at 8 clients: he sets `RADIANCE_AQL_CHAIN_FENCE=1` on 2.2.0 (`notes/friend-compare.md`
  §7); our branch carries the AQL fixes that address the cause.
- What he lists as open and we lack too: the shared expert beside the routed GEMMs (his row 8,
  +2–4 % estimated), IQR experts for fewer link bytes (his row 12).

So there is nothing to port for streaming. The comparison to make is the benchmark's (WP-X0.4).

## 4. Constraints

- **Lossless.** A lever may change where an expert's bytes are read from and when they are copied,
  never which kernel computes a row or in which order sums are taken. All levers below keep every
  routed row on the same `moe_gemm_q` launch with the same operands except table entries, and a
  redirected entry names a byte-identical copy (the property the mover, the stager and the
  prefetch already rely on). Proof per package: greedy text and logprobs equal with the lever on
  and off (1, 4, 8 clients, DFly2 on and off, the rollback gate of WP13), plus a debug switch,
  new with X3, that byte-compares a copy with its pool slot when its entry is redirected (the KV
  tiers have such a mode, `RADIANCE_DEBUG_TIER_VERIFY`; the expert path has none).
- **Every rank the same resident set** where the plan demands it (`planner.h`: the ranking uses
  only identity and the profile). The heat engines of both ranks see the same histograms and
  decide alike (the exit reports agree to a few units); a new policy must stay a pure function of
  the histograms. Copies landing at different times on the two cards are fine: a table entry is
  per rank and both addresses hold the same bytes.
- **No second compute queue beside a pass at TP2** (§2.2). Predictors run on the compute stream or
  on the host; copies run on the DMA engines.
- **Replay.** Decode passes are played from tapes. A lever changes tables only at refresh points
  on the host, or by a recorded device step (`kTapeXpStep`).
- **The link is shared.** Upstream carries in-place reads (critical path) and copies. Copies are
  paced into the stretches where no routed GEMM runs (the prefetch's mark) or accepted to slow the
  GEMMs they overlap; X0 measures which.
- `deterministic: true` keeps today's static plan; every lever is off there.
- Quantisation and formats are the friend's decisions: X6 offers options, nothing more.

## 5. Levers

### X1. Promotion latency and the copies' place in the pass (core; engine + mover)

**Mechanism (a).** One change of order, no new machinery: *publish before the pass is played.* The
rank thread's step is `run_step` (play pass s+2), sampler, `tiering_dispatch(s+2)`, and the
mover's retire — which publishes the copies that have landed — is the first thing
`tiering_dispatch` does (`Mover::begin_dispatch`, mover.cpp:471), i.e. behind the pass. A copy
issued behind pass s+1 (46 units are 85 MB, 3 ms of link) has landed long before pass s+2 is
played, yet pass s+2 still reads the pool and pass s+3 is the first to see the slot. Calling the
retire before `run_step` (engine.cpp:1181) gives that step back: the unit of §2.4 is missed in s,
s+1, s+2 instead of s … s+3, and one admitted at its first observation in s and s+1 only.

The second half of the reordering — crediting pass s's counts and issuing the swaps *before* pass
s+1 is played rather than a millisecond behind it — buys nothing by itself (the counts credited
are the same) and is the precondition of X3, which hands the pass a list of copies in flight.

**Where.** `Engine::rank_loop` (engine.cpp:1160–1210): `r.mover.retire(&r.heat)` ahead of
`r.ctx.run_step(b)` on non-draft passes; `Mover::begin_dispatch` keeps its retire for the slots
that freed meanwhile. The quarantine of a freed slot stays an event on the compute stream as today
(mover.h "Shadow slots and quarantine": a slot is reused only after the launches that could name
it have retired; publishing sooner does not shorten that, and the demoted unit's old slot is still
named by nothing newer than the pass before). For X3 later: `tiering_dispatch` split into a
crediting half before the pass and the fill / balance half behind it; check `find_histograms`'s
`before_step` and the routing ring's depth (`route_dropped`) there.

**Mechanism (b): pace the copies.** §1.4: the step's swaps are issued right behind the pass and
run under the in-place reads of its first five routed layers, 2.2 ms a pass at 8 clients. Two
ways, to be compared by stamps:
- *between the routed GEMMs*: the mover's copy list waits on a mark recorded behind each layer's
  routed down GEMM and sends a few units a gap (the prefetch's `rad_expert_prefetch_mark` and its
  DMA list that polls the mark, `xprefetch.h`; a gap at 8 rows is about 0.45 ms, seven units).
  The marks are records of the tape (`kTapeXpMark`), so this is replay-compatible; the plugin
  calls the mark behind `moe.step`;
- *in the turnaround*: issue the swaps when the pass's outputs are back and before the next pass
  is played, and let the next pass's first routed layer wait for nothing — the copies then overlap
  the embedding, the Engram lookups and layer 0 (dense, no experts) and the sampler's tail of the
  pass before. Simpler, but only about a millisecond of cover.
Demotions (d2h) ride the idle direction and need no pacing (`heat.h`: 25 GB/s down beside 28.5
up).

**Exactness.** Tables change only between passes, as today; pacing moves no byte.
**Gain (est.).** (a) p x 46 of 203 misses a step at 8 clients without speculation, 10–20 % for
p = 0.45–0.9, 1.3–2.7 ms; (b) up to the 2.2 ms measured. Together 196 → 214–222 tok/s; 4 clients
140 → 146–151. With DFly2 the promotions are budget-bound, (a) is 3–6 % of the misses and (b)
about 3 ms of 133: 188 → 195–200, 156 → 159–162. One warm client: nothing (2 promotions a step).
**Risk.** (a) small: the retire is a few event queries; if a copy's event has not signalled yet
the unit waits a step as today. (b) a paced copy lands later, which costs (a)'s step for the
last units of the list: order the list by heat and let what has not gone out by the pass's end go
out unpaced.

### X2. Admission for many streams (core; heat, mover, planner)

**Mechanism.** The policy was tuned on Flash-Next TP1 with 40 % of the experts resident (heat.h,
planner.h comments). Here 71 % are resident, eight sequences interleave, and the measured picture
is thrash (81–90 % readmits) plus rate limits (64 swaps a step at DFly2). Candidates, each a
replay experiment first:
- *Admit on the first miss* into slots whose eviction is free: the lent segments (5.4K slots)
  already keep host copies. A probation area (2Q's A1in: FIFO, say 5–10 % of the slab, lent slots
  only) takes every miss; a unit routed again while in probation moves to the main set by an
  address change (same slab class, no copy), one not routed again is dropped for free. The main
  set keeps LFU with decay and its hysteresis and stops trading with the probation's traffic.
  Cost: h2d bytes = missed bytes (370 MB a step and rank at 8 clients), in link time the misses
  leave free (duty 40 % → 75 %); at DFly2 with 8 clients (1.75 GB a step, duty 46 %) admission
  needs a bar (two rows, or heat of the layer's median resident) or the copies starve the reads.
- *Presence credit.* With many rows the credit "rows that chose it" favours experts several
  sequences share in one step; the cost of a miss is per (unit, step). Replay credit = 1 a step
  against today's.
- *Decay and hysteresis* for eight interleaved sequences (0.95 was chosen for one conversation).
- *Per-class budget above 64* when the pass is wide (DFly2 verify): the budget is sized for the
  free slots (`RAD_PLACE_TURNOVER_SLOTS = 3 x moves − shadow`), so both move together.
- *Host mirror.* Where the pinned pool can hold every expert (2 x 35.7 GiB; this host has 170 GB,
  the friend's 64 GB has not) every demotion should be an address change and `starved_pool`
  disappear: with `host-pool-mib: 76000` the expert floor is zero (`expert_vram_floor`). Whether
  the mover then keeps a pool slot for every unit is not verified (§1.5: prepared, not run); as a
  planner rule it would read "a unit in a budgeted slot keeps its pool slot while the pool has
  room".

**Where.** `HeatParams` (heat.h), `HeatEngine::step` / `observe`, `Mover::balance` / `fill` /
`drop` (the lent-slot paths exist), `planner.h` slot counts. Keys: none new for users; the chosen
values become defaults per "experts resident share" or stay behind environment variables until
they have run on Flash-Next too.
**Exactness.** Placement only.
**Gain (est.).** Admission at the first miss removes the s+2 miss of §2.4 as X1 removed s+3
(another p x 46 a step), and reaches units that recur at gaps the two-in-a-row bar never admits:
to 20–40 % of today's misses at 8 clients, 196 → 222–239 with X1. With DFly2 the gain is the
budget's: if a move budget of 128–192 a step and class with free evictions admits what recurs,
20–35 % of 961 misses, 188 → 212–230 (the least certain figure of this note; the copies then take
most of the link's idle time). The replay decides; a variant that does not beat "X1 alone" by 5 %
of the misses on the traces is dropped.

### X3. Early publish on the device (core; xprefetch generalised)

**Mechanism.** After X1 a copy issued before pass s+1 still helps only from pass s+2 on, although
it lands within milliseconds: the host has played the whole tape before the card reaches layer 2.
`xp_step` already solves this for the TP1 prefetch: a recorded kernel on the compute stream, right
before a layer's routed GEMMs, that swaps table entries to a copy if a done word says it landed
and the entry still names the pool slot. Generalise it:
- the copies are the mover's promotions and X2's probation admissions (destination = their final
  slab slot, so nothing is copied twice), issued in layer order before the pass, on the second
  DMA engine (`g_dma_xp`, aql.cpp:3036; free at TP2 today), each job tagged; the done words live in
  pinned memory as the prefetch's do;
- `xp_step` takes the layer's list of (entry, old, new, done word) from a per-step record buffer
  the host fills before the pass (the prefetch's `XpRec`, sized for the layer's admissions instead
  of two); the host publishes the same change in its residency table at the next pass's issue, so
  from then on the tables agree without the device step;
- pacing as in X1 (b): the DMA list waits on the mark behind each layer's routed down GEMM, so
  copies cross under attention and the trunk, not under the in-place reads; the list is ordered
  by layer first (a copy for layer 30 has 25 ms to land, one for layer 3 has two), then by heat.

No predictor kernel and no second compute stream: the 11 us penalty of the TP1 design does not
arise. The CAS kernel is one small launch a routed layer (41 a step, chained; about 0.2 ms a
step), later foldable into `router_sigmoid_topk_scatter`.

**Where.** `core/runtime/xprefetch.{h,cpp,hip}` (record sizes, a host-fed plan in place of
`xp_plan`), `core/device/aql.cpp` (the second DMA queue at TP2, done-word writes),
`core/place/mover.cpp` (`promote_to` handing its copy to that queue; publish semantics),
`engine_bringup.cpp:5138` (enable for `tp >= 2` in this host-planned mode), VRAM budget line
`expert prefetch` (records only, no ring: the copies land in slab slots).
**Exactness.** The prefetch's argument: an entry is redirected only to a landed, byte-identical
copy and only if it still names the source slot; the quarantine rule for the *freed* slot is the
mover's. New: a slab slot is named by kernels before the host has published it — the mover must
treat it as held from the moment the copy is issued (it does: shadow slot) and the victim's
drop must wait for the pass that used the old entry (it does: quarantine event).
**Gain (est.).** The remaining latency miss of pass s+1 for the layers the copy beats: with 85–370
MB a step and rank at 28 GB/s the copies need 3–13 ms of a 30–40 ms pass, so roughly the later
half of the layers: about 10 % of today's misses, to 30–50 % with X1 and X2. More at DFly2 (long
passes). 8 clients → 230–249, DFly2 221–243.

### X4. Router-ahead prefetch at TP2 (core + MoeFP8 + plugin), gated

**Mechanism.** First-time misses (a unit not routed in the last steps) are out of reach of X1–X3.
The core's predictor — layer l+1's router applied to layer l's router input — reaches them. To
run it at TP2 and on Hy4:
- prediction on the compute stream: fold the next layer's logits and the plan into the router
  launch (`router_sigmoid_topk_scatter` has the input row in registers; one more 512 x 3072 f32
  matrix read from VRAM, 6 MB, about 10 us a layer) or issue `xp_logits` + `xp_plan` unordered
  beside the routed gate_up GEMM (the shared arm's `haz_mark` trick, no second queue);
- f32 router weights and the sigmoid router's correction bias in the ranking (`XpNext` gains a
  bias weight; ranking by `sigmoid(logit) + bias`);
- copies a layer scaled by rows (2 at one row, up to 8 at 8–48 rows; `kXpSlots`, the ring
  `kXpGroups x kXpSlots` units: 24 units = 44 MB), row limit 16 → 64;
- the plugin chains `xp_next` through its `Ffn` layers and calls `rad_expert_prefetch_mark` behind
  `moe.step`.

**Gate (X0.6).** Offline, before any code: dump router inputs and expert ids of all layers for a
few hundred decode steps at 8 clients through the plugin's taps (`t_logits`, `t_ids`,
`RADIANCE_HY4_TAPS`), read the router weights from the container, and compute for k = 2, 4, 8 a
layer: the share of predicted pooled units that were routed (precision) and the share of missed
units predicted (recall), with the residency of the trace. Go if recall >= 30 % of the misses left
after X1–X3 at precision >= 40 % (the core's break-even was a quarter used at TP1, where the
predictor's stream was free).
**Exactness.** The prefetch's (§2.2).
**Gain (est.).** 10–20 % of the remaining misses; 8 clients 230–249 → 235–264.

### X5. Smaller items

- *Link-aware speculation window* (core sched, `draft-window: adaptive`): at 8 clients a verify
  row costs its trunk time plus 16 x 41 x miss-rate x 66 us of link (about 2.2 ms a row at a 5 %
  miss rate with no overlap between rows, against 0.95 ms of trunk). The window controller should
  price a row with the measured `stream_bytes` per routed row of the last steps, so a sequence
  with 50 % acceptance stops paying link time for rows 4–6. Lossless (window choice never changes
  tokens). Estimated +5–10 % at 8 clients with DFly2, nothing without it.
- *Shared arm beside the routed GEMMs* (plugin, `Ffn::step`): issue `shared.step` between the
  router and `moe.step`'s gate_up instead of after `lat_out`, left unordered against the routed
  GEMMs as `MoeFP8::pass` does with its own shared arm. The shared expert is 3.4 MB a layer in
  int8: 10–20 us a layer hidden under a link-bound GEMM, 0.4–0.8 ms a step. Byte-equal (same ops,
  same operands). Do it together with the stamps of X0.5.
- *Per-layer balance.* Layers hold 224–475 resident units; the global ranking is right for bytes,
  but a `/stats` row per layer (routed, missed) costs nothing and shows whether a few layers carry
  the misses (then a floor a layer is a one-line policy to replay).

### X6. Capacity: options, not decisions

- Each GiB of VRAM is 590 units a rank, 2.8 points of residency. The hit rate against capacity
  curve comes out of the replay (X0.2 runs it at 60–100 % residency).
- `kv-cache-dtype: i8` halves what a token of context costs; under real context load that is
  residency (the plugin refuses non-bf16 caches today, `kernels/perf.md` L9: lossy, the friend's
  call).
- Host KV tier (`prefix-cache-mib`, GUIDE §4.8) so finished conversations' blocks leave VRAM and
  are lent to the slab; to be measured on the multi-turn workload, config only.
- Experts below four bits (IQR, his PERF-OPTIONS row 12): 3.5 bits would put 81 % of the units on
  the cards and shrink every missed unit by an eighth. Format and quality are his.
- `num-speculative-tokens: 0` deployments should not load the drafter's weights into the static
  4.36 GiB (check what the 0.85B drafter costs a rank; if it is resident regardless, a
  `draft-expert-placement`-like key for a dense drafter is a small core item).

## 6. Considered and rejected

- **Host-CPU MoE for cold experts with few rows.** Bytes moved: 1.82 MB of weights a rank against
  about 3 KB of activations down and 3 KB of result up — the link would be free. But the work
  moves to the CPU: a unit is 3.54 M four-bit weights a row; matching the link (one unit in 66 us)
  needs 54 G weights/s a rank. An AVX2 nibble decode + `maddubs` path does 1.5–4 G weights/s a
  core, so 210 units x 2 ranks a step is 0.4–1.0 core-seconds: 19–50 ms on 20 cores against the
  link's 13–14 ms, on this Zen 2 host (the friend's CPU is unknown). It also needs a host round trip
  per routed layer inside a played pass (`kTapeHost` joins the stream: 41 joins a step), and
  bit-identity: libavx multiplies the stored w4nl levels, libr4d its int8 set (OPS `moe_gemm_q`:
  "inside the op's tolerance"), so a row's bytes would depend on where its expert lay. A
  libr4d-exact host kernel is possible (int32 sums per scale group, the same fold order) but
  pays only on an AVX-512 VNNI host with a slow link. Not for this box.
- **Copy, then compute, inside the layer** (the disk tier's routing-aware stager applied to the
  pool): exact, no prediction, but the in-place read already runs at the link's rate (§1.3), the
  DMA engine is no faster, and the layer would wait for the router's counts on the host (41 waits
  a step) and lose the played tape.
- **Arming the prefill stager at decode**: it stages every pooled expert of a layer (270 MB a
  layer and rank here); a step routes 4–44 % of them.
- **A second compute stream for prediction at TP2**: §2.2.
- **`expert-pin-top` / `expert-profile`**: static; the friend measured no change. X2's main set
  with a longer memory is the dynamic form.
- **Sharing a streamed expert across rows**: already the case (§1.1). Across steps it is X1–X3
  (the copy that follows the first miss).
- **Wider links / both links**: each rank already reads its own slice over its own card's link.

## 7. Measurement plan

1. **Trace** (X0.1): `RADIANCE_ROUTE_TRACE=FILE` — at `Engine::tiering_dispatch`, where
   `heat.observe` is called, append (step, layer, bank, reports, n, sparse (expert, count)) for
   rank 0; a few MB a minute. Workloads: `probe.py`'s phases; DFly2 on; the BetterBench mix; a
   multi-turn long-context session set (8 sessions x 20–60K tokens, the user's workload) so the
   KV loan is partly recalled.
2. **Replay** (X0.2): `tools/heat_replay` links `core/place/heat.cpp` unchanged, models the slab
   (budgeted / lent / shadow / turnover slots), the mover's latency in steps, the move budget and
   host copies; prints hit rate, missed units and link bytes (in-place, h2d, d2h) a step. It must
   reproduce the live run's hit rate within one point and its promotions within 10 % before any
   variant is believed. Variants: latency 4 / 3 / 2 / 1 / "layers the copy beats" (X1, X3);
   admission bars, probation size, presence credit, decay, budget (X2); capacity 60–100 % (X6);
   an oracle that knows the next step (the bound of X1–X3) and one that knows the layer's routing
   k layers ahead (the bound of X4).
3. **Cost model**: §1.4's fit per layer: duration of the routed gate_up and down launches against
   the layer's missed bytes (per-layer routed / missed counters in `/stats`), and the first-layers
   excess with the swaps switched off for a phase (`RADIANCE_HEAT_MOVES=0`, X0.3): it must vanish,
   which confirms §1.4's reading before X1 (b) is built.
4. **A/B of each package** on the real container: `work/xstream/probe.py` (extended by DFly2 and
   by the long-context phase), 3 repetitions, 210 W; report tok/s, ms a step, missed units and
   streamed MB a step, promotions, readmits, h2d / d2h bytes. Text and logprob equality as §4.
5. **H0** (X0.4): (a) `probe.py` with identical prompts already shows the effect; (b) run the
   friend's stack (image 2.2.0, his `example-tp2.yaml`) under our `probe.py`: his plugin with eight
   distinct prompts should land near ours (about 180–200 tok/s at 210 W), with one prompt near
   300–350; (c) ask him which prompts `bench_throughput` sends.
6. Regression guard: Flash-Next TP2 (all resident, when its model is back) and gpu-mini must not
   move; `deterministic: true` unchanged; `ctest --preset host` for `place_*` tests plus new unit
   tests of the heat variants (the heat engine is pure host code and already tested without a
   card).

## 8. Work packages for Opus implementers, in order

Each package: its own worktree off `hy4-perf`, core changes kept apart from the plugin, numbers and
a note in `notes/xs-N.md`, GPU only under `flock /root/work/claude-scratch/gpu.lock`.

**WP-X0 Instruments (first; no behaviour change).**
1. `RADIANCE_ROUTE_TRACE` (engine.cpp, about 40 lines) and the trace format.
2. `tools/heat_replay.cpp` + `scripts/heat_replay.py` (variants, plots); validated against two live
   runs (no spec and DFly2 at 8 clients).
3. Environment overrides for `HeatParams` and the mover's turnover / lead at engine_bringup.cpp:3903
   (`RADIANCE_HEAT_DECAY`, `_HYSTERESIS`, `_MIN_GAIN`, `_MOVES`, `_CREDIT_CAP`), logged at startup;
   per-layer routed / missed counters in `/stats`.
4. H0 (measurement plan 5) and the extended `probe.py`.
5. Stamps: routed GEMM time against missed bytes per layer; copies paced against unpaced.
6. The tap study for X4 (offline predictor precision / recall).
7. The host-mirror run of §1.5 and the long-context phase (KV loan recalled, with and without
   `prefix-cache-mib`).
Deliver: `notes/xs-0.md` with the replay's table for X1–X4 and a go / no-go for each.

**WP-X1 Promotion latency and pacing** (core). (a) The mover's retire before `run_step`; a place
test (a copy landed before step n is read from VRAM in step n). (b) The mover's copies paced by
the routed layers' marks or issued in the turnaround; stamps before and after (the first five
routed layers must fall to the others' time). A/B as plan 4. Accept: missed units a step at 8
clients −10 % or better, the 2 ms of §1.4 gone, promotions and readmits reported, texts equal.
(a) is a day's work including the measurement and needs no replay to be safe: do it while X0.2 is
being built.

**WP-X2 Admission** (core), only the variants the replay selected. Probation area over lent slots
with free eviction; parameters; optional planner rule for a host mirror. Accept: beats X1 by the
replay's predicted margin within a third; readmit share reported; Flash-Next TP1 replay (the
core's own tuning point) not worse than today.

**WP-X3 Early publish** (core). Host-planned records for `xp_step`, second DMA engine at TP2, done
words, pacing mark; selftest of the CAS kernel with a late and an early copy; the tape audit
(`kAuditPlays`) must pass with it. Accept: served-from-VRAM share of the copies issued before a
pass, per layer; tok/s as plan 4.

**WP-X4 Router-ahead at TP2** (core + `MoeFP8` + plugin), only on X0.6's go. Predictor on the
compute stream, f32 router + bias, k by rows, plugin wiring (`xp_next`, mark). Accept: copies used
>= 40 %, tok/s gain at 8 clients >= 4 %, one client not slower.

**WP-X5 Small items.** Shared arm beside the routed GEMMs (plugin, byte-equal, kbench-free);
link-aware window (core sched; needs DFly2 acceptance on real weights, so measure on the real
container only).

**WP-X6** is a short note to the friend with the capacity curve from X0.2 (residency against hit
rate and tok/s for 4-bit, 3.5-bit and an i8 KV cache). No code.

Order and dependencies: X0.1–X0.3 and X1 first (parallel), then X0's replay table → (X2 ∥ X3) →
X4; X5 any time after X0.5. X1 is a few lines and is measured before the others start.
