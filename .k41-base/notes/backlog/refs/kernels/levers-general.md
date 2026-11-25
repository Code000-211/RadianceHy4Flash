# Hy4-flash: levers that help at any TP size (prefill and decode)

Research and plan. Nothing here is implemented. Numbers marked **M** are measured by this note
(2026-11-22, real container `hy4-flash-iqr-v4-u3.rad`, 2 x R9700, TP2, 210 W, Engram in RAM,
sampled T 0.9 / top_p 1.0, code prompts, `src` @ dc680ce3 unless a line says `perf5`); everything
else is an **estimate** and says what it is derived from. TP-specific levers (the two-rank
junction / all-reduce wire, four-rank forms, expert streaming where the experts do not fit) are
the subject of the TP2 and TP4 notes and are only pointed at here.

Raw data: `work/levers/` — `probe.py` (phases: c1, c1b, c4, c8 of 256 sampled tokens, `think` =
one client with `reasoning_effort: high`, 384 tokens; `pf6` / `pf20` = one prompt of 6.7K / 24.9K
tokens), `run.sh` (one server under the GPU lock, probe, stop), `agg.py` (stamps -> per kind of
pass: wall, busy union, exclusive time per kernel and category), `layer.py` (one layer of a pass,
record by record), and per run `<tag>.json` / `.out` / `.server.log` / `.csv` / `.agg.txt`.
Runs: `ns-st` (no speculation, `RADIANCE_AQL_STAMP=7`), `cp-st` (DFly2 coupled, the `work/u3`
configuration, stamped), `p5-ns` (the same on the `hy4-perf5` build = integration + `hy4-adopt` +
`hy4-compat` + `hy4-xstream`), `fixed5` / `adapt5` (the depth-decision wait, A/B), `stop50` /
`stop30` / `cp2` (the depth threshold), `prof` (`profile-ops`, prefill), `cpv2` (`server: log: debug`, host phases), `greedy5`, `p5-cp`,
`lf2` / `cp3` (`draft-depth-load-from: 2`).

"Exclusive time" of a dispatch = its end minus the later of its start and the previous end on its
queue; the sum over a pass equals the busy union within 1 %. A stamped pass is slower than a
plain one (every dispatch gets a completion signal: the stamped wall is 2.4-4 ms above the real
step), so step times are taken from `/metrics` (`engine_step_seconds_decode_total` / steps) of
the same phases and only the *busy* time and the shares from the stamps.

## 0. Result in one page

1. **Where a decode step goes** (one client, no speculation, 15.7 ms; §1.1-1.3): 8.3 ms are
   kernels bound by the weight bytes they read (trunk linears 5.3, routed experts 2.7, head
   0.3), 5.9 ms are the other 1310 launches, most of them at the launch floor (quantisers, norms,
   glue, small projections; the two-rank junction's 1.1 ms is in here), 1.5 ms are waits in
   front of 458 quantiser launches that are not chained. At eight clients (28.7 ms) the routed experts are
   13.2 ms: bytes again (2.2 us a routed unit of 1.46 MiB), independent of rows.
   Attention is 11 % at one client and 5 % at eight; Engram, iHC, sampler and host are noise.
2. **A DFly2 step** (one client, 22-23 ms, 2.0-2.9 tokens): verify 15.8-16.6 ms, draft pass
   3.9 ms (head 1.5, IQ8R linears 1.05), 2.8 ms outside the kernels. A verified draft row costs
   1.0 ms.
3. **The 20 ms host wait of the adaptive depth is not a lever**: fixed depth 5 and adaptive
   depth forced to 5 (waiting every step) give the same step time within 0.3 ms (§1.5).
4. **The depth rule is.** `draft-depth-stop` 0.85 verifies far too little for one client:
   0.30 gives **+11 % on code and +22 % on thinking text** (§1.6). Config only. At two to
   four clients every setting tried (0.85, 0.5, 0.3, the load-aware rule from two sequences) is
   within the runs' scatter of 5 %; only fixed depth 5 loses there (-12 % at four).
5. **`hy4-adopt` was never merged into the integration branch**, so no measurement so far
   has it. On u3: **+17 % decode at one client, +12 % at four and eight, prefill +6-9 %**
   (§1.7). It also removes the unchained quantisers of point 1.
6. **Prefill** (3200 tok/s; §1.8): routed MoE 38 % (half of it link reads of non-resident
   experts: TP2), attention 20.5 %, trunk GEMMs 13.6 %, two-rank junction 9.2 %.

Ranked levers (lossless; details §2, packages §5):

| # | lever | where | gain | effort |
|---|---|---|---|---|
| L1 | `draft-depth-stop: 0.3` now; then the price rule from one sequence on, so the threshold follows the load by itself | config, then core sched | **M** +11 % code / +22 % thinking (1 client); neutral within 5 % at 2-4 clients | S |
| L2 | merge `hy4-adopt` (chained + shared quantisers, router fuse, arena lending, prefill stager) | plugin, arch/common, libr4d rows | **M** +17 / +12 / +12 % decode at 1 / 4 / 8 clients, prefill +6-9 %; a DFly2 step -10..-14 % (1 client), -17 % (4) | S-M |
| L3 | launch fusions: quantisers into their producers (junction, swiglu, expand-gate), q_nope+q_rope, router, routed tail, cache row + rope, sink + merge: 38 -> 27 dispatches a layer | libr4d, plugin | est. +10 % (1 client), +5 % (8), +5-6 % a DFly2 step | M-L, 4 packages |
| L4 | prefill attention (16.4b / 16.4c / 16.6) and `cast` / gather glue | libr4d | est. prefill +4-8 % | L |
| L5 | DFly2: q_temp for Q (sweep); chain stop on the card and short block are small after L1 | plugin, mtp op | unknown, +0-5 % | S |

Not lossless or format-bound (the friend's, §3): `attn_output` int8 -> IQR, trunk u5 -> u4,
`kv_b` int8, head and DFly2 linears below 8 bit, routed down as trellis: together about 8 % of a
DFly2 step, the down projection alone 4.5 % at eight clients.

## 1. Measured breakdown

### 1.1 Decode pass by category (**M**, ms a pass on card 0; launches in brackets)

| category | no spec c1 | no spec c4 | no spec c8 | DFly2 verify c1 (3.4 rows) | DFly2 verify, thinking (2.5 rows) | DFly2 verify c8 (~30 rows) |
|---|---|---|---|---|---|---|
| trunk linears, IQR u5 (`iqr_gemv` / `iqr_m16`) | 3.08 (251) | 3.30 | 3.40 | 3.25 | 3.24 | 3.61 |
| trunk linears, int8 (`gemm_i8_kpar`) | 2.24 (125) | 2.30 | 2.36 | 2.28 | 2.27 | 2.57 |
| routed MoE gate_up, u3 trellis (`moe_iqr`) | 1.63 (41) | 4.39 | 8.38 | 2.97 | 2.58 | 23.87 |
| routed MoE down, w4nl64 (`moe_down_dec`) | 1.03 (41) | 2.79 | 4.85 | 1.93 | 1.74 | 10.71 |
| MoE sort / gather / scatter / swiglu | 0.41 (123) | 0.44 | 0.42 | 0.41 | 0.43 | 0.62 |
| router (f32 GEMM + sigmoid top-k) | 0.68 (82) | 0.64 | 0.68 | 0.63 | 0.61 | 1.57 |
| all-reduce + iHC junction (two-rank) | 1.07 (89) | 1.19 | 1.34 | 1.12 | 1.10 | 2.22 |
| attention: `mla_project_kv` absorb / expand, latent projections | 1.09 (92) | 1.17 | 1.30 | 1.13 | 1.13 | 1.42 |
| attention: `mla_sink_rows` + `merge_sink` | 0.49 (84) | 0.58 | 0.72 | 0.52 | 0.52 | 1.75 |
| attention: cache row front, rope, copies | 0.43 (129) | 0.46 | 0.44 | 0.37 | 0.40 | 0.48 |
| activation quantisers (`quant_act_fp8`) | 0.87 (458) | 0.97 | 1.01 | 0.92 | 0.91 | 1.09 |
| norms, silu | 0.34 (101) | 0.36 | 0.39 | 0.29 | 0.31 | 0.35 |
| small bf16 GEMMs | 0.25 (54) | 0.32 | 0.37 | 0.28 | 0.27 | 0.28 |
| indexer / selection | 0.18 (36) | 0.22 | 0.29 | 0.20 | 0.19 | 0.36 |
| Engram | 0.12 (20) | 0.14 | 0.18 | 0.12 | 0.12 | 0.20 |
| head (`logits_gemm`, int8) | 0.30 (1) | 0.31 | 0.32 | 0.31 | 0.31 | 0.36 |
| **busy, sum** | **14.24** | **19.64** | **26.34** | **16.58** | **15.78** | **51.54** |
| real step (`/metrics`) | 15.74 | 21.28 | 28.65 | 23.27 (with the draft pass) | 22.46 | 58.1 |
| step minus the card's busy time | 1.50 | 1.64 | 2.31 | 2.78 | 2.83 | (5 passes only) |

A decode pass is 1729 dispatches (1734 as a verify). The sampler is not on the tape; §1.5 bounds
it at 0.2-0.5 ms a step.

The DFly2 draft pass (303 dispatches, 5 chain rounds for one sequence = 6 tokens) is **3.9 ms**
busy at one client, 5.6 at four, 6.1 at eight:

| draft pass, c1 | ms | launches |
|---|---|---|
| head, 5 x `logits_gemm` (int8 head, 186 MB a rank each) | 1.51 | 5 |
| draft linears IQ8R (`gemm_i8_kpar`) | 1.05 | 46 |
| quantisers, norms, silu | 0.48 | 94 |
| attention (sink, project, rope, store, copies) | 0.45 | 71 |
| all-reduce, gather, top-k, propose, the rest | 0.42 | 87 |

### 1.2 One trunk layer in order (**M**, no spec, one client, us; `layer.py ns-st.csv ns-st.json c1b 1729 8`)

`quant 1.1 · moe_down_dec 22.9 · moe_gather 3.4 · copy16 3.3 · quant 1.6 · gemm_i8 9.7 (shared
down) · quant 3.4 · iqr_gemv 7.9 · silu_mul 3.3 · quant 1.2 · gemm_i8 5.8 · junction 13.0 ·
quant 2.0 · iqr_gemv 6.0 (q_a) · rmsnorm 3.3 · quant 1.1 · iqr_gemv 6.5 · quant 1.1 · iqr_gemv 4.6
(q_nope, q_rope) · gemm_bf16 5.2 · row_front 3.4 · rope 3.3 · quant 2.0 · iqr_gemv 18.5
(attn_gate) · gemm_bf16 3.3 · rmsnorm 3.2 · qsa_work 3.2 · idx_pool_gated 8.9 · qsa_tail_store 3.3
· project absorb 3.3 · mla_sink_rows 9.1 · merge_sink 6.1 · project expand+gate 8.0 · quant 2.4 ·
gemm_i8 22.8 (attn_output) · junction 12.9 · quant 2.0 · iqr_gemv 10.2 (shared gate_up) ·
gemm_nt_f32 6.8 · router_sigmoid 9.8 · scatter_small 3.3 · quant 1.7 · moe_iqr 36.7 · swiglu 3.3`
= **294 us, 44 dispatches a layer** (weight names by size and position, not read from the tape).

### 1.3 What bounds what

- **Weight bytes** (the cards read 0.55-0.6 TB/s). A rank holds 3.50 GiB of static weights (the
  trunk about 2 GiB of it: per layer and rank `attn_output` 12.8 MB int8, `attn_gate` 7.9 MB u5,
  `kv_b` 7.3 MB bf16, the rest 1-3 MB each) and reads all of the trunk once a pass, whatever the
  row count: trunk linears 5.3 ms at one row, 5.8 ms at eight, 6.2 ms at thirty. `attn_output`
  (22.8 us for 12.8 MB) and `attn_gate` (18.5 us for 7.9 MB) run at the bandwidth; the small ones
  (4.6-10 us for 0.7-3 MB) are at two to three times it, i.e. near the launch floor.
  A routed expert unit is 1.46 MiB a rank (gate_up 0.88 u3, down 0.58 w4nl64). Eight streams
  route 4084 units a step: 3.5 GiB of gate_up in 8.4 ms and 2.4 GiB of down in 4.9 ms, the
  misses included, is the bandwidth. **At four streams and more the routed GEMMs are half of the
  pass (52 % at eight, 67 % in an eight-stream DFly2 verify) and they are bound by bytes read,
  not by rows**: 2.2 us a routed unit (it was 1.0 us with w4 gate_up; the u3 trellis is denser
  but each unit still costs its bytes), plus 66 us a missed unit (`experts-streaming.md`).
  Only fewer bytes (a format: the friend's) or fewer routed units (a depth rule that prices a verify row: §2 L1) help.
- **Launch floor.** About 30 of a layer's 44 dispatches take 1.1-3.4 us: the floor of a
  dispatch (3.2-3.3 us chained, `kernels/perf.md` measured 2.65 us in kbench). The pass's 1310
  launches outside the big GEMMs are 5.9 ms at one client (junction 1.07, projections 1.09,
  quantisers 0.87, router 0.68, sink 0.49, cache row / rope / copies 0.43, MoE glue 0.41, norms
  0.34, small GEMMs 0.25, selection 0.18, Engram 0.12), and the small IQR GEMMs carry a floor
  share of their own.
- **Gaps: the quantisers are not chained.** Of the 1729 dispatches 463 carry the barrier bit:
  all **458 `quant_act_fp8`** launches (one in front of every quantised GEMM: no producer writes
  codes on this branch), four `vit_gemm_wa`, the head. Each of them and the chained launch behind
  it start 4.4-4.6 us after the previous end in the stamps (every other dispatch overlaps its
  predecessor). Unstamped the gap is smaller, but the step's 1.5 ms (one client) to 2.3 ms (eight)
  outside the card's busy time is these 916 waits: 9.5 % / 8 % of the step.
- **Two-rank wire**: 84 junctions at 12.6-15.6 us, 1.07-1.34 ms a pass (7 % at one client).
  TP-specific; see the TP2 / TP4 notes.
- **Attention is small**: sink rows + merge 0.49 ms, project_kv 0.71 ms, latent projections
  0.39 ms: 11 % of a one-client pass together, 5 % at eight clients.
- **Experts' misses** at this container: hit 0.975-0.992 without speculation (5-53 missed units
  a step: 0.3-3.5 ms), 0.978 with DFly2 at eight clients (185 missed units = 12 ms of a 58 ms
  step). Where the experts fit (TP4) this term is zero; at TP2 it is `experts-streaming.md`'s.

### 1.4 The DFly2 step (**M**)

| phase | step ms | verify busy | draft pass busy | outside both | drafts verified a step | tokens a step | tok/s |
|---|---|---|---|---|---|---|---|
| c1 code (c1b) | 23.27 | 16.58 | 3.91 | 2.78 | 2.37 | 2.59 | 110.7 |
| c1 thinking | 22.46 | 15.78 | 3.85 | 2.83 | 1.46 | 2.04 | 90.7 |
| c4 code | 35.7 | 29.6 | 5.57 | ~0.5 | 8.5 (4 sequences) | 10.6 | 310 |
| c8 code | 58.1 | 51.5 | 6.13 | ~0.5 | 24.2 (8 sequences) | 22.3 | 407 |

- A verify row costs **1.0 ms** on the card at one client ((16.58 - 14.24) / 2.37 and
  (15.78 - 14.24) / 1.46), 0.95 ms of it the routed GEMMs: 390 more routed units a row (the
  draft rows share experts with their neighbours; a lone row routes 700). It is the same
  1.0 ms at four clients (fixed 5 against 0.85: 9.7 ms for 10.0 rows) and at eight
  ((51.5 - 26.3) / 24.2).
- The draft pass is 17 % of a one-client step and it always runs its five rounds, although the
  next verify takes 1.46 (thinking) to 2.37 (code) of the five drafts.
- Thinking text is slower only through the drafter: the no-spec step is the same (16.0 vs
  15.7 ms). Acceptance of the first draft is 0.57-0.64 either way; what differs is the depth the
  rule allows (depth 1 in 60-81 % of the steps of the BetterBench complete run, in 32 % of this
  probe's code steps).

### 1.5 The depth decision's host wait is not a lever (**M**)

The server log's `host waits N (20-29 ms each, all ended blocked)` is the host sleeping while the
card runs the step whose laws it needs; it is not card idle time. A/B at equal card work, one
client, sampled, same seeds: `draft-depth: fixed` (5 drafts verified, no wait) against
`draft-depth: adaptive` with `draft-depth-stop: 0` (5 drafts verified, the wait every step:
721 waits of 28.6 ms, all blocked):

| phase (two passes each) | fixed 5, ms a step | adaptive stop 0 (waits), ms a step |
|---|---|---|
| c1 | 27.22 / 25.75 | 27.22 / 25.51 |
| c1b | 26.57 / 25.64 | 26.41 / 25.30 |
| thinking | 26.39 / 26.25 | 26.28 / 26.19 |
| c4 | 46.35 | 46.19 |

Identical within 0.3 ms, tokens identical. The card starts on the first packets of the next
pass while the host still plays the rest of the tape, so the turnaround is the wake-up alone.
(`notes/wp13-8.md` measured 77.8 -> 84.5 tok/s greedy when the wait was skipped for greedy rows;
the waits were 93 ms each then, longer than a step. Now a wait is as long as the card's step and
`engine_steps_ahead_total` still counts every step as issued ahead.)
Greedy against sampled at fixed depth 5 is 0.2-0.5 ms a step (25.50-26.78 against 25.64-27.22):
the coupled sampler chain and its commit are not a lever either.

The step loop's own phases at debug level (`cpv2`, DFly2 coupled, mean of 128 steps): plan,
begin, build, pre, save, commit and draft bookkeeping together 0.03-0.14 ms; `ranks` (the rank
threads playing the 1734-record tape) 0.45-0.63 ms at one client, 1.23 ms at eight, overlapped
with the card; everything else is `collect`, the wait for the card. The host is not on the
critical path of a decode step.

### 1.6 The depth rule is the largest lever of the sampled DFly2 path (**M**)

One client pays 1.0 ms for a verify row on a 22-23 ms step that makes 2.0-2.9 tokens: a draft
pays its row when its whole prefix is accepted with probability above about 0.10
(1.0 ms x 0.11 tok/ms). The rule in force below five sequences verifies draft j only while the
product of the walk's top probabilities is at least 0.85. Sweep, same phases and seeds
(`cp2`, `stop50`, `stop30`, `fixed5`; tok/s, first / second pass):

| `draft-depth-stop` | c1 code | c1b code | c1 thinking | c4 code | drafts verified a step (c1 / c1b / thinking) |
|---|---|---|---|---|---|
| 0.85 (today) | 115.0 / 112.1 | 125.3 / 131.5 | 93.5 / 92.3 | 293.2 (289-312 over four runs) | 1.9 / 2.4 / 1.6 |
| 0.50 | 119.0 / 120.7 | 130.5 / 139.2 | 106.1 / 95.0 | 323.7 | 2.6 / 3.5 / 1.9 |
| 0.30 | **122.6 / 130.4** | **144.1 / 142.1** | **116.3 / 110.7** | 308.5 | 3.6 / 4.1 / 2.9 |
| fixed 5 | 107.6 / 113.7 | 137.0 / 124.2 | 115.1 / 104.2 | 263.9 | 5 / 5 / 5 |

0.30 against 0.85: **+11 % on code, +22 % on thinking text at one client**, in both passes of
all three phases. A run of 256-384 sampled tokens scatters by about 5 %, and at four clients the
settings are inside that (0.85 measured 289-312 over four runs, 0.5 324, 0.3 308): the same
1.0 ms row has to beat 0.3 tok/ms there instead of 0.11, so the marginal drafts about break
even. Only fixed 5 is clearly worse at four (-12 %). The engine already has the right rule for this
(`draft_depth_select`: take draft j while calibrated acceptance exceeds tok/ms x the row's ms,
tables calibrated online per stream count; the log shows `1 seq: 0.105 tok/ms, row 1.17 ms`
already counted) but only runs it from `draft-depth-load-from` = 5 sequences on, and the option
does not accept 1.

The load-aware rule from two sequences on (`draft-depth-load-from: 2`, `lf2` against `cp3`, same
phases and seeds, tok/s first / second pass):

| | c2 | c4 | c8 | drafts verified a step (c2 / c4) |
|---|---|---|---|---|
| stop 0.85 below five sequences (today) | 198.2 / 209.5 | 289.1 / 311.8 | 411.1 | 4.4 / 7.0-8.7 |
| load-aware from two | 216.3 / 192.4 | 287.5 / 297.3 | 429.9 | 6.3-6.6 / 11.7-11.9 |

No difference outside the scatter: the rule verifies half as many drafts again and gets the same
throughput, which is what "priced at the margin" looks like. So the two to four client range is
flat, and the one-client gain is the lever. The price rule at one sequence should land where
stop 0.3 does (its table for one sequence reads 0.10 tok/ms and 0.9-1.2 ms a row, i.e. take a
draft above 0.09-0.12 acceptance), without a hand-set threshold; that is to be measured when the
option accepts 1.

### 1.7 The adopted levers of `hy4-adopt` are not on the integration branch (**M**)

`claude/peaceful-wright-hhqnhi` @ dc680ce3 does not contain `hy4-adopt` (6 commits:
`shape_probe_ok`, shared int8 / fp8 codes with fused producers, `router_fuse`, the opt-in chained
quantisers, Engram read-ahead, the f32 Engram eps). The server says so at start: `activation
arena: not lent -- the 'hyv4flash' plugin does not declare at a smaller step size`. Every
figure of §1.1-1.6, of `notes/u3-perf.md` and of the BetterBench run is without it. `hy4-perf5`
(= integration @ 85331617 + adopt + compat + xstream) on the same probe, no speculation:

| | HEAD | perf5 | |
|---|---|---|---|
| c1, ms a step (two prompts) | 16.73 / 15.74 | 14.10 / 13.41 | 59.7 / 63.5 -> 70.9 / 74.5 tok/s, **+17 %** |
| c4, ms a step | 21.28 | 18.94 | 187 -> 210 tok/s, **+12 %** |
| c8, ms a step | 28.65 | 25.57 | 278 -> 311.5 tok/s, **+12 %** |
| prefill 6.7K, TTFT | 2.08 s (3211 tok/s) | 1.97 s (3393 tok/s) | +5.6 % |
| prefill 24.9K, TTFT | 7.82 s (3177 tok/s) | 7.18 s (3462 tok/s) | +9 % |
| dispatches a decode pass / unchained | 1729 / 463 | 1479 / 5 | |
| card busy, c1 / c8 | 14.24 / 26.34 ms | 13.08 / 24.89 ms | |
| step minus busy, c1 / c8 | 1.50 / 2.31 ms | 0.33 / 0.68 ms | the 916 gaps are gone |
| expert hit c1 / c8 | 0.992 / 0.987 | 0.997 / 0.991 | the arena lends 1.69 GiB (1206 units a rank) |

Per category (ms, launches), one client / eight clients:

| | HEAD c1 | perf5 c1 | HEAD c8 | perf5 c8 |
|---|---|---|---|---|
| trunk linears IQR | 3.08 (251) | 2.24 (251) | 3.40 | 2.66 |
| trunk linears int8 | 2.24 (125) | 2.00 (125) | 2.36 | 2.27 |
| routed gate_up + down | 2.66 | 2.12 | 13.23 | 12.32 |
| quantisers | 0.87 (458) | 1.11 (333) | 1.01 | 1.22 |
| norms / silu / MoE glue | 0.75 (224) | 0.33 (99) | 0.81 | 0.38 |
| junction (two-rank) | 1.07 | 1.35 | 1.34 | 1.54 |
| everything else | 3.57 | 3.93 | 4.19 | 4.50 |

With DFly2 (coupled, `draft-depth-stop` 0.85, same phases and seeds; `cp2` against `p5-cp`),
ms a step, first / second pass:

| | HEAD | perf5 | |
|---|---|---|---|
| c1 code | 23.33 / 22.05 | 20.76 / 19.44 | -11 % / -12 % |
| c1b code | 23.38 / 22.03 | 20.95 / 19.81 | -10 % |
| c1 thinking | 23.53 / 22.65 | 20.13 / 19.86 | -14 % / -12 % |
| c4 code | 36.66 (7.7 drafts a step) | 30.38 (7.0) | -17 % |
| c8 code | 58.1 (24.2 drafts, stamped run) | 48.94 (20.2) | about -10 % at equal rows |

Tokens a step differ between the two builds' runs by the sampling scatter (the coupled draws are
not reproducible across builds), so the step time is the comparison: **+11..+16 % tok/s at one
client, +20 % at four** at equal acceptance. tok/s as run: c1 113.6 -> 123.9, c1b 128.4 -> 131.8,
thinking 92.9 -> 121.3, c4 293 -> 335, c8 407 -> 441.

After the merge a layer is 38 dispatches, all chained, 270 us at one client; 19 of them run at
the floor of a chained dispatch, **3.2 us** each (`layer.py p5-ns.csv p5-ns.json c1b 1491 8`):
six `quant_act_fp8`, `gated_quant`, `rmsnorm_quant`, `swiglu`, `moe_gather`, `copy16`,
`mla_row_front`, `rope`, two small bf16 GEMMs, `rmsnorm`, `qsa_work`, `qsa_tail_store`, the
absorb projection. 1479 dispatches x 3.2 us is 4.7 ms: a third of the pass is launch floor.

### 1.8 Prefill (**M**, `prof`: `profile-ops`, 6681 tokens = three chunks of 2048 + 537, HEAD)

`profile-ops` restores a synchronisation around every op (TTFT 2.59 s against 2.08 s), so the
shares are what to read, rank 0, 2436 ms of op time:

| part | ms | share | |
|---|---|---|---|
| routed MoE GEMMs (`moe_gemm_q` gate_up 532, down 398) | 930 | 38 % | includes the in-place reads of the non-resident experts: hit 0.90, 2060 missed units a chunk and rank = 3.0 GiB over the link, about 110 ms of a chunk's ~215 ms |
| attention: `mla_attn_paged_sink` 204, `mla_attn_sel_sink` 177, `mla_expand_gate` 51, `attn_kv_sink` 38, `mla_project_kv` 22, `mla_kv_split` 9 | 501 | 20.5 % | sel: 3.6-3.9 ms a call at 2048 rows, 16 heads a rank (the 12 sparse layers once the context passes 2048 tokens); paged: 1.75 ms a call over the four absorbed chunks, 32 heads a rank (30 sliding layers); `attn_kv_sink` (first chunk) 0.8 ms |
| trunk GEMMs (`gemm_nt_q`) | 332 | 13.6 % | `attn_output` 91, `attn_gate` 72, `attn_q_a` 49 |
| two-rank junction (`ar_rmsnorm_ihc_junction`) | 225 | 9.2 % | 0.45 ms a call: 12.6 MB a direction over the link (TP note) |
| quantisers (`had_quant_act_i8g`, 2748 calls) | 83 | 3.4 % | |
| router GEMM f32 | 58 | 2.4 % | |
| `moe_gather` 46, `cast` 41, `swiglu_clamped` 39 | 126 | 5.2 % | |
| rest (kv_a / index / Engram GEMMs, rope, gather, embed) | 181 | 7.4 % | |

A 2048-token chunk is about 640 ms unprofiled (3200 tok/s). The BetterBench decline from 3992
tok/s at 1.5K to 3026 at 47K is the sparse layers' selection attention (no cost below 2048
tokens of context) plus the absorbed form of every chunk after the first.

## 2. Levers, ranked by expected gain x confidence / effort

All of §2 is lossless: no weight format, no rounding, no output law changes. Bits may move once
only where a line says so. What needs the friend's decision is §3.

Gains are for the user's case first (sampled code, DFly2 coupled, one to four clients, thinking
on), then no speculation and eight clients. "Step" figures are HEAD's unless stated.

### L1. Depth rule by price at every stream count — config today, then the rule from one sequence

- **Mechanism.** §1.6. Verify draft j when its calibrated acceptance times the tokens it yields
  exceeds the step time it adds; the price of a row is its routed experts' bytes (1.0 ms at one
  client), not a constant of the drafter.
- **Where.** (a) config only: `draft-depth-stop: 0.3` (`work/levers/stop30.yaml`); it is inside
  the scatter at two to four clients and the load-aware rule takes over from five. (b) core: `core/config.cpp:1081` accepts
  `--draft-depth-load-from` 2..64; let it take 1 and run `draft_depth_block_load` at n = 1
  (`core/engine_draft.cpp:1211`, `core/sched/draft_depth.h:306/639`: the tables for one sequence
  exist and are calibrated online already). Check `load_min` (2 rounds always) and the
  calibration bins at one sequence, where the stop rule has so far kept the low-confidence bins
  of drafts 3-5 empty (a rule that never verifies them never learns their acceptance: seed the
  bins from the built-in table or explore with a small probability).
- **Exactness.** Lossless: the rule reads the walk's laws only (the engine's own statement for
  `--draft-depth adaptive`), the verify is the same maximal coupling. Sampled tokens of a fixed
  seed change, as they do with any depth setting; greedy output cannot.
- **Gain.** (a) measured: one client +11 % code, +22 % thinking; two to four clients neutral
  within 5 %. (b) estimated equal to (a) at one client and neutral above (measured neutral at
  two and four); its value is that the threshold follows the row's price when L2 / L3 or a new
  container change it, instead of a constant tuned on this one.
- **Measurement.** `work/levers/run.sh <tag> <cfg> c1,c1b,think,c4,c1,c1b,think`, three seed
  sets (`SEED[0]` in `probe.py`), the BetterBench complete run for the thinking share.
  Report tok/s, drafts verified a step, acceptance by depth from the log's `draft depth adaptive
  (final)` lines.
- **Work package G-1** (S; Sonnet; GPU short). Config change in the serving yaml now. Then the
  one-sequence rule: option bound, a `sched_test` case at n = 1 (the choice equals
  `draft_depth_select` by hand for a given table), the exploration / seeding of empty bins,
  docs (GUIDE §4.6, the options table). Acceptance: one client >= stop 0.3 on code and
  thinking over three seed sets, two and four clients not below today's, `mtp_rrs_test` and the
  rollback gate unchanged.

### L2. Merge `hy4-adopt` into the integration branch

- **Mechanism.** §1.7: chained rotated quantisers (the 458 barrier-bit launches and the 916
  waits behind them), shared codes and fused producers (458 -> 333 quantiser launches, 224 -> 99
  norm / glue launches), `router_fuse`, `shape_probe_ok` (arena lent: +1206 resident units a
  rank at decode; prefill stager armed), Engram read-ahead for disk placement.
- **Where.** Plugin (`arch/hyv4flash/*`), `arch/common/rad_fp8.h`, `rad_block_moe_fp8.h`,
  libr4d rows (+ libref / libavx schemas), all on branch `hy4-adopt` (66c7d303), built and
  measured in `wt/perf5`.
- **Exactness.** Lossless. `notes/adopt.md`: greedy 64 tokens + logprobs identical base against
  new for bf16 / int8 / IQ8R, TP1 / TP2, drafter off and depth 5; `notes/perf.md`: greedy texts
  identical on the real w4nl64 container. Not yet compared on u3 (this note's runs are sampled).
- **Gain.** Measured on u3, no speculation: +17 % at one client, +12 % at four and eight;
  prefill +6-9 %. With DFly2: a step -10..-14 % at one client, -17 % at four
  (+11..+20 % tok/s at equal acceptance).
- **Measurement.** The table of §1.7 again on the merged branch (`run.sh ns ... c1,c8,c4,c1b,pf6,pf20`
  and the coupled set), plus greedy c1 texts HEAD against merged on u3 (`TEMP=0`).
- **Work package G-2** (S-M; Opus for the conflicts; GPU short). Merge `hy4-adopt` (6 ahead,
  8 behind) onto dc680ce3; `hy4-perf5` shows it merges with 85331617. Keep 28e35f5d (bf16 experts sharded) and dc680ce3 (slot wait) intact, rerun adopt's own list
  (arch tests, host parity matrix, gpu-mini greedy equal with `RADIANCE_HY4_SHAPE_PROBE=0/1`,
  `RADIANCE_HY4_CHAIN_Q8=0/1`), real-container greedy compare, the stamp table (every dispatch
  chained except the head and the four Engram GEMMs). `hy4-xstream` (X1) is a separate decision
  of the TP2 note; `hy4-compat` stays out (DECISIONS).

### L3. Fewer launches: quantisers into their producers, and six small fusions

- **Mechanism.** After L2 a chained dispatch still costs 3.2 us of card time whatever it does,
  and half of a layer's 38 dispatches do almost nothing else (§1.7). Each fusion below is the
  WP16 pattern (E6: the fused row declares the chain it replaces, bodies are the standalone
  kernels' device functions, kbench's fusion oracle compares bytes):

  | # | fusion (dispatch numbers of §1.7's layer) | launches a layer |
  |---|---|---|
  | a | `swiglu_clamped` + its `quant_act_fp8` (routed down's input) | -1 |
  | b | the junction writes the codes its consumers read: one quantiser behind the junction in front of the attention, two (the rotated pair and the routed arm's own) behind the one in front of the feed-forward; plain and two-rank forms | -3 |
  | c | `mla_expand_gate` + `quant` (input of `attn_output`) | -1 |
  | d | `q_nope` and `q_rope`: one IQR launch over the shared codes, two weights | -1 |
  | e | `mla_row_norm_rope_store` + the query's `rope` | -1 |
  | f | `gemm_nt_f32` + `router_sigmoid_topk_scatter` (WP16.5 skipped it: needs the multi-block f32 tile with the router's exact sum order) | -1 |
  | g | `moe_down_dec` + `moe_gather` (+ `copy16`): the routed down writes the per-token sum | -1..-2 |
  | h | `mla_sink_rows` + `merge_sink` in one launch for decode-class rows with one slot (short windows; ties into 16.4b's row tiles) | -1 (30 sliding layers) |

  Ten to eleven a layer: about -440 launches, 1479 -> 1040.
- **Where.** libr4d (`r4d_quant_act_fp8.hip`, `r4d_ihc.h` / the junction, `r4d_iqr*`, moe, mla
  files), libref + libavx schemas for each new op (hosts run the chain), plugin wiring behind a
  switch per fusion (`RADIANCE_HY4_*`, default on once green), `data/kernels_hy4.rkb` cases.
- **Exactness.** Byte-equal to the chain (E6), every row decode-class (E1: a row's bytes do not
  depend on M).
- **Gain (estimate).** 440 x 2.7-3.2 us (the fused kernel keeps the work, loses the dispatch)
  = **-1.2..-1.4 ms a pass**: no speculation one client 13.4 -> ~12.1 ms (+10 %), eight
  clients 25.6 -> 24.3 (+5 %); a DFly2 one-client step (verify + draft) about +5-6 %. Prefill:
  (a)-(c), (f), (g) remove 5 of the ~10 small ops a layer that the profile shows at 0.15-0.3 ms
  each at 2048 rows only if their fused forms serve every M: +1-2 %.
  Confidence: medium-high for the floor (measured, uniform across 19 kernels), medium for each
  fusion's cost (b needs the junction's four forms, f was skipped once for size).
- **Measurement.** `layer.py` before / after (dispatches and us a layer), stamps `agg.py` (busy
  a pass), `run.sh` c1 / c8 tok/s, greedy texts equal with each switch off / on.
- **Work packages** (each: emulator test first, selftest, fusion oracle, stamp table):
  - **G-3a** (M, Opus): fusions a, c (producer + quantiser; the model is adopt's `gated_quant`,
    `rmsnorm_quant`). -2 a layer.
  - **G-3b** (L, Opus): fusion b, the junction's quantised outputs in the plain, two-rank and
    four-rank forms (the junction is replicated: every rank must write identical codes, E5).
    -3 a layer. Coordinate with the TP notes, which touch the same kernels.
  - **G-3c** (M, Sonnet then Opus): d, e. -2 a layer.
  - **G-3d** (M-L, Opus): f, g. -2..-3 a layer.
  - **G-3e** with 16.4b: h.
  Order by gain / effort: 3a, 3c, 3b, 3d.

### L4. Prefill attention and glue (kernels, known open items re-ranked by the profile)

- **Mechanism / where.** The profile puts 20.5 % of prefill into the attention kernels and
  5 % into `moe_gather` / `cast` / `swiglu`. The open kernel items apply to exactly these rows:
  - `mla_sink` bit-neutral speed (perf.md L5 / 16.4b: column-major value tile, staging waves,
    row tiles, `hg` / `rt` / `sw` axes) on `mla_attn_paged_sink` and `mla_attn_sel_sink`
    (381 ms of 2436): the rows of one sequence select overlapping blocks; staging each block
    once a tile instead of once a row is where sel's 3.9 ms a call goes.
  - single-bf16 P and a consistent denominator for prefill rows (perf.md L6 / 16.4c), `stable 0`
    rows only.
  - `attn_kv_sink` 64-row tile (L7 / 16.6): 38 ms of 2436, first chunk only: 1.5 % at most; last.
  - `cast` 41 ms: a bf16 -> bf16 column copy of the 1024-wide query, 0.13-0.15 ms a call at 2048
    rows, six a layer-step; give the consumer the column slice as its operand (the pattern of
    `mla_row_norm_rope_store`) or fold the copy into the q GEMM's output placement.
- **Exactness.** 16.4b bit-neutral; 16.4c moves prefill rows inside their stated bound (E4;
  decode and verify rows untouched; tell the friend the build moved before KL numbers);
  `cast` removal byte-equal.
- **Gain (estimate).** sel + paged -20..-35 % (perf.md's estimates for L5 + L6 at 16 / 8
  heads against its attention-side budget) = -75..-130 ms of 2436: **prefill +3-6 %**, more the deeper the context (the sparse
  layers' share grows with the keys). `cast` + `moe_gather` fusion: +1-2 %.
- **Measurement.** `run.sh prof prof.yaml nowarm,pf6` (and `pf20`) before / after, `prof.last.txt`
  aggregation in this note's style; rad-kbench with selection tables taken from a real
  `qsa_select` (perf.md §2.2's caveat: random tables overstate sel).
- **Work package G-4** = 16.4b, 16.4c, 16.6 as planned in `kernels/perf.md` §4, with the
  acceptance figures restated at TP2 widths (16 heads a rank, 2048 rows): sel <= 2.8 ms,
  paged <= 1.0 ms a call. Opus; GPU short.

### L5. DFly2's draft pass

The draft pass is 3.9 ms of a 22-23 ms one-client step and 6.1 of 58 at eight. 65 % of it is
weight bytes (head 5 x 186 MB, IQ8R linears): formats, §3. What is left without them:

- **(a) q_temp for the proposal law** (open item of WP13.8). Q_k is the softmax at T 1 of the
  top-8 logits while the target samples at T 0.9; first-draft acceptance is 0.57-0.64.
  `mtp_couple_propose` takes the f32 shortlist values: scale them by 1 / q_temp before the
  propose (plugin, one `mul` on [S, 8] a round or a parameter of the op). Exact for any q_temp:
  the verify couples against whatever Q the state plane records. Gain unknown; a sweep of
  q_temp in {1.0, 0.9, 0.8, 0.6} on code and thinking costs four short runs once the parameter
  exists (`RADIANCE_HY4_DFLY_QTEMP`). Worth +0..+5 % by DFlash2's precedent (0.6 there).
  **G-5a** (S, Sonnet): parameter, `mtp_rrs_test` chi-square case with q_temp != 1, the sweep.
- **(b) Stop the chain on the card.** The rule can never take draft j + 1 once the product
  falls below the threshold at j, yet rounds j + 2..5 run their head pass and correction
  (0.42 ms a round at one client). With today's 0.85 that wastes 1.5 (code) to 2.2 (thinking)
  rounds = 0.65-0.9 ms a step (3-4 %); **after L1 the rule takes 3-4 drafts and the waste is
  under one round (1-2 %)**. Needs a device flag a round and a skip form of `logits_gemm` and
  the three correction GEMMs. Not worth a package after L1; listed so nobody re-derives it.
- **(c) Short block under load.** The plugin declares no `rad_drafter_short_block`; at eight
  clients the verify takes 3.0 drafts a row. A block of 3 from `draft-short-from` sequences on
  saves two rounds = 0.9 ms of 58 (1.5 %). Small; (S) if someone is in the file anyway.

### L6. Things measured and found not to be levers (so they are not built)

- The adaptive depth's host wait (§1.5): 0 +- 0.3 ms.
- The sampler chain at T 0.9 against greedy (§1.5): 0.2-0.5 ms a step.
- Attention at decode: sink rows + merge 0.5 ms, project_kv 0.7 ms of a 13-14 ms pass; 16.4b's
  decode gain is the one launch of L3 (h).
- Engram: 0.13 ms a decode pass (four `gemm_nt_bias` of 68 us are bf16 weight bytes: §3),
  1.2 % of prefill. iHC outside the junction: 0.04 ms.
- A shortlist head for the drafter: rejected by the user; not proposed.

## 3. Friend's decisions (formats; lossy or recalibration) — options with their measured price

Everything byte-bound in §1.3 moves only with the bytes. Per decode pass and rank (one client,
after L2: 13.1 ms busy; a DFly2 step ~21 ms):

| option | bytes today | time today | estimate |
|---|---|---|---|
| `attn_output` int8 -> IQR u5 (the one large int8 trunk weight) | 12.8 MB a layer | 21.8 us a layer = 0.92 ms | -0.3 ms (2.5 %) |
| trunk IQR u5 -> u4 | 0.68 GiB a pass | 2.24 ms (0.8 of it launch floor) | -0.25 ms (2 %) |
| `kv_b` bf16 -> int8 (DECISIONS U2 keeps it bf16; WP17 on request) | 7.3 MB a layer | absorb + expand 11.6 us = 0.49 ms | -0.2 ms (1.5 %) |
| head int8 -> u5 | 186 MB | 0.30 ms decode, **1.51 ms a draft pass** | -0.1 ms decode, -0.55 ms a DFly2 step (2.5 %) |
| DFly2 linears IQ8R -> u5 / u4 | ~0.55 GiB a draft pass | 1.05 ms | -0.35 ms a DFly2 step (1.5 %) |
| routed down w4nl64 (0.58 MiB a unit) -> u3 trellis (0.44) | 2.4 GiB a pass at 8 rows | 4.7-4.9 ms at eight clients, 10.7 in an eight-client verify | -1.2 ms (4.5 %) / -2.7 ms; at TP2 also 10 % more units resident |
| Engram key / value projections bf16 -> int8 | ~37 MB x 4 | 0.27 ms | -0.13 ms (1 %) |
| latent cache FP8 / int8 (perf.md L9) | | prefill selection | lossy, opt-in, own design note first |
| KV cache dtype i8 (experts-streaming X6) | | capacity at TP2 | TP2 note |

Together the trunk / head / drafter options are about -1.6 ms of a 21 ms DFly2 step (8 %); each
needs his KL run. We offer the tables; we do not change formats (DECISIONS 2026-11-22).

## 4. Pointers: not general, owned by the TP notes

- **Two-rank junction / all-reduce**: 1.07-1.54 ms a decode pass (7-10 % at one client), 9.2 %
  of prefill (12.6 MB a direction a junction at 2048 rows). TP2 / TP4 notes.
- **Experts that do not fit** (TP2): misses are 0.3-3.5 ms without speculation, 12 ms of a
  58 ms eight-client DFly2 step, and about half of the prefill's MoE time (18 % of prefill).
  `kernels/experts-streaming.md` X2-X4; L2 above already adds the arena's 1206 units.

## 5. Order and totals

| order | package | effort | measured / estimated gain |
|---|---|---|---|
| 1 | G-1 depth rule: `draft-depth-stop: 0.3` now, one-sequence price rule after | S | **M**: +11 % code, +22 % thinking at one client; neutral at 2-4 |
| 2 | G-2 merge `hy4-adopt` | S-M | **M**: +17 % (1 client), +12 % (4, 8) no speculation; prefill +6-9 % |
| 3 | G-3a, 3c, then 3b, 3d: launch fusions | M-L, four packages | est. +10 % (1 client), +5 % (8), +5-6 % a DFly2 step |
| 4 | G-4 = 16.4b / 16.4c / 16.6 prefill attention, `cast` | L (planned) | est. prefill +4-8 % |
| 5 | G-5a q_temp | S | unknown, +0-5 %; one sweep decides |

G-1 and G-2 act on different terms (tokens a step, time a step) and should multiply for a DFly2
client. Measured separately, not together: x1.11-1.22 from the rule, x1.11-1.16 from the merge,
so about +25-40 % for one sampled client: code 113-128 -> about 145-165 tok/s, thinking text
93 -> about 125-130. The first run after G-2 is the two together. After 1-3 a one-client decode
pass is about 12 ms of which 8 ms are weight bytes and 1.4 ms the two-rank
wire: the rest of the road is formats (§3) and the TP notes.

## 6. Files

`work/levers/`: `probe.py`, `run.sh`, `run5.sh` (perf5 build), `agg.py`, `layer.py`, configs
(`nospec`, `coupled`, `fixed5`, `adapt5`, `stop50`, `stop30`, `lf2`, `prof`, `cpv`, `nsv`),
results `<tag>.json` / `.out` / `.server.log`, stamps `ns-st.csv`, `cp-st.csv`, `p5-ns.csv` with
their `.agg.txt` / `.agg.json`, `prof.last.txt` (the last per-op table of rank 0), `radinfo-v.txt`
(the container's tensors). The stamp CSVs are 23-56 MB each; delete them when the tables above
are no longer questioned.
