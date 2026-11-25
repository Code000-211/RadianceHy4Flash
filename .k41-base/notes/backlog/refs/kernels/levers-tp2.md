# Hy4-flash at TP2 (2 x R9700): what is particular to two cards, measured, and the levers

Research and plan. Nothing here is implemented. Numbers marked **M** are measured in this note's
runs (2026-11-22, real container `/srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad`, TP2,
210 W, Engram in RAM, `src/build` @ dc680ce3, sampled T 0.9, code prompts) or cite the note they
come from; everything else is an **estimate** and says what it is derived from.

Scope: only what two ranks change. TP4-specific levers and TP-independent levers (launch gaps,
fusions, the adaptive depth's host wait, the draft head) are other notes; where a TP2 finding
points at one of them it is named and left there.

Data of this note: `work/tp2/` — `probe.py` (c1, c8, prefill phases with monotonic windows),
`probe2.py` / `probe3.py` (prefill only), `run.sh` / `run2.sh`, `profagg.py` (the last `profile-ops`
table summed by op), `attr.py` (frontier attribution of
`RADIANCE_AQL_STAMP` lines: a dispatch is charged `end - max(start, latest end before it)`, since
chained packets start early and wait for their chain), `agg.py`; runs `ns-st.*` (no speculation,
one pass in 12 stamped), `cp-st.*` (DFly2 coupled, adaptive depth, one pass in 12), `pf-st.*`
(prefill; no stamps, see §1.4), `pf-prof.*` (prefill under `profile-ops`).

## 0. Result in one page

1. **The cross-rank path is already the short one, and it is cheap.** TP2 runs the fused twin
   `ar_rmsnorm_ihc_junction` at every height by default (`RADIANCE_HY4_JUNCTION_TP4` gates four
   ranks only): 84 a pass, the only collectives of a trunk pass. Transport: P2P-BAR one-shot push
   into the peer's fine-grained IPC scratch, per-block flag handshake on the device, coded
   lossless wire from 512 KiB (86 rows). The start log confirms `--p2p peer to peer`. No host
   bounce, no host sync.
   Measured share of a decode pass (**M**, §1.1): 5.9 % at one stream (12.6 us each), 4.2 % at
   eight (15.2 us), 5.3 % / 3.6 % of a DFly2 verify pass (13.3 us at 6 rows, 27.5 us at ~48 rows).
   Against the plain junction's 7.3 us at TP1 (notes/wp16-2.md) the price of being two ranks is
   about 5.3 us x 84 = 0.45 ms a pass, **2.5 %** at one stream, under 2 % at eight.
   At prefill the same 84 junctions are **about 10 %** of a 2048-token chunk (0.67–0.73 ms each
   under `profile-ops`, §1.4), 7–8 points of that transport and wait for the peer.
2. **No rank skew, no barrier cost.** The junction's charged time is p10 12.5 / p50 12.8 / p90
   13.0 / p99 13.2 us at one stream (**M**): a rank never waits measurably for the other. Both
   ranks' heat engines and movers end a run with identical counters (**M**, exit report: 17084
   promotions, 12576 demotions, 23.86 GiB h2d on each), so the residency sets do not diverge. The
   host barrier is two condvar round trips a step (`core/runtime/rank_barrier.h`, 1–2 us each).
3. **What costs at two cards is the link to the host, not the link between the cards.** The u3
   container's routed experts are 29.3 GiB a rank in 20,992 units of 1.5 MB; the cards hold
   86–88 % of them with the whole KV loan out (**M**). The rest is read in place over a
   16 GT/s x16 uplink (27.6 GB/s measured in kernels/experts-streaming.md). Per rank and step
   (**M**, §1.3): 8.6 MB at one stream, 71–110 MB at eight, 35 / 232–377 MB with DFly2 at one /
   eight streams, **2.2–3.3 GB a prefill chunk**. As link time: 2 % of a step at one stream,
   9–13 % at eight, 15–22 % at eight with DFly2, **12–18 % of prefill**.
4. **The decode pass is launch-bound and replicated, so the second card buys capacity and
   bandwidth, not latency.** At one stream 19–21 % of the pass is gaps before ordered starts and
   most kernels sit at the launch floor on both cards in lockstep (**M**); the sharded,
   bandwidth-bound kernels (routed GEMMs, head) are 16 % of the pass at one stream and 41–65 % at
   eight streams / in a verify pass. Re-sharding (heads, indexer, drafter) therefore moves little
   (§3, closed items).
5. **The build measured (`claude/peaceful-wright-hhqnhi` @ dc680ce3) does not lend the activation
   arena and has no prefill stager** (`activation arena: not lent -- the 'hyv4flash' plugin does
   not declare at a smaller step size`): `shape_probe_ok` lives on `hy4-adopt` / `hy4-perf5`
   (528f9a0c), not on the integration branch. At TP2 that is 1.37 GiB a card (about 4 points of
   residency) and every prefill chunk reading its pooled experts synchronously. It is the first
   lever below; another session was measuring `wt/perf5` on the cards while this note was written,
   so its numbers are left to that session.

Ranking (gain x confidence / effort; all lossless):

| # | lever | what moves | expected gain (est.) | confidence | effort |
|---|---|---|---|---|---|
| T1 | residency a card: arena lending + prefill stager on the integration branch, then the slack items of §2.1 | VRAM budget, link | prefill +10–18 %; c8 +3–6 %, c8 DFly2 +5–10 % | high (mechanism measured on the w4nl64 container: prefill +30 %, decode +2–7 %, notes/perf.md) | S (merge + measure) |
| T2 | prefill: one routed pass over a spanned step (`prefill-span: 2`, `prefill-span-pass: step`, 4096 batched) | bytes read a token, VRAM and link | prefill +5–9 % on prompts ≥ 8K | medium (config only; Flash-Next +17 %) | S |
| T3 | expert prefetch at two ranks: re-time the `tp == 1` gate with chained launches, then router-ahead (X4) with the copies on each card's DMA engine | link, hidden behind compute | c8 +1–4 %, c8 DFly2 +2–8 %, c1 ±0 | medium-low (gated on a replay) | S for the re-timing, M–L for X4 |
| T4 | two-piece pipelined prefill (the junction's reduction on a second lane beside the other piece's compute) | all-reduce at prefill: 10 % of a chunk (**M**), 7–8 points of that transport and wait | prefill +3–6 % (one lone prompt) | medium-low | M–L |
| T5 | host mirror of the expert pool on this box (170 GB) | d2h copies, free eviction | c8 +1–3 % by itself; enables X2 | medium | S–M |
| T6 | drafter at TP2: fused reduction twins for the draft layers; shard-aware head shortlist | 10 standalone all-reduces + 5 gathers a draft pass | c8 DFly2 +0.5 %; the shortlist is the general note's | high / small | S |
| T7 | sampler chain at TP2: measure, then fold the gather | 1 gather + host merge a step | unknown, bounded by a few tens of us a step | low | S to measure |

Closed by measurement or arithmetic (§3): the junction's wire and handshake, rank skew and thread
pinning, the host barrier, replicated-vs-sharded drafter, heads / indexer re-sharding,
pipeline parallelism over two cards, the router on one rank, Engram rows through the peer.

Lossy or format-touching items are the friend's and are listed apart (§4).

## 1. Measurements

### 1.1 Collectives in a decode pass (**M**, card 0, frontier attribution)

| phase | tape (dispatches) | pass wall, stamped | idle gaps | collectives | each | share of pass |
|---|---|---|---|---|---|---|
| no spec, 1 stream | 1729 | 17.99 ms | 3.84 ms (21.3 %) | 84 `ar_ihc_junction_2rank` | 12.6 us (p10 12.5, p90 13.0, p99 13.2, max 125) | **5.9 %** |
| no spec, 8 streams | 1729 | 30.3 / 31.7 ms | 4.73 ms (15.6 %) | 84 | 15.2 us (p90 15.7, p99 28, max 47) | **4.2 %** |
| DFly2, 1 stream, verify (6 rows) | 1734 | 21.17 ms | 3.94 ms (18.6 %) | 84 | 13.3 us (p90 14.4, p99 15.4) | **5.3 %** |
| DFly2, 8 streams, verify (~48 rows; 2 passes) | 1774 | 64.7 ms | 4.87 ms (7.5 %) | 84 | 27.5 us (p90 28.2, p99 52) | **3.6 %** |
| DFly2 draft pass, 1 sequence | 303 | 4.1–4.25 ms | 0.2–0.3 ms | 10 `ar_oneshot_2rank_exact` + 5 gathers | 7.7 us / ~3–4 us | **2.2 %** |
| DFly2 draft pass, 8 sequences | 298 | 6.7 ms | 0.44 ms | 10 + 5 | 40–44 us (p90 58) / ~5 us | **6.5–7.0 %** |

- The stamped pass is slower than the served one (18.0 ms against a 15.6 ms step at one stream:
  every dispatch carries a completion signal), so the shares are of the stamped wall; the
  absolute times of the collectives are what to carry over: 1.06 ms a token at one stream,
  1.28 ms a step at eight, 2.31 ms a verify of 48 rows.
- The junction replaces `[add] -> all_reduce -> rmsnorm -> ihc_post -> ihc_pre -> rmsnorm`. The
  plain row is 7.3 us at n = 3072, M = 1 (TP1, notes/wp16-2.md). So the reduction itself (push
  6 KiB, one flag each way, fold) costs about 5.3 us, at the PCIe round trip's size; 84 of them
  are 0.45 ms a token.
- The drafter's reductions are the standalone exact one-shot followed by a separate norm
  (`hyv4flash_dfly.h:589`, `:982`, `:995`), not a fused twin: 7.7 us at 6 rows, but 40–44 us at
  48 rows (a message of about 290 KiB, under the coded wire's 512 KiB floor), where the trunk's fused
  junction at the same height takes 27.5 us including its norms and both iHC rows.
- Where the rest of the pass goes (for the reader of the other notes): at one stream the shared /
  latent `iqr_m16` GEMMs 17 %, `gemm_i8_kpar` 12 %, routed gate_up 9 %, routed down 6 %,
  `quant_act_fp8` (458 launches, 421 of them unchained) 5 %, gaps 21 %. At eight streams routed
  gate_up + down are 41–43 %, in a 48-row verify 65 % (28.4 + 13.6 ms of 64.7).
- A draft pass is 4.1 ms (1 sequence) / 6.7 ms (8), its five head passes over the rank's
  vocabulary shard 1.5–1.65 ms of it (36 % / 24 %).

### 1.2 TP1 against TP2

The real container does not fit one card. On gpu-mini (4 layers, dummy weights, notes/wp16-2.md,
**M** there): decode 1.10 / 1.16 / 1.23 ms a token at TP1 against 0.88 / 0.96 / 0.94 at TP2
(p511 / p2052 / p3000). The sharded kernels halve, the collectives cost less than that saves,
the launch-bound rest is unchanged. Not re-run here.

### 1.3 Link to the host (**M**, `/stats` deltas per phase, a rank)

| phase | missed units a step (rank 0) | read in place, MB a step a rank | as link time at 27.6 GB/s | of the step | promotions a step | h2d / d2h copies, MB a step a rank | VRAM hit |
|---|---|---|---|---|---|---|---|
| no spec, 1 stream | 5.8 | 8.6 | 0.31 ms | 2 % | 1.0 | 1.4 / 0.9 | 0.992 |
| no spec, 8 streams | 47–74 | 71–110 | 2.6–4.0 ms | 9–13 % | 12–16 | 18–24 / 14–19 | 0.982–0.988 |
| DFly2, 1 stream | 23 | 34.5 | 1.25 ms | 5 % | 4.4 | 6.6 / 5.3 | 0.986 |
| DFly2, 8 streams | 154–252 | 232–377 | 8.4–13.7 ms | 15–22 % | 32–37 | 48–56 / 34–40 | 0.969–0.978 |
| prefill 18–20K (2048 chunks) | 1970–2230 | 2950–3340 | 107–121 ms | 16–18 % | 45 | 85 / 65 | 0.87–0.88 |
| prefill 6–7.5K | 1490–1690 | 2240–2540 | 81–92 ms | 12–15 % | 15–21 | 61–69 / 48–58 | 0.86–0.88 |

- Units: 20,992 a rank (41 layers x 512, each rank holds its 768-column slice of every expert),
  1,499,904 B each, 29.32 GiB a rank. Resident 18,017–18,458 (86–88 %) with 6.89 of the 7.02 GiB
  lendable KV on loan. So 2,530–2,980 units, **3.8–4.5 GB a card, are missing**.
- A prefill chunk routes nearly every expert of every layer (16,664 unit reads a step), so its
  miss count is the non-resident set itself: every chunk re-reads those 3–4.5 GB a rank over
  the link, synchronously, once for gate_up and again for down.
- The two ranks stream over separate root ports (0000:40:01.1 and 0000:c0:01.1, each
  16 GT/s x16; the cards and their on-card switches run 32 GT/s). The uplink is the cap
  ("link-capped" at startup); P2P between the cards crosses both uplinks, so there is no faster
  path card to card than host to card. A Gen5 root port would double every figure of this table;
  the EPYC 7402P has none.

### 1.4 Prefill

`RADIANCE_AQL_STAMP` times recorded passes only and prefill chunks are issued, not played
(`pf-st`: no line written), so prefill was taken under `profile-ops` (barriers added; at
2048-row heights the kernels are long, the barriers cost little). Throughput (**M**): 2993–3241
tok/s for one prompt of 5.5–20K tokens, 1611–1725 tok/s each for two 6.5K prompts at once
(3340 tok/s together), i.e. about 650 ms a 2048-token chunk.

Per-op device time, rank 0, cumulative over three prompts (5.5K + 19.7K + 17.8K = 42,930 tokens
in 22 prefill chunks and 3 one-token steps; 2,870–2,890 tok/s under the profile against
3,084–3,121 without, so the barriers cost 7 %); `profagg.py pf-prof.server.log`, **M**:

| op | calls | total | share | a call |
|---|---|---|---|---|
| routed `moe_gemm_q` gate_up | 1025 | 2742 ms | 19.0 % | 2.68 ms |
| routed `moe_gemm_q` down | 1025 | 2157 ms | 14.9 % | 2.10 ms |
| `mla_attn_paged_sink` (30 sliding layers, 32 heads a rank) | 660 | 1642 ms | 11.4 % | 2.49 ms |
| **`ar_rmsnorm_ihc_junction`** | 2100 | **1414 ms (rank 1: 1543 ms)** | **9.8 % (10.7 %)** | 0.67 ms (0.73 ms) |
| `mla_attn_sel_sink` (12 sparse layers, 16 heads a rank) | 264 | 1371 ms | 9.5 % | 5.19 ms |
| `gemm_nt_q` `attn_output` + `attn_gate` | 2100 | 953 ms | 6.6 % | 0.45 ms |
| `had_quant_act_i8g` | 11450 | 501 ms | 3.5 % | 44 us |
| `mla_expand_gate` + `mla_project_kv` | 1848 | 561 ms | 3.9 % | |
| `all_gather` (f32, row 256, one a routed layer, prefill steps only) | 902 | 87 ms | 0.6 % | 97 us |
| everything else | | 3025 ms | 20.9 % | |
| total | | 14,453 ms | | |

- **Collectives are about 10 % of a prefill chunk**: 84 junctions of 0.67–0.73 ms, 57–62 ms a
  2048-token chunk. A chunk's message is 12.6 MiB a reduction a direction (9.5 MiB on the coded
  wire), 1.06 GiB a chunk a direction. The junction's own arithmetic at 2048 rows is not timed
  apart; from `ihc_pre`'s 60 us at 1024 rows (notes/wp16-2.md) about 0.15–0.2 ms (estimate), which
  leaves 0.5 ms of transport and wait a junction: 18–25 GB/s on a 27.6 GB/s uplink, **7–8 % of
  prefill**. Under the profile every op is synchronised, which widens the wait for the peer
  (rank 1's row is 9 % longer than rank 0's); the unprofiled share is at most this.
- The routed GEMMs are a third of prefill, 233 ms a chunk, and §1.3's in-place reads
  (107–121 ms a chunk at the link's rate) are about half of that.
- The attention rows (21 %) are head-sharded and already half of their TP1 size; they are the
  kernel note's (kernels/perf.md L5 / L6).
- The 41 `all_gather` a chunk (MoeFP8's prefill path) cost 4 ms a chunk; not worth a lever.

### 1.5 VRAM receipt a card (**M**, startup report)

| item | no drafter | DFly2 depth 5 |
|---|---|---|
| card | 31.86 GiB | 31.86 GiB |
| driver, code objects | 341 MiB | 341 MiB |
| hardware queues | 231 MiB | 231 MiB |
| headroom + drift | 144 MiB | 144 MiB |
| activation arena (2048-token step, not lent) | 1.37 GiB | 1.39 GiB |
| static weights (the rank's shard of everything that is not a routed expert) | 3.50 GiB | 3.97 GiB |
| KV cache (8 x 65536, replicated on both cards; 7.11 GiB of it lendable) | 7.33 GiB | 7.33 GiB |
| elastic (experts + KV) | 26.29 GiB | 25.80 GiB |
| routed experts at start (before the KV loan) | 18.96 GiB | 18.47 GiB |
| settled free after start | 338.5 MiB against 131 MiB asked | the same |
| host pool | 39,000 MiB for both ranks, 11.63 GiB a rank used at start | the same |

1 GiB a card is 716 units, 3.4 points of residency.

## 2. Levers

### T1. Residency a card: arena lending, the prefill stager, the slack

**Mechanism.** Every miss of §1.3 is a unit that had no room. Three sources of room a card, none
of them a format change:
- *Arena lending* (core, exists): the arena is sized for a 2048-token step; a decode step of up
  to 256 tokens uses about 190–210 MiB of it (notes/perf.md, `work/perf5/p5.server.log`: "256
  tokens use 211 MiB, 1024 tokens use 740 MiB of 1.41 GiB"). The core lends the top to the expert
  slab when the plugin answers sizing declares (`RadArchProbe::shape_probe_ok`,
  `core/engine_bringup.cpp:1710`). The plugin on the integration branch does not (only
  `hyv4flash_vision.h` reads `shape_probe`); `hy4-adopt` 528f9a0c does, with the one core
  precondition it found (`attn_kv_sink`'s `N` as a capacity). About 1.15 GiB a card at decode
  steps: 820 units, residency 86–88 % → 90–92 %.
- *Prefill stager* (core, exists, armed only with arena levels; `engine_bringup.cpp:5055`): copies
  a routed layer's pooled experts onto the card up to three layers ahead, on the DMA engine,
  while the layers in front compute. It does not remove link bytes, it hides them: the link is
  idle 82–88 % of a prefill chunk (§1.3). On `hy4-perf5` with the w4nl64 container the startup
  line is "prefill staging: a 906.66 MiB ring ... a step of more than 1024 tokens copies the
  pooled experts of 41 routed layers ... 3 layers ahead". Two-rank particulars to check: the
  stager's copies give way to collectives (`dma copy lists ... give way to the collectives`), and
  a chunk has 84 of them; the ring is charged to the arena a card.
- *Slack.* (a) Each card settles 207 MiB above what it was asked to keep (338.5 against 131 MiB):
  145 units. (b) `--gpu-headroom-mib 128` on a headless card pair with a measured 16 MiB driver
  drift: 64 would give 45 units; try only with the settle line watched over a long run.
  (c) A deployment without speculation should not carry the drafter (it does not: 3.50 against
  3.97 GiB static); a deployment with it pays 0.47 GiB = 336 units = 1.6 points, which the
  load-aware depth does not price.
  (d) `kv_index_keys` and the latent cache are replicated and sized for 8 x 65536; that is the
  operator's `max-model-len` x `max-num-seqs` and is lent while unused, so nothing to do beyond
  stating in the GUIDE's TP2 section that a served long context costs residency (0.92 GiB a card
  per 65K-token session, 3.1 points).

**Where.** `arch/hyv4flash/hyv4flash.cpp` (probe + declare, from 528f9a0c),
`libref` / `libavx` / `libr4d` `attn_kv_sink` parameter role, `core/engine_bringup.cpp` (levels,
stager; no change expected), config.

**Exactness.** Placement only. Greedy text + logprobs equal with lending on and off was shown on
hy4-perf (notes/perf.md); repeat on u3.

**Gain (est.).** From §1.3 with the missing set cut by 28–33 % (820 + 145 of 2,530–2,980 units)
and assuming misses fall in proportion (they fall faster for the hot tail, slower for first-time
misses; kernels/experts-streaming.md's replay is the instrument): eight streams −0.8…−1.3 ms of
30 ms (+3–5 %), eight streams with DFly2 −2.7…−4.5 ms of 55–63 ms (+5–8 %), one stream ±1 %.
Prefill: the stager hides up to the 12–18 % of §1.3 and lending does not apply at a 2048-token
step, so +10–18 %; notes/perf.md measured +30 % for both together on the larger container
(1645 → 2110 tok/s), where far more units were pooled.

**Measurement.** `work/tp2/run.sh` with the merged build, `nospec.yaml` and `coupled.yaml`,
`probe.py`: the three tables of §1 again, plus the startup's arena and staging lines; texts
equal against `ns-st` at T 0 (add a greedy identity phase from `work/xs1/xs.py`).

**Packages.** T1.1 merge / rebase 528f9a0c (and its test) onto the integration branch; arch test
that TP2 passes every level. T1.2 run the matrix, record the residency a card and the link table.
T1.3 headroom trial (config only), long run with the settle line. T1.4 GUIDE: TP2 budget section
with the unit arithmetic of §1.5.

### T2. Prefill: one routed pass over a spanned step

**Mechanism.** At 2048 rows a chunk reads every expert of every layer once (§1.3: 16.7K unit
reads): 31.5 GB a rank from VRAM and the 3–4.5 GB of pooled units over the link, each for gate_up
and down. With `max-num-batched-tokens: 4096`, `prefill-span: 2`, `prefill-span-pass: step` a lone
prefill's step is 4096 rows with one routed pass (MoeFP8's `rows` is the step), so the same bytes
serve twice the tokens: the per-token weight traffic of the routed block halves, on the link and
in VRAM. 016a19f3 removed the 65535-row refusal that stopped this at 19.5K prompts. Two-rank
particulars: (a) the junction's message doubles to 25 MiB a reduction, still one coded one-shot;
(b) the arena for a 4096-token step is larger (1.37 GiB at 2048; roughly twice that is the
estimate, the startup receipt states it), i.e. about a GiB less residency a card unless T1's
lending is in (then decode steps get it back); (c) the stager's ring (906 MiB a card on hy4-perf5
at 2048) is charged to the same arena. So T2 comes after T1.

**Where.** Config; `docs/ARCHITECTURES.md` already describes the keys.

**Exactness.** A spanned step is a wider chunk: prefill rows may differ from 2048-token chunks at
tie level (GUIDE §4.5, E4 of kernels/perf.md); deterministic for a given cut; decode rows
untouched. Checkpoints are written where a span of 1 writes them.

**Gain (est.).** Routed GEMMs' VRAM reads: 31.5 GB a rank a chunk is about 60–65 ms at the
card's 500–550 GB/s, halved a token: +5 %. Link: with T1's stager hidden already; without it
another +6–8 %. Flash-Next measured +17 % (6544 → 7650 tok/s) with its span.

**Measurement.** `probe2.py` at 2048 / span 1 against 4096 / span 2 / pass step, prompts of 6K,
18K, 40K; stream bytes a token; `profile-ops` table for the routed rows; decode c8 unchanged
(arena).

**Packages.** T2.1 the run. T2.2 if it pays: the deploy yaml and the GUIDE's Hy4 TP2 recipe.

### T3. Expert prefetch at two ranks

**Mechanism.** The core's next-layer prefetch is off above one rank
(`core/engine_bringup.cpp:5138`: "a pass with the prefetch's stream beside it starts every ordered
dispatch about 11 us later", Flash-Next at `--tp 2`, before chained launches). Two things changed:
Hy4's pass is chained (1267 of 1729 dispatches chained, and chained packets "take no acquire from
the packet processor"), and at two ranks each card has a DMA engine and an uplink that is idle
87–98 % of a decode step (§1.3). The design exists as X3 / X4 of kernels/experts-streaming.md
(early publish on the device; router-ahead prediction on the compute stream, copies on the DMA
lane). What this note adds:
- the gate's premise should be re-timed first, cheaply: an environment switch that opens
  `xp_open` at `tp == 2`, one stream and eight, stamps for the ordered-start delay. If the 11 us
  are gone the TP1 machinery is available as it is;
- both ranks must prefetch the same units in the same step (they do if the predictor reads
  replicated inputs: the router and its input are replicated), or one rank's miss stalls the
  other at the next junction. Assert it (compare the two ranks' prefetch counters at exit, as the
  movers' are equal today);
- the copies must honour the link gate of the collectives (docs/TP4.md, "any new collective row
  must honour the gate"; notes/xs-1.md saw the junctions lose 0.4 ms beside paced copies).

**Exactness.** Placement only.

**Gain (est.).** notes/xs-0.md's replay bounds what latency levers reach (admitting every miss for
the next pass removes 18 % / 29 % of the misses); prediction is the lever for first-time misses.
With 10–20 % of the remaining misses (X4's estimate) on §1.3's link time: eight streams +1–2 %,
with DFly2 +2–4 %; if the predictor reaches a third, twice that. Nothing at one stream.

**Measurement.** X0.6's offline precision / recall study on u3 traces first (go: recall ≥ 30 % at
precision ≥ 40 %); then §1.3's table.

**Packages.** T3.1 the switch and the re-timing (S). T3.2 = WP-X4 with the two-rank asserts.

### T4. Two-piece pipelined prefill

**Mechanism.** `qwen35_fp8` at two ranks runs a lone prefill step of ≥ 1024 tokens in two pieces;
the work between two blocks (reduction, norm, add) of one piece goes out on a second lane beside
the other piece's compute, with the narrow `overlap` all-reduce launch (GUIDE §4.5; 41K prompt
9.00 → 8.41 s, −6.6 %). For Hy4 the between-blocks work is exactly the junction, already one row.
A two-piece step would issue `ar_rmsnorm_ihc_junction` for piece A on lane 2 while piece B runs
the block. Needs: a lane-2 form of the junction that does not hold its blocks while it waits on
the link (the fused form "slows the GEMM beside it by a third"), the plugin's step split in two
row ranges (attention over the pieces' own keys: piece B attends piece A's rows of the same step,
so the cache rows of A must be stored before B's attention, as two chunks would), and the
indexer's tail between the pieces.

**Exactness.** A piece computes what a narrower chunk computes (prefill rows, E4).

**Gain (est.).** Bounded by the junction's transport and wait, 7–8 % of prefill (§1.4). The
qwen35 pipeline recovered 2.7–6.6 % where its reductions were a comparable share, so +3–6 %
prefill for one lone prompt, nothing for decode, mixed steps or several prompts at once.

**Measurement.** Before any code: the junction twin at M = 1024 / 2048 against the plain row at
the same height (`rad-kbench --timing-only` over the n = 3072 junction cases of
`data/kernels_hy4.rkb`, or a `r4d_selftest` timing mode beside `--perfhc`; the split of 0.67 ms
into arithmetic and transport),
and an unprofiled bound by running a prefill with `RADIANCE_HY4_JUNCTION=0` (standalone
reduction, so its `link.ar_bytes` / `ar_calls` counters are filled; they read 0 with the fused
twin).

**Packages.** T4.0 the two timings (S). T4.1 only if transport is ≥ 5 % unprofiled: a design note
(lane-2 junction form, the step's split, the cache rows' order), then the row, then the plugin.

### T5. Host mirror of the pool

**Mechanism.** A promoted unit gives up its pool slot, so a demotion copies it back (d2h 0.9 /
14–19 / 34–40 MB a step a rank at 1 / 8 / 8 + DFly2 streams, **M**; 12.7 GiB d2h against 23.9 GiB
h2d over the no-speculation run). With the pool holding every unit (2 x 29.3 GiB; `host-pool-mib`
about 61,000) a demotion is an address change, eviction is free, and X2's admission variants
(admit on first miss, probation) stop costing link time. This box has the RAM: 170.9 GiB, 60.2 GiB
Engram, 66 GiB available with today's 39 GiB pool. Check before relying on it: `ttm.pages_limit`
is 94.28 GiB and the process already pins 101.94 GiB (the Engram tables are pinned outside TTM);
a mirror adds 22 GiB of pinned pool.

**Exactness.** Placement only.

**Gain (est.).** d2h rides the idle direction, so by itself small: the copies' share of the
junction and shared-arm slowdown notes/xs-1.md saw, +1–3 % at eight streams. Its value is as
X2's precondition.

**Packages.** T5.1 config run (pool 61,000) with §1.3's table and the planner's slot counts;
T5.2 the planner rule "mirror where the pool holds all" is X2's.

### T6. The drafter at two ranks

Measured: a draft pass is 4.1–6.7 ms; its 15 collectives are 0.09 ms (1 sequence) to 0.47 ms
(8 sequences). The drafter is sharded as the trunk (heads, MLP columns, vocabulary shard of the
head; fusion, correction, selectors replicated).
- *Fused reduction.* Replace `all_reduce exact → rmsnorm` behind the draft layer's attention and
  MLP by the fused exact twin (`ar_rmsnorm` family; the draft layers have no iHC, so the junction
  row does not fit, the plain fused norm does). 10 launches fewer a draft pass and the coded /
  staged write at 48 rows: 40–44 → about 15–20 us each, −0.25 ms a step at eight streams
  (0.4 %), −0.03 ms at one. Byte-equal to its chain (E6).
- *Replicated drafter*: no. It removes the 10 reductions (≤ 0.44 ms) and doubles the draft GEMMs'
  bytes a rank (`gemm_i8_kpar` is 1.0–1.3 ms of the pass, the head 1.5–1.65 ms), and costs
  0.2–0.7 GiB a card of residency.
- *Head shortlist* (kernels/dfly2.md §9.2, the general note's lever): the head is 24–36 % of the
  draft pass. Two-rank design point only: after the first full pass and the merge, a shortlisted
  id belongs to one rank's vocabulary shard; each rank scores the ids it owns (a gather of its own
  head rows) and the existing `all_gather` + `row_topk_merge` carries the pairs, so no head row
  crosses the link and the merge stays deterministic (ties to the lower id).
- *Depth agreement.* The adaptive depth is decided once on the host from rank 0's readback
  (`engine_draft.cpp:1147`, the copy is issued on `r0.ctx.stream()`) and reaches both ranks in the
  one shared batch, so the ranks cannot disagree. The 20 ms blocked wait holds the next step's
  issue for both cards; rank 1 needs no copy of its own. Nothing two-rank-specific to fix.

### T7. The sampler chain at two ranks

Each rank reduces its vocabulary shard (60,416 columns) to its candidates, an `all_gather` carries
them, rank 0's host merges (`core/sample/sampler.cpp:333`, `:1478`). The chain is issued behind
the pass and is not in the stamped tape, so it was not timed here. From the op list it is one
gather a step and the candidate reduction over half the vocabulary a rank; the head itself
(`mvm` / `lm_wmma`, 0.30–0.38 ms a pass) is already halved by the shard. Measure with
`profile-ops` at one stream before proposing anything; expected a few tens of microseconds.

## 3. Considered and closed

- **A faster junction wire.** 5.3 us above the plain row is one P2P round trip; the coded wire
  already serves ≥ 86 rows; `wht6` is lossy and the sandwich norm reads the sum. More blocks lower
  the rate (docs/TP4.md: handshake-bound). Closed.
- **Fewer reductions a layer.** The two a layer sit in front of two norms that need the full sum.
  Replicating `o_proj`'s input instead (push 8192 / 4096 values a row, not 3072) is a larger
  message and a doubled GEMM. Closed.
- **Thread pinning, rank barrier.** p90 − p50 of the junction is 0.2 us at one stream; the
  barrier is 2–4 us a step. Closed.
- **Re-sharding heads or the indexer.** Attention is 1.46 ms of an 18 ms pass at one stream and
  1.9 ms of 30 ms at eight (**M**: `kv_project_wmma` 8–10 us x 84, `mla_sink_rows` 8–12 us x 42,
  `merge_sink`, the cache row's front, `rope`) and sits near the launch floor at 32 / 16 heads a
  rank; the indexer (replicated, 12 layers, `qsa_work` + `idx_pool_gated` + `qsa_tail_store`) is
  0.17 ms at one stream, 0.28–0.29 ms at eight (**M**). Running the indexer on one
  rank and pushing the selection saves no time on the other (it waits for the push) and 384 MiB
  of key cache on one card only, which a symmetric placement cannot use. Closed.
- **Whole experts a rank for serving** (the calibration layout). The same bytes a card, an
  unbalanced GEMM load a step, and different partial rounding. Closed.
- **Pipeline parallelism (layers 0–20 on card 0, 21–41 on card 1).** No collectives and no
  replicated work, the same VRAM. One stream: the two half chains run one after the other, the
  same wall as today. Many streams as two interleaved groups: a card's cycle is
  `2 x (5.3 + 0.475 r + G(r))` ms for groups of r rows against `10.6 + 0.95 x 2r + G(2r)` at TP2
  (fit of kernels/experts-streaming.md, G the routed GEMMs); with the measured G(4) ≈ 7.9 and
  G(8) ≈ 12.4 ms that is 30.2 against 30.3 ms. The routed bytes do not share across groups.
  Closed.
- **Router on one rank, ids pushed.** 258 MiB of f32 router a card, 129 MiB a card if alternated
  by layer (0.4 points of residency), no time. Not worth a new collective.
- **Engram rows through the peer.** The table is pinned once in host memory and both cards read
  it over their own uplinks; Engram and embedding kernels are 0.13–0.23 ms of a decode pass.
  Nothing to share.

## 4. Not lossless or format-touching: the friend's decisions

- **Expert bytes.** With the KV loan out, T1's lending and no drafter a card can hold about
  27.2 GiB of experts (elastic 26.29 GiB less 0.22 GiB of unlendable cache state plus 1.15 GiB of
  lent arena; 26.1 GiB without lending) against u3's 29.32 GiB a rank. A container whose routed
  experts are **7 % smaller (11 % without lending)** is fully resident at TP2 on 2 x 32 GB with
  idle KV, and §1.3's link column goes to zero at decode. With DFly2 loaded (elastic 25.80 GiB)
  the figures are 9 % and 13 %. Every served 65K-token session takes 0.92 GiB of that back.
- **KV dtype.** The latent cache is replicated on both cards (6.75 GiB of the 7.33 GiB). An FP8 /
  int8 latent row (kernels/perf.md L9, 656 B a token) returns 43 % of whatever the served context
  holds to the experts on each card.
- **`tp-wire: wht6`.** Not usable: the norms and the calibration read the exact sum. No gain to
  be had at decode anyway (§1.1).
- **`RADIANCE_MOE_VERIFY_BUDGET`** (dartree; fewer experts for verify rows): would cut the
  48-row verify's 65 % routed share, changes the target model.

## 5. Order

T1.1 → T1.2 (re-measure §1 on the merged build; everything below is re-ranked on those tables)
→ T2.1 and T3.1 and T5.1 (config or switch only, one GPU session each) → T6 fused twin (S)
→ T3.2 / T4.1 only on their gates.
