# Audit E: memory, placement, caches (Hy4-flash TP2, u3)

Read-only audit of `src` @ c655f58c (branch claude/peaceful-wright-hhqnhi). Nothing was built or run.
Sources: core/place/{planner,heat,mover,stager,rtrace,xstats}, core/runtime/{xprefetch,issue,ctx,arena},
core/engine.cpp, core/engine_bringup.cpp, core/mem/{vram_budget,pools,kv,kvtier,prefix,hostarena},
core/device/{aql,dispatch,hip}.cpp, core/format/{radfile,table_copy}.cpp, core/sched/{scheduler,batch,geometry}.cpp,
arch/common/rad_block_moe_fp8.h, arch/hyv4flash/hyv4flash.cpp, libr4d/r4d_moe_iqr.hip. Data I re-read:
work/longctx/{nospec.server.log, res/nospec.json}, work/churn/{summ.py,*.out,*.json}, prior notes in the brief.

Units: u3 unit = 1,499,904 B (1.43 MiB) a rank; **1 MiB = 0.70 units, 100 MiB = 70 units = 0.33 points;
1 GiB = 716 units = 3.41 points** (20,992 units a rank).

## 0. Summary: top tickets

| # | ticket | gain (est.) | conf. | effort |
|---|---|---|---|---|
| E-H1 | Size the mover's pinned pool to the host-pool budget (mirror), not to the plan's pooled set. **T5's "no-op" was this bug**: 8 GiB a rank of already-pinned host pool sits unused; demotions copy d2h and starve (17.9K `starved of a pool slot` in the longctx run) | d2h −72 GiB/run, free demotions, precondition of X2; c8 +1–3 % | high (code) | M |
| E-K1 | Turn the host KV tier on (`prefix-cache-mib`), sized from the freed pinned memory; idle conversations hold VRAM KV for `prefix-cache-ttl` 3600 s today, i.e. never lent to experts | 640 units (3 pts) per idle 65K session; removes the 32.9 s TTFT outlier at N8 D60000 | high (code) / medium (gain) | S (config) |
| E-S1 | Stage steps of 129–1024 tokens too (ring above the 1024 level, as the span level already does) and fix the 1024/1025 off-by-one | tail chunks, resume turns, held-back chunks stop reading 3.7–9 GB a rank synchronously | high (code) / medium (gain) | M |
| E-S2 | Ring sizing / heavy-layer staging: **at N8 long context 0.9K units a chunk overflow the ring and are read in place** (measured counter), and u3 gate_up has no packed map so in-place reads repeat per 64-row entry | prefill +5–15 % at long context, many sessions | medium | S–M |
| E-C1 | Snapshot on decode (rolling, every 256 output tokens) so the next turn does not re-prefill the whole previous reply ("decode writes no snapshot", scheduler.cpp:1717) | agentic/code turns: re-prefill ≤ 255 + new tokens instead of reply + new | medium | M |
| E-C2 | FP-2 + relax the in-step tail rule to non-spanned prefills: today the in-step path needs `prefill-span > 1` (scheduler.cpp:1276), so FP-2 alone does not remove the tail step or the I/4 = 512-token gap for Hy4 | measured: 31 % of resumes re-prefill +227 tokens; one extra ≤15-token step a prompt | high (code) | M |
| E-R1 | Queue scratch charged from kernel metadata, not 5 MiB × 32 CU | ≤160 MiB = ≤112 units (0.5 pt); explains most of the 207 MiB settle slack | medium | S |
| E-B1 | Startup: Engram tables pinned + filled serially (4 × 10.3 s = 42 s of 66 s) at 1.4 GB/s vs 4.2 GB/s disk; parallelise / overlap with the container pass; or `ngram-placement: ram=/dev/shm/...` to skip the 60 GiB read on restart | 66 s → ~35–40 s (overlap), ~25–30 s (tmpfs reuse) | medium | S–M |
| E-M1 | `min-gain 3` default at TP2 (churn sweep: promotions −40…−70 %, tok/s equal) | fewer copies under junctions/shared arm; link h2d −55 % | high (measured) | S |
| E-L1 | 64-row (DFly2 c8: 48 rows) / 128-row arena level | 86–130 MiB at decode = 60–90 units | medium | M |

Prior-work status in §5 (X0/X1 built, X2/X3 open with replay bounds, T3/X4 NO-GO, T1 done, **T5 reopened**, FP-2/FP-4 open,
FP-8 answered from code for gate_up).

## 1. What the code does today

### 1.1 Expert units, in-place reads, weight tables

- **Unit = one expert of one layer on one rank, gate_up + down together** (`grp_expert_slot(l, e, 0|1)`,
  rad_block_moe_fp8.h:40-45, :1204-1247). Planner/mover/stager move the unit; a slab slot holds both projections.
- A non-resident unit lives in the pinned pool (`Site::DeviceZeroCopy`); `moe_gemm_q`/`moe_gemm_iqr` get pointer
  tables (`RAD_WTAB`, 2 tables by parity per projection) and read pooled experts in place over PCIe.
- **Table refresh is incremental**: `Ctx::table_refresh` (issue.cpp:373-460) uploads the whole table once, then per
  issue resolves only `t.pend` entries and patches changed pointers through `wtab_patch` (≤ kWtabPatch per dispatch,
  gathered per op when `wtab_gather_`). In a recorded pass each table operand is a `kTapeTable` cut (328 a decode
  pass); an unchanged table costs a host check only.
- **Read multiplicity of an in-place expert** (libr4d/r4d_moe_iqr.hip): decode band = one block per (16-row tile, run),
  so an expert whose run straddles a tile boundary is read twice; prefill = run map at 64 rows an entry
  (kMapTiles = 4, :679) or packed map at 128 rows. **The packed map requires `fm.bits == 4 && fm.L == 16`
  (:928), so u3 gate_up (3-bit trellis) always takes the 64-row run map**: at 2048 rows × top 16 / 512 = 64 rows an
  expert on average, ~1.6–2 entries an expert (the file's own figure: 1.3 entries at Flash-Next top 10), and each
  entry re-reads the expert's weight tile. For VRAM experts that is decode work; **for in-place (unstaged) experts it
  multiplies link bytes** (unverified whether fine-grained host reads hit L2 across blocks; likely not). The
  `stream_bytes` counter (heat.cpp:258-260) prices one read per routed pooled unit, so it undercounts this.

### 1.2 Heat engine / mover / tiering_dispatch

- `Engine::tiering_dispatch` (engine.cpp:1000-1077), behind every non-draft pass: `begin_dispatch` (retire), credit the
  arrived histograms (`heat.observe`, credit cap 512 a histogram, heat.h:79), `heat.step` → swaps (**not behind a staged
  pass**, :1041), `pace_collect`, `issue`, `balance` (fill lent slots hottest first; turnover held filled for 4 unstaged
  steps after a staged one, :1066).
- Before each non-draft pass: `publish_landed` (X1a) and `pace_begin` (engine.cpp:1192-1195).
- HeatParams defaults (heat.h:40-94): decay 0.95, hysteresis 1.25, min_gain 1.5, 64 moves a step and class,
  credit_cap 512; env overrides `RADIANCE_HEAT_*` (engine_bringup.cpp:3912-3929).
- Pacing (X1b): held promotion copies go out from `Mover::pace_mark` (mover.cpp:1988-2001), riding the **next dispatch
  after the layer's routed down GEMM** (= the shared arm), default `RADIANCE_XS_PACE=-1` without a drafter, 0 with one
  (engine_bringup.cpp:3892-3899).
- Copy lists are `RAD_COPY_YIELD` (mover.cpp:1939, :1965; stager mover.cpp:658): the SDMA engine polls the link gate
  every 1 MiB (aql.cpp:3065, :3728-3746). **The gate is held only by collectives of staged passes**
  (`hold = staging_ && p.collective`, issue.cpp:1154). In played decode passes no one holds it, so the yield never
  fires and copies overlap the 84 junctions (xs-1: +0.4 ms junctions, +1.3 ms shared arm at c8).
- Promotion into a budgeted slot gives up the unit's pool slot; into a lent slot it keeps it (mover.cpp:1517-1521).
  Demotion of a budgeted unit = d2h copy into a free pool slot (or a donor's), else `starved_pool`.
- **Pool sizing**: `n_pool[c]` counts only the plan's HostPinned units (mover.cpp:135-152) and the class pool reserves
  exactly `n_pool × stride` (mover.cpp:203-205); but `Pools::host_` allocates the whole `host_pool_total_mib`
  (pools.cpp:175, :227). Log: `host_pool 19.73 GiB budget` a rank, `7878 pinned pool slots` (11.0 GiB) + 708 MiB
  embedding = 11.69 GiB used. **~8 GiB a rank (16 GiB total) of pinned host memory is allocated, locked and idle.**

### 1.3 Prefill stager (core/place/stager.{h,cpp}, Mover::stage_units)

- Ring = `vb.stage` = 2 × ceil(f × layer_max), f = share of experts the budget leaves off (vram_budget.cpp:262-287); here
  **523 MiB** (log; 906 MiB was the w4nl64 container). At the arena's top; lent to the slab in smaller levels.
- Armed (`begin_pass`, stager.cpp:121-138) only for a trunk pass with `n_tok >= min_rows` and a ring at this level.
  `min_rows = top_level + 1 = 1025` (engine_bringup.cpp:5086-5104), ring only at level 0 (full plan) or the span
  level (engine.cpp:2495-2503). **Steps ≤ 1024 tokens are never staged.**
- Per layer: all stageable (pooled, not in flight) units, **each unit copied once (gate_up + down together), published
  for the layer's whole op span, read twice from VRAM** (gate_up GEMM, down GEMM), retired at the next layer's first op
  (stager.cpp:274-321). Copies coalesced by pool-slot order (mover.cpp:620-650), one DMA list a layer, yielding.
- Pace: layer k released when the compute stream reaches layer k−3 (kLead = 3, stager.h:185); the next step's first 3
  layers are staged at the pass end when the next step continues a prompt at level 0 (stager.cpp:343-366).
- A layer that is not next goes only whole and only if `need <= ring/2` (stager.cpp:227); the next layer takes what the
  head can reach (partial); **what does not fit is read in place (`stage_overflow`)** (mover.cpp:636-642).
- Mixed steps: `n_tok` counts decode rows too, so chunk + decode rows > 1024 is staged. `--busy-prefill-chunk auto`
  floor = `max(Q, stage_rows/Q*Q)` = **1024** (scheduler.cpp:1449, :1484) against `min_rows` 1025 (see E-S1).
- The exit report prints `stats_.prefetched` as "staged" (mover.cpp:2151-2153): the longctx log says `0 staged`
  while `/stats` shows 2.0 TB staged on rank 0 (§2.3).

### 1.4 Activation arena and levels

- Liveness-packed (greedy interval colouring, largest first; rad_bufplan.cpp:1-150): persist/derived at the bottom
  (fixed region 12.2 MiB), transients packed above. Not one buffer per value.
- Levels (engine_bringup.cpp:1723-1807): `lo = max(256, pow2 >= max_seqs × (1+n_spec))`, ×4 up to the chunk:
  **256 and 1024** here (log: 185.82 MiB / 705.88 MiB of 1.37–1.41 GiB). A DFly2 verify (≤ 48 rows) uses the 256
  level; the 256 floor is for tile padding (comment :1712-1716).
- `arena_step` (engine.cpp:2470-2512): level from `max(n_tok, n_seq*(1+n_spec))`; if the level needs lent memory it
  reclaims synchronously (`Mover::reclaim`, drains both streams). `arena_tick`: the floor holds 2 s after the last
  large step (engine.cpp:2520-2533).
- Lends 1.69 GiB at ≤256 tokens (1210 slots), 1.19 GiB at ≤1024. Device arena 1.88 GiB + 41.13 MiB scratch.

### 1.5 VRAM receipt (vram_budget.cpp:38-426, startup log of work/longctx/nospec.server.log)

| line | MiB | what it is (code) | lever? |
|---|---|---|---|
| already held | 341.5 | measured after declare (engine_bringup.cpp:1849-1884; vram_budget.cpp:142); driver context + code objects loaded so far. libr4d's fat binary is 110.6 MB of gfx1201 code in 121 modules, loaded lazily per module on first use (aql.cpp:472-511) | E-R5 (small, unverified) |
| driver drift | 16 | constant kDriverDrift (vram_budget.cpp:25) | no |
| arena | 1.41 GiB | buffer plan, lent at decode | E-L1, E-L2 |
| step staging | 3.11 | block tables 3.01 MiB (8 seqs × 16384 blocks × 4 B × groups) | no |
| hardware queues | 230.7 | 2 queues × 35.36 MiB measured (create 2nd − 1st, :1855-1860) **+ 160 MiB queue scratch = 5 MiB × 32 CU constant** (dispatch.cpp:943-946) | E-R1, E-R3 |
| headroom | 128 | `--gpu-headroom-mib` | E-R4 (T1 b) |
| static weights | 3.50 GiB | rank shard of non-routed weights incl. f32 router 258 MiB (41 × 512 × 3072 × 4 B) | router: closed (levers-tp2 §3), dtype: friend's |
| KV | 7.33 GiB | 8 × 65536; 7.11 GiB lendable; 211 MiB sliding window + 96 KiB state unlendable | E-R6 |
| prefill staging | 523 | ring, lent at decode | E-S2 |
| settled free | 338.5 vs 131 asked | measured at the end of bring-up, **before any dispatch**, so the 160 MiB queue scratch the runtime has not yet taken is still free | E-R1/E-R2 |

### 1.6 KV, loan, reclaim

- Groups (log): `kv_latent_sparse` full, 12 layers, **block 4 tokens**, 54 KiB a block (= 4 × 12 × 1152 B), 131,089
  blocks = 6.75 GiB; `kv_index_keys` full, 12 layers, 3 KiB a block (64 B a token-layer), 384 MiB;
  `kv_latent_sliding` window, 30 layers, block 16, 540 KiB a block, 400 blocks (50 a sequence slot) = 211 MiB;
  `kv_index_tail` 12 KiB × 8 slots; `kv_engram_tokens` conv 32 B × 8. **Paged cost 14,592 B a token (0.89 GiB per
  65,536)**. Chunk quantum 16 (attention block 16).
- Loan: tokens from the top of every FULL group, above the highest held block (kv.h:465-495, min-heap free list
  :781). `loan_tick` every 100 ms, grows only in steps ≥ one chunk, shrinks at once (engine.cpp:2417-2455).
  Window groups never lent.
- Reclaim on admission: `KVManager::grow_for` (kv.cpp:927-957) → callback → `Mover::reclaim(kLendKV)` on every rank
  (engine_bringup.cpp:5346-5360). Reclaim (mover.cpp:1724-1846): flush + **sync the mover stream, retire, sync the
  compute stream** (drains the in-flight pass), then a host walk over all 20,992 units: lent-slot units revert to their
  host copy (address change, no copy); only *donors* (lent units whose pool slot went to a budgeted victim) are moved
  VRAM→VRAM or d2h, then a second mover sync. Exit report: 626 donations, 269 moved by a recall. So **a new long
  prompt does not wait for demotion copies**, it waits for both streams to drain once per recall, and recalls
  `shortfall + buffer` (buffer = max(one chunk, decaying shortfall peak)) per `grow_for`, i.e. roughly every other
  chunk of a long prompt.

### 1.7 Prefix cache, checkpoints, tiers

- Paged groups: content-hashed 4-token entries, shared by refcount on a hit (no copy; prefix.h:1-40).
- Hy4 has linear state (index tail, Engram CONV) → `window_ckpt = !has_linear` is **false** (geometry.cpp:151): the
  window rides in the checkpoint slot, snapshots follow the linear-state rules. Slots: 64 auto, 17.41 MiB a rank
  (window 33 blocks × 540 KiB + 96 KiB state), pinned host, 2.18 GiB for both ranks; interval 2048; geometric
  backoff, 8 a sequence.
- **Tail rule** (scheduler.cpp:1395-1403): a tail snapshot at the last 16-boundary only if it lies ≥ max(Q, I/4) =
  **512 tokens past an interval boundary**; the prompt is split there and the remainder (< 16 tokens) **runs as its
  own step**. In-step snapshot (`ckpt_cut`) needs the plugin's `ckpt_snap_ok` (hyv4flash.cpp:1238 sets only
  `shape_probe_ok`) **and** `geo_.span > 1 && prefill_span > 1` (scheduler.cpp:1276).
- Save (engine.cpp:2000-2045 → kv.cpp:1735-1850): on the **compute stream**, one `rad_memcpy_2d_async` per state group
  and per window page = ~35 blit kernels a rank, each d2h with a system-scope release fence (aql.cpp:2974-2995). Restore
  the same in reverse. ~17.4 MiB a rank each way (≈0.65 ms of link).
- Decode writes no snapshot (scheduler.cpp:1717 comment): each turn re-prefills the previous reply.
- Host tier (kvtier.h:1-60): copy on idle (write-back right after publish), free on demand (loan or allocation), restore
  by scatter; replicated groups stored once (TierExec::SlotLayout). **Off** here (no `--prefix-cache-mib`); TTL 3600 s
  (rad_core.h:1041). KV streaming (docs/KV-STREAMING.md) exists for QSA/GLM sparse attention; hyv4flash does not set
  `kv_stream_ok`.

### 1.8 Host memory and startup

- Pinned allocations: private anonymous mmap, 2 MiB-aligned, `MADV_HUGEPAGE`, `MADV_DONTFORK`, prefault with **4
  threads touching every 4 KiB**, then `hsa_amd_memory_lock_to_pool` in **1 GiB pieces, sequentially** (aql.cpp:4589-4720).
  NUMA: `prefer_numa_node(current card)`; this host reports NUMA −1 (single socket 7402P, NPS1): n/a.
- THP `madvise` (defrag madvise, shmem never); iommu passthrough. Process pins 101.95 GiB vs `ttm.pages_limit`
  94.28 GiB (userptr pins are not TTM).
- Container: one pread pass, 16 workers rearranging, `posix_fadvise(DONTNEED)` behind the read (radfile.cpp:144-147),
  66.63 GiB in 17.1 s (4.2 GB/s).
- Engram: `copy_table` (engine_bringup.cpp:380-445) → `rad_dev_alloc(HOST_MAPPED)` (prefault + lock) then
  `table_copy_fill` (16 readers, 64 MiB pieces, DONTNEED, sample check; table_copy.cpp:36-62), **one table after the
  other**: 4 × 14.75 GiB in 10.3–10.5 s each (1.43 GB/s) + 4 × 315 MiB scale planes 0.2 s. Log order puts all four
  before the container pass. `loaded in 66.0 s` ≈ 42 s Engram + 17 s container + ~7 s rest.
- Engram bytes a token: 4 layers × 16 heads × 96 B + row scales ≈ 6.3 KB a token a rank, 96-byte random rows
  (13 MB a 2048-chunk a rank: negligible link time).

## 2. What the runs measured (summaries)

### 2.1 Churn (work/churn, heat knobs, sampled T 0.9; summ.py)

| run | c1 | c4 | c8 | c8b | promotions / 1K tok (c8) | link MiB/tok (c8) | hit (c8) |
|---|---|---|---|---|---|---|---|
| base nospec (2 runs) | 73.8–75.5 | 205.8–207.2 | 301.4–302.0 | 300.6–301.6 | 1135–1183 | 3.2–3.4 | 0.992 |
| min-gain 3 | 74.0 | 201.9 | 297.6 | 306.1 | 407 | 1.2 | 0.992 |
| hysteresis 2 + min-gain 3 | 73.9 | 202.2 | 297.3 | 306.8 | 469 | 1.3 | 0.991 |
| min-gain 6 | 70.6 | 192.0 | 293.4 | 302.7 | 285 | 0.8 | 0.987 |
| decay 0.97 | 75.1 | 205.5 | 297.1 | 300.8 | 1189 | 3.4 | 0.992 |
| base DFly2 coupled (2 runs) | 152.1 | 296–308 | 374–384 | 388–397 | 1067–1125 | 3.0–3.2 | 0.984 |
| DFly2 + min-gain 3 | 152.2 | 302.2 | 386.9 | 391.9 | 572 | 1.6 | 0.985 |

Reading: min-gain 3 cuts promotions 40–70 % (link h2d −55…−65 %) at equal tok/s within run noise (±1–2 %); 6 loses at c1
(−5 %). Decay 0.97 does nothing. readmit_16 (re-promotion within 16 steps) 0.11–0.30 at c8.

### 2.2 Checkpoint slots and resume (work/churn rs*.out, resume.py: 4 or 16 conversations × 3 turns, ~4K tokens,
48-token greedy replies, 3 × 8K fillers after each turn)

| run | slots | turn-2 / turn-3 hit | evict full / retention | hits cut (no snapshot), tokens cut |
|---|---|---|---|---|
| rs16 (4 conv) | 16 | 0.000 / 0.000 | 55 / 0 | 20, 33,676 |
| rs64 (4 conv) | 64 | 0.935 / 0.927 | 0 / 0 | 12, 1,252 |
| rs256 (4 conv) | 256 | 0.935 / 0.927 | 0 / 0 | 12, 1,260 |
| rsB64 (16 conv) | 64 | 0.942 / 0.929 | 23 / 18 | 24, 4,128 |
| rsB256 (16 conv) | 256 | 0.942 / 0.929 | 0 / 0 | 24, 4,120 |

- 16 slots collapse to no hits (the fillers evict everything); auto 64 = 256 for these shapes. Nothing to do on slots.
- **Re-prefill per resume**: 213–227 tokens when the previous prompt ended ≥ 512 past a 2048 boundary (= the 48-token
  reply + ~170 new tokens + the < 16-token tail), but **246–679 (mean ≈ 445) on the 10 of 32 rsB64 resumes whose
  previous prompt ended within 512 tokens past 4096** (cached exactly 4096): the I/4 tail rule of §1.7. Mean
  re-prefill 290 vs ≈ 218 possible: +33 %. The reply itself (48 here) is always re-prefilled (no decode snapshot).

### 2.3 Long context (work/longctx/nospec, sweep.py; from res/nospec.json, rank 0 `/stats` deltas of the prefill phase)

| N × depth | prompt tokens | chunk passes (stage_layers/41) | staged a chunk a rank | overflow (in place) a chunk | decode tok/s (B) | hit (B) | VRAM units pre→post |
|---|---|---|---|---|---|---|---|
| 1 × 2K | 1.8K | 1 | 4.0 GiB | 0 | 69.7 | 0.959 | 18147→19181 |
| 1 × 60K | 56K | 27 | 5.0 GiB | 0 | 69.4 | 0.972 | 18725→18172 |
| 4 × 16K | 68K | 34 | 6.0 GiB | 0.1 | 190.4 | 0.971 | 18017→17411 |
| 4 × 32K | 123K | 60 | 7.2 GiB | 49 units | 176.6 | 0.958 | 17417→16214 |
| 4 × 60K | 232K | 113 | 8.9 GiB | 409 units | 148.3 | 0.929 | 16229→14403 |
| 8 × 2K | 15K | 7 | 9.5 GiB | 819 units | 213.7 | — | — |
| 8 × 32K | 252K | 124 | 9.4 GiB | 950 units | 249.9 | — | — |
| 8 × 60K | 446K | 218 | 9.3 GiB | **882 units (1.23 GiB)** | 184.5 | — | — |

- A chunk stages 4–9.5 GiB a rank (145–345 ms of link at 27.6 GB/s; chunks are 0.5–0.75 s): at many long sessions the
  link is ~50–65 % busy on staging alone, and 0.4–1.2 GiB a chunk is read in place synchronously (×1.6–2 for gate_up,
  §1.1).
- N4 D60K decode: residency −1,826 units, hit 0.929, −22 % tok/s vs 16K: the KV the live sessions hold (0.89 GiB per
  64K). kv used/total mid-run 5.43/6.98 GiB: 1.55 GiB not lent beyond what is held (E-K3).
- N8 D60K: one B request got cache 0 → TTFT 32.9 s (re-prefill of 48.8K), one cut to 49,152: prefix entries evicted by
  the pool (no host tier).
- Exit report: 92 % readmits, 6.2 TB h2d (stager included), 72 GiB d2h, 17,897 `starved of a pool slot`, per-layer
  residents 255–447.

## 3. Tickets

Fields: location · change · why/numbers · lossless · confidence / effort / deps · verify.

### 3.1 Placement and streaming

**E-H1 Mirror pool: size the mover's pool to the host-pool budget (reopens T5).** area core/place
- `mover.cpp:135-152` (pool slots only for plan-HostPinned units), `:203-205` (reserve `n_pool × stride`),
  `mover.cpp:1517-1521` (budgeted promotion drops the pool slot), `pools.cpp:175,227` (host pool allocated whole).
- Give each class `min(all its units, host_pool_room / stride)` pool slots; at load also write the resident units'
  bytes into their pool slot (the container pass already has them in hand); a promotion keeps its pool slot whenever
  it has one; demotion of a mirrored unit = address change (republish + quarantine), like a lent-slot drop.
  Reclaim's donor path disappears for mirrored units.
- Today: 19.73 GiB a rank pinned, 11.69 GiB used → 8 GiB a rank = 5.6K extra slots **at zero extra RAM**; with
  `host-pool-mib 61000` every unit mirrored (+11 GiB a rank pinned; 65 GiB available). Removes d2h copies (72 GiB a
  long run, 14–19 MB a step at c8), the 17.9K `starved_pool` refusals, donor moves in reclaims; makes X2's
  admit-on-first-miss free to evict. notes/tp2t1.md T5 "no-op (pool slots unchanged: 7848)" is explained by this
  sizing: T5 was never tested.
- lossless (placement only). high confidence on the mechanism; gain c8 +1–3 % alone (levers-tp2 T5 estimate), more with
  X2. Effort M (load path writes, planner field, mover invariants, place_test).
- Verify: exit report `d2h`, `starved of a pool slot`; `/stats experts` demotions vs d2h_bytes; `ctest place_*`;
  greedy identity (xs.py / cmp.py).

**E-H2 Alternatively shrink the host pool to what is used.** core/mem
- `pools.cpp:175` sizes `host_` from `host_pool_total_mib`; if E-H1 is not done, size it to the plan's pool need +
  staging. Frees 16 GiB pinned: less pin time at startup (~0.2–0.3 s/GiB, unverified), room for the KV host tier
  (E-K1) or container page cache (E-B3). lossless. high / S.

**E-M1 Default `min_gain` 3 at TP2 with ≥ 85 % residency.** core/place
- `heat.h:61` (default 1.5; its comment says traffic is U-shaped — measured on Flash-Next TP1 at 40 % residency).
- Churn sweep (§2.1): promotions −64 % (c8 no spec), −49 % (c8 DFly2), tok/s within noise (c8b +1.8 %, c4 −2.5 %).
  Fewer copies = less of xs-1's junction/shared-arm slowdown and fewer quarantines. Make it residency-dependent
  (e.g. 1.5 below 75 % resident, 3 above) so TP1/Flash-Next keeps its tuning.
- lossless. high (measured, 2 runs each), S. Verify: churn.py A/B 3 runs, greedy identity.

**E-M2 Pace rides after the post-FFN junction, by bytes (xs-1 "Next", refined).** core/place + runtime
- `mover.cpp:1988-2001` rides the dispatch after the routed down GEMM = the shared arm (`gemm_i8_kpar<2>` +0.8 ms,
  `gated_quant` +0.5 ms, junctions +0.4 ms in xs-1).
- Record the pace mark behind the layer's second junction (the next layer's attention stretch: q/kv projections,
  attention kernels, no PCIe) and cap a share at `stretch_us × 27 GB/s` (at c8 ~1–2 units a layer; 46 promotions a
  step over 41 layers ≈ 1.1 a layer). Leftovers unpaced at the next dispatch as today.
- lossless. medium; +1–2 % c8 no spec on top of X1b's +2–3 % (xs-1 measured the 1.8 ms moving, not vanishing). M.
  Verify: `RADIANCE_AQL_STAMP=24`, stlay.py/kdiff.py per-kernel deltas.

**E-M3 Hold the link gate around decode junctions when paced copies are in flight.** core/runtime, device
- `issue.cpp:1154` holds only in staged passes; `aql.cpp:3732` yields only if `ever_held`. In a pass with paced
  copies, record hold/release around `ar_rmsnorm_ihc_junction` (the kernel-side release `RAD_LINK_ENDED` path exists,
  `issue.cpp:1253`), so a yielding list stops within 1 MiB (36 us) of a junction start.
- Cost: 2 barrier packets a junction unless the kernel sets/clears the gate itself (84 a pass); only for passes with
  held copies. Gain ≤ 0.4 ms a step at c8 (xs-1). lossless. medium-low; do after E-M2 shows what is left. M.

**E-M4 One patch kernel per pass for residency changes.** core/runtime
- `issue.cpp:415-447`: changed table entries are patched per op (`wtab_patch`, per table or gathered per op). With
  `publish_landed` before the pass every change is known at pass start: write all of a step's patches in one launch
  at the pass's head. At c8 promotions touch ~30–40 routed layers a step → up to ~40–80 small launches (est.
  0.1–0.25 ms). Needs: kTapeTable cuts still resolve in-flight transfer waits (stager publishes mid-pass, so keep the
  per-op path for staged passes). Count first: `RADIANCE_DEBUG_ISSUE` prints walked entries / patches.
- lossless. low-medium (count unverified), S–M.

**E-M5 Copy a staged unit into its slab slot VRAM→VRAM instead of h2d again.** core/place
- After a staged pass no swaps run (engine.cpp:1041); the next unstaged dispatch promotes ~45 units (prefill rows of
  §1.3 of levers-tp2: 45 promotions a step at prefill) with h2d copies of units that crossed the link minutes earlier
  … but the ring is recycled by then. Variant that works: when `heat.step` behind a staged pass *would* promote a unit
  that is in the ring right now, issue its promotion as a d2d copy from the ring place before the place is lapped (or
  stage it straight into a free slab slot). 64 MB a step a rank of h2d → d2d at 500 GB/s.
- lossless. medium-low (needs swap decisions behind staged passes, which were switched off on purpose to keep the
  transfer stream for the stager), M. Verify: h2d bytes per prefill chunk.

**E-M6 Per-layer floor or a ranking term for the layer's pooled share.** core/place (X5 per-layer item, now with data)
- Exit report: residents a layer 255–447 (sd 69); layers 28 and 24-ish hold 255–303. Heavy pooled layers are exactly
  the ones that overflow the ring at prefill (E-S2). Replay (bin/rad-heat-replay) a per-layer minimum (e.g. 300) to see
  the decode-hit cost; if ≤ 0.1 point, it buys staging fit. lossless. low, S (replay only).

**E-M7 Exit-report fix.** core/place `mover.cpp:2151-2153` prints `stats_.prefetched` as "staged"; print
`staged_units`, `staged_bytes`, `stage_overflow`, `stage_copies` (the counters exist, `/stats` has them). Also the
per-layer table shows layer 0 (dense) as 1 resident unit, so `min 1` is meaningless (unverified source of the 1).
lossless, high, S.

### 3.2 Prefill-chunk expert handling

**E-S1 Stage 129–1024-token steps; fix the 1024/1025 off-by-one.** core/engine + sched
- `engine_bringup.cpp:5086-5104` (`min_rows = top_level + 1` = 1025), `engine.cpp:2495-2503` (ring only at level 0 or
  span level), `scheduler.cpp:1449,1484` (busy floor `stage_rows/Q*Q` = 1024).
- (a) Off-by-one: a held-back chunk of exactly 1024 tokens alone in its step (reserve for a waiting request that does
  not enter, or the interval split into two 1024 halves) selects the 1024 level → unstaged. Round the floor up
  (`(stage_rows+Q-1)/Q*Q` = 1040) or let the stager arm at `n_tok >= top_level` when the step is a prefill at level 0.
- (b) Give the 1024 level a ring right above its buffers (`arena_low_off` mechanism of the span level: `lv.end +
  kArenaGuard`, 706 MiB + 1 MiB + 523 MiB < 1.41 GiB + 523 MiB) and arm for prefill steps ≥ ~128 tokens: at 128
  rows × 16 / 512 an expert is routed with p = 98 %, so such a step reads the whole pooled set (3.7–4 GiB a rank at
  86–88 % residency; 9 GiB with long contexts live) synchronously in place today: 135–330 ms of link (more with the
  gate_up re-read factor). Cases: every prompt's last chunk < 1025 (16,506 = 8 × 2048 + 122), every resume turn
  (§2.2: ~220–450 tokens), the tail split's remainder step if it stays (E-C2).
- Cost: while the floor is at the 1024 level (2 s after such a step) the slab keeps 523 MiB less of the arena loan
  (366 units), and reclaiming it is the existing arena recall.
- lossless (stager copies byte-identical, published per op). high on mechanism; gain: such a step becomes
  max(link, compute) instead of link + compute → est. −50…−150 ms TTFT per resume turn / prompt tail. M.
  Verify: `/stats experts stage_layers`, TTFT of resume.py; texts identical with RADIANCE stager off/on.

**E-S2 Ring sizing and heavy layers.** core/mem + place
- `vram_budget.cpp:262-287` sizes the ring as 2 × ceil(f × layer_max) with one global f; per-layer pooled shares are
  uneven (exit report residents 255–447 → pooled 65–257 units = 93–368 MiB vs ring/2 = 261 MiB), and with the KV loan
  recalled they grow. `stager.cpp:227` keeps a layer above ring/2 out of the look-ahead; `take()` then gives it only
  the room past the previous layer's place; the rest is `stage_overflow` (read in place, synchronous).
- Measured (§2.3): overflow 0 at one session, 49 → 409 units a chunk at N4 32K → 60K, **819–950 units (1.2–1.4 GiB)
  a chunk at N8**. In place, u3 gate_up re-reads per 64-row entry (§1.1).
- Changes, in order: (1) let a heavy layer go ahead partially (stage `ring/2` of it at its look-ahead slot, the rest
  when it is next) instead of not at all; (2) size the ring from the planner's per-layer maximum at the KV floor
  (`max_l pooled_l(at kv_one)`) instead of f × layer_max, bounded by e.g. 1 GiB: the ring is arena top, lent at decode,
  so decode residency is unchanged; at prefill it costs residency 1:1 but those units are staged anyway;
  (3) order a heavy layer's staged units by routed count of the last chunk (hot first) so what overflows is the coldest.
- lossless. medium (the in-place cost of overflow at 1.2 GiB a chunk ≈ 45–90 ms of a 0.5–0.75 s chunk → est. +5–15 %
  prefill at N8 long context; zero at one session). S–M. Verify: `/stats experts stage_overflow` per chunk
  (work/longctx sweep), prefill tok/s.

**E-S3 Packed map (or a 128-row entry) for the 3-bit trellis gate_up (FP-8, answered from code).** kernel owner
- `r4d_moe_iqr.hip:928`: `g.packed = fm.bits == 4 && fm.L == 16 && …`; Hy4's routed gate_up is u3 → never packed:
  64-row entries, ~1.6–2 decodes a routed expert at 2048 rows (Hy4 top 16 = 64 rows an expert on average; the
  file's Flash-Next figure is 1.3 entries at top 10). Every extra entry is one more decode of the weight tile (VRAM)
  and, for an in-place unit, one more PCIe read. Building the 8-tile form for bits 3 is the kernel owner's
  (VGPR budget 237 at 4-bit, :684-686).
- lossless (sums in the same order? the file states r4d_selftest compares maps byte for byte). medium, M.
  Verify: `r4d_selftest --perfmoeiqr` at Hy4 shapes (K 1536/3072, top 16), map tiles pinned 4 vs 8.

**E-S4 Stage the drafter's routed layer?** — not a lever: stager.h:76-82 (only the last chunk's few rows draft).
Checked.

### 3.3 VRAM receipt

**E-R1 Queue scratch from kernel metadata.** core/device + bringup
- `dispatch.cpp:943-946` charges `n_cu × 5 MiB` = 160 MiB whenever the device is a dGPU; comment: the runtime hands a
  queue scratch at the first dispatch of a kernel with a private segment (`step_advance`'s local array, spills).
  `aql.h:45` / `aql_code.cpp:302` already parse `private_segment_fixed_size` per kernel.
- Charge `max private_size over the kernels the program can launch × waves the runtime sizes for` (or 0 when no
  selected kernel has a private segment). The settle line is measured before any dispatch (engine_bringup.cpp:5428),
  so today it cannot tell; the 207 MiB slack is ≈ 160 + 16 (drift) + ~31.
- Up to 160 MiB = 112 units (0.53 pt). lossless. medium (whether a Hy4 kernel spills is unverified; DFly2's
  `step_advance` may), S. Verify: rocm-smi / sysfs `mem_info_vram_used` after c1, c8, prefill, DFly2 warm-up vs the
  settle line; list kernels with private_size > 0 from the code objects.

**E-R2 Soak the remaining settle slack with expert flex in single-model mode.** core/engine
- `xflex_tick` (engine.cpp:564-630) grows the slab in 1 GiB steps into memory no model holds, only with a fleet broker
  and `flex.experts.max-mib` (engine_bringup.cpp:5298-5330). After warm-up, commit `free − headroom − 32 MiB` (VMem
  granularity) once, as a revocable `kLendFleet` loan. ~30–200 MiB = 20–140 units depending on E-R1.
- lossless. low-medium (broker presence in a one-model fleet unverified), S–M.

**E-R3 CWSR off on this dedicated box (operator).** 2 compute queues × 35.36 MiB measured (`queue_bytes`, mostly the
context-save area with `cwsr_enable 1` in the hardware report). `amdgpu.cwsr_enable=0` should drop this to ~1–2 MiB a
queue: ~68 MiB = 48 units. Loses mid-wave preemption (debuggers, priority queues). lossless numerically. low-medium,
S (config + reboot). Verify: the "hardware queues" line.

**E-R4 `--gpu-headroom-mib 64` (T1 b, still open).** 64 MiB = 45 units. Watch the settle line and first-launch
allowance (`engine_bringup.cpp:5441-5470` warns below `Config{}.gpu_headroom_mib`). S, config.

**E-R5 Code objects: measure, then trim.** libr4d carries 110.6 MB of gfx1201 code in 121 modules; a module loads
whole at its first kernel (aql.cpp:493-511). How much of the 341 MiB "already held" is code is unverified; if tens of
MiB, a deployment build listing only the forms Hy4 uses (e.g. R4D_MOEIQR_FORMS 6 forms, r4d_moe_iqr.hip:767) saves them.
low, S to measure (sum `im.size` of loaded modules at the settle line).

**E-R6 Lend the sliding-window pages of unused sequence slots.** core/mem
- `kv_latent_sliding` 400 blocks = 8 slots × 50 = 211 MiB, never lent (kv.h:472-474: WINDOW groups are per sequence).
  At 1–4 live sequences 105–185 MiB idle = 73–129 units (0.35–0.6 pt). Lend slot k's pages from the top when fewer
  than k slots are live; reclaim at admission (same recall path).
- lossless. medium, M (window ring addressing per slot; checkpoint restore writes these pages).

**E-L1 64/128-row arena level.** core/engine + build
- `engine_bringup.cpp:1735` floors levels at 256 rows for tile padding. At ~0.69 MiB a row, a 64-row level ≈ 56 MiB vs
  186 MiB (−130 MiB = 91 units), a 128-row level ≈ 100 MiB (−86 MiB = 60 units). Decode at 1–8 clients without spec
  (≤ 8 rows) and DFly2 c8 (48 rows) would run there.
- Safe form: in `plan_level` (rad_bufplan.cpp:150-215) round every transient's rows up to the largest row tile of any
  kernel that writes it (or a global 128), keeping the "multiple of every tile" property the comment needs.
- lossless. medium (tile audit), M. Verify: arch test that every level binds; greedy identity.

**E-L2 Drop the routed `edn` buffer (G-3 g, moe_down + gather).** `rad_block_moe_fp8.h:331` `edn [rows*top_k,
n_embd] bf16` = 2048 × 16 × 3072 × 2 B = **192 MiB** at the full plan (24 MiB at the 256 level). Fusing the gather
into moe_down removes it: −192 MiB of prefill-time arena (134 units resident during prefill, or a 192 MiB larger ring
for E-S2) and −24 MiB at decode. Kernel owner's G-3 g; byte-identity depends on the fused sum order. medium, M–L.

**E-L3 Arena packing report.** `plan_buffers` computes `peak_live` (rad_bufplan.cpp:99-104) but the startup line prints
only the packed size; print `peak_live` per level. If packed > peak_live + 5 %, try best-of-N orderings (deterministic).
low, S.

**Checked, nothing to gain**: step staging 3.11 MiB (block tables for 8 × 65536 at 4-token blocks); arena fixed region
12.2 MiB; driver drift 16 MiB; the drafter's static weights when speculation is off (the u3 container carries none:
3.50 vs 3.97 GiB, levers-tp2 T1 c, closed); router sharding (levers-tp2 §3, closed); kv_index_keys / latent replicated
on both ranks (indexer re-sharding closed, and it is lent while unused); per-kernel scratch 41 MiB (unverified split).

### 3.4 KV loan, reclaim, prefix cache, checkpoints

**E-K1 Host KV tier on (`prefix-cache-mib`).** config (+ docs)
- Without it a finished conversation's blocks stay held in VRAM until the cache evicts them or `prefix-cache-ttl` 3600 s
  expires (rad_core.h:1041, kvtier.h:1-30), and `spare_tokens` (kv.cpp:959-972) counts only blocks above the highest
  held one: idle sessions are never lent to the experts. With the tier, entries are written back at once and released
  on the loan's demand (engine.cpp:2427-2433 comment).
- Per idle 65K session 0.89 GiB = 640 units (3 points); restore 0.89 GiB h2d ≈ 33 ms a rank vs a re-prefill of
  16–20 s. Fixes N8 D60K's 32.9 s TTFT (§2.3). Size it from E-H2's 16 GiB of idle pinned pool (≈ 18 full sessions)
  or make the tier allocate from the host pool's unused tail.
- lossless. high on mechanism; gain depends on the agentic workload (multi-turn). S (config), then measure with
  work/longctx multi.sh and resume.py. Watch: write-back d2h rides the idle direction; the tier's gather/scatter
  launches are one per (group, layer) per batch (kvtier.h:38-43).

**E-K2 Recall the whole prompt's KV at admission, once.** core/mem
- `grow_for` (kv.cpp:927-957) recalls `shortfall + buffer` (buffer ≈ one chunk) per call, so a 60K prompt recalls ~15
  times; each recall drains both streams on both ranks and walks 20,992 units (mover.cpp:1732-1745). At admission the
  prompt length is known: recall `ceil(prompt/chunk) × chunk` once (as a lower bound on the loan), so later chunks
  find blocks in hand. Also skip the compute-stream sync when the stream is known idle (prefill chunks are issued
  after readback). lossless. medium, S. Gain: ~15 drains a long prompt (each ≤ one step of queued work; est. 5–50 ms
  total), and no lent-slot churn between chunks. Verify: count `reclaim` calls a prompt (add a counter).

**E-K3 KV loan fragmentation: measure, then compact.** core/mem
- The loan reaches only above the highest held block (kv.h:487-491). N4 D60K mid-run: 5.43 GiB used of 6.98 GiB not
  lent → 1.55 GiB (1,100 units) neither used nor lent; the buffer is one chunk (29 MiB) plus a decaying shortfall peak.
  Either cached prefix blocks above the live ones pin the mark or the peak term had not decayed (unverified). Add
  `/stats kv held_top` and `buffer_blocks`; if fragmentation, relocate the highest cached-only entries downward (d2d
  copy + prefix-cache remap) when the gap exceeds e.g. 256 MiB. lossless. low-medium, S to measure / M to fix.

**E-C1 Rolling decode snapshot.** core/sched + engine
- scheduler.cpp:1717: "decode writes no snapshot", so every turn re-prefills the previous reply. For code/agentic
  replies (hundreds to thousands of tokens) that dominates the resume prefill. When a sequence's decode position
  crosses a multiple of 256 (on the quantum), save a snapshot into the sequence's running-turn slot (the window rows
  behind the position are held during decode; linear state is the live slot at that position — a decode step advances
  1–6 tokens, so the save goes behind the step that reaches the boundary, with spec only if the accepted count lands
  exactly; otherwise the next boundary).
- Cost: one ~17.4 MiB d2h a rank per 256 tokens a sequence (35 blits; E-C3 makes it one) ≈ 0.7 ms of link in the idle
  direction; at c8 one snapshot every ~32 steps. Gain: next-turn re-prefill ≤ 255 + new tokens; with E-S1 a ~300-token
  resume step is cheap anyway, so the gain is largest for long replies (≥ 1K).
- lossless (snapshot bytes are the state's). medium (the chat template must re-render the reply token-identically:
  verify with resume.py and `reasoning_effort none`, and with thinking on), M.

**E-C2 FP-2 plus non-spanned in-step tail snapshot.** plugin + core/sched
- FP-2 (plugin `ckpt_snap_ok`, hyv4flash.cpp:1238 sets only `shape_probe_ok`) is necessary but not sufficient: the
  in-step path is taken only with `geo_.span > 1 && span_now_ && cfg_.prefill_span > 1` (scheduler.cpp:1276-1277), and
  `ckpt_cut` only for `n_seq == 1` prefill steps (batch.cpp:1133). Hy4 runs without span (T2 not recommended), so
  today every prompt ≥ 2560 tokens takes the split: tail snapshot at the last 16-boundary only if ≥ 512 past an
  interval (scheduler.cpp:1403), plus a separate step for the < 16-token remainder.
- Change: with `ckpt_inside_`, use `tail_checkpoint(n_target, in_step=true)` for any prefill alone in its step.
  Removes (a) the extra ≤ 15-token step a prompt (≈ a full trunk pass + in-place reads of ~40 % of pooled experts:
  est. 60–80 ms, FP-1's open number) and (b) the I/4 gap: §2.2 measured 31 % of resumes re-prefilling +227 tokens.
- lossless (snapshot bytes identical to the tail step's, flashnext-port.md row 9). high (code), M (plugin: ring cut,
  Engram CONV cut, per FP-2). Verify: `RADIANCE_DEBUG_CKPT_HASH=1` equal with/without; resume.py cached counts.

**E-C3 One copy-list launch per checkpoint save/restore.** core/mem + device
- kv.cpp:1785-1840: per window page one pitched blit with n_layers rows (33 pages) + 2 state groups ≈ 35 blit kernels a
  rank, each with a system-scope fence (aql.cpp:2974-2995), on the compute stream. Use one ranges copy
  (`rad_memcpy_ranges_async`, a kernel reading a range list, aql.cpp:2997-) or one SDMA list on the mover lane with
  an event the next step waits on. Saves ~34 launches + fences a checkpoint (est. 0.2–0.5 ms a rank on the compute
  stream; one a 2048-token chunk and one a resume). lossless. medium, S.

**E-C4 Hash/lookup cost of 4-token prefix entries (checked, small).** A 60K prompt is 15K entries hashed and looked up
at admission (prefix.h BlockHash 128-bit chained): ~1–3 ms host, once a request. Not a lever unless profiles show it.

**E-C5 KV streaming for the 12 sparse layers (research).** docs/KV-STREAMING.md: host-authoritative KV with a VRAM
ring for indexer-selected attention; hyv4flash does not set `kv_stream_ok`. Hy4's `mla_attn_sel_sink` reads a
selection, so `kv_latent_sparse` (13,824 B a token, 6.75 GiB at 8 × 65K) could stream; the indexer keys (768 B a token)
stay resident (dense-read). At N4 D60K that returns ~2.5 GiB = 1,800 units (8 points) to the experts, at the price of
per-step resolve copies of missed selected blocks over the same link the experts use. lossless. low (cost unmodelled),
L. Only after E-K1.

### 3.5 Host memory and startup

**E-B1 Pin and fill the Engram tables in parallel, overlapped with the container pass.** core/bringup
- engine_bringup.cpp:380-445 + table_copy.cpp:36-62: 4 tables one after the other, each prefault (4 threads,
  aql.cpp:4589-4601) + 15 sequential 1 GiB locks (aql.cpp:4650-4680) + a 16-reader fill: 10.3 s each = 1.43 GB/s
  against 4.2 GB/s disk, ≈ 42 s of the 66 s start. Run the four tables concurrently (or lock piece k+1 while filling
  piece k) and start them beside the container pass: disk total 127 GiB at 4.2 GB/s ≈ 30 s floor. First add phase
  timers (prefault / lock / fill) to the log line.
- est. 66 s → 35–40 s. lossless. medium (pin cost per GiB unverified), S–M.

**E-B2 `ngram-placement: ram=/dev/shm/radiance-ngram` (config).** engine_bringup.cpp:388-425: a tmpfs file is filled
once and on later starts only checked by samples and pinned in place (`rad_dev_map_file`, aql.cpp:4728-4770), skipping
the 60 GiB read: restarts save ~15–25 s more. Needs SVM (present here). Caveat: shmem huge pages follow
`shmem_enabled` (`never` here) → 4 KiB pages for the table; mount the tmpfs with `huge=always` or set
`shmem_enabled=advise`. Engram rows are 96-byte random reads, so page size matters little for bandwidth (GPU TLB only;
iommu pt). S (operator). Verify: startup time, Engram kernel time in a stamped pass.

**E-B3 Keep the container in page cache when RAM allows.** radfile.cpp:144-147 drops every range behind the read.
170.9 GiB RAM − 102 GiB pinned = 69 GiB; with E-H2 (−16 GiB pinned) the 66.6 GiB container fits; a restart would read
at memory speed (16 workers) instead of 17 s from disk. Make the DONTNEED conditional on MemAvailable after pinning
(> container + margin). Conflicts with E-K1/E-H1 use of the same RAM: pick one. low-medium, S.

**E-B4 Startup phase timestamps.** The log has no timestamps; `loaded in 66.0 s` is the only total. Add elapsed
seconds to the bring-up lines (pool pin, Engram per table, container pass, arena levels, settle). S, diagnostics.

**Checked**: NUMA (single socket, NUMA −1 on both cards; n/a); THP on pinned pools (MADV_HUGEPAGE on private memory,
defrag madvise; fine); IOMMU (passthrough, so no IOTLB cost on in-place reads); `ttm.pages_limit` 94 GiB < 102 GiB
pinned is fine (userptr pins); container read is disk-bound (4.2 GB/s, rearrangement keeps up: 359 MiB held at most).

## 4. Format / lossy (friend's decisions)

- f32 router (258 MiB a card) in bf16: −129 MiB = 90 units; changes routing.
- FP8/int8 latent KV (`kv-cache-dtype i8`): −43 % of the 13.8 KB a token; at long context +1,000–1,800 units.
- Smaller experts (below u3's 1.43 MiB a unit) or a container whose gate_up takes the packed map (4-bit L16).
- Sliding-window KV in FP8 (211 MiB fixed).

## 5. Prior items: status

| item | status | evidence |
|---|---|---|
| X0 (trace, replay, knobs, per-layer counters) | done | notes/xs-0.md; rtrace.h, `RADIANCE_HEAT_*` engine_bringup.cpp:3912-3929 |
| X1 a (publish before the pass) | done, nearly inert | xs-1: 0–0.05 early publishes a step at c4–c8 |
| X1 b (pacing) | done, default −1 without drafter, 0 with | xs-1: c8 +2–3 %; time moved to shared arm/junctions → E-M2, E-M3 |
| X2 (admission, 2Q probation) | open | replay bound: oracle −18 % / −29 % misses; moves 128 −2…−5 %; presence credit no gain. Needs E-H1 for free evictions. Live churn: min-gain 3 = E-M1 |
| X3 (early publish on device) | open | replay bound lat 0: −5…−8 % misses |
| X4 / T3 (router-ahead prefetch at TP2) | CLOSED, NO-GO | notes/tp2t3.md: machinery −4…−6 %, copies −2…−5 % |
| X5 link-aware window, shared arm beside routed GEMMs | open (scheduler / plugin) | not in this area |
| X6 capacity options | friend's | §4 |
| T1 arena lending + stager | done | tp2t1: prefill +15–25 %, c8 +5 % / +12 %; log shows levels and the 523 MiB ring |
| T1 slack (a) settle 207 MiB | open → E-R1/E-R2 | explained by the 160 MiB queue-scratch constant |
| T1 slack (b) headroom 64 | open → E-R4 | config |
| T1 slack (c) drafter weights without spec | CLOSED | 3.50 vs 3.97 GiB |
| T1 slack (d) replicated latent/index keys | CLOSED (docs) | lent while unused |
| T2 span pass | not recommended | tp2t1 |
| T5 host mirror | **REOPENED** → E-H1 | the run changed `host-pool-mib` but the mover sizes pool slots from the plan (mover.cpp:135-205) |
| FP-2 in-step tail snapshot | open → E-C2 (needs the non-span relaxation too) | hyv4flash.cpp:1238, scheduler.cpp:1276 |
| FP-4 persistent prefix store | open | restart value grows with E-B1/E-B2 shortening the start |
| FP-8 packed map at Hy4 shapes | answered for gate_up: never (bits 3) → E-S3 | r4d_moe_iqr.hip:928; down (`moe_down_dec`) not checked |
| expert prefetch at TP2, heat damping, pin-top/profile, host-CPU MoE | CLOSED (brief) | — |
