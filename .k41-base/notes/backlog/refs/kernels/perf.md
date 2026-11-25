# WP16 performance: design and work packages

Design note. Nothing here is implemented. Measured numbers are marked **M** (this note's runs, one
R9700, 210 W, 2026-11-21) or cite the package note they come from; everything else is an
**estimate** and says how it was derived. Repo state: `claude/peaceful-wright-hhqnhi` @ 1774a48a.

Raw data of this note: `work/wp16/` — `kbench-report.md` (full `rad-kbench --timing-only` report of
`data/kernels_hy4.rkb`, 448 cases), `speed-decode.txt` (its M ≤ 16 rows), `stamps-tp1.csv` +
`stamp.py` + `agg.py` (every dispatch of 15 recorded decode passes of gpu-mini TP1,
`RADIANCE_AQL_STAMP=8`, summarised with `scripts/stampgaps.py`). The kbench run used `src/build`
(03:30 build, predates WP8's last rows; the Hy4 attention / iHC / project kernels are the ones of
HEAD), the stamps `wt/ready/build` (08:52, HEAD).

## 0. Result in one page

1. **Decode at Hy4 is launch-bound, not attention-bound.** gpu-mini TP1 (4 layers): a decode pass
   is 169 dispatches, 130 of them ordered behind the one before, 0 chained; 35 % of the pass
   nothing runs (gap before an ordered start: p50 4.36 us) (**M**). That is about 31 ordered
   launches a layer; at 42 layers about 1300 a token, 5.7 ms of gaps plus about 2 ms of kernels
   that run under 2 us. The two attention kernels together are 13.5 us a layer (**M**).
2. **`mla_project_kv` costs more than the attention it wraps**, at decode (absorb + expand 56 us
   against 14 + 4.5 us at 64 heads, **M**) and far more at prefill: at M = 1024 and 64 heads
   absorb 1.65 ms, expand 6.14 ms, the attention between them 1.4–1.9 ms (**M**). It is a scalar
   kernel (one thread a column, 2-byte loads, a 192-step fma chain).
3. **Every prefill row with a cached prefix runs absorbed** (`hyv4flash.cpp:879`: non-absorbed
   only when every prefill sequence of the step starts at position 0), i.e. every chunk after the
   first, every prefix-cache hit and every context past 2048. So finding 2 is the prefill cost of
   the real workload: about 360–390 ms of attention-side kernels per 1024 tokens at TP1, 93–132 ms
   a rank at TP4 (sum of **M** rows, §2.2).
4. **`ihc_pre` at n = 3072 is 18.7 us a row** (**M**, 5.7 us at n = 512): 85 launches a token,
   1.6 ms. Scalar strided loads; the bandwidth floor is about 1 us.
5. **Exactness defect to fix with the attention work:** `mla_sink` drops to the unsplit form above
   16 (window) / 32 (window 0, sel) rows *of the launch*. With several streams a row's bytes then
   depend on how many other rows the step has, and a verify of ≥ 3 streams x 6 rows no longer
   equals the single-row decode. A `rad-tune` axis on `splits` would do the same. §3.4 replaces the
   launch-dependent rule by a law on an absolute token grid.
6. wp6b's expectation "a multi-row tile cuts M = 8 roughly by the row count" does not hold: the
   WMMA work is per (row, head) and is not shared, only the key staging is. Multi-row tiles pay at
   16 and 8 heads a rank (TP2/TP4), nothing at 64 heads (HG 4 already fills the workgroup).

Ranking by expected gain. The real container (w4nl64 about 147 GB) does not fit the cards of a
box (2 x 32 GB): it runs at TP2 with most experts on the host tier, or at TP4 on four cards. The
decode column is the TP4 rank (16 / 8 heads); at TP2 (32 / 16 heads) L3 is −0.75 ms a token and
−119 ms per 1024 prefill tokens, the other rows are the same. Levers that are plugin or engine
settings rather than kernels (replicated KV groups, expert placement, the audit fixes) are in §3.9
and in the package order of §4.

| # | lever | decode, ms a token (est., 42 layers) | prefill with prefix, ms per 1024 tokens (est.) | bits |
|---|---|---|---|---|
| L1 | chained launches (`chain_pays` + chained forms of the Hy4 rows) | −2.0 (bound −5.7; Flint: −13..−24 % of a step) | ~0 | none |
| L2 | `ihc_pre` vectorised + fused junction (`[all_reduce] + rmsnorm + ihc_post + ihc_pre + rmsnorm`) | −1.1 and −0.8..−1.3 | −5 | `ihc_pre` reduction order changes once; fusion byte-equal to its chain |
| L3 | `mla_project_kv` as WMMA, latent-major key block, `mb` axis | −0.76 (TP4), −0.2 (TP1) | −56 (TP4 rank), −254 (TP1) | new sum order, row-independent |
| L4 | small fusions: expand + `gate_mul`, cache-row front, sigmoid router | −0.35..−0.55 | ~0 | byte-equal to chain |
| L5 | `mla_sink` bit-neutral speed: column-major value tile, staging waves, row tiles, axes | −0.17 (M = 1), −0.3 (verify, TP4) | −6..−20 | none (byte-compare with today's kernel) |
| L6 | `mla_sink` fast prefill law: single-bf16 P, consistent denominator | 0 (not used for decode rows) | −11..−34 | prefill rows only |
| L7 | `attn_kv_sink` `bq` axis (64-row tile at window 512) | 0 | −1.7 (one-shot from 0 only) | moves prefill bits within the bound |
| L8 | libavx attention tiling | host only | host only | libavx only |
| L9 | FP8 / int8 latent cache (656 B a token) | ~0 | up to −40 % of selection prefill at 8 heads | lossy, opt-in, friend's decision |

Baseline the estimates sit on (experts resident; a token that pulls experts from the host tier is
slower by the placement's miss cost, §3.9): a TP4 rank's decode token ≈ 17 ms (launch-bound part 7.6 ms
measured-and-scaled, weight traffic about 64 MiB a layer a rank ≈ 5 ms at 550 GB/s, project_kv,
iHC, attention, two reductions a layer). The GEMM share is not measured (no real-width container);
WP16.0 measures it before anything else is ranked finally.

## 1. Constraints every lever keeps

- **E1 verify == decode in bytes.** A decode-class row's output bytes are a function of its own
  operands, the cache content it sees and the op's parameters. Not of M, the launch's other rows,
  the head group, a tuned axis, scratch availability or the rank count's workgroup layout.
  Today's exception (unsplit above 16 / 32 rows) is removed by §3.4.
- **E2 paged (window 0) == sel over an in-order table in bytes** (6b's acceptance; a sequence
  crosses 2048 without a seam).
- **E3 cache rows.** Both paths write the same cache row; no lever touches `kv_store` bytes except
  L9, which is a different cache dtype and a different prefix-cache key.
- **E4 prefill rows** may differ from decode rows in the last place (PLAN §1.3) and may move within
  the kbench bound when a tuned axis moves; they must be deterministic for a given launch.
- **E5 TP.** Replicated ops (iHC, norms, junction) run the same kernel on the same bytes on every
  rank; the reduction stays exact bf16. A fused junction must give every rank identical `h`.
- **E6 fusions** declare the chain they replace (`RadFuseFn replaces`) and pass kbench's fusion
  oracle byte for byte. A fusion never introduces a new rounding or sum order of its own.
- **E7 rad-tune axes on decode-class rows are bit-neutral**, proven by an emulator test that walks
  the axis cross product and compares bytes. Axes that move bits are allowed on prefill-only rows.
- **Lossless** against today's build means: L1, L4, L5 change no byte; L2 (`ihc_pre` order), L3,
  §3.4 change last-place bits once, inside the bounds the rows already state against libref
  (1 bf16 ulp on `x`, rel_l2 ≤ 1e-3 on attention). Land those three before the friend's final KL
  numbers or tell him the build moved; none needs recalibration.
- gfx1201: wave32, gfx12 WMMA (bf16 and f16 16x16x16), LDS ≤ 64 KiB a workgroup, ≤ 256 VGPR a
  wave with **no spills in a hot loop** (state VGPR / LDS / occupancy in each commit as 6a–6d
  did), ≤ 1024 threads a workgroup. 210 W cap; clock about 2.8 GHz warm.

## 2. Measurements

### 2.1 Decode pass, gpu-mini TP1 (M, `stamps-tp1.csv`, 8 passes of 169 records)

Wall 1.66 ms stamped (1.34 ms a token unstamped at context 511, 1.41 at 2052). Sum of dispatch
times 1.71 ms, gaps 0.58 ms. Per layer of the pass (sliding MoE layer, records 46–83): 38
dispatches, 31 ordered.

| kernel | n a pass | us each | note |
|---|---|---|---|
| bf16 GEMMs (`gemm_bf16_nt_m64`) | 40 | 25.2 | gpu-mini's trunk is bf16 at n = 512; the real trunk is quantised at n = 3072 |
| `kv_absorb_rows` | 4 | 38.1 | overlaps the gate GEMM (unordered), 16 us (32 heads) / 59 us (64 heads) |
| `kv_expand_rows` | 4 | 27.3 | 20.7 / 34.2 us |
| `rmsnorm` | 27 | 1.9 | 120 us of gaps a pass in front of them |
| `rope` | 8 | 5.9 | |
| `mla_sink_rows` + `merge_sink` | 4 + 4 | 9.1 + 4.5 | context 511: window 0 at 511 keys |
| `ihc_pre` (n = 512) | 9 | 4.5 | 18.7 us at n = 3072 (kbench) |
| `ihc_post` | 8 | 1.7 | |
| `idx_pool_gated`, `qsa_tail_store` | 2 + 2 | 12.9, 14.2 | sparse layers |
| 16 other kinds | 57 | 1.1–5.7 | each waits 4.4 us |

The step's shape between two GEMMs is a run of 1–2 us kernels each behind a 4.4 us gap:
`o_proj → rmsnorm → ihc_post → ihc_pre → rmsnorm → GEMM`, `… → binary(add) → rmsnorm → ihc_post →
ihc_pre → rmsnorm → GEMM`, `sink_rows → merge → expand → gate_mul → GEMM`.

### 2.2 Kernels at Hy4 geometry (M, rad-kbench, cold weights, us a launch incl. the 2.65 us floor)

| kernel | heads | M = 1 | M = 16 | M = 128 | M = 1024 |
|---|---|---|---|---|---|
| `mla_attn_paged_sink` window 512 | 64 | 13.7 (8.9 short ctx) | 71 (51) | 220 | 1400 / 1874 |
| | 16 | 13.7 | | | 533 / 856 |
| `mla_attn_paged_sink` window 0, 2048 keys | 32 | 45.9 (9.4 short) | 99 | 498 | 1229 / 3542 |
| | 8 | 45.9 | | | 724 / 2247 |
| `mla_attn_sel_sink` 2048 keys | 32 | 46.1 | 127 | 576 | 5311 |
| | 16 / 8 | 46.0 / 46.1 | | | 3953 / 3926 |
| `mla_project_kv` absorb | 64 / 32 / 16 / 8 | 23.4 / 16.1 / 15.1 / 15.0 | 38 / 28 | 198 / 98 | 1654 / 805 / 370 / 195 |
| `mla_project_kv` expand | 64 / 32 / 16 / 8 | 33.0 / 19.8 / 12.8 / 20.9 | 83 / 44 | 685 / 329 | 6139 / 2942 / 1393 / 678 |
| `attn_kv_sink_h256` window 512 (N = M) | 64 | 6.7 | 6.8 | 26.8 | 574 |
| `attn_kv_sink_h256` window 0 (N = M) | 32 | 5.8 | 5.9 | 20.1 | 361 |
| `mla_kv_split` | 64 | 2.5 | 21.2 | 12.0 | 212 |
| `ihc_pre` rows 8, n = 3072 | | 18.7 | | | 92 |
| `ihc_post` n = 3072 | | 2.8 | | | 37.8 |
| `gate_mul` n = 16384 | | 2.5 | 3.0 | 9.6 | 168 |
| `engram_gate` n = 3072 | | 17.4 | | | 176 |

Two values in a cell are two fixture cases of the same parameters (different context / table
draw). The window-0 M = 1 time at 2048 keys (45.9) is 27–28 us with warm index tables (wp6b).
Readings:
- project_kv at 64 heads and M = 1 sits at the card's bandwidth (12.6 / 16.8 MiB a launch at
  about 600 GB/s); at 32, 16 and 8 heads it is bound by the 192-step scalar chain, not by bytes
  (3.2 MiB should take 5 us, takes 15).
- sel at M = 1024 does not get cheaper with fewer heads: 1024 rows each stage their own 2048
  tokens (2.4 GiB a launch at any head count), so it runs at memory bandwidth. kbench draws the
  tables as random permutations; rows of one real sequence select overlapping groups, which the
  card's cache serves. **Measure with tables from a real `qsa_select` before spending on it.**
- Small anomalies to look at when the files are open anyway: `mla_kv_split` 64 heads M = 16 (21 us,
  slower than M = 128), `engram_norm` M = 128 (20.9 us, slower than M = 1024), `mla_project_kv`
  expand at 8 heads (20.9 us, slower than 16 heads).

### 2.3 Derived budgets (estimates)

Decode, a token, 30 sliding + 12 sparse layers, TP4 rank (16 / 8 heads):

| part | us a layer | ms a token | basis |
|---|---|---|---|
| gaps in front of ordered launches | 31 x 4.4 = 136 (+ 2 reductions) | 5.7–6.1 | 2.1, scaled by layers |
| kernels under 6 us on the path | about 50 | 2.1 | 2.1 |
| `ihc_pre` x 2 | 37 | 1.6 | 2.2 |
| project_kv absorb + expand | 28 (sliding), 36 (sparse) | 1.3 | 2.2 |
| attention + merge | 18 (sliding), 32–50 (sparse at 2048) | 1.0 | 2.2, wp6b |
| weights (GEMMs, experts, router) | about 64 MiB a rank | about 5 | parameter count at 4–5 bit, kv_a / kv_b bf16, router f32; **not measured** |
| total | | about 17 | |

Prefill of 1024 rows with a cached prefix (all rows absorbed), attention side only:

| | TP1 (64 / 32 heads) | TP4 rank (16 / 8 heads) |
|---|---|---|
| sliding layer: absorb + attention + expand | 1.65 + 1.4..1.9 + 6.14 = 9.4 ms | 0.37 + 0.53..0.86 + 1.39 = 2.5 ms |
| sparse layer ≤ 2048: absorb + paged + expand | 0.81 + 1.2..3.5 + 2.94 = 6.1 ms | 0.20 + 0.72..2.2 + 0.68 = 2.4 ms |
| sparse layer > 2048: absorb + sel + expand | 0.81 + 5.31 + 2.94 = 9.1 ms | 0.20 + 3.93 + 0.68 = 4.8 ms |
| 30 + 12 layers | 357–392 ms | 93–132 ms |

For comparison the non-absorbed one-shot path at TP1: `attn_kv_sink` 0.57 ms + split 0.21 ms +
the kv_b GEMM a sliding layer.

## 3. Levers

### 3.1 L1 chained launches

**What.** `RadArchProbe::chain_pays` is unset for hyv4flash, so chaining is off for the whole
model (`chained launches: off` in the startup report) although its GEMM, norm, rope, `kv_store`,
`add`, `silu_mul`, MoE and gather rows already have chained forms. None of the Hy4 rows is in
`kChained` (`r4d_rows.cpp:11463`).

**Gain.** Precedent (GUIDE.md "chained launches"): Flash-Next, about 930 dependent launches,
13.9 → 12.5 ms, i.e. 1.5 us a launch; the wait behind the barrier bit is 3 us after a short
kernel. Hy4: about 1300 ordered launches a token → −2.0 ms a token (−12 % of the estimated TP4
token; upper bound the whole 5.7 ms of gaps). gpu-mini: 130 launches, expect 1.34 → about 1.15 ms.

**Precondition.** The eight commits of `fix/sync-audit` (/root/src/radiance, not upstream:
9208ae12 a chained kernel's argument slot is held until its blocks have counted, 6fc46ed9
`s_wait_kmcnt 0` after `s_dcache_inv`, bf6a9840 system acquire on a reused spare stream's first
dispatch, 3022ab69 / bf697c5b wall-clock deadlines of the flag waits, 0275adfc TP4 tail counters,
bb85d00a, 8647d0e9) are correctness fixes of exactly the paths this lever switches on (chained
launches, collectives at TP4). They come first (WP16.0b). Flint's `chain_pays` commit (819ae5a5,
flint-* branches only) is the model for the plugin change: 13–24 % of a step there, bit-identical.

**Design.**
- Step a (no kernel work): set `chain_pays = RAD_CHAIN_ALWAYS` in the plugin's probe, run gpu-mini
  TP1 / TP2 with stamps. Rows without a chained form keep their barrier; nothing else changes.
  Keep it only if the step gets faster (LFM2's and Qwen3.8-27B-bf16's notes in `kChained` show
  long bf16 GEMMs can lose; Hy4's real trunk is quantised, gpu-mini's is bf16 — judge on the
  quantised gpu-mini container (`hy4-flash-w4nl64` or `-i8`) as well).
- Step b: chained forms (`template <… , typename CH>` with `r4d_chain_wait()` /
  `r4d_chain_done()`, `r4d_chain.h`) for `ihc_pre_bf16`, `ihc_post_bf16`, `gate_mul_bf16`,
  `mla_project_kv_bf16`, `mla_attn_paged_sink_bf16`, `mla_attn_sel_sink_bf16` (both kernels of the
  launch: `mla_sink_rows` and `merge_sink`), `idx_pool_gated_f32`, `idx_score_bf16`,
  `router_sigmoid_topk_f32`, `swiglu_clamped_bf16`, `engram_*`, `embed_lookup_q_bf16`,
  `quant_act_fp8_grp`, `dequant_fp8_bf16`, `cast` / the copies, and the fused rows of 3.2 / 3.4.
  A row joins `kChained` only when every kernel its launch emits has the form; every block must
  reach `r4d_chain_done` exactly once (`mla_sink_rows` returns early for `!mine` lanes and empty
  ranges: move the body into a `__device__` function as the header of `r4d_chain.h` says).
  The answer stays 2 (the wait ends in `r4d_chain_acquire`); `sinks`, the block table and
  `cu_seqlens` are read with scalar loads, which that acquire covers.
- Prefill-height launches stay plain (as the other attention rows do): chain only below the
  decode cap.

**Exactness.** None moved: the chained form runs the same body.
**Validation.** `r4d_selftest` chained cases as the existing rows have; WP8's spec test
(`work/wp8/spec.py`: text and logprobs bit-equal to the plain greedy run, TP1 and TP2) with
chaining on; 20 minutes of repeated decode with `RADIANCE_AQL_CHAIN_FENCE=1` against the default
compared by tokens (the stale-scalar-read class of bug shows as a rare token flip).

### 3.2 L2 iHC: vectorised `ihc_pre`, fused junction

**`ihc_pre_bf16`, vectorised.** Today a thread strides the row's `hc*n` columns by 256 with 2-byte
loads of `h` and nine 4-byte loads of `fn`: 48 serial iterations at n = 3072.
- New form: a workgroup of 512 threads a token row, thread `t` owns 16-byte pieces
  `t, t + 512, …` of the row (1536 pieces at the real geometry, 3 a thread); a piece is 8 bf16 of
  `h` (one load) and for each of the `rows` projections two `float4` loads of `fn`. Sums:
  per thread, piece ascending, element ascending within the piece; lanes by `r4d_xor_sum<32>`;
  the 16 waves in wave order through LDS. `rows + 1` f32 accumulators (9), no WMMA (the
  projection is 8 x 12288, a matvec against f32 weights; the weight is 393 KiB, read at
  bandwidth in under 2 us).
- The collapse `x = sum_j pre[j] * h[j*n + i]` reads `h` again as 16-byte pieces (it is in the
  cache). Order `j` ascending as today, so `x` given the gates is unchanged.
- Expected: 18.7 → 5–6 us at n = 3072 M = 1 (−13 us x 85 launches = −1.1 ms a token), M = 1024
  92 → about 40 us. VGPR small (under 64), LDS under 1 KiB.
- Exactness: the reduction order changes, so `c`, the gates, `x` and `post` move within the bound
  the row already states (x within 1 bf16 ulp of libref, post 2e-4 relative). Row-independent
  (E1); every rank runs the same kernel (E5). One law change, do it before the junction so the
  junction copies the final order.

**Fused junction `rmsnorm_ihc_junction_bf16`** (and its collective twin). Replaces, between the
attention's `o_proj` and the feed-forward's first GEMM,
`[all_reduce] → rmsnorm(post norm) → ihc_post → ihc_pre(next) → rmsnorm(next norm)`, and at the
layer's end `add → [all_reduce] → rmsnorm → ihc_post → ihc_pre → rmsnorm` with the next layer's
`hc_attn` and `attn_norm` (the last layer: the head form, `rows == hc`, no `post` output, then
`output_norm`). Where an Engram sits in front of the next layer (1, 11, 21, 31) the chain stops
after `ihc_post` (form `tail = 0`), since `engram_gate` writes `h` between.
- One workgroup of 1024 threads a token row (the thread count of `rmsnorm_kernel<…, 1024u>`, so
  its reduction tree can be copied literally). Phases, a barrier between each:
  1. `y_raw` = input (`a + b` rounded to bf16 where the chain has the `add`); sum of squares in
     the standalone rmsnorm's order → `y = bf16(y_raw * rsqrt * gain)`, kept in LDS as bf16
     (6 KiB at n = 3072).
  2. `h[j] = bf16(post[j] * f32(y) + f32(h[j]))` written to memory and, from the same registers,
     the next connection's `rows + 1` reductions in the vectorised `ihc_pre` order.
  3. gates and `post_next`; `x = bf16(sum_j pre[j] * h[j])` kept in LDS (6 KiB); its sum of
     squares in rmsnorm's order; `xn = bf16(x * rsqrt * gain_next)` stored. `x` is stored only
     when a tap asks (an optional output operand).
- LDS about 13 KiB; VGPR under 96. 4 launches → 1 (5 → 1 with the `add`): −7 ordered launches a
  layer. Gain: 294 launches a token x 4.4 us of gap = −1.3 ms unchained; about −0.8 ms once L1 is
  in (a chained short launch still costs the 2.65 us launch floor).
- Collective twin at world > 1 (`ar_rmsnorm_ihc_junction`): the same phases behind the exact
  reduction, built like `ar_hc_write` / `ar_rmsnorm_quant_i8` (`r4d_ar_hc.h`,
  `r4d_ar_oneshot_2rank_exact.hip`, `r4d_ar_oneshot_Nrank_exact.hip`): the two-rank one-shot at
  decode width, the four-rank tail form. Do the plain row first, the twin as the second commit.
  The twin is wanted for a second reason (notes/flashnext-levers.md): the staged / coded two-rank
  wire (12.1 bit a value, lossless) exists only for the *fused* writes (`ar_hc_write`,
  `ar_gather_hc_write`, `ar_rmsnorm_*`); Hy4's standalone `all_reduce exact → rmsnorm → ihc_post`
  cannot use it. On Flash-Next at two ranks it was worth +2.6..2.9 % prefill. The twin's wire
  must stay exact (E5; the sandwich norm and the calibration read the sum; `wht6` is not allowed);
  the coded form for an exact norm wire exists only on the flint-perf branch (affd49b0) and has
  to be ported with it. Four ranks: exact tail form, no coded wire.
- The row reads weights of two layers at the layer boundary (`ffn_post_norm` of l, `hc_attn` and
  `attn_norm` of l + 1). If the placement groups (`grp_layer`) do not allow that, end the boundary
  junction after `ihc_post` and open the next layer with a fused `ihc_pre + rmsnorm` row
  (2 launches instead of 1; still −5 a layer).
- Exactness: E6, byte-equal to the chain it declares, including the two bf16 roundings of `y` and
  `x` that the chain has. E5: replicated.
- Validation: `tests/r4d_emu/test_ihc.cpp` extended (junction against the four reference ops in
  sequence, bytes; (hc, n) = (4, 3072), (4, 512), (2, 128); M = 1, 6, 17, 1024); kbench fusion
  oracle; fixture rows at n = 3072; the harness's stream parity after every layer; TP2 run
  compared with TP1 by `work/wp8/cmpdirs.py` (layer 0 `attn_hc_x` stays bit-identical).

### 3.3 L3 `mla_project_kv` on WMMA

**Law (both modes, every M):** tokens are the WMMA rows (16 a tile, unused rows zero), f32
accumulators, K in steps of 16 ascending, one rounding to bf16 at the store. An output row depends
only on its own input row, so the bytes are independent of M, the tile, `mb` and the grid (E1).
Replaces the scalar fma order; against libref it stays inside the present bound (not byte-equal
today either).

**absorb** `Y[t, col] = sum_{c<192} X[t, h, c] * Wk[h][c][col]`:
- `C += A·B`, A = X (lane = token, 16 contiguous `c`: one 16-byte load a lane half from `x`),
  B = `Wk[c0..c0+15][col0..col0+15]` (lane = output column, elements along `c`).
- **Latent-major key block.** In the checkpoint layout (`[448 rows][512]` a head) B's elements are
  strided by a row, eight 2-byte loads a lane. Stored as `WkT[h][col][c]` (512 x 192 bf16 a head,
  `c` contiguous) a lane half is one 16-byte load. Build it once at load, either as a second
  declared weight filled through a layout hook (`r4d_fp8_layout.cpp` style; kv_b itself must keep
  the checkpoint layout for `gemm_nt` and expand) or by a bring-up pass; +12.6 MiB a sliding layer
  and +6.3 MiB a sparse layer at TP1 (454 MiB over the model, divided by the world a rank). The
  relayout moves no bit (same operands into the same WMMAs), so: phase A the WMMA law on the
  checkpoint layout (strided loads, correct, slower), phase B the relayout, byte-compared with A.
- Wave = one 16-column tile, 12 k-steps. Workgroup 4 waves; grid (8, heads, ceil(M / (16 mb))).
- `mb` (16-row tiles a wave, 1 / 2 / 4 / 8, each with its own accumulator, B loaded once for all
  of them): at M = 1024 the weight is otherwise read 64 times (806 MiB at 64 heads, 1.3 ms of pure
  traffic). `mb` is a bit-neutral rad-tune axis; default by M (1 up to 16 rows, 8 from 128).
- VGPR at `mb 8`: 64 accumulators + 32 A fragments + 4 B: about 110. No LDS.

**expand** `Y[t, c] = sum_{j<512} X[t, h, j] * Wv[h][c][j]`: A = X (lane = token, contiguous `j`),
B = `Wv` rows (lane = output column `c`, elements along `j`: contiguous in the checkpoint's value
rows, no relayout). 32 k-steps a 16-column tile, 16 tiles a head; a wave a tile; same `mb` axis.

**Fused `mla_expand_gate`**: the expand store computes `ao = bf16(sum)` and at once
`bf16(f32(ao) * sigmoid(f32(g)))` with `gate_mul`'s own formula (same `expf`), writing `ag`
(and `ao` only if tapped). −1 launch a layer; E6 byte-equal to expand → `gate_mul`.

**Gain.**
- Decode: 16 heads absorb 15.1 → about 6 us, expand 12.8 → about 7; 8 heads 15.0 + 20.9 → about
  5 + 5; −0.76 ms a token at TP4. At 64 heads both are at bandwidth already: −0.2 ms a token.
- Prefill with prefix: absorb is 6.4 G MAC, expand 8.6 G MAC at M = 1024, 64 heads; at the
  22 TMAC/s `attn_kv_sink_h256` reaches on this card (17 G MAC in 0.57 ms, **M**) about 0.3 +
  0.4 ms against 1.65 + 6.14. Over the model: −254 ms per 1024 tokens at TP1, −56 ms a rank at
  TP4 (two thirds / half of the whole attention side).
- With L3 and L6 the absorbed sliding prefill (about 1.8–2.1 ms a layer at TP1) is level with a
  non-absorbed path over a gathered prefix (kv_b GEMM over M + 511 rows about 1.0 ms + split 0.3 +
  attention 0.76 ms, wp6c). **So the paged-gather op for a non-absorbed prefix path is not worth
  building**; the existing split (non-absorbed from position 0, absorbed otherwise) stays.

**Validation.** `tests/r4d_emu` case for both modes (heads 8 / 16 / 32 / 64; M = 1, 2, 15, 16, 17,
64, 1024; a row's bytes equal across M and across `mb`; relayout phase B == phase A in bytes);
`r4d_selftest`; kbench cases exist for every head count; harness parity; WP8 spec test.

### 3.4 `mla_sink`: the stable law (exactness), then L5 and L6

**3.4.1 Stable law on an absolute grid (fixes §0.5).**
- Chunks are the absolute 16-token groups `[16c, 16c + 16)` of the sequence (paged) or of the
  row's table order (sel), masked at both ends of the visible range. Super-chunks are absolute
  too: 4 chunks (64 tokens) with a window, 8 chunks (128 tokens) at window 0 and sel. A row's
  partials are its super-chunks: 8 or 9 with a window of 512, up to 17 at 2048 + margin keys
  (`kMaxSplits` 20, partial slot count a function of the op and window only).
- A partial is what it is today (`value[512]`, max, den, relative to its own max). `merge_sink`
  unchanged in formula: all maxima first, partials with `den > 0` added in ascending super-chunk
  order, sink once.
- The law never falls back to another summation because of M or scratch. The op gets an optional
  parameter `stable` (0 / 1; libref and libavx accept and ignore it, they are row-independent
  already). `stable 1`: always this law; the scratch hook sizes for the declared M range. The
  plugin declares two instances a layer: decode-class rows `[0, DT)` with `stable 1` and
  `M ≤ max_seqs * (1 + speculative rows)`, prefill rows with `stable 0`. Scratch at 9 partials:
  1.19 MiB a row at 64 heads, 0.30 MiB at 16; 48 rows at TP1 57 MiB, fine. If the scratch is
  short the launch is refused at declare, never silently unsplit.
- Why absolute: a row's chunk and split boundaries must not move when its neighbours in the
  launch change, and rows of one sequence must be able to share staged key tiles (3.4.2 row
  tiles); with boundaries relative to `lo = p + 1 − window` each row has its own grid, shifted by
  one token a row.
- E2 holds as before: both forms anchor at token 0.
- Cost: one more partial at a window (9 against 8) and a masked first chunk; expect M = 1 within
  ±1 us of today. Bytes change once (last place, same bound).
- Validation: emulator test: (a) a row's bytes with M = 1 equal its bytes inside launches of 2, 6,
  16, 17, 33 and 64 rows of 1, 3 and 8 sequences; (b) paged window 0 == sel in bytes; (c) window
  edge, lengths 0, 1, 15, 16, 17, 511, 512, 513, 2048, 2049; (d) HG 1 / 2 / 4 equal. On the card:
  `work/wp8/spec.py` at 1 stream and a new run at 4 streams x 6 rows against 4 plain runs.

**3.4.2 L5 bit-neutral speed** (each step byte-compared with the kernel before it):
- **Column-major value tile.** The value product's `fragT` reads a lane's 16 tokens of one column
  from the row-major tile: 128 two-byte LDS loads a lane a chunk (wp6b's 1.8 us a chunk). Stage
  the latent a second time as `vt[512 cols][16 tokens]` (16.4 KiB), so a fragment is one 16-byte
  LDS load a lane half. The transposition is done in registers at stash time: a staging thread
  loads the same 16-byte column piece of 8 tokens (8 loads), stores them row-major as today and
  stores 8 column pieces `[col][tok 0–7]` or `[tok 8–15]` (the WMMA lane halves).
- **LDS.** Row-major double buffer 37.4 KiB + one column-major buffer 16.4 KiB + partial dots
  4.1 KiB a head group: 57.9 KiB (HG 1), 62.0 KiB (HG 2). HG 4 (70 KiB) needs both tiles
  single-buffered with a third barrier in front of the stash: 18.7 + 16.4 + 16.4 = 51.5 KiB. The
  column-major tile is single-buffered in every form, so the stash of the next chunk must wait
  for every wave's value product: either the third barrier everywhere (simplest, try first) or
  two column-major buffers at HG 1 only (74 KiB: no).
- **Staging waves** (`sw` 0 / 4): four extra waves a workgroup that only fetch and stash. The
  compute waves then hold no staged registers (HG 1 today: 9 x 4 = 36 of its 245 VGPR), which
  raises the waves a SIMD the row can keep resident, and a staging thread has 8 or 9 loads in
  hand. At 128-thread workgroups (16 or 8 heads, TP4) this is what makes the 8-load transposition
  fit; without it LOADS would be 16 (64 VGPR, over the budget).
- **Row tiles** (`rt` 1 / 2 / 4, paged form only): one workgroup stages a super-chunk once for
  `rt` consecutive rows of one sequence; waves = `rt * HG * 4` (+ `sw`), ≤ 512 threads. Tiles are
  numbered sequence by sequence by a block scan of `cu_seqlens`, as `attn_kv_sink_h256` numbers
  its query tiles; grid.x = `ceil(M / rt) + seqs − 1`, tiles past the last return at once. A
  tile's slot `s` is the absolute super-chunk `first(tile) + s` (one slot more than a single row
  needs); a row writes an empty partial (den 0) where the slot is outside its range, which the
  merge skips. 8 heads: two rows share the 16 WMMA columns (today half the lanes carry zeros).
  sel keeps `rt 1` (a table a row).
- **rad-tune axes** (all bit-neutral, E7): `hg` {0, 1, 2, 4} (0 = today's rule), `rt` {1, 2, 4},
  `sw` {0, 4}. `RadTuneValidFn`: `(rt * hg * 4 + sw) * 32 ≤ 512` (1024 where LDS allows),
  `hg * 16 ≤ heads` rounded up, `rt == 1` for sel. No `splits` axis: the split is the law.
- Gain (estimate): chunk 1.8 → about 1.2 us. M = 1: window 13.7 → about 11 us, 2048 keys 27 →
  about 20 us: −0.17 ms a token. Verify of 6 rows at TP4 (16 heads, `rt 4`): 48 → 18 workgroups,
  about −5 us a sliding layer, −0.2..−0.3 ms a verify step. At 64 heads `rt` gives nothing.
  M = 1024: LDS operations a chunk −44 % (HG 1) to −81 % (HG 4), expect −15..−25 % of
  1.4–1.9 ms.

**3.4.3 L6 fast prefill law** (`stable 0` rows only): P enters the value product as one bf16, and
the denominator accumulates the *rounded* p, so the result is an exact softmax of logits perturbed
by at most 2^-9 relative in p (0.002 nat). 17 MMAs a chunk a wave instead of 25. "Where the bound
allows": HF rounds every score to bf16 before the softmax (up to 0.06 nat at |s| near 16, plan
D4), so this sits well inside the reference's own noise; the kbench bound of the row (rel_l2
≤ 1e-3) is expected to hold with worst cases near 5e-4 — measure on the fixture and fall back to
hi + lo if a case exceeds it. Decode-class rows keep hi + lo: they are latency-bound, the saving
there is under 5 %, and it keeps one fewer law change in E1's path. Unsplit, as today. Gain: −30 %
MACs on the absorbed prefill attention: −0.4..−0.6 ms a sliding layer and −0.4..−1.6 ms a sparse
layer at TP1 M = 1024 (−17..−34 ms per 1024 tokens), about a third of that a TP4 rank.
The f16 route (`attn_kv_sink_h256` style, V widened to f16, 11-bit P) was considered and dropped
for this kernel: the latent is also the score operand, and a second converted copy does not fit
the LDS budget next to the column-major tile.

### 3.5 L4 small fusions (each E6, byte-equal to its chain)

- **Cache-row front `mla_row_norm_rope_store_bf16`**: `rmsnorm` on the row's first 512, `rope`
  (gptj) on its last 64, `kv_store` of the 576 row, and the row left in the wire buffer for the
  non-absorbed path. Today: rmsnorm (ordered), rope, rope (unordered), kv_store (ordered).
  Pattern: `qkv_norm_rope_kv_store_bf16`. −1 ordered launch and −2 dispatches a layer. `q_rope`'s
  rotation stays its own launch (unordered beside it).
- **Sigmoid router `router_gemm_sigmoid_topk_scatter`**: the f32 router GEMM, `router_sigmoid_topk`
  and the sort's scatter in one launch, after `router_gemm_topk_scatter_bf16` (Flash-Next). −1 to
  −2 ordered launches on 41 layers. Selection must be byte-equal to the chain (ties to the lower
  index, bias for the selection only, weights the plain sigmoids normalised times 3.998): a
  different GEMM sum order would flip near-tie picks, so copy `gemm_nt_f32_tile_kernel`'s order.
- **expand + gate_mul**: in 3.3.
- Gain together: −3 ordered launches a layer: −0.55 ms a token unchained, −0.35 ms with L1.

### 3.6 L7 `attn_kv_sink_h256`: `bq` axis

A workgroup of 128 query rows reads keys from its first row's window start to its last row's own
key: 639 keys at window 512, of which each row sees 512 (20 % of the key tiles are masked work).
With 64 rows (8 waves, 256 threads) it is 575 keys (11 %). Axis `bq` {128, 64}, default 128;
`RG` becomes a template parameter (`kThreads = bq * 4`). LDS and VGPR unchanged a wave (168 VGPR,
LDS under 37 KiB). The key tiles start at the workgroup's lowest visible key, so `bq` moves a
row's chunk grid and therefore prefill bits within the bound (E4, allowed). Expected −8..−10 % at
window 512 (574 → about 520 us at 64 heads, M = 1024), nothing at window 0: −1.7 ms per 1024
tokens of a one-shot prefill at TP1. Low priority; it is the cheapest package.

### 3.7 L8 libavx attention

`mla_attn_*_sink` and `attn_kv_sink` in `libavx/avx_hy4.cpp` run one (row, head) at a time and
convert every key from bf16 for every head. Absorbed rows: convert a token's latent once and feed
all heads (the key is shared by the heads; a 32–64x cut of the conversions), 8 query rows a tile.
Dense rows: 8-row tiles over a key block. Keep: output independent of the thread count (the
checker reruns on 4 threads byte for byte); libavx against libref within the checker's 6e-3.
Matters only for host-placed layers and host-only runs; no card. Sonnet-tier.

### 3.8 L9 quantised latent cache, 656 B a token (design outline, not scheduled)

Row `[512 codes, 1 B each | 4 f32 scales, one a 128 columns | 64 bf16 rotated key]` = 656 B
against 1152 B. Sparse cache 13.8 → 7.9 KiB a token over 12 layers; at 1 M context 13.2 → 7.5 GiB
on **every** rank (the latent is replicated). Speed: decode about 0 (latency-bound); selection
prefill at 8–16 heads is traffic-bound (§2.2) and would gain up to 40 %, if real tables do not
already overlap.
- Kernel fit: the 128-column scale groups are exactly the wave quarters of `mla_sink_rows`. Codes
  are decoded to bf16 exactly at stash (E4M3 and int8 both fit bf16's 8-bit mantissa); the score
  is `sum_q s_q(token) * partial_q` at the LDS meet, in quarter order; the value product takes
  `p * s_q(token)` as its P. No fp8 WMMA needed, the query stays bf16.
- Codes: E4M3 (the `quant_act_fp8` group-128 rule, already a kernel) or int8 with the same scales
  (better SNR on a normed latent). The choice is a quantisation decision.
- It is **lossy** against the bf16 cache (E4M3: about 2 % rms a column), changes every attention
  output, needs its own prefix-cache key and `--kv-cache-dtype` value, and must be in the friend's
  KL run. Do not build before he asks for it; then a Fable design pass first (store op, both
  attention forms, the non-absorbed path reading the decoded latent, libref / libavx twins).

### 3.9 Plugin and engine settings that are not kernels (from notes/flashnext-levers.md)

Checked against the tree at 1774a48a: the plugin sets no `chain_pays` and never calls
`rad_kv_group_replicated` (`glm5next.cpp:561` does, for its latent and index groups).

- **Replicated KV groups (WP16.0c).** Every Hy4 cache is whole on every rank (MLA latent, window
  latent, indexer keys; PLAN §1.3). Without `rad_kv_group_replicated` a host-tier slot and a
  prefix-store entry hold every rank's copy end to end: 2x at TP2, 4x at TP4, of 13.8 KiB a token
  (sparse latent) plus 768 B (indexer keys) plus the window. Declaring the groups replicated
  stores one window, written by rank 0 and read by all: 2–4x more cached prefix tokens in the
  same host pool and half / a quarter of the tier traffic. Three calls beside the group
  declarations (`hyv4flash.cpp` near 516 / 612 / 624). The export covers **paged groups only**:
  the sparse latent and the indexer keys certainly; the window group (`RAD_KV_WINDOW`) only if the
  engine accepts it (flint's window group is the precedent to read); the indexer tail ring and the
  Engram CONV group are not paged and stay as they are. No effect at one rank. Lossless.
  Validation: `RADIANCE_DEBUG_TIER_VERIFY` at TP2 on gpu-mini (followers hash what they would have
  written); a prefix-cache hit after eviction to the host tier gives the tokens of the uncached
  run. Precondition of the claim: every rank computes the cache row with the same kernel on the
  same bytes (`kv_a`, norm, rope are replicated ops: holds; it must keep holding under L4's fused
  front row, which is replicated too).
- **Expert placement at the real size (WP16.9).** 147 GB of w4nl64 experts against 64 GB of VRAM a
  box: most experts live in the host pool and a decode token's cost is set by how many of its
  16 x 41 routed experts are resident. This outweighs every kernel lever above if it is wrong.
  The planner is generic over `MoeFP8` (`expert_tiered`, `expert-profile`, `expert-pin-top`,
  `host-pool-mib`, next-layer prefetch; prefetch is off at tp ≥ 2). To do: (a) record a decode
  profile with `expert-stats` on the user's workload (code prompts; the container's own
  activation profile is prefill-heavy) and feed it as `expert-profile` with `expert-pin-top`;
  (b) check the planner with Hy4's serving form, TP-sliced experts (768 / 384 columns of every
  expert a rank, wp7 / wp10): a rank's slice of a hot expert must be resident on that rank, and
  the slices of one expert must be promoted together; (c) report resident share and miss cost a
  token. Needs the real container, so it is the friend's box or a real-size skeleton with a
  synthetic Zipf profile; here only the declare-time placement (`--debug-placement`) can be
  checked. Lossless (placement moves no bit).
- **Engine levers already in the branch** (prefix resume, prefill span, MoE prefill GEMMs at
  K = 1536, checkpoints carrying the indexer tail ring and the Engram CONV group) and the
  drafter-dependent ones (coupling, traversal verify, adaptive depth as a block prefix) are
  listed with their preconditions in notes/flashnext-levers.md §1; they are WP13's and the
  coordinator's, not kernel work. One interaction with this note: verify rows of a block drafter
  are exactly the decode-class rows of E1, so 16.4a (stable law at any row count) must land
  before a drafter runs several streams.

## 4. Work packages (Opus unless noted), in order

Common rules: AGENT_RULES.md; own worktree; emulator test first, card under the lock for
`r4d_selftest`, `rad-kbench --fixture data/kernels_hy4.rkb --kernels libr4d:libref --bench` and
the harness; commit message states VGPR / LDS / spills / occupancy of every new kernel; every
package ends with the gpu-mini stamp table (`work/wp16/stamp.py`, `agg.py`) before and after.
Order: 16.0, 16.0b, 16.0c first (measurement, correctness fixes, a three-line lossless win), then
16.1a, then the kernel packages, 16.1b last; 16.9 as soon as a real-size container exists, since
at the real size it bounds everything else. Packages 16.2, 16.3, 16.4 touch different `.hip` files and separate blocks of `r4d_rows.cpp` /
`r4d_shapes.cpp`; they can run in parallel after 16.0. 16.1b goes last because it wraps their
kernels.

| WP | content | depends | acceptance |
|---|---|---|---|
| 16.0 (S or O, GPU short) | Measurement base. (a) harness preset `gpu-wide`: n = 3072, real trunk widths (q 1024, kv 512, 64 / 32 heads, layer-0 MLP 13824, latent 1536, expert ff 1536), 4 layers (sparse, sliding, sparse, sliding), 16 experts, dummy weights, one quantised recipe (`w4nl64` or `i8`) beside bf16. (b) decode stamps and prefill timings (one-shot, chunked 1024, context 3000) at TP1 and TP2, written as a per-layer table x 42 in `notes/wp16-0.md`. (c) fixture rows: M = 6 and 16 at 16 / 8 heads for both sink ops, `ihc_*` at n = 3072 M = 6 / 16, sel cases with tables taken from a real `qsa_select` run. | - | the table replaces §2.3's estimates; ranking of §0 confirmed or corrected in the note |
| 16.0b (O, GPU short) | Bring the eight `fix/sync-audit` commits (/root/src/radiance, libr4d / aql) onto the Hy4 branch: cherry-pick, resolve against upstream's squashed history, no other dartree content. | - | builds; `r4d_selftest`; ctest baseline unchanged; a chained model that exists today (a flint or Qwen fixture) still passes its chained selftests |
| 16.0c (S, GPU short) | §3.9 `rad_kv_group_replicated` for the sparse latent and indexer-key groups (window group if the engine takes it); arch test at worlds 1 / 2 / 4. | - | `RADIANCE_DEBUG_TIER_VERIFY` clean at TP2; prefix hit from the host tier token-equal; slot bytes a cached token halved at TP2 |
| 16.1a (S, GPU short) | `chain_pays = RAD_CHAIN_ALWAYS` in the plugin probe (model: flint 819ae5a5); list which Hy4 rows still take the barrier bit; measure bf16 and quantised gpu-mini / gpu-wide, TP1 and TP2. Keep only if faster. | 16.0, 16.0b | step time and stamp table; spec test bit-equal |
| 16.2 | §3.2: vectorised `ihc_pre`; `rmsnorm_ihc_junction_bf16` (all forms: with / without `add`, `tail` 0 / 1, head form); plugin wiring behind `RADIANCE_HY4_JUNCTION` (default on once green); then the collective twin `ar_rmsnorm_ihc_junction` (two-rank one-shot with the coded exact wire ported from flint-perf affd49b0, four-rank exact tail). | 16.0 | emu bytes against the chain; kbench fusion oracle; `ihc_pre` n = 3072 M = 1 ≤ 7 us; ordered launches a layer −7 (TP1); harness parity; TP2 == expectations of wp7 |
| 16.3 | §3.3: WMMA `mla_project_kv` (phase A law, phase B latent-major block), `mb` axis, fused `mla_expand_gate`. | 16.0 | row bytes independent of M and `mb`; phase B == phase A bytes; M = 1024: absorb ≤ 0.5 ms, expand ≤ 0.7 ms at 64 heads; M = 1 at 16 heads ≤ 8 us each; spec test bit-equal |
| 16.4a | §3.4.1: stable law, `stable` parameter (libref, libavx, libr4d schemas identical, OPS.md), two instances in the plugin, scratch sizing, refusal instead of fallback. | 16.0 | emu list of 3.4.1; 4-stream verify equals 4 plain runs in tokens and logprobs; M = 1 time within 10 % of today |
| 16.4b | §3.4.2: column-major value tile, staging waves, row tiles, axes `hg` / `rt` / `sw` with the valid-fn; `rad-tune` run on the card. | 16.4a | every step byte-equal to 16.4a's kernel (emu sweep over the axis product); chunk ≤ 1.3 us; no spills, LDS ≤ 64 KiB in every form |
| 16.4c | §3.4.3: single-P prefill law on `stable 0` rows. | 16.4b | fixture bound holds (state worst rel_l2); M = 1024 time −25 % or the package is dropped |
| 16.5 (S) | §3.5: cache-row front, sigmoid router fusion. | 16.0 | fusion oracle bytes; router picks identical to the chain on the harness prompts |
| 16.1b | Chained forms for every Hy4 row and every row added by 16.2–16.5; `kChained`. | 16.1a, 16.2–16.5 | all Hy4 decode dispatches chained in the stamps; spec test; fence A/B run of §3.1 |
| 16.6 (S) | §3.6 `bq` axis. | - | selftest, kbench; window 512 M = 1024 −8 % or dropped |
| 16.7 (S, host) | §3.7 libavx tiling. | - | `rad-avx-check` green, thread-count independence |
| 16.8 (F then O) | §3.8, only on the friend's request. | 16.4b | own design note first |
| 16.9 (O; real container or skeleton) | §3.9 expert placement: decode `expert-stats` profile, `expert-profile` + `expert-pin-top`, planner check with TP-sliced experts. | 16.0, real-size container | resident share and miss cost a token reported; slices of an expert promoted together |

Targets after 16.1–16.5 on gpu-mini TP1 (today 1.34 ms a token, 169 dispatches, 130 ordered, 0
chained): ≤ 1.0 ms a token, ≤ 115 dispatches, every dependent one chained. Prefill of a second
1024-token chunk: attention side per sliding layer ≤ 2.2 ms at 64 heads (today 9.4).

## 5. Risks and open points

- The decode baseline's GEMM share is an estimate; if quantised GEMMs at real widths dominate more
  than assumed, L1–L4 shrink in percent (not in ms). 16.0 settles it.
- Chained launches and TP4 collectives without the `fix/sync-audit` commits risk hangs and silent
  races (stale scalar reads, argument slots reused early); 16.0b is a hard precondition of 16.1.
- Decode at the real size is bounded by expert residency (§3.9), which nothing here measures.
- Chaining can lose behind long bf16 GEMMs (two documented cases). Decide on the quantised
  container, per the probe's "with and without a drafter" switch.
- The junction row spans two layers' weight groups; the fallback (two rows) is in §3.2.
- The latent-major key block needs a second stored form of part of kv_b; if the loader's "one
  stored form a weight" rule blocks a derived weight, phase A alone still gives most of the
  prefill gain (the strided loads cost at M = 1, not at M = 1024 with `mb 8`).
- 16.4a's third barrier / ninth partial may cost a little at M = 1; it buys E1 at any stream
  count, which today's build does not have.
- sel at prefill height: kbench's permutation tables are a worst case. Do not optimise it (or
  argue for L9 with it) before 16.0's real-table cases exist.
- VGPR: `mla_sink_rows` is at 245 (HG 1). Staging waves lower it; the row-tile forms must be
  checked form by form in the disassembly, as wp6b did.
