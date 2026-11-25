# Audit A: libr4d decode-hot kernels (Hy4-flash, TP2, 1..48 rows)

Read-only audit of `src/libr4d` at c655f58c. Nothing built or run. Line numbers verified by reading.
Sections: 0 byte-floor table, 1 trunk / routed / head GEMM findings (K-*), 2 junction + chain
machinery (J-*, C-*), 3 attention rows (T-*), 4 small kernels (S-*), 5 checked / not a lever,
6 format / lossy (friend's decision). Estimates are marked "est."; every measured number is from
COMMON.md or kernels/levers-general.md §1.2 (one sparse layer, 1 client, pre-G-3 tape).

## 0. Where each weight kernel sits against its byte floor (per rank, one sparse layer, M = 1)

Floor = bytes / 570 GB/s (1 MB = 1.75 us). Shapes from the graph (n 3072, q_lora 1024, 16 heads a rank
on a sparse layer, latent 1536, expert ff 768 a rank) and the iqr recipe (u5 `tcq_code=mad scale=f32` =
`FormT<kTrellisMad,5,0,16,0,kRowF32>`; int8 g128 bf16 scales; routed gate_up u3, down w4nl64).
Labels of the levers-general §1.2 list were re-assigned by issue order (`hyv4flash_moe.h:390-402`):
the "gemm_i8 9.7 (shared down)" there is latent_out, "iqr_gemv 10.2 (shared gate_up)" is latent_in.

| kernel (form) | weight | MB | meas. us | floor us | x floor | regime |
|---|---|---|---|---|---|---|
| attn_output (i8 kpar) | 3072 x 4096 | 12.6 | 22.8 | 22.1 | 1.03 | bandwidth |
| attn_gate (u5 gemv) | 4096 x 3072 | 7.9 | 18.5 | 13.8 | 1.34 | bandwidth + decode VALU |
| q_a (u5 gemv) | 1024 x 3072 | 1.97 | 6.0 | 3.4 | 1.75 | latency (3 units a wave) |
| q_nope (u5 gemv) | 3072 x 1024 | 1.97 | 6.5 | 3.4 | 1.9 | latency (1 unit a wave, kw 8) |
| q_rope (u5 gemv) | 1024 x 1024 | 0.66 | 4.6 | 1.2 | 4.0 | launch floor |
| latent_in (u5 gemv) | 1536 x 3072 | 2.95 | 10.2 | 5.2 | 2.0 | latency (runs beside router) |
| latent_out (i8 kpar) | 3072 x 1536 | 4.7 | 9.7 | 8.2 | 1.18 | near bandwidth |
| shared gate_up (u5 gemv) | 1536 x 3072 | 2.95 | 7.9 | 5.2 | 1.5 | latency |
| shared down (i8 kpar) | 3072 x 768 | 2.36 | 5.8 | 4.1 | 1.4 | latency (2 groups a wave) |
| routed gate_up (moe_iqr u3) | 16 x 1536 x 1536 | 14.8 | 36.7 | 26 | 1.41 | block-head latency (K-2) |
| routed down (moe_down_dec w4nl64) | 16 x 1536 x 768 | 9.7 | 22.9 | 17 | 1.35 | block-head latency |
| head (logits_gemm i8, mvm) | 60416 x 3072 | 186 | 300 | 326 | 0.92 | at bandwidth (spec 640 GB/s) |
| junction fn (f32, 1 block) | 8 x 12288 | 0.39 | 7.3 plain | 0.7 | 10 | one WGP (J-1) |

Excess over floor on the listed weight kernels: about 41 us a sparse layer, ~1.7 ms a token (est.,
sliding layers have 2x the q/gate/output bytes and were not measured separately). Two inconsistencies
to re-measure before ranking anything on them: `gemm_bf16 5.2` (kv_a, bf16 576 x 3072 = 3.5 MB) and
`gemm_nt_f32 6.8` (router, f32 512 x 3072 = 6.3 MB replicated per levers-tp2 §3) imply 680 / 925 GB/s,
above the card's 640 GB/s spec, so either the attribution or the shapes are off.

M = 8..48: every weight kernel here reads its weight once for all rows (iqr_m16 up to 64 tokens a
decode, kpar NT <= 4, moe_iqr a run's rows within a 16-row tile, moe_down_dec a 16-row run tile, head
one read per 128 rows). The one exception is K-1.

## 1. Trunk, routed and head GEMMs

### K-1 moe_iqr decode grid splits an expert's run at fixed 16-row tile boundaries (weight read twice)
- area: kernel. `r4d_moe_iqr.hip:270-322` (block x = (16-row tile of sorted rows, run in it),
  `r0 = (blockIdx.x / kRows) * kRows`, the run clamped to `[r0, r0+16)`), `:911` (`g.gx = T`).
- Mechanism: a run [oa, ob) that straddles r0+16 becomes two runs in two tiles; each of the two
  blocks (for every y tile) decodes and streams the expert's whole weight tile. moe_down_dec already
  fixed this for down (`r4d_moe.hip:1521-1545`, `:1529`: "tiles start at the run's first row"; block j = j-th
  run tile in expert order, one round of offsets, one extra zeroing block). Port that numbering to
  moe_iqr's decode grid: x = min(T, n_expert) + T/16 + 1 slots, block j finds the j-th run tile.
  It also removes the idle blocks: today x = T slots per y tile, and every slot past the tile's run
  count still runs the 17-load offset search before returning (`:318`).
- Why: decode band reaches T = 768 (`r4d_moe.hip:2671`, 2T <= 3 n_expert). At T = 768 / 512 experts
  (c8 DFly2 verify, 48 rows) about 398 distinct experts and 48 boundaries, each inside a multi-row run
  with p ~ 0.48 -> ~23 extra expert decodes a layer (+6 % of the routed gate_up bytes; that phase is
  23.9 ms a c8 verify: est. -1.2..-1.4 ms a step, ~2 %). At T = 128 (c8 no spec) +0.7 %. Idle
  blocks at T = 768: ~350 of 768 slots x 24 y-tiles.
- Lossless: moves bits within f32 rounding (a row that today sits in a 1-2-row split piece takes the
  dot path; joined to its run of >= 3 it takes the WMMA path, `:351`). Note a row's bytes already
  depend on where the tile boundary falls, i.e. on the other rows of the step; the run-aligned grid
  makes them depend only on the expert's own row count, which is what moe_down_dec does. Byte-compare
  is impossible by construction; compare against the map grid / libref bounds.
- Confidence high (code), gain medium; effort M. Verify: `r4d_selftest` moe_iqr cases and
  `--perfmoeiqr`, kbench `moe_gemm_q` rows of `data/kernels_hy4.rkb` at T = 16, 128, 480, 768.

### K-2 moe_iqr: each block pays a 3-round-trip head before its first weight byte, 96x per run at M = 1
- area: kernel. `r4d_moe_iqr.hip:278-318` (wave 0: 17 `off[]` loads a lane, scan, `lds_barrier`),
  `:333-334` (`C.w[e >> cshift]` pointer-table load, dependent), then `D::fetch` (`:141` dot path, `:450` tile path).
- Mechanism: at M = 1 (T = 16 < kOneWaveMinRows) the wide-K form runs KW = 4, RTB = 1: grid
  16 x 96 blocks of 128 threads, 3 units a wave (nkb 12). Every one of the 96 y-blocks of a run
  repeats the search and the pointer load; `static_assert(TPW == 1u || KW == 1u)` (`:228`) forbids
  several tiles a block in this form. Options (each byte-identical, same per-tile arithmetic):
  (a) let a KW = 4 block take 2-4 tiles in turn, fetching the next tile's first unit before folding
  the current one; (b) have the scatter (`moe_scatter_i32`) write a run descriptor list
  (expert, rs, re, W ptr, S ptr) so a block does one scalar load instead of search + table load.
- Why: gate_up 36.7 us vs 26 us floor at M = 1; with ~1536 short blocks in ~2-6 residency rounds
  each round pays ~3 dependent latencies before streaming. Est. -4..-8 us a layer (-0.15..-0.3 ms a
  token at 1 client). Same head exists in moe_down_dec (`r4d_moe.hip:1595-1654`: offset round,
  2 lds_barriers, `wtab_at`), so (b) serves both.
- Lossless byte-identical. Confidence medium (mechanism read; size of the win unmeasured). Effort M.
  Verify: selftest moe_iqr (decode band cases), kbench moe rows M = 1..4, stamps of a pass.

### K-3 q_nope + q_rope as one launch, byte-identical (refines G-3d)
- area: kernel + plugin. `r4d_gemm_iqr.hip:954` (`kw = N >= 4096u ? 4u : 8u`), `:1031` (same rule in
  m16_launch), `:959-963` (grid N/16, block kw waves); plugin `hyv4flash_attn.h:509-510`.
- Mechanism: G-3d was stopped because sliding layers give q_nope (N 6144) kw 4 and q_rope (N 2048)
  kw 8. A dual-weight form with 8-wave blocks keeps each part's order: blocks of the kw-4 part take
  two 16-row tiles x 4 waves (waves 0-3 tile 2b, 4-7 tile 2b+1, each tile's LDS partials summed over
  its own 4 waves ascending, as the plain kernel does), blocks of the kw-8 part one tile x 8 waves.
  Per tile the unit assignment (`kb = wave, wave+KW, ..`), fold and LDS order are the plain kernel's.
  On the 12 sparse layers (16 heads: N 3072 / 1024) both are kw 8 already: a plain concatenated grid.
  Two Ctx (W, S, N), two Y; same activation (the fused q-residual codes).
- Why: q_rope is 0.66 MB in 4.6 us (4x floor: pure fixed cost); merged it rides q_nope's stream.
  Est. -3..-4 us a layer, -0.13..-0.17 ms a token. G-3 measured ~0.3-0.5 us per plain launch
  removed; a weight launch removed also drops its ramp/tail, which is most of q_rope's 4.6 us.
- Lossless byte-identical (per-tile arithmetic unchanged). Confidence medium-high on bytes, medium
  on gain. Effort M. Verify: a selftest case comparing the dual launch against two gemm_iqr_nt_m64
  calls at M = 1..16 (u5 kRowF32), kbench fusion oracle.

### K-4 q_a + attn_gate as one dual-weight launch (both read the attention-norm codes)
- area: kernel + plugin. `hyv4flash_attn.h:502` (`q_a.step(c, act(xa), ..)`) and `:549`
  (`gate.step(c, act(xa), ..)`; `gate.feed = &q_a` at `:338`, so they share the codes); kernel as K-3.
- Mechanism: one launch over q_a (N 1024, kw 8) and attn_gate (N 4096 / 8192, kw 4) with K-3's
  tiles-per-block trick. q_a's consumer (q-norm) then waits for the gate's bytes too, but the gate's
  18.5 us leaves the critical path later (it is consumed only by mla_expand_gate).
- Why: q_a runs at 330 GB/s (6.0 us for 1.97 MB); merged ~ (1.97 + 7.9) MB at ~550 GB/s ~ 18 us vs
  6.0 + 18.5 today: est. -4..-6 us a layer, -0.2 ms a token. The engine cannot overlap them otherwise:
  a dependent chained launch waits for every chained block issued before it
  (`core/device/aql.cpp:1676`, target `chain_issued`), so issuing the gate earlier would only make
  q-norm wait for it.
- Lossless byte-identical. Confidence medium (gain depends on the merged launch holding bandwidth).
  Effort M (shares K-3's kernel). Verify: as K-3; stamps of the attention front before/after.

### K-5 latent_in + shared gate_up as one dual-weight launch (both read the junction's xf codes)
- area: kernel + plugin. `hyv4flash_moe.h:390` (`lat_in.step(c, act(xf), ..)`) and `:399`
  (`shared.step(c, xf, ..)`), both u5 kRowF32, K 3072, N 1536 -> kw 8 both: a concatenated grid is
  byte-identical without K-3's trick.
- Why: latent_in 10.2 us + shared gate_up 7.9 us for 5.9 MB (floor 10.4): est. -5..-7 us a layer x 41
  = -0.2..-0.3 ms a token. This is the kernel-level form of X5 (KNOWN open: "shared arm beside routed
  GEMMs"); it does not need a second queue.
- Interaction: the shared arm uses gated_had_quant_i8g after its gate_up (`hyv4flash_moe.h:26-90`);
  with K-7 the shared part would be the gated form, so a dual launch of a plain + a gated part needs
  the gated epilogue per part (the gated grid is N/32 with 2 x kw waves: compatible with 8-wave
  blocks at kw 4 only, so pick one of K-5 / K-7 first; K-5 pays more).
- Lossless byte-identical. Confidence medium. Effort M. Verify: as K-3.

### K-6 latent_out + shared down as one dual-weight kpar launch
- area: kernel + plugin. `r4d_gemm_fp8a8.hip:1162-1183` (kpar grid N/16, `KW = 4` for ngrp >= 4);
  latent_out K 1536 (12 groups), shared down K 768 (6 groups), N 3072 both: KW 4 both, so a
  concatenated grid (blocks [0,192) weight A, [192,384) weight B, two outputs) is byte-identical.
  Plugin `hyv4flash_moe.h:397-402` (lat_out, shared, then `op_join` add).
- Why: 9.7 + 5.8 us for 7.1 MB (floor 12.4): est. -2..-3 us a layer. Needs the shared arm's gate_up
  done earlier (K-5) or the launch waits for it anyway.
- Lossless byte-identical. Confidence medium. Effort S-M after K-3's dual entry exists.

### K-7 the shared expert (and layer 0's MLP) do not use the gated GEMM epilogue that exists
- area: plugin + kernel. `r4d_rows.cpp:9366-9382` (`gemm_iqr_gated` = `gemm_nt_q_gated`, M <= 16,
  "byte-identical to gemm_iqr_nt_m64 followed by gated_had_quant_i8g", fusion oracle
  `r4d_fuse_gemm_nt_q_gated_iqr`, r4d_fuse.cpp:482); kernel `r4d_gemm_iqr.hip:99-129, 304-320`
  (gated_put / gated_tail, last of a 128-column group's 8 blocks quantises). The plugin's GatedMlp
  issues gate_up then `gated_had_quant_i8g` (`hyv4flash_moe.h:31-78`, 88-90).
- Mechanism: declare `gemm_nt_q_gated` for decode-class rows of the shared expert (gated width 768 a
  rank: a multiple of 256; runs of 128 rows: fine) and layer 0's dense MLP; keep the pair for prefill
  (the row serves M <= 16 only, so two instances by row range).
- Why: -1 launch a layer (42) and the [M, 1536] bf16 plane's write + read; est. -0.3..-0.5 us a launch
  removed (G-3's measured figure): -0.02 ms a token. Small; listed because it is plumbing only.
- Lossless byte-identical (the row's claim). Confidence high (code), gain low. Effort S.

### K-8 chained GEMMs wait before issuing loads of their immutable weights
- area: kernel. `r4d_gemm_iqr.hip:157` (wait) then `:184` (first `D::fetch`); `:358` / `:411` (m16);
  `r4d_gemm_fp8a8.hip:868` (kpar wait) then `:790` (first `aload`); `r4d_chain.h:107-128` (thread 0
  polls, `__syncthreads`, then the acquire).
- Mechanism: the packet processor launches chained packets ahead "as far as the card has slots"
  (`aql.cpp:1380-1386`), so a GEMM's blocks are often resident while the kernel in front runs; they
  then sit in the poll. For trunk weights (static during a step; not routed experts, whose pointer
  depends on the router) issue the wave's first one or two weight units (and the scale loads that
  do not depend on the activation) before `r4d_chain_wait*`. Registers are not touched by the
  acquire's global_inv / s_dcache_inv, so the values stay valid. Activation loads stay behind it.
- Why: hides one DRAM latency (~1-2 us) per trunk weight launch that becomes resident early: ~9 a
  layer, ~380 a token. Est. -0.2..-0.5 ms a token (wide error bar: depends on residency of the
  waiting blocks next to the running kernel). The "fixed ~4.5 us" term of `libr4d/isa/README.md`
  "General results" is mostly this ramp for the small weights of §0.
- Lossless byte-identical. Confidence medium-low on gain; the mechanism (wait before any load) is
  certain. Risk: a weight that a tier copy can rewrite mid-step (LoRA, trunk tiering) must keep the
  wait; gate it on the row's weight being VRAM-static. Effort S-M.
  Verify: r4d_selftest chained cases, spec test bit-equality, stamps (gap before GEMM starts).

### K-9 one-unit-ahead prefetch leaves short K loops latency-bound
- area: kernel. `r4d_gemm_iqr.hip:184-186` (GEMV), `:411-417` (m16), `r4d_gemm_fp8a8.hip:789-805`
  (kpar).
- Mechanism: each wave holds the current unit and one in flight. q_a / latent_in / shared gate_up:
  nkb 24 / kw 8 = 3 units a wave; shared down kpar: 6 groups over 4 waves (2,2,1,1). With depth 1 the
  wave is ~2 latencies + compute; issuing all of a wave's units up front when nkb / KW <= 4 makes it
  ~1 latency. Raw unit (B = 5): 10 code dwords + 4 continuation dwords = 14 VGPR a unit: 3-4 units fit.
- Why: est. ~1 us on 4-5 launches a layer (-0.15..-0.2 ms a token); overlaps with K-8 (do K-8 first,
  then measure whether K-9 still pays).
- Lossless byte-identical. Confidence medium-low. Effort S. Verify: kbench gemm_iqr / gemm_i8a8 rows
  at the shapes of §0.

### K-10 logits_gemm at M <= 2 has no chained form (head keeps its barrier bit)
- area: kernel. `r4d_gemm_fp8a16_nt_m1.hip:806-807` ("the matvec below three rows stays plain"),
  `:879-886` (`mvm_kernel` launched plain; only `lm_wmma` has `r4dchain_lm_wmma`, `:708`).
- Mechanism: add `r4dchain_mvm` (wait_vec: the activation is read by lane-varying uint4 loads) and pass
  `chain` through the M <= 2 branch.
- Why: levers-general §1.3 lists the head among the 463 barrier-bit dispatches (4.4-4.6 us gap each,
  stamped). One a no-spec step, five a DFly2 draft pass at one sequence (M = 1 each round) plus the
  verify head at <= 2 rows: est. -10..-20 us a DFly2 step (~0.1 %). Small, S.
- Lossless byte-identical. Confidence high (code). Verify: selftest logits chained case, stamps.

### K-11 moe_down_dec + moe_gather (+copy16) fused by a last-arriver fold (refines G-3g, KNOWN open)
- area: kernel. `r4d_moe.hip:1851-1860` (moe_down_dec stores bf16 Y per sorted row at the tile's last
  K block), gather unverified here (see section 4).
- Mechanism: per (token, 64-column tile) an arrival counter; the block that brings the token's last
  slot reads the token's top-k rows of Y (bf16, as stored today) in slot order and applies gather's
  weighted sum, then resets the counter (the split-K finisher pattern of `r4d_gemm_iqr.hip:716-745`).
  Byte-identical iff the finisher reproduces gather's order and roundings exactly.
- Why: -1..-2 launches a layer (moe_gather 3.4 + copy16 3.3 us measured). Est. -2..-4 us a layer.
- Lossless byte-identical by construction. Confidence medium. Effort M.

## 2. Two-rank junction and the chain machinery

### J-1 the M = 1 junction is one 512-thread block; fn (393 KB f32) streams on one WGP
- area: kernel. Launcher `r4d_ar_ihc_junction.hip:287-293` (`nb <= M`), kernel `:137-200`; arithmetic
  `r4d_ihc.h:207-260` (`acc_piece` float4 loads of 8 rows of fn, 3 pieces a thread at hc 4, n 3072).
- Mechanism (a, S): thread 0 spins on the peer flag (`:178`, the ~5.3 us P2P round trip of
  levers-tp2 §1.1) while the other 511 threads wait at `__syncthreads`. Let them touch fn (all 8 x
  12288 floats, 24 float4 a thread), the two gain vectors and h (24 KB) during the wait, so the
  junction's `acc_piece` / `norm_row` loads hit L2. Loads only, no values used: byte-identical.
  (b, M): split the plain part's piece sums over several blocks at small M: the order is per wave
  (thread pieces ascending, xor_sum over lanes) then waves 0..15 ascending (`r4d_ihc.h:73-105`), so
  blocks that each own some waves write part[wave][k] and a last arriver finishes: same bytes.
- Why: the plain row is 7.3 us at M = 1 (fn at ~54 GB/s on one WGP); 84 junctions a token. (a) est.
  -2..-4 us each = -0.17..-0.34 ms a token; (b) more, more effort.
- Lossless byte-identical. Confidence medium. Not the closed "faster wire": the wire is untouched.
  Verify: `tests/r4d_emu/test_ihc.cpp`, r4d_selftest junction cases, TP2 bit-equal greedy text.

### C-1 the chain wait is all-or-nothing for dependent launches (informs K-3..K-6)
- `aql.cpp:1676`: a dependent chained launch's record targets `chain_issued` (every chained block
  issued before it), an independent one `chain_frontier`. So two independent weight streams cannot
  overlap across a dependent launch on one queue; only one launch over both weights overlaps them.
  `r4d_chain.h:107-128`: thread 0 polls (s_sleep 1 for <= 16-block grids, else 4 / 16), then
  `__syncthreads`, then global_inv + `s_dcache_inv` (wait_vec skips the latter). Per aql.cpp:1335 a
  dependent small launch costs 2.1 us with the agent acquire and 1.2 without. Not a ticket: context.

### Checked in the GEMM / head / junction part, not a lever (so nobody re-derives them)
- **Weight re-streaming at M > 1**: none (see §0's last paragraph). The u5 trunk takes the GEMV only at
  M = 1 (`gemv_max`, `r4d_gemm_iqr.hip:1047`: B = 5 gives 1) and the WMMA 16-token tile from 2 rows
  (`iqr_m16_kernel`, iu8 16x16x16, one decode a 128-K unit for up to 4 token tiles, transposed through a
  2176-byte LDS tile a wave): trunk linears 3.08 -> 3.40 ms from 1 to 8 clients (+10 %).
- **Decode tables to LDS**: the trunk and routed forms are the computed trellis (`kTrellisMad`): no
  table (`Dec::kTab` = 4, `stage` a no-op); moe_down_dec's w4nl codebook arrives as a kernel argument
  (`MoeTab tab`). Nothing to hoist.
- **Load widths**: IQR units are read as uint4 / uint2 runs of the lane's 40-byte (u5) / 24-byte (u3)
  segment (`load_dw`, `r4d_iqr_dec.h:82-104`); kpar weight fragments as two 32-byte supertile loads a
  unit, scales one uint4 a group; head 16 B a lane a pass with one-pass-ahead prefetch; junction float4
  / uint4 throughout. No 2-byte loads on these paths.
- **Trellis continuation reads** (`r4d_iqr_dec.h:276-280`): each lane also reads 4 dwords of the unit
  above (+40 % L2 requests at u5, +67 % at u3); DRAM bytes once when the neighbour block's unit is in
  L2. Non-resident experts read in place run at the link's rate (66 us for 1.82 MB = 27.6 GB/s,
  experts-streaming §0), so the continuation does not cross the link twice. Not a lever unless a
  profile shows L2-bound.
- **Head** (`mvm_kernel<1,..>`, 2048 blocks x 8 waves, a wave a row, 6 passes of 512 B a row at
  K 3072): 186 MB in 0.30 ms = 620 GB/s, at the wall. The 5 draft rounds are serially dependent;
  lossless levers are only K-10 and a fused top-8 epilogue (skips the 242 KB f32 logits write and the
  row_topk launch a round: est. a few us a round, S-M; not worth a ticket before K-10).
- **attn_output / latent_out kpar**: at 1.03 / 1.18 x floor. KW = 4 for any K >= 512; for shared down
  (6 groups) the waves carry 2,2,1,1 groups: balancing (KW 3 or 6) moves bits (new sum order) for
  ~0.5 us: not worth it.
- **iqr kw rule** (`N >= 4096 ? 4 : 8`, `:954`, `:1031`) fixes the sum order per N; it cannot be a
  bit-neutral tune axis. K-3's tiles-per-block trick is the only way to mix kw in one launch.
- **moe_down_dec G** (column tiles a block, `r4d_moe.hip:2705-2717`): 1 below 200 rows; the file's own
  measurement says TP2 is within noise for G > 1. A pin exists (`r4d_moe_gemm_pin_down_dec`) if
  someone wants to re-check at M = 1 (2 residency rounds of 384 blocks at ~11 KB LDS each).
- **Host pool coherence** (`core/device/aql.cpp:4323` says the coarse-grained pool reads at 27 GB/s vs
  15 GB/s fine-grained; `aql.cpp:4513-4515` keeps fine-grained coherence for the pinned pool): the
  measured in-place rate (27.6 GB/s) shows the experts' path already gets the link's rate. Checked.
- **Expert unit alignment vs TLB**: not verified (planner slab classes, `core/place/planner.cpp:457`);
  no evidence of a TLB cost in the measured 2.2 us a resident routed unit (bytes-bound).
- **Chained forms present**: iqr GEMV / m16 / m64 for the trellis forms (`chained_form`, `:895-899`),
  kpar, moe_gemm_iqr (decode band only; the map grid stays plain), moe_down_dec, lm_wmma, both junction
  twins, ihc_pre / ihc_post / gate_mul. Missing on this path: the M <= 2 head (K-10).

## 4. Small decode kernels (quantisers, norms, router, MoE glue, sampler, drafter glue)

Read by a helper auditor and condensed here; F2 and the F18 condition were spot-checked against the
source. Saved time is given either as launches removed (G-3 measured ~0.3-0.5 us each) or as body
latency, which counts in full. Important: the "router_sigmoid 9.8 / scatter_small 3.3" figures are from
the pre-adopt tape. Decode now runs the fused `router_sigmoid_topk_scatter_f32`
(`RADIANCE_HY4_ROUTER_FUSE` default on, `hyv4flash_common.h:319`). From the code its body is about
2-3 us (16 dependent rounds of 64-bit compare-select + DPP max, `r4d_glm.hip:160-199`). Re-measure in
kbench, chained and plain, at M = 1 before ranking anything on the 9.8.

| # | title | where | change / why | lossless | conf / effort |
|---|---|---|---|---|---|
| S-1 | copy16 is the routing-histogram copy | `hyv4flash_moe.h:128` (`w.ecnt` one row, no pitch); `rad_block_moe_fp8.h:2183-2213`; `core/runtime/ctx.cpp:694-696` -> `r4dchain_copy16` | Declare `ecnt` as [n_layer, n_expert] PERSIST plus `rad_route_table` and `ecnt_pitch`, as qwen4exp_fp8.cpp:1001-1008 does. -41 launches a pass. Side risk: today's transient `ecnt` may be aliased (unverified). | yes (counts only) | high / S |
| S-2 | moe_gather serialises its 16 slot loads | `r4d_moe_gather.h:78-86` (runtime `top_k` loop: LDS slot, a 2-byte `ye` load, fmaf) | Template TOPK = 16, issue all 16 loads first, 16-byte column loads, same fmaf order. Est. 3.4 -> ~1 us a layer (-0.1 ms a pass). | byte-identical | med-high / S-M |
| S-3 | no `inv` plane declared | `hyv4flash_moe.h:106-146`; `r4d_moe.hip:3441-3456, 3500` | Above 512 slots (M > 32) gather adds two unchained prepass launches and loses its chained form. Declare `inv` so the router writes it and `moe_gather_inv` runs. Matters for the c8 DFly2 verify. | byte-identical | high / S |
| S-4 | fuse lat_out's quantiser into the gather | gather writes `w.lat.x` (`rad_block_moe_fp8.h:2096-2101`), then a quantiser | A wave per (row, 128-group): sum, bf16, store, `r4d_fwht_wave<128>` + quant as `gated_quant_fp8_body`. -41 launches. Overlaps G-3g / K-11: pick one. | byte-identical | med / M |
| S-5 | the fused router idles 7 of 8 waves at M <= 8 | `r4d_glm.hip:305-328, 425-432` (`dim3(1)`) | Run the latent's had_quant (12 groups) in the idle waves, or as extra blocks of the router GEMM grid. -41 launches, ~1 us hidden a layer. | byte-identical | med / M |
| S-6 | shorten the selection chain | `r4d_glm.hip:170-181` | Pairwise max tree (depth 4) instead of a 16-deep chain, or a sorted list a lane popped per round (rowtopk's `tk_round`, `r4d_rowtopk_bf16.hip:106-120`). Est. 2-3 -> < 1 us a layer. | same winners and order | med / M |
| S-7 | at verify heights the fused router runs on one CU | `r4d_glm.hip:425, 316` (32 waves at M > 16) | Above M = 8: a block per 4 rows plus an arrival counter for the scatter (`r4d_moe.hip:3746-3818` RgInst pattern). Est. -2..-4 us a layer at c8 verify (router 0.68 -> 1.57 ms a pass from c1 to c8 verify). | byte-identical | med-low / M |
| S-8 | G-3f exactness worry is unfounded (KNOWN open) | `r4d_glm.hip:13-77, 116-127` | Every `gemm_nt_f32` tile form has the same per-lane k order and butterfly, so router GEMM + sigmoid + scatter fuse like `router_gemm_topk_scatter_kernel` (`r4d_moe.hip:3669-3767`). -41 launches. Combine with S-7. | byte-identical | high (exactness) / M |
| S-9 | gemm_nt_f32 at 17-63 rows issues 8 two-byte x loads per k step | `r4d_glm.hip:46-47, 125` | Stage x transposed in LDS (48 KiB) with 16-byte loads; fma order unchanged. c8 verify only. | byte-identical | med-low / M |
| S-10 | re-tune the router tile at M = 1, N 512 | `r4d_glm.hip:115-127` (`<2,2,16>` duplicates row 0 at M = 1; 64 blocks x 4 waves) | Try `<1,2,16>` / `<1,1,16>` with `--perff32`. The 6.8 us measured for 6.29 MB is above HBM rate (see §0): re-measure. | byte-identical | med / S |
| S-11 | standalone quantiser: 3 groups serial a wave at K 3072 | `r4d_quant_act_fp8.hip:65, 154` (a block of 8 waves a row, 24 groups) | Grid (M, nkb / 8) or 768 threads so all groups go at once; hoist loads. ~-0.4 us body each. | byte-identical | med / S |
| S-12 | quantiser bf16 loads are 2-byte | `r4d_quant_act_fp8.hip:71-75`; `r4d_fused_quant_fp8.hip:290-294, 368-371` | uint2 loads (K % 128 == 0 under ROT; check pitch % 4 on the host). Small. | byte-identical | med / S |
| S-13 | rmsnorm_had_quant_i8g (and rmsnorm_bf16) re-read the row from global | `r4d_fused_quant_fp8.hip:211-246`; `r4d_model_bf16.hip:270-276` | Stage the 6 KiB row in LDS; same partition order. ~2 L2 round trips (0.3-0.5 us) a call. | byte-identical | med / S-M |
| S-14 | embed_lookup copies 2-byte elements | `r4d_model_bf16.hip:82` | uint4, all loads before stores. 1x a pass + 4x a draft pass. | byte-identical | med / S |
| S-15 | DFly2: f32 -> bf16 `cast` before row_topk | `hyv4flash_dfly.h:761-764`; `r4d_rowtopk_bf16.hip:148-176` | A stage-1 form reading f32 and rounding in registers. -5 launches a draft pass. Check that RNE / NaN match cast_f32_bf16. | byte-identical if the rounding matches | med / S-M |
| S-16 | DFly2 correction chain copies gate and up into `cgu` | `hyv4flash_dfly.h:731-732, 750-753, 1025-1029` | Write the two linears into column slices of `cgu` (pitched output). -10 copy2d a draft pass. | byte-identical | high (copies) / S |
| S-17 | DFly2 draft layers use no fused producers | `hyv4flash_dfly.h:502-546, 675-682` | `rmsnorm_had_quant_iq8r` (RES form absorbs the add), `gated_had_quant_iq8r`, `gate_had_quant_iq8r` all exist and are chained. -15..-25 launches a draft pass. Check that RES add rounding equals add_bf16. | byte-identical | med / M |
| S-18 | shared expert's silu_mul is fused only with a rotated int8 down | `hyv4flash_moe.h:57-58` (`gfuse` needs `fd.rot`) | The iqr recipe's down is `transform=fwht128` (rotated), so this likely does not apply to u3. Check `code_form` on the container; if unrotated, add an unrotated chained `gated_quant` arm. | byte-identical | low-med / S |
| S-19 | row_topk is two launches | `r4d_rowtopk_bf16.hip:217-258` | The last stage-1 block runs stage 2 via an arrival counter. -5 launches a draft pass. | byte-identical | med / M |
| S-20 | sample_topk: one block a row, LDS-atomic contention | `r4d_sample_stages_f32.hip:413-470, 1192` | Wave-aggregated or per-wave histograms, or a multi-block stage 1. Probably most of the CLOSED "sampler 0.2-0.5 ms": code detail only. Separate note: `sample_chain_f32`'s top_p uses float LDS atomicAdd, which is order-dependent (a determinism issue). | byte-identical (int counts) | med-low / M |

Checked, not a lever (small kernels):
- swiglu_clamped_had_quant_i8g: G-3a, CLOSED; chained; only S-12 remains.
- scatter_small no longer runs at decode: router fuse covers up to 64 rows (`rad_block_moe_fp8.h:1724`).
- gemm_nt_f32 at M = 1 is bytes-bound. 16-byte lane mapping would change the sum order. Router on one rank is CLOSED.
- gated_had_quant_i8g / gate_quant: a wave a group, chained.
- The rmsnorm width table is author-measured and partition-fixed.
- copy16 / copy2d16 bodies use 16-byte loads (only their existence matters: S-1, S-16).
- had_quant_act_i8 (per row), r4d_rmsnorm_had_quant_i8.hip, r4d_gated_had_quant_i8.hip, r4d_rmsnorm_quant_i8.hip and pair_argmax are not on Hy4's path.

## 3. Decode-class attention rows (sink, merge, project_kv, expand, row front, indexer)

Read by a helper auditor and condensed here. The T-1 cliff rule and the T-4 dependency were
spot-checked against the source.

Kernel map at TP2 (all chained at decode heights; `r4d_rows.cpp:11939-12002`):
- `mla_sink_rows` (`r4d_mla_sink.hip:147-437`):
  - Sliding (32 heads, window 512, 8 slots), M = 1: (1, 2, 8) = 16 workgroups x 128 threads.
  - Sliding, M 8..16: HG2, (M, 1, 8) x 256.
  - Sparse (16 heads): (M, 1, slots) x 128.
  - Bound by a serial chunk chain (4 / 8 chunks of 16 tokens) plus about 3 dependent prologue round trips. 245 VGPR. Uses WMMA bf16.
- `merge_sink`: (M, heads) x 128.
- absorb (`r4d_mla_kv_bf16.hip:121-229`): (8, H) x 128, with the whole weight slice in registers (KC 6).
- expand / expand_gate: (16, H) x 128.
- `mla_expand_gate_had_quant_i8g` (`:262-345`): (2, H, ceil(M/16)) x 1024.
- row_front: T x 256.
- query rope: (M, 4) x 256.
- qsa_work: ceil(T/4) + S workgroups.
- idx_pool_gated: W x 128.
- qsa_tail_store: T x 256.
- idx_score: (M, ceil(live/64)) x 128.

Absorb and expand are bytes-bound: expand 4.2 MB has a 7.3 us floor against 8.0 measured at 16 heads. Absorb's 3.3 us for 3.15 MB is below its floor, so that stamp is unreliable.

| # | title | where | change / why | lossless | conf / effort |
|---|---|---|---|---|---|
| T-1 | **expand launches sit exactly on the full-card cliff at 32 heads** | `r4d_mla_kv_bf16.hip:530` (grid (2, 32, ceil(M/16)) x 1024 threads = 2048 waves), plain expand / expand_gate (16 x 32 x 4 waves = 2048); rule `r4d_common.h:349-375` (2048 waves or 2017..2047 run ~5x slower; measured 27.6 vs 5.2 us; wp16-2.md:30) | No expand launcher applies `r4d_cliff_rows`. Add one idle workgroup (it returns at once but still reaches chain_done) when the wave count falls in the window. This hits every one of the 30 sliding layers at every decode height. Also on the cliff: absorb 32 heads M 17..32, merge_sink sliding M = 16 / sparse M = 32, sel_sink 16 slots M = 32, expand 16 heads M 17..32. The sliding expand time was never measured (the 8.0 us figure is a 16-head layer). **Check first: a kbench A/B at heads 32, M = 1.** | byte-identical | med (wave count certain; cliff at 1024-thread workgroups unverified) / S |
| T-2 | static kv_b loads before `r4d_chain_wait` | `r4d_mla_kv_bf16.hip:236, 351` (waits); `:169-172, 292-295` (loads) | K-8's idea, here with a large payoff. Expand's whole kv_b quarter (KC 4) and absorb's slice (KC 6) fit in the registers they already use. Issue them before the wait so they stream under the latency-bound sink + merge (~15 us on 16-32 workgroups). Est. up to 7-14 us a sliding layer; absorb gains ~5 us on sparse layers only. | byte-identical | med / S-M |
| T-3 | fold merge_sink into the expand (alternative to G-3 h, KNOWN open) | `r4d_mla_sink.hip:443-482, 673-678`; `r4d_mla_kv_bf16.hip:262-299`; `hyv4flash_attn.h:665-676` | When the sink is split, the expand workgroup (a head, 16 tokens) merges its head's partials into LDS (16 KB) with `merge_sink_body`'s fold, then multiplies. -1 launch a layer (merge 6.1 us measured) and the `lat` round trip. Cost: the merge runs twice, once per column workgroup. | byte-identical | med / M |
| T-4 | merge_sink: value loads wait on the denominator load | `r4d_mla_sink.hip:452-455` | Load `part[s]` unconditionally (the fold already skips den <= 0, `:461`). Load `sinks[head]` early. Store 4 bf16 as one 8-byte store (`:478-481`). Saves one L2 round trip (~0.5-1 us) x 42. | byte-identical | high / S |
| T-5 | expand's gate load sits behind the partial-sum barrier | `r4d_mla_kv_bf16.hip:211` (after `:184`), `:308` (after `:302`) | Load `gv` with the first fragments (4 VGPR). ~0.5-1 us x 42. | byte-identical | high / S |
| T-6 | `stable 1` sink with 17..48 rows falls back to the one-workgroup-a-row walk | `r4d_mla_sink.hip:76` (kRowsWindow 16, kRowsFull 32), `:598`, `:267-311` | That cap was justified by a `stable 0` measurement. Raise the split cap for `stable 1` to 48-64 rows; scratch is 25 MB at 48 rows x 32 heads x 8 slots. Split and fused forms are byte-equal by law. Est. 35-58 -> 5-9 us a sliding layer at c8 verify (~30 layers x 30-50 us a verify step). | byte-identical | med / S |
| T-7 | idx_pool_gated does full work for items with no page | `r4d_idx_pool_gated_f32.hip:72-182` (page check at `:177-180`); `hyv4flash_indexer.h:360-365` | After the wait, read the page and return when no group of the workgroup closes and `out` / `q8` are null. At 1 client 3 of 4 steps close nothing: 8.9 us -> floor, ~50 us a step. | byte-identical | high / S |
| T-8 | idx_pool_gated: overlap the f64 trig with the serial f64 norm sum | `:133-137` (one thread, 128-term f64 sum), `:159-163` (pow / cos / sin after barriers `:138, 145, 148`) | Compute the rotation early in waves 2-3, and drop one barrier. ~1-2 us a closing launch. | byte-identical | med / S |
| T-9 | idx_pool_gated: weight rows read as 512-byte-strided gathers | `:89-105` | Stage j-tiles coalesced and transposed in LDS (32 KB); same mul-add order. Est. 1-2 us of 8.9. | byte-identical | med-low / M |
| T-10 | fuse qsa_work + idx_pool_gated, and the decode-class tail store | `r4d_qsa_work_bf16.hip:57-114`; `r4d_qsa_tail_store_bf16.hip:41-61`; `hyv4flash_indexer.h:354-376` | The pool workgroup finds its item and gathers its keys directly. Rows with qlen <= ring - ratio + 1 cannot alias the gather, so the tail store runs in the same kernel; prefill keeps the separate store. -2 launches a sparse layer. Check the alias proof against the rollback contract (`qsa_tail_store` header `:9-11`). | byte-identical | med / M |
| T-11 | rotate q_rope inside the decode sink kernel (alternative to G-3 e) | `r4d_mla_sink.hip:186-206`; `rope_body` `r4d_model_bf16.hip:662-698`; `hyv4flash_attn.h:523` | The query fragment holds whole gptj pairs, so apply rope_body's exact formula (M-RoPE planes included) on the `stable 1` instance. -1 launch a layer (rope 3.3 us). sel_sink needs the same change. | byte-identical if copied | med / M |
| T-12 | row_front: 3 dependent global phases, 2-byte copies | `r4d_model_bf16.hip:817-830`, `kv_store_body` `:770-782` | Write the cache slot from the registers that produce the values, and use wider accesses. ~0.3-0.8 us of 3.4, x 42. | byte-identical | med / S |
| T-13 | sink: value product over 8 waves a head group (a new 16.4b axis) | `r4d_mla_sink.hip:391-396` | Waves 4-7 take half of the value tiles: the 2-byte `fragT` LDS reads and the value VGPRs a wave are halved. M = 1 leaves 48 CUs idle, so the extra waves cost nothing. | byte-identical | med / M |
| T-14 | row tiles under the lo-anchored law (detail for 16.4b `rt`, KNOWN open) | `r4d_mla_sink.hip:21-29, 337, 392` | Stage [lo_0 + 16c, +15 + rt) once; row i reads WMMA rows from offset i. Until the window fills, the grids are identical. | byte-identical | med / M-L |
| T-15 | sink: one barrier a chunk with triple-buffered keys (HG1) | `:152, 324-401` | ~61 KB of LDS, which conflicts with 16.4b's column-major tile: choose one. | byte-identical | low-med / M |
| T-16 | sink: 64-token super-chunks at window 0 / sel | `r4d_mla_sink.hip:72` (kShiftFull 7, kMaxSplits 20) | kShiftFull 6, kMaxSplits 40: sparse sink ~9 -> 5-6 us, x 12 layers. **Law change** (moves bits once, within the bound; the owner decides). | within bound | med / S-M |
| T-17 | read host-uploaded step metadata before the wait (vector loads) | sink prologue `:162, 313-321`; also qsa_work, tail, row_front, rope, idx_score | Only for operands no kernel in the pass writes. Unverified for DFly2's device-written tables and positions. Takes 1-2 dependent round trips off short kernels. | byte-identical if the precondition holds | low / M |
| T-18 | `r4d_chain_wait` -> `_vec` for kv_project / kvq | `r4d_mla_kv_bf16.hip:236, 351` | Drops `s_dcache_inv` if `tests/chain_scalar_audit.py` agrees. Sub-us. | byte-identical | med / S |
| T-19 | small width fixes (check the ISA first) | sink stores `:411, 424, 432`; query loads `:200-203`; qsa_work `:113`; tail `:61` | Use uint4 / float4 where the compiler did not merge the accesses. | byte-identical | low / S |
| T-20 | idx_score: waves past `nc` still load and run WMMA | `r4d_idx_score_bf16.hip:60-111` | Wave-uniform early-out that stores -inf. Small (the grid is already sized to `live`). | byte-identical | high / S |

Checked, not a lever (attention):
- Duplicate token lanes in WMMA at M = 1 are harmless.
- HG at M = 1 is latency-bound, so HG 2 would not help.
- merge_sink's arithmetic is ~0.5 us; the cost is T-4.
- The `sink_seq` and qsa_work sequence scans are scalar-cache hits.
- Query rope at M = 1 sits at the floor; only fusion helps (T-11).
- Already done: L3 WMMA project_kv, the L4 row front, G-3 c, the stable law, and chaining these rows.

## 5. Ranking (est., lossless only; c1 = one client, no spec)

1. **T-1** expand cliff (S; kbench decides; potentially tens of us a sliding layer).
2. **K-8 / T-2** pre-wait weight prefetch in chained GEMMs and project_kv (S-M; -0.3..-0.8 ms a token together, wide error bar).
3. **K-3 / K-4 / K-5 / K-6** dual-weight launches over shared activations (M, one kernel entry; together est. -14..-20 us a layer, ~-0.6 ms a token).
4. **J-1a** warm fn during the junction's flag wait (S; -0.17..-0.34 ms a token).
5. **K-2** moe_iqr / moe_down_dec block head (M; -0.15..-0.3 ms c1).
6. **K-1** run-aligned moe_iqr grid (M; ~2 % of a c8 DFly2 step), and **T-6** stable split cap (S; c8 verify).
7. Launch removals, each S-M at ~0.3-0.5 us a launch x 41-42: S-1 (copy16), S-3 (inv, c8), S-4 / K-11, S-5 / S-8, T-3, T-10, T-11, K-7, K-10.
8. Body-latency fixes: S-2 gather, T-4 / T-5, T-7 idx_pool idle, S-6 selection, S-11 / S-13 quantiser and norm.

## 6. Format / lossy (the friend's decision)

- Int8 head with an int8-quantised activation (iu8 WMMA / sudot4): lossy (activation quantisation). The head is already at the byte wall, so the gain is VALU only: not worth it.
- Router weight f32 -> bf16: halves 258 MB a pass (~-0.2 ms) and changes routing bits.
- kv_b in a narrower format: absorb and expand are at their byte floor in bf16.
- kv_a bf16 -> int8 / IQR: the 576 rows are no whole 128-row tile (HY4FLASH-QUANT.md).
- attn_output int8 -> IQR (already in levers-general §3).
- iHC `fn` f32 -> bf16: halves the junction's 393 KB a call; changes gates.
- f32 instead of f64 in idx_pool's norm / trig (within bound, not byte-identical).
- T-16's super-chunk law change: the owner's call (moves bits, not a format).
