# B — libr4d kernels on the Hy4-flash PREFILL path (M = 256..2048 rows a step, TP2)

Auditor B. Read-only audit of `src/` at `c655f58c` (branch claude/peaceful-wright-hhqnhi). Nothing was built
or run. Every line number below was read in this tree. "M" in a number = measured by an earlier note (cited);
everything else is arithmetic from the code (marked *est.*). Card facts used: gfx1201, 64 CUs / 32 WGPs,
wave32, bf16 WMMA 16x16x16 = one every 16 clk per SIMD (fp8/i8: every 8 clk, the figure in
`r4d_gemm_8a8_prefill.h:31-34`), 128 SIMDs, sclk 2.1-2.4 GHz under the power limit
(`r4d_gemm_8a8_prefill.h:36-37`), LDS 64 KiB a workgroup / 128 KiB a WGP.

Profile used for shares: `levers-general.md` §1.8 (2436 ms op time for 6681 tokens = 3.26 chunks, rank 0,
`profile-ops`); a "call" below is that total / (3.26 x layers).

---------------------------------------------------------------------------------------------------------

## 0. Summary (top items, ranked by expected prefill gain x confidence / effort)

| # | finding | where | est. gain (prefill) | lossless | conf. | effort |
|---|---|---|---|---|---|---|
| K1 | **FP-8 answered from the code: Hy4's u3 gate_up can never take the packed 128-row run map** — the packed grid is compiled and selected only for `bits == 4 && L == 16`; u3 runs the 64-row run map (4 token tiles a decode), i.e. ~1.3-1.5 decodes (and, for non-resident units, link reads) of each expert per chunk | `r4d_moe_iqr.hip:773, 928` | gate_up -6..-15 % of 532 ms (Flash-Next precedent -6..-9 % at 20-40 K rows; Hy4 has 32 K rows at 64 rows/expert, more doubled entries) = **+1.5..3 %** | yes (packmap bytes = run-map bytes, comment `:686-700`) | high | S-M |
| K2 | `moe_gather` at prefill never takes the wide kernel: `wide_ok` requires bf16 routing weights, Hy4 declares them f32 → 2-byte-a-lane `gather_kernel` with serial slot loads | `r4d_moe.hip:3476`, `hyv4flash_moe.h:126` | 46 ms → ~25 ms *est.* = **+0.8 %** | yes (F32 law: fmaf) | high | S |
| K3 | Junction arithmetic: every row re-reads the whole f32 `fn` table (8 x 12288 f32 = 384 KiB) from L2; a block owns 4 rows at 2048 (512 blocks) but walks them one by one | `r4d_ihc.h:207-262`, `r4d_ar_ihc_junction.hip:107-122` | arithmetic 0.29 → ~0.12-0.15 ms a call x 84 = **-12..-14 ms a chunk (+2 %)** | byte-identical (same per-row sum order) | high (mech.) / med (size) | M |
| K4 | `mla_sink_rows` value product reads V through `fragT` = 8 `ds_read_u16` a lane a fragment (64 a wave a chunk); the score side 18 b64 reads. Column-major value tile (16.4b part 1) is the first lever, not row tiles; sel is **not** at the WMMA floor (WMMA floor 1.4-1.6 ms vs 3.6-3.9 M) | `r4d_mla_sink.hip:392`, `r4d_gdn_wmma.h:85-96` | refines 16.4b: sel+paged -25..-35 % = **+3..5 %** (as G-4) | bit-neutral | high (mech.) | M |
| K5 | `cast` = `hipMemcpy2DAsync` → `aql_copy2d16` with one 256-thread block per **row** and `ceil(width/16B/256)` blocks across: the indexer's copies are 128-byte rows (8 of 256 lanes busy, 32 768 one-row blocks), the attention's 384-byte head rows (24 of 256 lanes, 65 535 blocks) | `core/device/aql_blit.hip:108-119, 260-275`, `r4d_model_bf16.hip:2059-2101` | 0.13-0.15 ms → ~0.02 ms a call (bytes are 4-25 MB) = **+1.0..1.5 %** (41 ms of 2436) | yes | high | S |
| K6 | Remove the casts entirely: (a) `q_b` as one GEMM writing `[T,H,256]`, consumers read pitched heads (the sink kernels already take a head stride), (b) indexer query likewise, (c) `kv_b` GEMM with a pitched A instead of `op_lat_copy`, (d) `attn_kv_sink` reads `kvb` + `row` directly (no `mla_kv_split`) | `hyv4flash_attn.h:427-437, 682-699`, `hyv4flash_indexer.h:233-234, 384-390`, `r4d_mla_sink.hip:531-533` | 2-5 launches a layer-step and their bytes; with K5 most of the time is already gone | yes | high | M |
| K7 | Prefill `mla_attn_*_sink` occupancy is LDS-bound: 41.6 KiB (HG 1) / 45.7 KiB (HG 2) a workgroup = 3 / 2 workgroups a WGP = 6 / 8 waves a CU; the K tile is double-buffered in LDS although the next chunk already sits in registers | `r4d_mla_sink.hip:152-156, 330, 397` | part of K4's gain; single LDS buffer + 3rd barrier → 2x workgroups a WGP *est.* | bit-neutral | med | S-M |
| K8 | `rope` not `rope_rows` at prefill: Hy4 declares `rope` (one accurate `sincosf` a (token, head, pair)) though the byte-identical per-(token,pair) form exists and is opt-in by op name | `hyv4flash_common.h:188-200`, `r4d_model_bf16.hip:1530-1541`, `r4d_rows.cpp:7384` | 32x fewer sincos for q_rope (32 heads), 16x indexer; ~1-2 ms a chunk *est.* | yes | high | S |
| K9 | `mla_expand_gate` at prefill: x (the latent output, 64 KiB a 64-token tile a head) is re-read by each of the 16 column-tile workgroups of a head (1 GiB of L2 a call at 32 heads); and the had_quant fusion is capped at M ≤ 64 | `r4d_mla_kv_bf16.hip:330-365, 530`, `hyv4flash_attn.h:133, 466` | expand 0.4 → ~0.3 ms a call; quant fusion -44 us a sliding/sparse layer = **+0.5..1 %** | yes | med | M |
| K0 | **Row-tiled `idx_score`**: every row re-reads all L/4 pooled keys (512 B a (row, group), no LDS); 16 rows a WG sharing key fragments → 16x fewer bytes. Byte-identical. The largest term that grows with context | `r4d_idx_score_bf16.hip:49-99` | at 47K ~-33 ms a chunk (**+5 %**), ≥ 100 ms at 128K | yes | high | M |
| K10 | Down projection (w4nl64, K = 768 a rank) runs ~6x under the int8 WMMA rate at prefill (Flash-Next: 26 TMAC/s, comment `r4d_moe.hip:2826-2830`); a 64-row entry re-stages its activation tile in each of the 6 N-blocks (256 cols) | `r4d_moe.hip:3117-3131` | unknown; candidate: a block walks 2-6 N tiles over one staged A tile | yes | low-med | M |

Context-length decline (§9): from the kernels, the terms that grow with context are the indexer's per-row
score over L/4 groups and its per-row top-k (both linear in L per row, so O(T·L) a chunk), the score buffer
(T x L/4 f32), and (outside the kernels) the KV footprint taking VRAM from expert residency. `mla_attn_sel_sink`
itself is **flat** in L beyond 2048 (2048 selected keys a row whatever the context).

---------------------------------------------------------------------------------------------------------

## 1. Kernel fact sheet (what runs at M = 2048, TP2)

| op (row) | kernel, file:line | grid x block at M = 2048 | tile / WMMA | LDS | staging | bound (est.) |
|---|---|---|---|---|---|---|
| `mla_attn_paged_sink` (sliding, 32 h, window 512, `stable 0`) | `mla_sink_rows<HG 2>`, `r4d_mla_sink.hip:146-437` | (2048, 1, 1) x 256 (hg rule `:611-616`: 32 heads > 16 and 2048 WGs ≥ 64 → HG 2) | a WG = 1 row x 32 heads; 16-token chunks; per wave per chunk 9 bf16 WMMA (8 latent + 1 rope) + 16 for P·V (hi + lo) = 25 | keys[2][16x584] bf16 37 376 B + partial 8 KiB + small = 45.7 KiB | each row stages its whole window (32 chunks x 18 KiB = 576 KiB) from L2: 1.18 GB a call; 2048 rows of one sequence share 2047/2048 of it | LDS-instruction + latency (§2) |
| `mla_attn_sel_sink` (sparse, 16 h, ctx > 2048) | same, `<HG 1, SEL>` | (2048, 1, 1) x 128 | 1 row x 16 heads, 128 chunks (2048 selected tokens) | 41.6 KiB | 2.36 MB a row, 4.8 GB a call from L2/MALL (the latent of 47 K tokens is 52 MB a sparse layer: MALL-resident) | WMMA floor 1.4-1.6 ms (26.2 M WMMA), measured 3.6-3.9 (M) |
| `attn_kv_sink` (first chunk only) | `attn_kv_sink_kernel`, `r4d_attn_kv_sink_h256.hip:167-360` | (16 q-tiles, 32 heads) x 512 | BQ 128 rows x 1 head, 16-key tiles; per wave per tile 8 bf16 (scores) + 8 f16 (P·V) WMMA | s_k 8.25 KiB + s_vt 12 KiB + s_x 16 KiB = 36.4 KiB | K/V tile loaded once a WG (128 rows) | 3 barriers a 16-key tile (`:244, 268, 284`) |
| `mla_project_kv` absorb | `kv_project_wmma<true, MB 4, NC 2>`, `r4d_mla_kv_bf16.hip:126-270` | (512/128 = 4, 32 h, 32 token groups) x 128 | wave = 2 col tiles x 4 token tiles, K = 192 one chain | none | weight fragments by `global_load_tr_b128` straight to registers: kv_b's key block read once per 64 tokens (32x a call) | L2 |
| `mla_expand_gate` (prefill) | `kv_project_wmma<false, MB 4, NC 1, GATE>` | (16, 32, 32) x 128 | WG = 1 col tile x 4 K-quarters x 4 token tiles; quarters meet in LDS | part[4][4][32][8] f32 = 16 KiB | token fragments re-read by each of 16 col-tile WGs a head | L2 (§5) |
| `mla_expand_gate_had_quant_i8g` | `kv_expand_gate_quant`, `:285-330, 530` | only M ≤ 64 (`kFuseRows`, `hyv4flash_attn.h:133`) | 1 token tile / WG, 1024 threads | 16 + 4 KiB | — | not used at prefill |
| `mla_kv_split` (first chunk) | `kv_split_kernel<8>`, `r4d_mla_kv_bf16.hip:402-428` | N·heads·(56) threads / 256 | pure copy, 16 B a thread | none | reads kvb + rope, writes k, v: 2 x 33.5 MB + 58.7 MB at 32 heads | HBM |
| `mla_row_norm_rope_store` | `mla_row_front_kernel`, `r4d_model_bf16.hip:817, 2789` | (T) x kThreads | a WG a token | — | — | tiny |
| `rope` (q_rope 32 h, idx 16 h) | `rope_kernel`, `r4d_model_bf16.hip:1543-1556` | (M, ceil(H·32/kThreads)) | thread a (token, head, pair): `sincosf` each | — | — | VALU (K8) |
| `gemm_nt_q` IQR u5 (o, gate, q_a, ...) | `iqr_prefill_kernel` → `gemm_8a8_prefill_body`, `r4d_gemm_iqr.hip:767-865`, `r4d_gemm_8a8_prefill.h` | (ceil(M/128), N/128) x 256, raster groups of 8 M bands | 128 x 128 block, 8 waves 4(W) x 2(A), wave 2 x 4 tiles of 16x16, iu8 WMMA, K tile 64 B | 2 x 128 x 72 B + scales ≈ 19 KiB | weight from the fragment plane straight to VGPRs; activation via LDS; weight decoded **once a call** into scratch by the first blocks of each N tile (`IqrPre::begin`), others spin | WMMA / power (o_proj 77 TMAC/s M-derived) |
| `moe_gemm_q` gate_up (u3) | `moe_iqr_kernel<KW 1, RTB 4, MAP, NT 4>`, `r4d_moe_iqr.hip:276-500, 779` | (run-map entries ≈ 1.3-1.5 x 512, 1536/64 = 24) x 128 | wave = 1 weight tile (16 rows) down all K (12 units), each unit decoded (≈300 VALU a lane) and multiplied against ≤ 4 token tiles | 4 x kTrBytes | activation fragments read from global (L2) by every wave of the block (4x) | decode VALU + link for misses |
| `moe_gemm_q` down (w4nl64a8i8h) | `moe_gemm_q_kernel<MT 4, NTL 16, 2, 8, W4 2, AL, SK, UW, I8>`, `r4d_moe.hip:3117-3131` | (entries at 64 rows, 1536/256 = 6) x 512 | 64 rows x 256 cols, K = 768 (6 x 128) | activation + decoded weight staging (not read in detail) | each expert's down slice read once per 64-row entry | latency / prologue (K10) |
| `moe_gather` (prefill) | `gather_kernel<NTHR, F32>`, `r4d_moe_gather.h:47-93` (not the wide one, K2) | (M, n/256) | thread a column, 2-byte loads | 512 B | inverse built by two extra launches (`r4d_moe.hip:3449-3456`) | latency |
| `had_quant_act_i8g` | `quant_act_fp8_kernel<ROT, I8>`, `r4d_quant_act_fp8.hip:52-120` | (M) x 256 | wave = one 128-group, lane 4 values, `r4d_fwht_wave<128>` in registers | none | 4 x 2-byte loads a lane (`:72-75`) | HBM (44 us at 2048 x 3072 ≈ 430 GB/s) |
| `ar_rmsnorm_ihc_junction` (2 ranks) | `ar_ihc_junction_2rank_kernel`, `r4d_ar_ihc_junction.hip:279-330` + `junction_row` `r4d_ihc.h:207` | (min(M, 512) = 512) x 512 → 4 rows a block | row by row, 6-8 barriers a row | n bf16 dynamic + ~1 KiB | `fn` 384 KiB read per row (K3) | L2 |

---------------------------------------------------------------------------------------------------------

## 2. Absorbed sink attention (`mla_attn_paged_sink`, `mla_attn_sel_sink`) — 381 ms of 2436 (15.6 %)

Facts (read in `r4d_mla_sink.hip`):
- Prefill rows run `stable 0`: split form up to `kRowsWindow 16` / `kRowsFull 32` rows (`:76`), above that
  **unsplit** (`:602-604`), one workgroup a row walking all of its chunks, the sink at the store (`:416-426`).
- HG rule `:611-616`: 32 heads → HG 2 (256 threads, one K stage serves 32 heads); 16 heads → HG 1 (128 threads).
- Per chunk and wave: 8 latent + 1 rope score WMMA (`:336-338`), partials through LDS + **barrier** (`:339-340`),
  softmax in registers, P as hi + lo bf16 (`:365-389`), 8 x 2 value WMMA with `fragT` (`:391-396`), stash of the
  next chunk + **barrier** (`:397-399`). 2 barriers a chunk.
- Masked work: none inside a row's range (lo-anchored slots, `begin = lo`, 16-token chunks from it); a window
  row is exactly 32 chunks. No padding rows (one row a WG). Empty-lane waste only at 8-head groups (not at TP2).
- WMMA count (*est.*): sel 2048 rows x 128 chunks x 4 waves x 25 = 26.2 M bf16 WMMA = 205 K a SIMD x 16 clk =
  3.3 M clk = **1.37 ms at 2.4 GHz, 1.56 at 2.1**: measured 3.6-3.9 ms (M) → ~40 % of the WMMA rate. Paged
  (32 h): 2048 x 32 x 2 groups x 4 x 25 = 13.1 M → 0.68-0.78 ms floor against 1.75 ms (M).
  **So neither kernel is WMMA-bound today; and with single-P (16.4c) the floor drops by 32 % (17 of 25).**

### F2.1 Column-major value tile first (16.4b, part 1) — evidence that it is the dominant LDS cost
- Where: `r4d_mla_sink.hip:391-396` uses `fragT(tile, kPitch, ...)`, `r4d_gdn_wmma.h:85-96`: 8 `ds_read_u16` a
  lane a fragment. Per wave per chunk: 8 fragments x 8 = **64 ds_read_u16**, against 18 `ds_read_b64` for the
  9 score fragments and 9 (HG 1) / 5 (HG 2) `ds_write_b128` for the stash. The value side is ~3/4 of the LDS
  instructions a chunk.
- Change: perf.md §3.4.2's `vt[512][16]` tile written at stash time (8 extra b128 stores a staging thread), the
  value fragment one `ds_read_b128` a lane: 64 → 8 LDS reads a wave a chunk.
- Why first: it is bit-neutral, independent of the row-tile bookkeeping (cu_seqlens scan, slot law), helps decode
  too, and removes the largest LDS instruction stream. Row tiles (F2.2) then divide what is left.
- Lossless (same values into the same WMMAs). Conf. high on the instruction counts, medium on the share of time.
  Effort M (LDS budget: +16 KiB; see F2.3 for where to get it). Verify: `tests/r4d_emu/test_mla_sink.cpp`
  bytes vs the current kernel; kbench `mla_attn_sel_sink` / `_paged_sink` rows in `data/kernels_hy4.rkb`
  (34 / 93 cases), with tables from a real `qsa_select` (perf.md §2.2 caveat).

### F2.2 Row tiles (16.4b part 2 / FP-7) — what they buy, quantified
- `rt` rows of one sequence a WG: the score's A operand (keys) and the value's A operand (`vt`) are loaded from
  LDS **once for rt WMMAs** (B = each row's query / P), so LDS reads a WMMA fall by rt; staging (global → LDS) of
  the paged window falls by ~rt (consecutive rows share 511/512 of the window). sel: rows share selected groups
  only partially (union staging as Flash-Next `attn_usel.h`).
- At 32 heads HG 2 x rt 2 = 16 waves (512 threads): VGPR budget (comment `:94`/perf.md: 245 at HG 1) means 1 WG a
  SIMD pair; check spills. 16 heads HG 1 x rt 4 = 16 waves.
- Note for the planner: the WMMA count does not change, so after F2.1 + F2.2 the floor is §2's 1.4-1.6 ms (sel)
  unless 16.4c lands; **G-4's acceptance "sel ≤ 2.8 ms" is reachable by F2.1/F2.2 alone (≈ 1.8x of floor),
  "≤ 2.0" needs 16.4c.**

### F2.3 Single LDS buffer for K (occupancy) — `r4d_mla_sink.hip:152, 330, 397`
- The next chunk is already fetched into registers (`staged[]`, `fetch(buf ^ 1)` at `:330`) and stored only after
  the value product (`:397`); the second LDS buffer exists only to avoid a barrier before that store.
- Change: one `keys[16 x kPitch]` buffer (18.7 KiB), a third `__syncthreads()` in front of `stash`. LDS: HG 1
  41.6 → 22.9 KiB, HG 2 45.7 → 27.0 KiB, which frees exactly the 16.4 KiB F2.1 needs (HG 1: 39.3 KiB with `vt`).
- Occupancy today (*est.*): 128 KiB a WGP / 41.6 KiB = 3 WGs (HG 1, 12 waves, 3 a SIMD); / 45.7 = 2 (HG 2, 16 waves).
  VGPRs (~150-245) allow 6 a SIMD. Prefill is not latency-critical per row; more WGs a WGP hide the 9 global loads
  and 2 barriers a chunk.
- Bit-neutral. Conf. medium (gain must be measured: a third barrier costs). Effort S. Make it an axis (`kb` 1/2)
  instead of a replacement, see §11.

### F2.4 Prefill-only: drop `visible[]`/per-element masking work in interior chunks — minor
- `:347`: every score element tests `visible[buf][...]` (an LDS read a lane a chunk) even when the whole chunk is
  visible (all interior chunks of a window row; all chunks of a sel row except the table's -1 pages). A
  per-chunk "all visible" bit (OR of `staged_ok`) lets the hot path skip 8 LDS-dependent selects. Bit-neutral.
  Tiny (VALU), S. Conf. high on the mechanism, low on the gain.

### F2.5 The paged form's table read for sel prefill rows — minor
- `kSpanPages 160` (`:78`): a sel row's whole range is 512 pages (2048 tokens / block 4), so `spanned` is false for
  every unsplit prefill row and `token_at` reads `table[block * bc]` from global two chunks ahead (`:217-224,
  332`). It is hidden by the 2-chunk lookahead; no change proposed unless a profile shows table stalls.

### F2.6 L6 / 16.4c (single bf16 P for `stable 0` rows) — restated with the WMMA arithmetic
- 25 → 17 WMMA a wave a chunk (`:394-395`, the `pl` product), -32 % of the floor computed above. Lossy within the
  stated bound (moves prefill rows' bits; decode rows untouched): **friend/owner decision**, already planned.

---------------------------------------------------------------------------------------------------------

## 3. `attn_kv_sink_h256` (first chunk, 38 ms = 1.5 %) and `mla_kv_split` (9 ms)

Facts: `r4d_attn_kv_sink_h256.hip`: BQ 128 rows, one head, 16 waves (`:52-58`); key tiles of **16** (`:242`), each
tile: barrier, stash (K row-major, V transposed to f16 `[dim][key]`, `:244-267`), barrier, 8 bf16 score WMMA a
wave, slice partials through `s_x` + barrier (`:282-284`), softmax, 8 f16 P·V WMMA. **3 barriers a 16-key tile**
for 16 waves x 16 WMMA = 256 WMMA. Interior tiles skip the per-element mask (`:306`). Window 512 at BQ 128:
639 keys a WG for 512 a row (20 % masked tiles: the 16.6 `bq` axis, already planned).

### F3.1 32-key tiles (two sub-tiles a barrier pair)
- Stage 32 keys (2 x 8.25 KiB K, 2 x 12 KiB V^T: +20 KiB → 56 KiB, still one WG a CU at 512 threads), run the two
  sub-tiles' score/softmax/PV back to back: 3 → 1.5 barriers per 16 keys. Bit pattern: the online softmax still
  sees 16-key steps in the same order → byte-identical if the max/den update stays per 16-key sub-tile.
- Gain: first chunk only; ≤ 0.5 % of prefill. Conf. medium. Effort S-M. Low priority (like 16.6).

### F3.2 Read `kvb` and the rope row directly; drop `mla_kv_split` — `hyv4flash_attn.h:432-434, 682-699`
- `attn_kv_sink` takes `k`/`v` with row and head strides (`Args` `:71-81`: `k_rs, k_hs, v_rs, v_hs`). `v` is
  exactly `kvb[:, h*448+192 : +256]` → pass `v = kvb + 192`, `v_hs = 448`, `v_rs = H*448`: no copy needed for V
  at all. `k` needs `[nope 192 from kvb | rope 64 from row]`: give the kernel a second base for chunks ≥ 24
  (`fetch` `:226-228`: `ch < 24 ? kvb : rope_row`).
- Removes `mla_kv_split` (one launch, 58.7 MB read + 67 MB written at 32 heads x 2048 rows) on every first chunk.
  Byte-identical. Conf. high. Effort S.

### F3.3 `op_lat_copy` (the n = 512 `cast`) — `hyv4flash_attn.h:427, 684-685`
- Copies the 512 latent columns of `w.row` (pitch 576) into a contiguous `latc` because `gemm_bf16_nt_tiled`
  takes no A pitch (`r4d_gemm_bf16_nt_tiled.hip:222-235`, A must be `[M, K]`). Add an `a_ld` to the bf16 tiled
  GEMM's staging (its loads are 16-byte row pieces; 576 x 2 B is 16-byte aligned). Byte-identical. S.

---------------------------------------------------------------------------------------------------------

## 4. `cast` (41 ms = 1.7 %) — the copy kernel, and which casts exist

Which casts run at prefill (`grep decl_copy`):
- first chunk (non-absorbed) of every layer: `op_lat_copy` (n 512, `hyv4flash_attn.h:427`), `op_qn_copy`
  (H·192 = 6144 cols as head rows of 384 B into a 512-B head pitch, `:436`), `op_qr_copy` (H·64 as 128-B rows, `:437`);
- every selecting step of a sparse layer: the indexer's `op_qn_copy` / `op_qr_copy` (16 heads x 64, 128-byte
  rows into a 256-B head pitch, `hyv4flash_indexer.h:233-234, 384-390`);
- DFly2 conditioning at prefill (`hyv4flash_dfly.h:628-629, 570`): drafter q_nope/q_rope copies, embedding copy.
- `hyv4flash.cpp:964` `op_stream` (embedding into the stream), taps (off).

### F4.1 The copy kernel is mis-shaped for narrow rows — `core/device/aql_blit.hip:108-119, 260-275`
- `r4d_rad_cast` (`r4d_model_bf16.hip:2059-2101`) → `r4d_p2p_copy2d` → `hipMemcpy2DAsync` → intercepted
  (`core/device/aql.cpp:5139`) → `rad_aql_blit_copy2d`: grid `(ceil(w16 / 256), min(height, 65535))`, 256 threads,
  **thread = one 16-byte piece of one row**. For a strided head copy the cast collapses to inner rows
  (`r4d_model_bf16.hip:2077-2095`): indexer qr: 128-B rows → w16 = 8 → 8 of 256 lanes busy, 32 768 blocks;
  attention qn: 384-B rows → 24 of 256 lanes, 65 535 blocks + a second y-loop; n 512 latent: 64 of 256.
  Bytes per call are only 4-25 MB (0.01-0.05 ms at HBM rate); 0.13-0.15 ms (M) is block-launch / occupancy cost.
- Change: flatten: total = height x w16 pieces, `i → (r = i / w16, c = i % w16)` (one `udiv` by a launch constant,
  or a magic reciprocal), grid `min(ceil(total/256), 2048)` grid-stride. Same bytes. Also helps every other
  narrow 2D copy in the engine (DFly2, taps, state copies).
- Lossless. Conf. high. Effort S (core, not libr4d: hand to the core auditor if that is their file).
  Verify: rad-kbench `cast` row if present; else the harness's tap parity (`RADIANCE_HY4_TAPS=1`).

### F4.2 Remove the casts (K6) — refinement of G-3d "q_nope+q_rope dual launch"
- (a) Attention, first chunk: declare `q_b` once (`q_nope` + `q_rope` rows interleaved per head in the checkpoint's
  `[H][192|64]` order) writing `w.q_full [T, H, 256]`; give the absorbed consumers views: `mla_project_kv` absorb
  reads `x` with head stride 256 (its `KvProjectMeta.xh` is already a head stride, `r4d_mla_kv_bf16.hip:56-58,
  148`); `mla_attn_*_sink` reads `q_rope` with head stride 256 (`row_tensor_ok` accepts `stride[1] ≥ width`,
  `r4d_mla_sink.hip:531-533`); `rope` must rotate the trailing 64 of a 256 head (today `rotary_dim` rotates the
  leading dims — **unverified** whether a trailing offset exists; add an `offset`). Saves 1 GEMM launch a layer
  every step and both copies on first chunks.
- (b) Indexer: same for `index_query_{nope,rope}` (`hyv4flash_indexer.h:227-234`): one linear into `[T, 16, 128]`;
  `idx_score` reads it as is. Saves 1 GEMM + 2 casts on every selecting step of 12 layers.
- (c), (d): F3.3, F3.2.
- Lossless (same GEMM rows, same products). Conf. high for (b)-(d), medium for (a) (rope offset). Effort M.

---------------------------------------------------------------------------------------------------------

## 5. `mla_project_kv` (22 ms) and `mla_expand_gate` (51 ms)

Facts (`r4d_mla_kv_bf16.hip`): weight as the WMMA A operand straight from global (absorb: `load_tr_b128`
transposing loads, `:93-96`), tokens as B from global, no LDS except expand's K-quarter meet (`:208-226`).
`mb` law `:436-439`: 4 token tiles a wave from 17 rows (the only rad-tune axis on the path, `tMlaKv`). Absorb
past 16 rows takes NC 2 column tiles a wave (`:385-394`). Grid `(OUT/PER, heads, ceil(M/64))` `:364`.

### F5.1 Expand: token fragments are re-read by 16 workgroups a head
- Expand WG = 1 column tile (16 of 256 output cols) x 4 K-quarters x 64 tokens; each of a head's **16 column-tile
  WGs** reads the same 64 x 512 bf16 token block (64 KiB) → 16 x 2048 x 32 x 1 KiB = **1.07 GB of L2 a call**,
  plus kv_b's value block 32 times (14.7 MiB x 32 = 470 MB). At 0.4 ms a call (*M-derived*: 51 ms / ~126 calls)
  that is ~3.8 TB/s of L2 — the likely bound (the WMMA floor is 8.6 G MAC ≈ 0.14 ms).
- Change: a WG of 16 waves = 4 column tiles x 4 quarters (or 8 x 2), stage the 64-token x 512 block once in LDS
  (64 KiB is too much: stage the WG's K-quarter, 16 KiB, per wave group), or swap roles (tokens as A, more column
  tiles a wave: NC 2-4 with MB 4 = 8-16 accumulators). Token traffic /4. Keep the `((q0+q1)+q2)+q3` order.
- Gain *est.* 0.4 → 0.25-0.3 ms a call x ~126 = -13..-19 ms of 2436 (+0.5..0.8 %). Lossless. Conf. medium. M.
  Verify: `tests/r4d_emu/test_mla_kv.cpp` (walks `mb`), kbench `mla_expand_gate` (48 cases).

### F5.2 `mla_expand_gate_had_quant_i8g` at prefill heights
- Capped at `kFuseRows 64` (`hyv4flash_attn.h:133, 466, 648`) and its kernel takes one token tile a WG
  (`:530`, 1024 threads, kv_b read once per 16 tokens). At prefill the pair `mla_expand_gate` → `had_quant_act_i8g`
  (o_proj's input, 2048 x 8192 bf16 = 33.5 MB written + read, one 44-us-class launch) runs.
- Change: give `kv_expand_gate_quant` an MB axis (2-4 token tiles a WG; VGPR: 1 accumulator a tile a wave) and lift
  the cap. Then F5.1's re-read pattern applies there too.
- Gain: ~50-70 us a layer-step on 42 layers (*est.* +0.4 %). Lossless (G-3c law). Conf. medium. M.

### F5.3 Absorb — nothing big
- NC 2 x MB 4 a wave, weight read 32x a call (6.3 MB x 32 at 32 heads = 200 MB L2), tokens 4x. 22 ms total.
  Checked, not a lever beyond retuning `mb` (§11).

---------------------------------------------------------------------------------------------------------

## 6. Routed MoE at prefill (930 ms = 38 %)

Layout at TP2 (`hyv4flash_moe.h:206-231`, `place`): every expert's width sliced, `n_ff_exp_r = 768`; each rank runs
**all 2048 x 16 = 32 768 routed rows** against its slices: gate_up `[1536 rows, K 1536]` u3 (0.86 MiB a unit),
down `[1536, K 768]` w4nl64. Mean 64 rows an expert.

### F6.1 (K1, FP-8) u3 gate_up never takes the packed map — `r4d_moe_iqr.hip:773, 928`
- `g.packed = fm.bits == 4 && fm.L == 16 && ...` and the packed instantiation is `if constexpr (F::B == 4 &&
  F::L == 16)`. Hy4's gate_up is a 3-bit trellis → the run map at `kMapTiles 4` (64-row entries, `:679`, built by
  `r4d_moe_blockmap_emit`, `:931-934`), `moe_iqr_kernel<F, 1, 4, true, 4, 1>` (`:779`).
- Consequence: an expert with 65-128 rows takes **two entries = two full decodes** of its 16-row tiles (the decode
  is "the cost, ~300 VALU a unit and lane", `:6-7`), and — for a unit not resident — **two in-place reads over
  the link** (each entry's blocks fetch the codes, `D::fetch` `:441-445`; host-mapped memory is, as far as I know,
  not cached in L2/MALL: unverified). Flash-Next at 2048 tokens: 1.3 entries an expert at 64 rows vs 1.05 at 128
  (comment `:681-686`); Hy4's 64-row mean puts more experts across the 64 boundary.
- Change: build the packed path for `B == 3` (and 2, 5, 6 as needed): `R4D_MOEIQR_FORMS` already instantiates
  `X(3, 16)` (`:723-724`); lift the two `B == 4` gates. Check VGPRs at NT 8 (237 at 4 bits, `:686`; a 3-bit
  `Raw` is narrower) — no spill allowed (R4D_WAVES 6).
- Gain *est.*: Flash-Next 4 → 8 tiles: -6..-9 % of gate_up at 17-41 K rows (`:688-692`); plus fewer duplicate link
  reads. On 532 ms: **-30..-80 ms a 3.26-chunk profile (+1.5..3 %)**. Lossless (comment `:686-690`: the packed
  map gives the run map's bytes; `r4d_selftest` compares them byte for byte: `--perfmoeiqr`,
  `R4D_PERF_IQR_PINS`, `r4d_moe_iqr_pin_map_tiles`, `r4d_selftest.cpp:7940, 8297-8303`).
- Conf. high (selection logic), medium (gain). Effort S-M.
- Note: the env vars `RADIANCE_MOE_IQR_MAP_TILES` / `RADIANCE_MOE_DOWN_MT` named in the task **do not exist** in
  this tree; the pins are dlsym test hooks only (`r4d_moe_iqr_pin_map_tiles` `:982`, `r4d_moe_gemm_pin_down_mt`
  `r4d_moe.hip:3232`). FP-8's "kbench check with RADIANCE_..." must use `r4d_selftest --perfmoeiqr` /
  `R4D_PERF_MOE_MT` instead.

### F6.2 Down: `dmt 4` already taken; the block re-stages per 256 columns — `r4d_moe.hip:2841-2845, 3117-3131`
- At 32 768 rows ≥ 32 x 512, `dmt = 4` → 64-row entries, 256-col tile, 16 waves, grid `(entries, 6)`. K a rank is
  only 768 (6 passes of 128): each block pays the map read, run search, activation staging (64 x 768 int8 = 48 KiB
  from L2) and epilogue for 6 K passes. The 6 N-blocks of one entry each re-stage the same activation tile.
- The comment `:2826-2836` records that a block taking several **row** tiles in turn was no faster. Not tried (as
  far as the comments say): a block walking **2-3 N tiles** over one staged activation tile (the decode-side TPW
  idea of `moe_iqr`). Lossless (each element same sum). Conf. low on gain. Effort M. Verify `--perfmoe`,
  `R4D_PERF_MOE_MT`.

### F6.3 Order the run map so non-resident experts' entries dispatch first
- Map entries are in expert order (`moe_iqr_packmap_kernel` / `moe_blockmap_kernel`: exclusive scan over experts).
  A block of a non-resident expert waits ~66 us (M, a unit) on the link while holding a CU slot; resident blocks
  are ~1 us. If the misses cluster in expert-id ranges, they dispatch in bursts; dispatched first, their link reads
  start at t = 0 and overlap all resident work.
- Change: the map builder takes a residency bitmask (the mover knows it; one u32 x 16 a layer) and places miss
  entries first. Same bytes (entries are independent; the output rows are disjoint).
- Gain unknown (depends on how far the link reads already overlap): upper bound is the non-overlapped part of
  ~110 ms of link time a chunk (M, levers-general §1.8). Conf. low-medium. Effort M (core+libr4d). Owner overlap:
  experts-streaming (memory auditor).

### F6.4 gate_up → down "chaining while the unit is in cache" — checked, not a lever
- gate_up and down read **different** weights (the unit's two halves), so a chained launch saves no weight bytes;
  it would only keep the `[rows, 1536]` intermediate on chip, which needs the whole expert's gate_up output for a
  row (all 96 tiles) in one block — a per-expert serial block. The intermediate is 100 MB a layer write + read
  (bf16) and is already halved by the fused `swiglu_clamped_had_quant_i8g` (G-3a, M ≤ 65535 covers 32 768 rows:
  `r4d_rows.cpp:5923-5926`). Not worth it.
- Sorting rows by expert: already (rows are expert-sorted; gate_up gathers 1536-byte int8 token rows, down reads
  sorted rows contiguously, `r4d_moe.hip:624-628`).

### F6.5 (K2) `moe_gather` wide kernel refused for f32 routing weights — `r4d_moe.hip:3476`
- `wide_ok = ew->dtype == RAD_BF16 && ...`; Hy4's `w.ew` is f32 (`hyv4flash_moe.h:126`), so the 2048-token gather
  runs `gather_kernel` (a thread a column, 2-byte `ye` loads, slot loads serialised on the table: "about two thirds
  of cold-stream rate", `:2245-2248`) plus two inverse-table launches (`:3449-3456`).
- Change: `gather_wide_kernel<NW, F32>`: read `ew` as f32 (`wl` as a float lane value), keep `gather_kernel<F32>`'s
  law (`acc = fmaf(w, x, acc)`, `r4d_moe_gather.h:84`) so the bytes match. Also fold the inverse build (already
  given when `moe_scatter_inv` ran: check `gather_inv` path is taken at prefill).
- Gain: 46 ms → ~25 ms (+0.8 %). Lossless. Conf. high. Effort S. Verify: `r4d_selftest` gather pins
  (`r4d_moe_gather_pin_form`, `r4d_moe.hip:2366-2370`) byte-compare form 1 vs 2.

### F6.6 Gather + `had_quant_act_i8g` for `lat_out` (G-3 "moe_down+gather(+copy16)" refinement)
- After the gather the latent `[T, 1536]` bf16 is quantised (`w.lat` codes, `hyv4flash_moe.h:149-151`) for
  `ffn_latent_out`. A gather-wide wave owns 256 contiguous columns = two 128-groups → the rotation
  (`r4d_fwht_wave<128>` over 16 lanes x 8) and amax fit in-wave. Saves one quantiser launch and 6 MB a layer.
  Lossless (G-3 law). Conf. medium. Effort M. Gain ~0.3 %.

### F6.7 Things checked
- `swiglu_clamped` (39 ms in the old profile): the fused G-3a row covers M ≤ 65535 → should not appear in a new
  profile; if it does, `fuse_on("RADIANCE_HY4_FUSE_SWIGLU")` or `hi8` (`rad_fp8.h:655`) failed — check.
- The **all_gather "used by MoeFP8's prefill path" does not exist for Hy4**: no `all_gather` in
  `rad_block_moe_fp8.h` or `arch/hyv4flash/` except DFly2's draft top-k (`hyv4flash_dfly.h:782`).
- MoE sort/scatter at prefill: `scatter_hist/scan/place` (`r4d_moe.hip:333-404`), small; not read in detail.

---------------------------------------------------------------------------------------------------------

## 7. Trunk GEMMs at large M (332 ms = 13.6 %), quantisers (83 ms), norms

Facts: `gemm_iqr_tiled` (`r4d_rows.cpp:9384-9395`, NOTUNE) = the int8 prefill tile (`r4d_gemm_8a8_prefill.h`):
128 x 128 block, 8 waves, weight from a fragment-order plane in VGPRs (never LDS), activation through LDS, K tile
64, kGroupM 8 raster, one `lds_barrier` a staged tile. The IQR form decodes the weight **once a call** into the
op's scratch (`IqrPre::begin`, `r4d_gemm_iqr.hip:782-842`): the first blocks of each N tile claim the 8 16-row
parts, decode, and every block of that N tile spins (`s_sleep 2`) until all 8 are done.

### F7.1 Small-N trunk GEMMs run far below o_proj's rate (M-derived)
- o_proj: 91 ms / 137 calls = 0.66 ms for 51.5 G MAC → 77 TMAC/s. q_a: 49 ms / 137 = 0.36 ms for 6.4 G MAC →
  **18 TMAC/s** (4x worse). q_a is N 1024 (replicated, `RAD_SHARD_NONE`, `hyv4flash_attn.h:221`): grid 16 x 8 =
  128 blocks for 64 CUs (one wave of blocks, no tail balance), and only 8 N tiles x 8 parts = 64 decode parts, so
  half the blocks of each N tile wait in `begin` while 8 decode 16 x 3072 parts each.
- Candidates: (a) a 128 x 64 or 64 x 128 arm for N ≤ 1024 (the int8 prefill tile has **no** tile axis: NOTUNE; the
  w4a8 rows have `tGemmPrefill`/`tGemmTiled` with 14-15 arms); (b) for small-N weights decode all parts before
  the K loop with every block helping (the claim loop already lets any block decode, but blocks of N tile j only
  claim j's parts — a global claim over all N tiles would spread the decode); (c) keep the decoded int8 plane of
  the small replicated weights resident (q_a 3 MiB, kv_a 1.7 MiB, index_* < 1 MiB, lat_in/out 4.5 MiB each at
  int8 → ~16 MiB a layer = 0.7 GiB: a VRAM trade, memory auditor).
- Gain: q_a + kv_a + lat_in/out + index linears at 18 → 50 TMAC/s would save ~25-35 ms of the 332 (*est.*,
  +1..1.5 %). Conf. medium (cause not proven: kbench `gemm_nt_q` at (2048, 1024, 3072) vs (2048, 3072, 8192) first).
- Which row serves kv_a (N 576) and index_head (N 16) at M > 64 is **unverified**: both prefill tiles need
  N % 128 (`cGemmNtIqrTiled`, `cGemmNtI8A8Tiled`, `r4d_rows.cpp:5718-5764`). Check the resolve report.

### F7.2 `had_quant_act_i8g` (83 ms, 44 us a call M) — `r4d_quant_act_fp8.hip:52-120`
- One 256-thread block a row, a wave a 128-group, four **2-byte** loads a lane (`x[xrow + c]`, `:72-75`) → the
  compiler cannot prove 8-byte alignment through `const uint16_t*`; one `uint2` load would quarter the load
  instructions. At n = 1536 (latent: 12 groups) waves 4-7 of 8 run one group, 0-3 two → 25 % idle.
- At 2048 x 3072: 12.6 MB in + 6.3 MB out in 44 us = 430 GB/s (≈ HBM), so the load width matters less than the
  fusions: the remaining standalone calls at prefill are o_proj's input (F5.2), `lat_out`'s input (F6.6), and
  the shared expert's / dense paths. 520 calls a chunk (11 450 / 22) = ~12 a layer: list them from a new profile
  before fusing more.
- Lossless. Conf. high (mechanism), low (gain). S.

### F7.3 `rmsnorm` at 2048 rows — checked
- The standalone norms left on the prefill path (indexer key norm `[T, 128]`, post norms where the junction is
  off) are small; `rmsnorm_rows` exists (`rows_form`, `r4d_model_bf16.hip:1241`) and is opt-in like `rope_rows`.
  Not a lever.

---------------------------------------------------------------------------------------------------------

## 8. Junction arithmetic (`ar_rmsnorm_ihc_junction`, 0.29 ms arithmetic of 0.45-0.67 ms a call, 84 a chunk)

### F8.1 (K3) Share the `fn` reads across a block's rows — `r4d_ihc.h:207-262`, `r4d_ar_ihc_junction.hip:107-122, 1681`
- `nblocks` default = M capped at `R4D_AR_MAX_BLOCKS 512` (`r4d_ar_entry.hip:1681`,
  `r4d_ar_oneshot_2rank_exact.hip:36`) → at 2048 rows a block owns 4 rows and `arij_rows` runs them **one after
  the other** through `junction_row`.
- Per row, step 2 (`:224-242`) reads `h` (4 x 3072 bf16 = 24 KiB) and, for the TAIL form, `acc_piece` reads the
  f32 `fn` rows: R = 2·HC = 8 rows x 12 288 f32 = **384 KiB a row** (`acc_piece`, `:41-66`). 2048 rows → 805 MB of
  L2 reads a call; at 0.29 ms (M, t4.md) that is ~2.8 TB/s: the arithmetic is L2-bound on `fn`.
- Change: process the block's rows in groups of RT = 4: step 1 (the norms) per row into RT `ybuf`s (4 x 6 KiB LDS),
  step 2 loops pieces once, loads `fn` once a piece and accumulates `acc[RT][R + 1]` (36 floats), `h` per row; the
  `finish`/collapse/norm/quant per row as today. Every row's sums keep their order (piece order per thread, same
  wave/LDS reduction) → **byte-identical**.
- Gain *est.*: `fn` traffic /4 → the arithmetic toward its `h`/row bytes (2048 x (24 KiB x 2 + 12 KiB) ≈ 125 MB):
  0.29 → 0.10-0.15 ms, **-12..-16 ms a chunk (+2..2.5 %)**. VGPR +27; LDS +18 KiB (fine at 512 threads).
- Also applies to the plain `rmsnorm_ihc_junction` row (TP1) and the 4-rank twin. Conf. high (mechanism and
  bytes), medium (time). Effort M. Verify `tests/r4d_emu/test_ihc.cpp` (bytes vs the chain), kbench
  `rmsnorm_ihc_junction` (69 cases in `kernels_hy4.rkb`), TP2 run vs TP1 by `work/wp8/cmpdirs.py`.

### F8.2 Barriers a row
- `junction_row` has 6-8 `__syncthreads` a row (`:211, 218`, `finish` x2, `norm_sum` x2 a norm, `:248, 256`);
  with RT rows batched, the barriers of step 1 and 3 can be shared across rows (one reduction tree for RT
  sums). Folded into F8.1.

---------------------------------------------------------------------------------------------------------

## 9. Indexer, Engram, router, embedding at prefill — and why prefill slows with context

A second reader went through this section's kernel bodies for me. I did not re-check its line numbers. Its
times are estimates that rest on one L2-rate anchor (`notes/idxint8.md` `--perfidx`: 33.5 MB in 9.6 us, about
3.5 TB/s) and on the select timings in the code comments (`r4d_qsa_select_bf16.hip:369-372, 392-393`).

### 9.1 Facts

| kernel | file:line | grid @ 2048 | block | tile / LDS | bound (est.) |
|---|---|---|---|---|---|
| `idx_score_bf16` | `r4d_idx_score_bf16.hip:49-99`, launch `:172-175` | (M, ceil(live/64)); at 47K: 2048 x 184 | 128 | a WG = 1 row x 64 keys; a wave = 16 heads x 16 keys, 8 bf16 WMMA (`:84`); **no LDS** | L2: a (row, group) costs 512 B (256 B key + the row's query re-read every 16 keys, `:72-78`) for 2048 MAC |
| `qsa_select_i32` wide form | `r4d_qsa_select_bf16.hip:417-760` (M > 64, `:51, 1225`) | M | 256 | radix select (12-bit first digit, then 8/8/3 bits), ~17 KiB LDS | memory: per pass, 4 B score + 4 B table a candidate |
| `idx_pool_gated_f32` | `r4d_idx_pool_gated_f32.hip:62-235`, launch `:360-374` | ceil(W/4) = 129 | 128 (a thread a channel) | 4 groups, 10 KiB; scalar f32, f64 `pow` / `cos` / `sin` | latency / occupancy |
| `qsa_work_bf16`, `qsa_tail_store_bf16` | `r4d_qsa_work_bf16.hip:44-115`; `r4d_qsa_tail_store_bf16.hip:39-60` | 512+S; T (only S·ring do work) | 128; 256 | 2-byte copies | trivial |
| `gemm_nt_f32_bf16` (router) | `r4d_glm.hip:17-127` | TP2 and M ≥ 256: N 256 a rank (`rad_glm_tp.h:43, 110-141`), grid (16, 256) | 128 | wave 8 rows x 4 cols, lane sums k ≡ l mod 32, scalar FMA | load-issue bound: 12 loads per 32 FMA; 0.43 ms (M-derived) = ~7.5 TFLOP/s |
| `engram_gate_bf16` | `r4d_engram_bf16.hip:230-303` | T | 256 | scalar, 4 barriers, f32 weights re-read for every row | near HBM |
| `engram_norm_bf16` / `embed_lookup_q_bf16` (Engram gather) | `r4d_engram_bf16.hip:178-198`; `r4d_model_bf16.hip:115-157` | 4096 / 3072 | 256 | 2-byte loads; the norm reads its row twice (`:188, 196`) | the gather's `raw` plane is host-domain (`hyv4flash_engram.h:74-78`): goes over PCIe |
| Engram GEMMs `gemm_bf16_nt_bias_tiled` | `r4d_vit_bf16.hip:451, 827-842` | 128x128 | — | bf16 WMMA, double-buffered `s_a` | compute (48 GMAC a layer, replicated) — internals not read |
| `embed_lookup_bf16` | `r4d_model_bf16.hip:69-90` | T | 256 | 2-byte copies | ~25 MB once a chunk |

### 9.2 Why prefill slows from ~4000 tok/s at 1.5K to ~3000 at 47K

- **Turns on above 2048** (`dense_limit`, `hyv4flash.cpp:1091`; `p.select`). At 1.5K nothing on the selecting path
  runs: no index_query GEMMs, no rope, no casts, no idx_score, no qsa_select, no `mla_attn_sel_sink`. The 1.5K
  baseline also uses the first chunk's cheap non-absorbed attention on every layer. Fixed costs that switch on
  past 2048:
  - sel attention: 12 x 3.6-3.9 ms (M) ≈ 45 ms a chunk, against the window-0 paged form it replaces;
  - absorbed attention on the 30 sliding layers for every chunk after the first: paged 1.75 ms against
    `attn_kv_sink` ~0.3-0.8 ms, plus absorb/expand;
  - the indexer query side.
- **Grows linearly in L per row** (so O(T·L) a chunk, O(P²) over a prompt):
  - **idx_score**: every row re-reads every pooled key from L2/MALL. No row sharing, no LDS.
  - **qsa_select**: 1 pass at n ≤ 511, 2 passes for 511 < n < 6144, a 2048-key sample + 1 pass from n ≥ 6144.
  - **The f32 score buffer**: T x L/4.
- **Not growing in L**:
  - `mla_attn_sel_sink`: 2048 selected keys a row at any context. It may lose MALL locality at large L: the
    sparse latent cache is ~54 MB a layer at 47K against the 64 MB MALL (unverified).
  - paged window 512.
  - block tables (a row's table slice is 32 pages for the window).
- **Outside the kernels (hypothesis, memory auditor)**: the KV cache grows ~12 x 1152 B + index keys a token. At
  47K that is ≈ 0.68 GB a rank that expert residency cannot use: ~460 more non-resident 1.46 MiB units. If
  residency shrinks with the KV loan, that is more 66-us link reads a chunk. Unverified.

| L | G = L/4 | idx_score L2 bytes a layer | score buffer | est. score ms a layer | est. select ms a layer | indexer ms a chunk (x12) |
|---|---|---|---|---|---|---|
| 2K | 512 | 0.54 GB | 4.2 MB | 0.15 | ~0.04 | ~2.3 |
| 8K | 2048 | 2.1 GB | 16.8 MB | 0.6 | ~0.09 | ~8.3 |
| 32K | 8192 | 8.6 GB | 67 MB (≈ MALL) | 2.4 | ~0.24 | ~32 |
| 47K | 11750 | 12.3 GB | 96 MB (> MALL → HBM) | 3.5 | ~0.30 | **~46** |
| 128K | 32768 | 34 GB (keys 8.4 MB > L2) | 268 MB | ≥ 9.8 | ~0.59 | ≥ 125 |

Reading at 47K: the chunk goes 512 → 683 ms (+171). The indexer's L-dependent part is ~46 ms of that, ~42 of
it idx_score (~7x above its WMMA floor of 0.51 ms a layer). The remaining ~125 ms is the switched-on fixed costs
above plus the unverified residency effect. Profile per op at 2.5K and 47K to split "turns on" from "grows".

### 9.3 Findings

**F9.1 Row-tiled `idx_score` (largest context-dependent lever)** — `r4d_idx_score_bf16.hip:49-99, 172-175`
- Mechanism:
  - A WG of 8 waves takes R = 16 rows of one sequence x 256 keys.
  - Each wave keeps its 16-key B fragments in registers (32 VGPR, as today) and loops over the R rows.
  - Each row's 16 x 128 query (4 KiB) is staged once in LDS (double-buffered, 8 KiB).
  - Each (row, 16 keys) is the same 8-WMMA chain and the same epilogue (relu, weight, swap16), so the output
    is byte-identical.
  - A tile that crosses sequences takes the per-row path.
- Gain: bytes per (row, group) 512 → ~32 B. At 47K: 12.3 → 0.8 GB a layer, ~3.5 → 0.6-0.8 ms, about **-33 ms a
  chunk (+5 % at 47K)**, ≥ 100 ms at 128K.
- Lossless. Conf. high / effort M.
- Verify: `idx_score_case` (`r4d_selftest.cpp:28656`) old-vs-new bytes; `--perfidx 49152 2048`; kbench
  `idx_score` (8 cases).

**F9.2 Causal skip in idx_score** — `:64, 98`
- Mechanism: pass `pos` and skip tiles with b0 ≥ (pos+1)/4. qsa_select never reads columns at or past `own`
  (`r4d_qsa_select_bf16.hip:451-453`). Keep the -inf fill when the `t_score` tap is on.
- Gain: the wasted fraction is 256/G: 50 % at 2K, 2 % at 47K.
- Lossless for the selection. S. Fold into F9.1.

**F9.3 Keep the score buffer in MALL by row slices** — `hyv4flash_indexer.h:390-409`
- Mechanism: issue score + select per 1024 rows (256 at 128K). Needs a row-offset parameter in both kernels,
  because they look up sequences by absolute row in `cu`.
- Gain: ~4 ms a chunk at 47K, ~12 at 128K.
- Lossless. Conf. medium. M.

**F9.4 qsa_select: no block-table load in threshold passes** — `r4d_qsa_select_bf16.hip:466, 484-485, 556, 578`
- Mechanism: `keyof()` reads `tb[b]` only to zero absent pages, which idx_score already writes as -inf. Add a
  `masked 1` option for Hy4 (not the default: other scorers may not mask). Halves the select's loads.
- Gain: ~1 ms a chunk at 47K. Lossless given idx_score's contract. S.

**F9.5 Router f32 GEMM (17.8 ms a chunk)** — `r4d_glm.hip:17-78, 126`
- (a) Stage x as f32 in LDS once and widen the wave tile to 16 x 8, keeping each lane's k order and the butterfly:
  byte-identical. *Est.* 1.5-2x.
- (b) A three-part bf16 split + WMMA: f32-class but not bit-equal (lossy list).
- Conf. medium. M. Verify `--perff32 256 3072 2048`, `f32_forms`.

**F9.6 Engram gather into a host-domain plane** — `hyv4flash_engram.h:74-78`, norm `r4d_engram_bf16.hip:188-197`
- Mechanism: with the table on the card, declare `raw` (and `ids`) device-domain, or fuse gather + norm (a wave a
  (row, head)). Keep the norm's ≤ 3 values a lane in registers instead of re-reading them.
- Gain: ~2-3 ms a chunk (PCIe write + double read). Conf. medium (mapping unverified). S-M.

**F9.7 idx_pool_gated** — `r4d_idx_pool_gated_f32.hip:89-105, 161, 360`
- 129 WGs on 64 CUs. Each lane reads its own 512-B weight row, so a float4 load touches 32 lines; 16.5 MB of L2
  a call.
- Fixes:
  - a transposed weight copy, keeping j ascending (byte-equal);
  - 2 groups a WG;
  - an `inv_freq` table (byte-equal only if host and device `pow` agree after the cast: unverified).
- Gain: 0.2-0.5 ms a chunk. Conf. low. S.

**F9.8 Small launch waste**
- `qsa_tail_store` launches T WGs where only S·ring work (`:41-51`).
- `qsa_work` could fold into the pool's load.
- `embed_lookup_bf16` uses 2-byte copies; use uint4 (`r4d_model_bf16.hip:80-84`).
- `engram_gate` re-reads its 160 KB of f32 weights a row and uses 2-byte loads (`r4d_engram_bf16.hip:250-301`).
- Total ~0.1-0.3 ms a chunk. Lossless, S.

Checked, not a lever: `router_sigmoid_topk` (4 MB), `engram_ids`, the token embedding, and qsa_select's algorithm
(already a sampled radix select; its floor is reading the score row once).

---------------------------------------------------------------------------------------------------------

## 10. Format / lossy (friend's decision, listed only)

- 16.4c single-bf16 P for `stable 0` rows (F2.6): -32 % of the sink attention's WMMAs; moves prefill bits within
  the bound.
- L9 quantised latent cache (perf.md §3.8): halves sel's staged bytes; lossy.
- fp8/int8 indexer (`RADIANCE_HY4_INDEXER=fp8|int8`, already built): halves/quarters idx_score's key bytes, moves
  near-ties.
- `attn_kv_sink` `bq 64` (16.6) moves prefill bits within the bound (E4).

---------------------------------------------------------------------------------------------------------

## 11. rad-tune axes on the prefill path (and those missing)

Present and relevant (none has a recorded gfx1201 Hy4 tuning in the notes I read):
| row | axis | values | default | note |
|---|---|---|---|---|
| `mla_project_kv_bf16`, `mla_expand_gate_bf16` | `mb` | 0, 1, 2, 4, 8 | 0 = 1 to 16 rows, 4 above (`r4d_mla_kv_bf16.hip:436-439`) | measured "at 8-64 heads, M 1-1024"; never at M 2048 x 32/16 heads with the gate fold |
| `gemm_bf16_nt_tiled` (kv_b first chunk; Engram bf16) | `tile` | 0, 1 (128x128), 2 (64x64) | 0 = by grid | |
| `rmsnorm_quant_i8*` | `reg` | 1, 0 | 1 | decode heights only |
| all-reduce / `ar_rmsnorm_ihc_junction` | `nblocks`, `nthreads`, `drain`, `acq`, `ts_min` | see `r4d_rows.cpp:6310-6331` | 0 (= M capped 512), 1024, 3, 0 | rad-tune cannot reach two-rank rows (comment `:6300-6305`); `RADIANCE_TUNE=` at serve |
| `row_topk` | `nch`, `pt1` | 0/1024/2048/4096, 2/4/8/16 | 0, 8 | DFly2 head |

Missing (hard-coded choices that should be axes, all bit-neutral):
- `mla_attn_*_sink`: `hg` (only the test hook `g_sink_form`, `r4d_mla_sink.hip:93-96`), and the planned `rt`, `sw`,
  plus F2.3's K buffers `kb` {1, 2}.
- `gemm_iqr_tiled` / `gemm_i8a8_tiled`: no tile arm axis (the w4a8 rows have 14-15 arms); F7.1.
- `moe_gemm_q` (IQR): map tiles {4, 8} (test hook only), KW/RTB; down: `dmt` {2, 4} (test hook only).
- `moe_gather`: form {1, 2} (test hook only).
- `attn_kv_sink`: `bq` (planned 16.6).
- `had_quant_act_i8g`: none needed.

---------------------------------------------------------------------------------------------------------

## 12. Checked and found already optimal / not a lever

- The int8 prefill tile's layout (`r4d_gemm_8a8_prefill.h:22-58`): a recorded list of measured alternatives; no
  new idea from the code beyond F7.1's small-N case.
- IQR decode "per row tile": the prefill GEMM decodes each value once a call (not per M tile); the MoE IQR decodes
  once per run-map entry (F6.1 is the remaining duplication).
- `mla_project_kv` absorb (F5.3); `mla_row_norm_rope_store` (one WG a token, already the fused front).
- MoE row sort (expert-sorted, contiguous) and gate_up/down chaining (F6.4).
- Sel attention's per-row cost does not grow with context beyond 2048 (§9).
- `swiglu_clamped` at prefill: fused (G-3a) at every height used.
