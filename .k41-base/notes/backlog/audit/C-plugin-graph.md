# C: the Hy4-flash plugin graph (arch/hyv4flash, arch/common MoE/FP8, plugin/spechub)

Auditor C, read-only. Source: `/root/work/claude-scratch/hy4flash/src` @ c655f58c (branch
claude/peaceful-wright-hhqnhi). All `file:line` below are in that tree and were read for this note.
Abbreviations: **R** replicated at TP2 (both ranks compute the same thing), **S** sharded,
**M-dep** cost grows with rows, **B** cost is weight bytes (row count barely matters at decode heights),
**F** launch-floor op (tiny work, ~3.2 us chained dispatch; per notes/g3.md a removed chained launch is
worth only ~0.3-0.5 us of layer time). "chained" = has a chained (barrier-free) form on the tape;
after hy4-adopt + G-3 a decode pass has 1330 dispatches, all chained except the head and the four
Engram key GEMMs (`vit_gemm_wa`) (levers-general §1.7, g3.md).

Container assumed: u3 (`hy4-flash-iqr-v4-u3.rad`): trunk linears IQR u5 / int8 (attn_output), kv_a and
kv_b bf16, indexer linears bf16, router f32 (declared; **bf16 in the checkpoint**, see F1), Engram
table E4M3 rows, head int8, DFly2 linears IQ8R.

---------------------------------------------------------------------------------------------------

## Part 1. Ordered op lists

### 1.0 Step dispatch (hyv4flash.cpp `step`, 1106-1213)

`draft_pass > 0` -> `Dfly::query` (1118); `draft_pass < 0` (context pass) issues nothing (1113-1119);
`enc` -> vision; else the trunk step below. Pass key (1249-1259): decode steps recorded under
`1 | select<<1 | nonabs<<2`, draft query pass `9`, context pass `17`, prefill / mixed passes
unrecorded (0) - so **prefill and mixed steps are issued op by op from the host, every step**.

### 1.1 Front of a trunk step

| # | op | site | operands | switch | fused / chained | TP2 | cost |
|---|---|---|---|---|---|---|---|
| 1 | `embed_lookup` | hyv4flash.cpp:1137 | token_ids (raw) [T], `token_embd` bf16 -> `x0` [T,3072] | - | chained | R (table replicated) | M-dep, tiny |
| 2-4 | media: `cast` copy, `all_reduce` (exact), `scatter_rows` | 1140-1144 | `mm_embd` raw -> `mm_rows`; mm_rows idx | only with n_mm_rows | - | AR across ranks (rank 0 holds rows) | media steps only |
| 5-8 | `cast` bf16->bf16 x4 (`op_stream`) | 1147-1148 (decl 964) | `x0` -> column j of `streams` [T,4*3072] | - | 4 separate launches | R | F, M-dep (see F12: one `hc_enter`) |
| - | taps (`t_embed`) | 1146 | - | RADIANCE_HY4_TAPS | no-op when off (common.h:463) | | |

### 1.2 Engram (in front of layers 1, 11, 21, 31) - hyv4flash_engram.h `Engram::step` 328-395

| # | op | site | operands | switch | fused / chained | TP2 | cost |
|---|---|---|---|---|---|---|---|
| E1 | `engram_ids` (decode form, `num_accepted`) | engram.h:352 | token_ids, CONV state `kv_engram` (this layer's slot), cu, positions, `engram_token_map` i32, `mult` i64, `vocab_sizes`, `offsets`, num_accepted, state_index -> `eg.ids` [T,16] i32 (**host domain**), `eg.mask` [T] | - | one per Engram layer (4 a step) | R | F |
| E1' | `engram_ids` (prefill form, + `ahead` ids of up to 2 x max_tok following prompt rows) | engram.h:365-368 | as E1 + `eg.ids_ahead` (host) | RADIANCE_HY4_ENGRAM_AHEAD (on) | | R | M-dep |
| E2 | `embed_lookup_q` | engram.h:371 | ids [T*16], table E4M3 [rows,96] + scale -> `eg.rows` [T,1536] (**host domain**) | --ngram-placement (disk: libavx host gather = host join) | | R (both ranks gather over their own uplink) | M-dep, link |
| E3-E4 | `quant_act_fp8` (group 96) + `dequant` | engram.h:376-377 | E4M3 round trip of a bf16 table | only bf16 table (`engram_fp8_embedding`), not u3 | | R | |
| E5 | `engram_norm` | engram.h:380 | rows -> `eg.normed` [T,1536], `head_gain` f32 | RADIANCE_HY4_ENGRAM_EPS | chained | R | F |
| E6 | `gemm_nt_bias` key | engram.h:382 | normed [T,1536] x key [12288,1536] bf16 (37.7 MB) + f32 bias -> `eg.key` [T,12288] | - | **unchained (`vit_gemm_wa`, barrier)** | R | B (37.7 MB) |
| E7 | `gemm_nt_bias` value | engram.h:384 | normed x value [3072,1536] bf16 (9.4 MB) -> `eg.value` [T,3072] | - | | R | B |
| E8 | `engram_gate` | engram.h:387 | key, val, mask, gains; streams h inout | - | chained | R | F, M-dep |

The junction behind the previous layer stops at `ihc_post` (tail 0, hyv4flash.cpp:772-776), so an Engram
layer also issues `ihc_pre` + `rmsnorm` (1160, 1164) and q_a quantises its own input (no junction codes).

### 1.3 One trunk layer, decode step (T = rows, sliding MoE layer first), post adopt + G-3

| # | op | site | operands | switch | fused / chained | TP2 | cost |
|---|---|---|---|---|---|---|---|
| L1 | (`ihc_pre` + `rmsnorm` attn_norm) | hyv4flash.cpp:1160,1164 | only layer 0 and the 4 Engram layers; else inside the previous junction | RADIANCE_HY4_JUNCTION | | R | F |
| L2 | `attn_q_a` GEMM (iqr_gemv u5) | attn.h:502 | codes `trunk_codes.junc_attn_in` (written by the junction) -> `q_a_raw` [T,1024] | FUSE_JUNCTION_Q, SHARE_CODES | chained, no quantiser | R | B (~2 MB) |
| L3 | `rmsnorm_had_quant_i8g` (q residual) | attn.h:503-505 (decl 235-243) | q_a_raw, gain -> `q_a` bf16 + `q_res` codes | FUSED_QUANT, CHAIN_Q8 | fused norm+quant | R | F |
| L4 | `attn_q_nope` GEMM | attn.h:509 | q_res codes -> `q_nope` [T,H*192] | (FUSE_QPROJ documented, **never read**: F10) | | S (heads) | B |
| L5 | `attn_q_rope` GEMM | attn.h:510 | q_res codes -> `q_rope` [T,H*64] | | unordered beside L4 | S | B/F |
| L6 | `attn_kv_a` GEMM (bf16 in u3) | attn.h:511 | `block_in` bf16 -> `kv_row` [T,576] | | | R | B (3.5 MB bf16) |
| L7 | `mla_row_norm_rope_store` | attn.h:517 (decl 284-299) | row, kv gain, rope pos, slot_mapping, cache | RADIANCE_HY4_CACHE_ROW (on; off under taps or streamed KV -> rmsnorm 520 + rope 525 + kv_store 529 / kv_store_streamed 542) | fused (WP16.5) | R (latent cache replicated) | F |
| L8 | `rope` (gptj / imrope) on q_rope | attn.h:523 | q_rope in place | - | separate (G-3e open) | S | F |
| L9 | `attn_gate` GEMM (u5, 7.9 MB) | attn.h:550 | junction codes -> `attn_gate` [T,H*256] | | | S | B (18.5 us) |
| L10 | `mla_project_kv` absorb | attn.h:650 | q_nope x kv_b (bf16, per-head) -> `q_latent` [T,H*512] | | | S | B (kv_b 7.3 MB a layer shared with L13) |
| (L10b) | `kv_resolve` | attn.h:661 | streamed sparse group only | --kv-resident | | R | |
| L11 | `mla_attn_paged_sink` stable 1 (decode rows [0,DT)) | attn.h:563 | q_lat, q_rope, cache, block_table, seqused, cu, sinks -> `attn_latent` | - | 2 dispatches (`mla_sink_rows` + `merge_sink`, G-3h open) | S | M-dep (keys) |
| L11' | sparse layer, selecting step: `mla_attn_sel_sink` stable 1 | attn.h:584 | + `ix.selected`, `ix.length` | | | S | M-dep |
| L12 | `mla_expand_gate_had_quant_i8g` (T <= 64, all rows absorbed) | attn.h:668 | lat, kv_b, gate -> o_proj codes | FUSE_EXPAND_QUANT, EXPAND_GATE | fused G-3c | S | F |
| L12' | else `mla_expand_gate` (671) or `mla_project_kv` expand (674) + `gate_mul` (682) | | | | | S | |
| L13 | `attn_output` GEMM (int8, 12.8 MB) + its quantiser unless L12 | attn.h:685 (rad_fp8.h:1926-1931) | -> `o_proj` [T,3072] partial | | | S (columns) | B (22.8 us) |
| L14 | `ar_rmsnorm_ihc_junction` (j_mid) | hyv4flash.cpp:1176 -> common.h:654-664 | o partial, post gain, streams, hc_ffn fn/scale/base, ffn_norm -> `block_in` + `junc_ffn_in` codes | JUNCTION, FUSE_JUNCTION_Q | fused AR + post norm + ihc_post + ihc_pre + norm + quant | AR; arithmetic R | link ~13 us |
| L15 | `ffn_latent_in` GEMM | moe.h:390 | junction codes -> `moe.latent` [T,1536] | | | R | B (~3 MB) |
| L16 | `gemm_nt_f32` router (rows < 256) | moe_fp8.h:1883 -> glm_router.h:83 | `block_in` bf16 x router **f32 [512,3072] = 6.29 MB** -> logits f32 [T,512] | - | | R | B (F1) |
| L17 | `router_sigmoid_topk_scatter` | moe_fp8.h:1884 | logits, correction -> ids, ew, sorted, eoff, ecnt | ROUTER_FUSE | fused select+sort | R | F |
| L18 | `had_quant_act_i8g` (latent, rotated) | moe_fp8.h:1925 (decl 1282) | lat -> `moe.latent_rot` codes | CHAIN_Q8 | separate (reads L15's output) | R | F |
| L19 | `moe_gemm_q` gate_up (u3 trellis) | moe_fp8.h:1974 | codes, 2 class tables, sorted, eoff -> `expert_gate_up` [T*16, 1536] | | | S (768 of 1536 cols each) | B (16 units x 0.88 MiB) |
| L20 | `swiglu_clamped_had_quant_i8g` | moe_fp8.h:1979 | egu -> eff codes | FUSE_SWIGLU | fused G-3a | S | F |
| L21 | `moe_gemm_q` down (w4nl64) | moe_fp8.h:2016 | eff codes -> `expert_down` [T*16, 1536] | | | S (K slice) | B |
| L22 | `moe_gather` | moe_fp8.h:2084 | edn, ew, sorted -> `moe.latent` (partial routed sum) | | separate (G-3g open) | S partial | F |
| L23 | `copy16` route report | moe_fp8.h:2212 -> ctx.cpp:706-707 | `moe.expert_counts` -> core routing row | - | **one copy dispatch a layer, avoidable (F3)** | R | F |
| L24 | `had_quant_act_i8g` (lat_out input) | rad_fp8.h:1926 via moe.h:397 | lat -> codes | | separate | R | F |
| L25 | `ffn_latent_out` GEMM | moe.h:397 | -> `moe.routed` [T,3072] partial | | | R weight, partial data | B (~3 MB) |
| L26 | shared `ffn_gate_up_shexp` GEMM | moe.h:399 -> 88 | junction codes -> sgu [T,1536] | | unordered beside L25 only | S | B |
| L27 | `gated_had_quant_i8g` | moe.h:89 | sgu -> sff codes | FUSED_QUANT | fused | S | F, ordered |
| L28 | shared `ffn_down_shexp` GEMM | moe.h:90 | -> `moe.shared_out` partial | | ordered (X5, F5) | S | B |
| L29 | `ar_rmsnorm_ihc_junction` (j_end, with the add of routed + shared) | hyv4flash.cpp:1193 | a = routed, b = shared_out, ... -> next layer's `block_in` + `junc_attn_in` codes | as L14 | fused add + AR + ... | AR | link |

Sliding MoE layer: 29 dispatches (L11 counts 2). **Sparse layer** adds after L9 (hyv4flash.cpp:1171,
indexer.h `step` 324-422):

| # | op | site | operands | switch | chained | TP2 | cost |
|---|---|---|---|---|---|---|---|
| X1 | `index_key` GEMM (bf16 [128,3072]) | indexer.h:351 | block_in -> `ix.token_keys` [T,128] | | | R | B small |
| X2 | `rmsnorm` key | 352 | in place | | | R | F |
| X3 | `qsa_work` | 354 | tk, positions, cu, tail ring, state_index, index block table -> stage, page, bpos, nc | | | R | F |
| X4 | `idx_pool_gated` (exact; `_fp8` / `_i8` under RADIANCE_HY4_INDEXER) | 359-366 | stage, wv/wg (f32 128x128), ape, gain -> index cache | INDEXER | runs over `work` = ceil(T/4)+S rows even when no group closes | R | F |
| X5 | `qsa_tail_store` | 374 | tk -> tail ring | | | R | F |
| selecting step only (context > 2048): | | | | | | | |
| X6-X7 | `index_query_nope`, `index_query_rope` GEMMs (bf16, K=1024) | 380-381 | `q_a` (bf16) -> iqn [T,16*64], iqr [T,16*64] | | | R | B small |
| X8 | `rope` on iqr | 382 | | | | R | F |
| X9-X10 | `cast` copies iqn, iqr -> `ix.query` [T,16,128] | 387-388 | head-interleaved assembly (F13) | | | R | F, M-dep |
| X11 | `index_head` GEMM [16,3072] | 389 | block_in -> iw | | | R | F |
| X12 | `idx_score` (exact) / fp8: `cast` iq->iqf (no-op relayout, F21), `dsa_query`, `cast` gate f32, `dsa_score` | 393-405 | over `live` columns only (390) | INDEXER | | R | M-dep x context |
| X13 | `qsa_select` (topk 511, own) | 407 | score -> sel, nsel, len | | | R | **154 us / launch at 64K, 305 us at 200K (notes/idxint8.md perfidx)** |
| D1-D2 | DFly snapshot: `cast` i32 sel [T,512] + nsel -> `dfly.sel_snap.<l>`, or `fill` nsel 0 on non-selecting steps | hyv4flash.cpp:1172 -> dfly.h:232-236 | only layers 25/29/33/37/41 with the drafter served | RADIANCE_HY4_DFLY | | R | F; 4 MB a layer at 2048 rows |

Decode sparse, non-selecting: 34 dispatches (+1 fill on a DFly layer); selecting: +8..+10.

### 1.4 Tail of a trunk step

| # | op | site | operands | notes | TP2 |
|---|---|---|---|---|---|
| T1 | last layer's `ar_rmsnorm_ihc_junction` with hc_head collapse + final norm | hyv4flash.cpp:993-998, 1193 | -> `final_hidden` [T,3072] **PERSIST** (616) | final norm fused; runs on **all T rows** | AR |
| T2 | (`ihc_pre` hc_head + `rmsnorm` output_norm) | 1201-1204 | only if the junction is off | | R |
| T3 | `gather_rows` | 1207 | final_hidden, out_ids (raw, n_out) -> `logit_input` [n_out,3072] | logits rows = `batch->n_out` only (non-final prefill rows skipped) | R |
| T4 | `logits_gemm` (int8 head, 186 MB a rank) | 1209 | -> `logits` f32 [n_out, 60416] | **unchained** (barrier) | S (vocab rows) |

The sampler, its staging and `mtp_couple_verify` / `mtp_trav_verify` are the core's (core/sample/sampler.cpp:313-316),
behind T4, not on the plugin's tape.

### 1.5 Prefill chunk differences (T up to max_tok = 2048; pass unrecorded)

- **Attention path** (attn.h `core` 636-693, `plan` hyv4flash.cpp:1075-1104):
  - first chunk (every prefill sequence at position 0, not selecting): **non-absorbed**: `cast` latent
    copy (598), `gemm_nt` kv_b over the step's latent rows (600), `mla_kv_split` (602), two `cast` copies
    of q_nope / q_rope into `q_full` (609-610), `sub_i32` rebase of cu (612), `attn_kv_sink` (614),
    `gate_mul` (682). Decode rows of a mixed step stay absorbed (L10-L12).
  - later chunks, or any selecting step: **absorbed**: absorb (650), `sub_i32` (570), `mla_attn_paged_sink`
    stable 0 (572) or `mla_attn_sel_sink` stable 0 (588), `mla_expand_gate` (671; L12's quant fusion is
    only for T <= 64), `attn_output` with its own quantiser.
  - `sub_i32` is issued **once per layer** (42 a chunk) though cu_p is the same in every layer (F14).
- **Router** (MoeFP8 `pass` 1890-1919): rows >= 256 (`GlmTp::kSplitRows`, rad_glm_tp.h:43) and > the fused
  sort's bound: `GlmRouter::step` -> `SplitF32Proj` (rad_glm_tp.h:135-142): `gemm_nt_f32` over this rank's
  256 router columns + **`all_gather`** f32 [T,512] (2 MB a direction at 2048 rows), then
  `router_sigmoid_topk` and `moe_scatter` (separate). Router GEMM is f32 compute-bound here: 58 ms of
  2436 profiled (2.4 % of prefill, levers-general §1.8) (F2).
- **Routed arm**: `moe_gemm_q` gate_up writes `expert_gate_up` [T*16, 1536] bf16 (**100 MB at 2048 rows**),
  `swiglu_clamped_had_quant_i8g` reads it back; down writes `expert_down` [T*16, 1536] (100 MB) that
  `moe_gather` reads back (F6, G-3g).
- **Indexer**: selecting chunks run X6-X13 over all T rows; `ix.score` is [T, blocks] f32 with
  blocks = max_ctx/4 + 1 (indexer.h:63,81): **134 MB at 64K context, 268 MB at 128K** for 2048 rows (F17).
- **Engram**: as decode plus `ahead` hashing (E1') - only useful for a disk-placed table (F15).
- **Casts**: the 4 stream copies (5-8), the non-absorbed latent and q copies, the indexer's query copies
  (X9-X10), the DFly snapshot copies (D1). levers-general §1.8: `cast` 41 ms of 2436 (1.7 %).
- **Last layer**: all its work after the cache writes runs for all T rows although only `n_out` rows
  (and, with the drafter, the anchor rows) are read (F4).

### 1.6 DFly2 query (draft) pass - hyv4flash_dfly.h `query` 884-1058 (n drafting sequences, M = 6n rows, depth 5)

Stage A/B (once):

| op | site | notes | TP2 |
|---|---|---|---|
| `embed_lookup` M rows | 907 | -> `dfly.embed` | R |
| `cast` emb -> x (M x 3072) | 909 | kept raw for chain k=0 (F24) | R |
| `iota` row0 | 911 | constant 0, 6, 12, ... | R |
| `gather_rows` emb[row0] -> ea | 912 | | R |
| `gather_rows` final_hidden[anchor_rows] -> th | 913 | the trunk tap | R |
| `rmsnorm` x2 into cat halves | 914-915 | | R |
| fusion linear [3072, 6144] IQ8R (quantiser + GEMM) | 916 | 18.9 MB | R |
| `mul_weight` (fusion gate), `add`, `scatter_rows` into x | 917-920 | | R |

Per draft layer (x5; Lkv = Lidx = 25/29/33/37/41), 36 dispatches at TP2 with IQ8R linears (each IQ8R
linear = its own `had_quant_act_i8g` + `gemm_i8`, rad_fp8.h:1926-1931):

`dfly_sel` (926) · `rmsnorm` attn_norm (933) · q_a quant+GEMM (935, R) · `rmsnorm` q_a_norm (936) ·
q_b quant+GEMM (938, S) · 2 x `cast` split of q into q_nope / q_rope (943-944) · `rope` q (945) ·
kv_a quant+GEMM (946, R) · `rmsnorm` latent (947) · `rope` key (948) · `kv_store` into trunk layer Lkv's
cache (955) · gate quant+GEMM (972, S) · `mla_project_kv` absorb (974) · `mla_attn_sel_sink` (985) ·
`mla_project_kv` expand (990) · `gate_mul` (991) · attn_output quant+GEMM (992, S) · **standalone
`all_reduce`** (993) · `rmsnorm` post (996) · `add` (998) · `rmsnorm` ffn_norm (1001) · gate_up
quant+GEMM (1003, S) · `silu_mul` (1004) · down quant+GEMM (1005, S) · **standalone `all_reduce`**
(1006) · `add` (1008).

Stage E: `rmsnorm` output_norm over M rows (1012).

Correction chain, round k = 0..4 (1015-1055), M per launch = n (all drafting sequences batched):
`iota` rowk (1017) · `gather_rows` z[k+1] (1018) · k=0 `gather_rows` emb[row0] / else `embed_lookup` prev
(1019-1022) · `rmsnorm` x2 into ccat halves (1023-1024) · correct_gate quant+GEMM [3072,6144] (1025, R) ·
correct_up quant+GEMM (1026, R; quantises the same ccat again) · 2 x `cast` into cgu halves (1027-1028) ·
`silu_mul` (1029) · correct_down quant+GEMM [3072,3072] (1030, R) · `add` (1031) · **`logits_gemm`**
(1033; head shard 186 MB a rank, unchained) · `cast` f32->bf16 logits (1035) · `row_topk` R=8 with
vocab_off (1037) · **`all_gather`** pairs (1039) · `row_topk_merge` (1040) · `mtp_couple_propose` (1049,
coupled only; spechub.hip:583-588, one thread a row) · `cast` cidx[:,0] -> tokens[:,k] (1053) · `cast`
cidx[:,0] -> prev (1054). = 24 dispatches a round at TP2 coupled.

Pass total ~ 2 + 9 + 5x36 + 1 + 5x24 = 312 (measured 303, levers-general §1.1). Per pass and rank:
5 x 186 MB head + 5 x 47.2 MB correction (R) + 18.9 MB fusion (R) + 5 x ~77 MB draft layers.
Head read is sharded (vocab rows, hyv4flash.cpp:902-903); one `logits_gemm` a round covers every drafting
sequence (M = n); the first draft cannot reuse the trunk logits (they predict the bonus token, the chain's
round 0 predicts the token after it).

Verify side: the verify pass is an ordinary trunk step with depth+1 rows a sequence (the core picks the
depth); every op of 1.3 runs on every row incl. KV stores of rows later rolled back (overwritten next step,
dfly.h:25-27). `mtp_trav_verify` (spechub.hip:945-972) is launched by the core's sampler: one workgroup of
128 a row, non-root rows return at once (spechub.hip:954).

---------------------------------------------------------------------------------------------------

## Part 2. Findings

Ordered by expected value. Estimates use the measured costs of COMMON.md / levers-general.md; where a
number is my arithmetic it says so.

### F1. Router weight declared f32 although the checkpoint stores bf16: 2x the bytes, 129 MiB a pass a rank
- area: plugin + kernel row (memory, bytes)
- location: `arch/common/rad_block_glm_router.h:44-45` (`decl_w(..., RAD_F32, {experts, n_embd})`),
  read by `gemm_nt_f32` (glm_router.h:49-54, 83) / `SplitF32Proj` (rad_glm_tp.h:118-124);
  name map row `ARCHITECTURES.md` "glm_router.weight ... f32".
- fact checked: `work/wp11/rel/model-00071-of-00148.safetensors` header: `model.layers.{1,10..14}.mlp.gate.weight
  BF16 [512, 3072]` (the hc fn and indexer pool weights are genuinely F32 there). HF does
  `F.linear(h.float(), self.weight.float())` (modeling_hy_v4_flash.py:226), i.e. the reference is the bf16
  weight widened exactly.
- change: declare the router at the stored dtype (bf16) and serve it with a `gemm_nt_f32` row (and the
  fused `router_gemm_*` rows when G-3f lands) that loads bf16 weights and widens them in registers, with the
  same f32 product and summation order as today's f32 row.
- why it pays: 41 x (6.29 - 3.15) MB = **129 MiB less read a decode pass a rank** (~0.22 ms at 0.58 TB/s,
  ~1.6 % of a 13.4 ms one-client pass; also every verify pass) and **129 MiB of VRAM a rank** returned
  (~88 routed units of 1.46 MiB more resident at TP2, where misses cost 66 us a unit). levers-tp2 §3
  already counted "258 MiB of f32 router a card".
- lossless: byte-identical (bf16 -> f32 is exact; same products, same order).
- sure: high (header read); effort S-M (libr4d row + libref/libavx schema dtype + plugin declare); verify:
  r4d_selftest router rows bf16-weight vs f32-weight bytes equal, greedy texts equal, `/stats` residency.

### F2. Prefill router GEMM on f32 FMA: use bf16 WMMA with f32 accumulation for prefill rows
- area: kernel + plugin (prefill)
- location: `rad_glm_tp.h:119-124` (split router GEMM, rows >= 256), `glm_router.h:49-54`.
- change: for prefill-class rows only (pass rows >= kSplitRows), a bf16 x bf16 -> f32 WMMA router GEMM.
  Products of two bf16 values are exact in f32, so only the summation order differs from the f32 kernel.
- why: router GEMM is 58 ms of 2436 profiled prefill (2.4 %), f32-FMA compute-bound at 2048 rows
  (0.43 ms a layer a chunk by the profile); WMMA would make it a few percent of that: ~+2 % prefill.
- lossless? moves bits: f32 sum order only (the reference itself is an unspecified-order f32 GEMM); router
  near-ties on prefill rows can flip. Decode/verify rows keep today's kernel (E1: a row's bytes
  independent of M). Needs the user's OK on "prefill rows may differ in sum order" (as G-4's 16.4c).
- sure: medium (profile share measured; WMMA speed estimated); effort M; verify: prefill KL vs HF on the
  harness, prefill-profile router line.

### F3. Routed-count report costs one `copy16` dispatch a MoE layer: declare a routing table
- area: plugin
- location: `hyv4flash_moe.h:128` (`w.ecnt` a plain [n_expert_r] buffer, `ecnt_pitch` 0),
  `rad_block_moe_fp8.h:2203-2212` -> `core/runtime/ctx.cpp:706-707` (async copy when the pointer is not the
  core row). The pattern: `arch/qwen4exp_fp8/qwen4exp_fp8.cpp:1005-1007, 1137` (`rad_route_table`,
  `mw.ecnt_pitch = n_expert_all`); ABI doc `abi/rad_builder.h:840-851`.
- change: declare `moe.expert_counts` as [n_layer(+1), 512] i32 PERSIST, `rad_route_table(b, buf)` once,
  set `MoeWire::moe.ecnt_pitch = 512`.
- why: removes 41 `copy16` dispatches a pass (the "copy16 3.3" of every layer in levers-general §1.2/§1.7);
  also fixes a latent hazard: `w.ecnt` is transient today, and qwen4exp's comment says a transient histogram
  can be aliased by a wide prefill chunk (corrupted counts for the heat engine / file stager).
- lossless: byte-identical (same counts, no op output depends on them).
- sure: high; effort S; verify: `/stats` expert counts identical, dispatch count -41 in a stamped pass.

### F4. Prefill: the last layer computes its post-cache work for every row of the chunk
- area: plugin (prefill)
- location: `hyv4flash.cpp:1153-1211` (every layer over T rows), output rows only at 1207-1210.
- change: in layer 41 (sparse), keep the cache writers on all T rows (q_a ... only what the indexer key side
  and `mla_row_norm_rope_store` need: kv_a, row front, index_key/norm/work/pool/tail) and run everything
  else (q path, indexer query side, attention, o_proj, j_mid, MoE, shared, j_end/final norm) on the
  `n_out` output rows plus, with the drafter, the rows that can be anchors (each prefill sequence's last
  row). A `gather_rows` of the streams' rows [hc*E] into a compact buffer after the cache writes, then the
  rest at M = rows kept; `out_ids` remapped to 0..n-1. A chunk that ends no prompt (n_out = 0, no anchor)
  skips that tail entirely.
- why: a layer is ~14 ms of a ~600 ms 2048-row chunk; the routed MoE of that layer at 2048 rows touches
  nearly every expert (with misses over the link: half of prefill MoE time is link reads). Estimate
  ~2-2.5 % of prefill per chunk (arithmetic, not measured), more on non-final chunks.
- lossless: the kept rows' bytes equal today's only where every op is row-local (GEMMs, router, MoE
  per-row, junction); the prefill attention (`stable 0`) may depend on M -> "moves bits within the prefill
  bound" at worst; KV/index caches byte-identical. Interacts with FP-2 (`ckpt_cut`) and `final_hidden`
  (drafter reads anchors only, dfly.h:913).
- sure: medium; effort M; verify: harness chunked prefill KL, rollback gate, prefill profile.

### F5. X5 refined: Hy4's shared expert never gets MoeFP8's interleaving; two ordered launches a layer
- area: plugin (+ small MoeFP8 hook)
- location: `hyv4flash_moe.h:178` (`c.n_ff_shared = 0`: the shared arm is the plugin's GatedMlp), issued
  after `lat_out` at `moe.h:399` (GatedMlp::step 88-90). MoeFP8's interleave (moe_fp8.h:1855-1861,
  1939-1993, buffers `rad_buf_concurrent` 1571-1605) therefore never applies. With `Ctx::haz_mark`
  (core/runtime/ctx.cpp:1164-1219) only shared gate_up runs unordered (beside lat_out); `gated_had_quant_i8g`
  and the shared down GEMM are ordered launches.
- change: give `MoeFP8::pass` a `SideOps` hook (the pattern of rad_block_attn_gated_fp8.h:194-199,
  rad_arch.h:1527-1543) and pass the shared arm's three stages, issued behind the router, `op_hq` and the
  routed down GEMM; declare the shared arm before `moe.declare` (moe.h:321 vs 326) so its buffers' declared
  lifetimes cover the routed span (or mark sgu / sff codes / sout `rad_buf_concurrent`).
- why: removes ~2 ordered launches a MoE layer (act 3.3 us + down 9.7 us + gaps, levers-general §1.2) from
  the critical path: estimate 0.1-0.3 ms a one-client pass (1-2 %), more at 8 clients where the routed
  GEMMs are longer. MoeFP8's own arrangement measured -0.2..-0.3 % on Flash-Next (a smaller arm).
- lossless: byte-identical (order only; no shared buffer between the arms).
- sure: medium (overlap gain is estimated); effort M; verify: stamped pass, greedy texts equal.

### F6. MoE prefill: the gate_up output makes a 100 MB round trip through HBM a layer
- area: kernel (prefill + decode launch)
- location: `rad_block_moe_fp8.h:1964-1979` (`moe_gemm_q` -> `w.egu` [T*16, 2F] -> `swiglu_clamped_had_quant_i8g`);
  buffer `hyv4flash_moe.h:130`.
- change: a gate_up GEMM epilogue that computes the clamped SwiGLU and the rotated int8 codes (a tile owns
  128 gate columns and the matching 128 up columns, `parts 2`), rounding gate/up to bf16 first exactly as
  the stored `egu` is rounded today.
- why: at 2048 rows `egu` is 32768 x 1536 x 2 B = 100 MB written and read again per layer (~0.35 ms at
  0.58 TB/s, ~14 ms a chunk = ~2 %), plus `swiglu_clamped` 39 ms of 2436 (1.6 %) profiled; decode: -1
  launch a layer. Same structure for down + gather (G-3g, open): `edn` is another 100 MB at 2048 rows.
- lossless: byte-identical if the epilogue rounds as the stored plane does.
- sure: medium; effort L (libr4d moe_iqr epilogue, both class tables); verify: fusion oracle, kbench prefill.

### F7. DFly2 correction chain: 47 MB of replicated linears read 5 times a pass a rank
- area: drafter (TP2)
- location: `hyv4flash_dfly.h:731-733` (correct_gate/up/down `RAD_SHARD_NONE`), 1025-1031; manifest
  hyv4flash.cpp:549-551 (world 1 Decl at 540).
- arithmetic: IQ8R ~1 B a weight: gate 18.9 MB + up 18.9 MB + down 9.4 MB = 47.2 MB a round, **236 MB a pass a
  rank** (~0.4 ms), i.e. ~37 % of the draft linears' bytes, both ranks reading all of it.
- change (a, bits move in drafts only): the hidden half of `ccat` is known for all 5 rounds before the chain
  (z[1..5]); precompute `W_gate[:, :E] @ hnorm(z[k+1])` and `W_up[:, :E] @ ...` for all rounds in one
  M = 5n GEMM pair, and per round read only the `enorm(prev)` half + down: 18.9 + 5 x 28.3 = 160 MB
  instead of 236 MB (-76 MB, ~0.13 ms a DFly2 step). The K split changes f32 sum order -> proposals'
  bits move; the coupled verify is exact for any Q (state plane), so the output law is unchanged.
  (b) shard correct_gate / correct_up by rows and down by columns (as the draft MLP), plus one exact
  all_reduce of [n, 3072] a round before the `add`: -118 MB a pass a rank (-0.2 ms) for 5 new two-rank
  reductions (~13 us each): net ~-0.14 ms. (a) and (b) combine.
- why: a DFly2 one-client step is ~20 ms; together ~1 %. 
- lossless: (a) moves draft bits only (law exact); (b) changes rounding of the reduction (exact AR in rank
  order) -> draft bits only.
- sure: medium; effort M; verify: mtp_rrs chi-square, acceptance per depth, stamped draft pass.

### F8. DFly2 layers have none of the trunk's launch fusions (quantiser sharing, fused producers, row front, expand+gate+quant, add+norm)
- area: drafter (plugin only; every kernel exists)
- location: `hyv4flash_dfly.h:502-548` (`linear`: no `lead` / `feed` / `fed_codes`, each IQ8R Lin declares its own
  quantiser), 933-1008. Known open: notes/wp13-9.md "Not done / for later".
- change, per draft layer (byte-identical, existing ops):
  - kv_a and gate `feed = &p_qa.lin` (same input `xn`, same form): -2 quantisers;
  - `rmsnorm` q_a_norm -> `rmsnorm_had_quant_i8g` writing q_b's codes (attn.h:235-243 pattern): -1;
  - `rmsnorm` latent + `rope` key + `kv_store` -> `mla_row_norm_rope_store` (attn.h:284-299): -2;
  - `mla_project_kv` expand + `gate_mul` + o_proj quantiser -> `mla_expand_gate_had_quant_i8g` (attn.h:461-471): -2;
  - `silu_mul` + down's quantiser -> `gated_had_quant_i8g` (moe.h:57-78 pattern): -1;
  - `add` + `rmsnorm` -> `add_rmsnorm` (OPS.md:3645-3655, libr4d "the drafter's residual add and the norm
    after it", byte-identical to the pair): add1+ffn_norm, add2+next attn_norm / final norm: -2;
  - attn_norm / ffn_norm writing the codes of q_a and gate_up (`rmsnorm_had_quant_i8g` with a gain): -2.
  Total ~-12 a layer, **~-60 of the pass's ~303 dispatches**.
- why: the draft pass is launch-bound (wp13-7: 33 % idle; 94 quantiser/norm/silu launches = 0.48 ms,
  levers-general §1.1). At G-3's realised ~0.3-0.5 us a chained launch: ~20-30 us; if the draft pass's
  small launches behave like their measured 5 us average: up to ~0.3 ms. Estimate 0.05-0.3 ms a DFly2
  step (0.3-1.5 %).
- lossless: byte-identical (each fusion is a fusion oracle row already used by the trunk).
- sure: high that the launches exist, medium on the gain; effort M (plugin only, mirror attn.h); verify:
  dfly parity harness (taps), greedy + coupled e2e identical, kbench fusion oracle (already green for trunk).

### F9. DFly2 chain round: 24 dispatches, ~9 avoidable
- area: drafter
- location: `hyv4flash_dfly.h:1015-1055`.
- change (byte-identical unless noted):
  - one `iota` [depth, n] (period) before the chain instead of 5 `iota` rowk (+ row0): -5 a pass;
  - the 5 `hnorm(z[k+1])` and 5 `gather_rows` z[k+1] in one M = 5n launch each before the chain: -8 a pass;
  - correct_up fed by correct_gate's codes (same `ccat`, same form): -1 a round;
  - `row_topk` reading f32 logits and rounding to bf16 in-kernel (today `cast` f32->bf16 1035 then
    `row_topk`): -1 a round (row_topk rounds the same way the cast does);
  - `mtp_couple_propose` writing `tokens[:,k]` and `prev` itself (or the merge doing so when uncoupled):
    -2 a round (the two `cast`s 1053-1054);
  - the two `cast`s into `cgu` halves (1027-1028) exist because libr4d GEMMs refuse a strided output
    (dfly.h:448-450): a `silu_mul` / `gated_had_quant_i8g` variant with two input operands removes both and
    fuses the down quantiser: -3 a round (a concatenated gate|up weight would also do it but is a
    container change, DECISIONS U4: friend's).
  ~-9 a round, ~-50 a pass with the hoists.
- why: launch-bound chain, 5 rounds strictly dependent: ~0.42 ms "rest" + 0.48 ms small launches a pass.
  Estimate 20-150 us a DFly2 step.
- lossless: byte-identical (row_topk change: identical if the kernel rounds as `cast` does).
- sure: medium; effort S-M (plugin + small op variants); verify: dfly taps (`dfly.cand_*`), mtp_rrs.

### F10. RADIANCE_HY4_FUSE_QPROJ is documented and tested but never read
- area: plugin
- location: `hyv4flash_common.h:311` (documents it, default on), `tests/hyv4flash_arch_test.cpp:120` (sets it);
  no `fuse_on("RADIANCE_HY4_FUSE_QPROJ")` anywhere in arch/. G-3d (q_nope + q_rope one launch) is not done
  (notes/g3.md: per-part K-wave counts differ for sliding layers).
- change: either implement G-3d behind it or drop the switch from the comment / the test list so nobody
  A/Bs a no-op.
- lossless: n/a. sure: high; effort S (doc) / M-L (G-3d); verify: grep.

### F11. Engram: host-domain ids/rows also when the table is not on disk
- area: plugin (memory/link)
- location: `hyv4flash_engram.h:74-78` (`eg.ids`, `eg.ids_ahead`, `eg.rows` `RAD_DOMAIN_HOST`), comment 45-49.
  notes/adopt.md "Open": "Engram host-domain planes cost ~8 us a pass a layer at vram/ram placement (3 more
  ordered launches)".
- change: declare a device-domain pair as well and pick at step by the table's placement (or ask the core for
  the placement at declare time), so `engram_ids` -> gather -> `engram_norm` stay on the card for vram/ram
  placement (the u3 setup runs "Engram RAM").
- why: ~8 us a layer x 4 = ~32 us a decode pass (0.25 %), plus 3 KiB of rows written and read back over the
  link a token at prefill (12 MB a chunk a layer: ~0.45 ms a chunk for the 4 layers).
- lossless: byte-identical. sure: medium (cost from adopt.md); effort S-M; verify: stamps, greedy equal.

### F12. Four `cast` launches copy the embedding into the four streams; one `hc_enter` does it
- area: plugin
- location: `hyv4flash.cpp:964` (decl), 1147-1148 (issue).
- change: `hc_enter` (OPS.md:198; libr4d `hc_enter_bf16` r4d_rows.cpp:8848; libref "gated residual entry:
  hc copies of x") `x0 -> streams`, or have `embed_lookup` write stream 0 and one copy fill 1..3.
- why: -3 launches a step (decode: ~1-2 us; prefill 4 x 12.6 MB copies -> 1 pass).
- lossless: byte-identical. sure: high (if hc_enter's layout is [T, hc*n] stream-major, check); effort S.

### F13. Indexer query assembly: two `cast` copies a sparse layer on selecting steps
- area: plugin + kernel operand
- location: `hyv4flash_indexer.h:383-388` (iqn, iqr -> `ix.query` [T,16,128]); same for the trunk's non-absorbed
  prefill `attn.h:605-610` and the drafter's q split `dfly.h:939-944`.
- change: let `idx_score` (and `dsa_query`) read the query as two operands (nope part pitch 16*64, rope part
  pitch 16*64) instead of one interleaved plane; for `attn_kv_sink` likewise q_nope / q_rope as two operands.
  (Splitting the weights the other way round is a container change.)
- why: 24 launches a selecting decode step (12 layers x 2), ~3 ms a prefill chunk of copies (2048 x 4 KB x 2 x 12
  plus the attention's q copies in the first chunk) - part of levers' "cast 41 ms".
- lossless: byte-identical. sure: medium; effort M (kernel operand forms in three libraries).

### F14. `sub_i32` cu rebase issued once per layer instead of once per step
- area: plugin
- location: `hyv4flash_attn.h:379` (decl, per layer), 570 and 612 (issue in every layer of a prefill / mixed step);
  buffers `AttnWire::cu_p` per geometry (attn.h:92).
- change: one `sub_i32` at the step's start into one model-level `cu_prefill` both wires read.
- why: -41 host-issued launches a prefill or mixed step (these passes are unrecorded, pass_key 0) - small.
- lossless: byte-identical. sure: high; effort S.

### F15. Engram `ahead` hashing runs for every prefill step at every placement
- area: plugin
- location: `hyv4flash_engram.h:359-368` (A = n_ahead unless RADIANCE_HY4_ENGRAM_AHEAD=0), 371-373.
- change: hash ahead rows only when the table is disk-placed (the only reader: libavx's reader thread;
  "a device gather ignores them", engram.h:54-55).
- why: up to 2 x 2048 x 16 ids a layer hashed and written to host memory over the link per chunk (256 KB a
  layer); small.
- lossless: byte-identical. sure: high; effort S.

### F16. DFly snapshot: copies on every selecting trunk step, whole-step tall, PERSIST
- area: plugin (drafter context taps)
- location: `hyv4flash_dfly.h:202-238`; buffers `dfly.sel_snap.<l>` [max_tok, 512] i32 + nsel PERSIST x 5 layers
  (**~21 MB PERSIST a rank at max_tok 2048**), `final_hidden` [max_tok, 3072] PERSIST (hyv4flash.cpp:616,
  12.6 MB, PERSIST **even when the drafter is not served**).
- change: (a) non-streamed groups: let layer l's `qsa_select` write directly into `dfly.sel_snap.<l>` /
  `nsel_snap.<l>` (the attention then reads them): no copy, -10 launches a selecting decode step;
  streamed groups keep the copy (kv_resolve rewrites `sel` in place). (b) the 5 nsel `fill`s of a
  non-selecting step as one fill over a [5, T] plane: -4. (c) declare `final_hidden` PERSIST only when
  `m.df.served`. (d) prefill chunks: only each sequence's last row can be an anchor; copy [0, DT) plus those
  rows (or skip on chunks that end no prompt).
- why: launches (small), 12.6-34 MB of VRAM a rank (~9-23 routed units).
- lossless: byte-identical. sure: high; effort S (a-c), M (d); verify: dfly parity taps, rollback gate.

### F17. Indexer score plane sized [max_tok, max_ctx/4] f32 for the prefill chunk
- area: plugin (arena)
- location: `hyv4flash_indexer.h:63, 81` (`ix.score` [T, blocks] f32), issue 390, 403-409.
- change: on prefill steps run `idx_score` + `qsa_select` in row tiles (e.g. 256 rows) over one [256, blocks]
  plane, or a fused score+select row tile.
- why: 134 MB at 64K context, 268 MB at 128K, for a 2048-row chunk, in the arena's peak; with single-interval
  liveness (core/build/rad_bufplan.cpp:10-11, 91) it adds to every other step-long plane. Memory returned
  goes to expert residency during prefill (when the arena is not lent). Gain small in time.
- lossless: byte-identical (rows independent). sure: medium; effort M.

### F18. Arena: every wire plane is live for the whole step (shared across layers, single interval)
- area: core planner / plugin buffer plan
- location: `core/build/rad_bufplan.cpp:10-11, 91` (one [first_def, last_use] interval a buffer); the wires
  `AttnWire` (attn.h:71-97, two geometries), `MoeWire` (moe.h:111-147), `IndexWire`, `EngramWire`, Codes pairs.
- observation: attention scratch (sliding wire at 2048 rows: q_latent, attn_latent 67 MB each, kv_b 59 MB,
  q_full / k / v / ao / gate / ag ~34 MB each; ~0.44 GB; sparse wire ~0.22 GB) and MoE scratch (egu 100 MB,
  edn 100 MB, eff 50+25 MB, ...) are never needed at the same time inside a layer, but each is used by every
  layer, so both span the whole step and the arena at max_tok holds their sum (my arithmetic, ~1 GB+).
- change: core: let a buffer whose every use-run starts with a full write be dead between runs (multi-interval
  liveness), or a plugin-declared union scratch (no ABI today).
- why: arena peak at max_tok (prefill) down by roughly the smaller of the two groups (~0.3 GB a rank) ->
  more experts resident during prefill. Decode levels are lent already, so decode gains only at the levels.
- lossless: byte-identical. sure: medium (sizes are arithmetic); effort M-L (core); owner: core auditor.

### F19. `gather_rows` before the head on steps whose output rows are all rows
- area: plugin
- location: `hyv4flash.cpp:1207-1210`.
- change: when `n_out == T` and `out_ids` is the identity (pure decode and verify steps; known on the host),
  issue `logits_gemm` on `final_hidden` directly (or let `logits_gemm` take the row ids). The pass key must
  include the choice (pass_key 1258).
- why: -1 launch a decode step (~3 us, ordered just before the unchained head). Small.
- lossless: byte-identical. sure: medium (identity check on out_ids host mirror needed); effort S.

### F20. Engram: four identical CONV histories and four `engram_ids` launches a step
- area: plugin + op parameter
- location: `hyv4flash.cpp:969-981` (one CONV group, a state slot per Engram layer via
  `rad_bind_layer_kv` engram.h:140), issue engram.h:352/365. kernels/engram.md §1 designed one shared
  history ("only the first issue of a step advances it").
- change: one history (bind the first Engram layer only) and one `engram_ids` over the 4 layers' constants
  (a weight table operand) writing [T, 4*16] ids; or keep 4 launches but one state.
- why: -3 launches a step, 3 fewer CONV state rows to checkpoint / restore per sequence (prefix cache,
  rollback). Small.
- lossless: byte-identical. sure: high; effort S-M.

### F21. FP8/int8 indexer: `cast` iq -> iqf is a no-op relayout
- area: plugin (only under RADIANCE_HY4_INDEXER=fp8/int8, not default)
- location: `hyv4flash_indexer.h:243-246, 393`: [T,16,128] bf16 contiguous is byte-for-byte [T*16,128].
- change: pass `brows(w.iq, T)` reshaped as [T*16, 128] to `dsa_query` (a RadOperand with rows T*16, cols 128).
- why: -1 launch a selecting sparse layer, -T*4 KB copy. lossless: byte-identical. sure: high; effort S.

### F22. Short block not declared (L5c, known): concrete code
- area: drafter
- location: `hyv4flash_dfly.h:853-865` (no `rad_drafter_short_block`); `query` refuses `M != n * block`
  (886-890); rounds fixed at `depth` (1015). Pattern: qwen35_fp8.cpp:328, 522.
- change: declare `rad_drafter_short_block(b, 3)` (or the engine's choice) and run `rounds = M / n - 1`
  in `query` (stages C/D at M rows, chain to `rounds`), `tokens` pitch unchanged.
- why: levers-general L5c: 0.9 ms of 58 at eight clients (1.5 %). lossless: yes (verify unchanged).
- sure: high; effort S-M.

### F23. Drafter all_reduce x10 a pass: T6 concretely
- area: drafter + kernel
- location: `hyv4flash_dfly.h:993-998` (AR o -> post_norm -> add), 1006-1008 (AR y -> add), then norm.
- change: with F8's `add_rmsnorm`, a remaining twin `ar_add_rmsnorm` (exact AR of `a`, `x += a`, `y =
  rmsnorm(x)`) built from the junction's AR device functions makes a layer's two reductions + four
  elementwise ops two launches. The 5 `all_gather`s of the top-8 pairs are needed (round k+1's input).
- why: -15..-20 launches a pass (T6 open item). lossless: byte-identical (E6). sure: medium; effort M.

### F24. Draft pass stage A/B small items
- area: drafter
- location: `hyv4flash_dfly.h:907-920, 1019-1020`.
- change: `embed_lookup` straight into `x`; gather the anchor embeddings `ea_raw` once (S rows) before the
  scatter and use it as round 0's `pe` (no `cast` of M x 3072, no second gather); the two fusion norms
  into `cat` as one launch only if a two-segment norm exists (else keep).
- why: -2 launches a pass. lossless: byte-identical. sure: high; effort S.

### F25. Engram key/value GEMMs are the only unchained trunk GEMMs besides the head
- area: kernel row
- location: `hyv4flash_engram.h:286-293` (`gemm_nt_bias`, resolves to `vit_gemm_wa` for the key's N 12288 per
  levers-general §1.2/§1.7: "all chained except the head and the four Engram GEMMs").
- change: a chained bf16 `gemm_nt_bias` row at decode heights (and the head's `logits_gemm` chained form, 1 a
  trunk step + 5 a draft pass).
- why: each barrier launch costs the measured 4.4-4.6 us gap (levers §1.3): 4 + 1 a trunk step, 5 more a DFly2
  draft pass: ~20-45 us a DFly2 step. lossless: byte-identical. sure: medium; effort S-M (libr4d).

### F26. Indexer key side: five launches a sparse layer every step, two of them usually idle
- area: kernel fusion (plugin wiring)
- location: `hyv4flash_indexer.h:351-375` (key GEMM, key rmsnorm, `qsa_work`, `idx_pool_gated` over
  `work` rows even when no group closes - 3 of 4 decode steps of a lone sequence -, `qsa_tail_store`).
- change: one op doing norm + work + pool (for the groups that close) + tail store from the key GEMM's output.
- why: -3 x 12 = -36 launches a decode step; at ~0.3-0.5 us realised a launch: ~10-20 us. Low.
- lossless: byte-identical. sure: medium; effort M (libr4d + libref/libavx).

### F27. Junction + Engram + ihc_pre: four launches around each Engram layer
- area: kernel fusion
- location: `hyv4flash.cpp:772-776` (tail-0 junction), 1156, 1160, 1164; q_a quantises itself (attn.h:219-223).
- change: a junction form that applies `engram_gate` between `ihc_post` and `ihc_pre` (all row-local),
  so the Engram layers get the same one-launch junction with codes as the others.
- why: -4 launches x 4 a step. Low. lossless: byte-identical if built from the same device functions.
- sure: medium; effort M-L.

### F28. Coupled propose passes its pointers in a by-value struct (check)
- area: spechub (correctness risk that would show as acceptance)
- location: `plugin/spechub/spechub.hip:583-588` (`R4D_EMIT(..., propose, ..., c)` with `c` a
  `RadMtpCoupleProposeCall` holding `idx`, `val`, `card` = `batch->seq_ids` raw, `draw`, `state`), against
  spechub.hip:976-978 for the verify: "Separate pointer arguments let graph replay rebind each operand
  address independently: a struct on the host would freeze the addresses it was built with."
- risk: the draft pass is recorded (pass key 9). If tape replay rebinds raw operands (e.g. `seq_ids` from a
  staging ring) by operand and cannot see inside the struct, `card` is stale -> `s.draw.card == c.card[row]`
  fails -> status -1 -> the row keeps the argmax proposal and an unbound state (lower sampled acceptance,
  still exact). Unverified: I did not read how the tape patches kernarg bytes.
- check: count `status == 1` states in replayed vs first-recorded draft passes (or diff kernarg addresses);
  fix = separate pointer parameters, as the verify kernels do. sure: low; effort S.

### F29. qsa_select dominates the indexer at long context (pointer to the kernel auditor)
- area: kernel
- location: issued at `hyv4flash_indexer.h:407`, 12 a selecting decode step.
- numbers: notes/idxint8.md perfidx: `qsa_select` 154 us a launch at 64K (16384 groups, M = 1 or 4), 305 us at
  200K, against 6-12 us for `idx_score`: **1.8-3.7 ms a decode step** once a session passes ~32-64K tokens
  (code-agent histories). The plugin issues it once per sparse layer, row-wise; nothing to drop in the graph.
- lossless fix is kernel-side (multi-workgroup radix/threshold top-k). Not mine; flagged because it is the
  largest per-step term the graph reaches at long context.

---------------------------------------------------------------------------------------------------

## Host joins and barriers in a step (complete list from the plugin's side)

Trunk decode / verify step (recorded tape):
1. `logits_gemm` (T4): unchained, barrier on its predecessor.
2. Engram key `gemm_nt_bias` x4 (`vit_gemm_wa`): unchained barriers (F25).
3. Engram gather at `--ngram-placement disk`: libavx host kernel on host-domain buffers = a host op in the pass
   (host join behind `engram_ids`, x4 a step); vram/ram placement: device gather over host-domain planes (F11).
4. Streamed sparse KV (`--kv-resident`): `kv_stage_wait` / `kv_stage_layer` / `kv_stage_next` per sparse layer
   (attn.h:535, 652, 692; dfly.h:958, 975, 1000) - host checks and mover waits; `kv_resolve` device op.
5. Route report: `rad_route_report` per MoE layer enqueues an async copy (F3), no host wait; the heat engine
   reads it next step. Expert prefetch (`rad_expert_prefetch*`, moe_fp8.h:1866-1867, 1948, 2019) is inactive at
   TP2 (T3 closed) - no work.
6. Calibration taps (`m.calib.*`, `calib_experts_pass` with `rad_stream_sync`, moe_fp8.h:2121-2149): only in a
   calibration run.
Draft pass: the 5 `logits_gemm` barriers; `all_gather` x5 and `all_reduce` x10 are device collectives (two-rank
handshake, not host joins); streamed KV as in 4. Context pass: empty (still a barrier round of the core,
dfly2.md §3.5: core export open). Prefill / mixed steps: unrecorded, every op host-issued.

## Host -> device per-step operands the plugin binds (the core stages them)

Trunk step: token_ids [T], positions [T] (+ positions_h host mirror read in `plan`, hyv4flash.cpp:1084-1090,
and again in `pass_key`), rope_pos [3,T] (vision builds only), cu_seqlens [S+1], out_ids [n_out],
num_accepted [D]; per KV group: sliding {slot_mapping, block_table, seqused}, sparse {slot_mapping,
block_table, seqused}, index keys {block_table}, index tail {state_index}, Engram {state_index}; = 14 arrays.
The two block-4 tables (sparse latent and index keys) are each ctx/4 i32 a sequence: 64 KB a sequence at
64K context each. Draft pass: token_ids, positions, cu, anchor_rows, seq_ids, sparse slot_mapping,
sparse block_table (another ctx/4 a sequence). Whether the core re-uploads whole tables every step is
the core's (unverified here).

## RADIANCE_HY4_* switches (all read in arch/hyv4flash unless noted)

| switch | default | gates | note |
|---|---|---|---|
| CACHE_ROW | on | `mla_row_norm_rope_store` (common.h:556) | off under taps / streamed KV |
| CHAIN_Q8 | on | chained quantiser rows (common.h:302) | |
| DFLY | on | `=0` declines the drafter (dfly.h:122) | |
| DFLY_QTEMP | 1 | `mtp_couple_propose` q_temp (dfly.h:136) | sweep: no winner (g1g5) |
| DFLY_SABOTAGE, DFLY_TEST_BREAK, TEST_RING | off | tests only | |
| ENGRAM_AHEAD | on | ahead ids (engram.h:27) | pays only on disk placement (F15) |
| ENGRAM_EPS | f32 | Engram norm eps (engram.h:32) | numerics (U1) |
| EXPAND_GATE | on | `mla_expand_gate` (attn.h:399) | |
| FUSED_QUANT | on | rmsnorm/gated `_had_quant_i8g` producers (common.h:297) | |
| FUSE_EXPAND_QUANT, FUSE_JUNCTION_Q, FUSE_SWIGLU | on | G-3c / b / a | |
| FUSE_QPROJ | "on" | **nothing** (F10) | dead switch |
| INDEXER | exact | fp8 / int8 indexer (hyv4flash.cpp:865) | lossy-ish (moves selections; friend/user) |
| JUNCTION | on | fused junction (common.h:548) | |
| JUNCTION_TP4 | off | 4-rank junction twin | untested on 4 cards; TP2 unaffected |
| PREFILL | (non-absorbed first chunk) | `=absorbed` forces absorbed | |
| ROUTER_FUSE | on | `router_sigmoid_topk_scatter` | |
| SHAPE_PROBE | on | sizing declares / arena lending | |
| SHARE_CODES | on | Lin::feed code sharing | trunk only; drafter has none (F8) |
| TAPS | off | debug taps (all free when off) | |

None defaults off while measured good at TP2; the default-on ones were each measured byte-identical.

---------------------------------------------------------------------------------------------------

## Format / lossy (friend's decision) - seen while reading, not proposed as ours

- `kv_a` is bf16 in u3 (3.5 MB a layer, R, 149 MB a pass a rank); `gm-i8kva` recipe exists (wp13-9).
- Engram key/value projections bf16 (47 MB a layer R, levers §3 has the int8 row).
- Fold `fusion_gate` (mul_weight, dfly.h:917) into the fusion weight's rows, or concatenate correct_gate|up
  and the drafter's q_b split, or one index_query weight: container / manifest changes (DECISIONS U4).
- Head below int8, draft linears below 8 bit, routed down as trellis: levers §3.

## Checked and found already optimal / not a lever

- Logits only for `n_out` rows (out_ids gather) - non-final prefill rows already skipped (1206-1211).
- Final norm on all rows: fused into the last junction (cheap row-local); its real cost is F4's.
- Indexer query/score/select only on selecting steps and sparse layers (indexer.h:376-379, hyv4flash.cpp:1171);
  no indexer on sliding layers. Engram only on its 4 layers.
- Taps and calibration hooks issue nothing when off (common.h:454, 463-465).
- q_a / q_b / kv_a / gate order in `front` (attn.h:502-550): under `haz_mark` q_rope, kv_a, the q rope and the
  gate already run unordered behind their predecessors; the big GEMMs are bandwidth-bound and gain nothing
  from more overlap with each other.
- First draft reusing the trunk logits: impossible (different position, see 1.6).
- Round k's head only for live sequences: the head is weight-bound (186 MB) at any M <= 8; only skipping a whole
  round helps (L5b, not worth after L1: levers-general L5).
- Head read is sharded by vocabulary and batched over sequences (M = n a round).
- Router replicated, indexer replicated, Engram replicated, q_a / kv_a / lat_in / lat_out replicated: closed in
  levers-tp2 §3 (no sharding without a new collective that costs more). The drafter correction is the
  exception (F7).
- The KV rows of verify rows that are rolled back: overwritten by the next step before any read (dfly.h:25-27);
  nothing to skip.
- `chat.h`: a static `RadChatFormat` descriptor (chat.h:23-52); no per-request work in the plugin. Template
  rendering / tokenisation / prefix matching are the server's and core's.
- MoeFP8 prefill router `all_gather` (SplitF32Proj): halves the f32 router compute for 2 MB a direction; a win
  at 2048 rows as long as the router stays on f32 FMA (re-check if F2 lands: with WMMA the split may not pay).
- `mtp_trav_verify` launches M workgroups with non-roots returning at once (spechub.hip:954): fine at M <= 48.
- Closed and not re-proposed: q_temp (g1g5), expert prefetch at TP2 (T3), two-piece prefill (T4), shortlist head,
  sampler chain, depth-rule host wait, re-sharding (levers-tp2 §3), G-3 a/b/c (done), arena lending / stager (done).
