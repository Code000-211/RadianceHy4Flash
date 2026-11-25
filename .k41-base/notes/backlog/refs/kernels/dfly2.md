# DFly2, the speculative drafter of Hy4-flash: forward, ops, caches, rollback, validation

Design note for WP13 (PLAN §5, "13 DFly2"). Nothing here is implemented or measured. Every
statement about radiance marked **(verify)** was read from the sources but not run.

Sources read:
- vLLM @820fce0: `vllm/models/hy_v4/nvidia/dfly.py` (model, fusion, correction),
  `dfly_speculator.py` (`HYV4DFlyQueryModel.forward`, the query pass), `dfly_inputs.py` (query
  block, slots, reuse indices), `dfly_target_reuse.py` (KV aliasing, index snapshots),
  `hy_v4/dfly_config.py`, `hy_v4_flash/dfly_config.py`, `hy_v4/nvidia/model.py`
  (`HYV4DecoderLayer._forward_normal`), `hy_v4/nvidia/attention.py`,
  `hy_v4_flash/nvidia/{attention,compressed_indexer,indexpool_ops}.py`,
  `tests/v1/spec_decode/test_dfly_prepare_inputs.py`, `docs/features/speculative_decoding/dfly.md`.
- SGLang @a8ac865: `srt/models/hunyuan_v4_flash.py` **skips** `model.dfly_layers.*` at load
  (line 1258). This checkout has no DFly forward; vLLM is the only reference.
- radiance @1774a48a: `abi/rad_builder.h` (drafter ABI), `abi/rad_runtime.h` (`RadBatch`),
  `core/engine_draft.cpp` (block driver), `core/sched/scheduler.cpp` (`block_begin/context/query/end`),
  `core/sched/advance.h`, `arch/common/rad_block_dflash2.h`, `rad_block_lilicorr.h`,
  `arch/hyv4flash/*`, `libref/ref_qsa.cpp`, docs GUIDE §4.6, PLUGIN §11, OPS, QSA, notes/wp*.md.
- Not found where the task pointed: `arch/flint` declares no drafter (`docs/FLINT.md`: "No draft
  head"); the DFlash2 block drafter lives in `arch/common/rad_block_dflash2.h` (used by
  `qwen35_fp8`, `gemma3`, `nemotronh`). `docs/DESIGN-adaptive.md` is not in this tree (only in
  `/root/src/radiance`); GUIDE §4.6 covers the same ground.
- `notes/flashnext-levers.md` (which Flash-Next / Flint levers reach Hy4) and
  `docs/DESIGN-block-spec.md` at `flint-spec` 12716103 in `/root/src/radiance` (the speculation
  levers over a block drafter). §8 says what of it applies to DFly2 and what must be ported.
- Paper (arXiv 2607.25852, "AngelSpec"): hidden-correction formula (its Eq. 15) and "D-cut", which
  uses a per-token draft confidence to choose the verify depth. It does not give the confidence
  head's forward.

---

## 1. What DFly2 computes (the reference, exact)

### 1.1 Configuration (real `config.json`, top level)

| key | value | meaning |
|---|---|---|
| `dfly_num_hidden_layers` | 5 | draft layers `i = 0..4` |
| `dfly_target_kv_layers` | 25, 29, 33, 37, 41 | draft layer `i` reads and writes the latent cache of trunk layer `Lkv[i]` |
| `dfly_target_index_layer` | 25, 29, 33, 37, 41 | draft layer `i` reuses the selection of trunk layer `Lidx[i]` |
| `dfly_target_index_topk` | 2048 | width of the trunk's selection in tokens (512 groups x 4); must equal `index_topk` |
| `dfly_context_local_window` | null (0) | no extra local window |
| `dfly_mask_token_id` | 120009 | the id every undrafted block position embeds |
| `dfly_mtp_fusion` | `slot0` | the trunk's hidden state is fused into the anchor row only |
| `dfly_enable_hidden_correction` | true | the autoregressive head of §1.3 step 5 |
| `speculative_config` | `num_speculative_tokens 5`, `draft_sample_method probabilistic` | block = 6 rows, 5 proposals |

vLLM requires: every `Lkv`/`Lidx` in `[1, layers)` and a **sparse** layer, `Lkv` pairwise distinct,
all selected layers with identical rotary parameters (so the draft uses the sparse layers' theta
1e7), draft `mlp_type dense`, no iHC, no Engram, no indexer of its own, causal attention, heads /
dims as the sparse trunk layers (32 heads, q latent 1024, kv latent 512, 192/64/256), learned sinks,
elementwise gate. The MLP width is not in the config; it is the checkpoint's `mlp.down_proj` column
count (9216 in the release, as `hyv4flash.cpp:drafter` already reads it). `post_proj_norm` exists
iff the checkpoint has `self_attn.post_proj_norm.weight` for every draft layer (the manifest
declares it; keep it mandatory and say so in the refusal).

The draft has **no embedding and no lm_head of its own**: both are the trunk's
(`maybe_share_target_embed`, `share_target_lm_head`). `confidence_head.*` is in the checkpoint and
**ignored by vLLM** (`_ignored_prefixes`); see §8.3.

### 1.2 Inputs of one draft

After a trunk step, for each sequence that sampled a token:
- `a` (anchor position) = position of the last **accepted** row + 1. The token at `a` is the
  token the trunk just sampled (the "bonus" token); it is known, not drafted.
- `src` = the trunk step's row of position `a - 1` (vLLM: `query_end - num_rejected - 1`).
- `t` = the trunk's **final** hidden state at `src`: `norm(hc_head(h))`, exactly the lm_head's
  input (radiance: `final_hidden`).
- `Sel_l(src)` for `l` in `Lidx`: the token set layer `l` attended at row `src`
  (all tokens `<= a-1` when the trunk attended densely).
- The latent cache of layers `Lkv` for positions `< a`.

The draft is inactive when `a + 6 > max_model_len` (radiance: `ok_ctx` in `block_begin`).

### 1.3 Forward

`B = 6` rows a sequence at positions `a .. a+5`, tokens `[bonus, mask, mask, mask, mask, mask]`.
All arithmetic bf16 in vLLM except where stated; `E = 3072`.

```
1  e[j]   = embed(tok[j])                                   j = 0..5     trunk table
2  x[0]   = e[0] + gate * eh_proj( [ enorm(e[0]) | hnorm(t) ] )          gate: [E] vector, eh_proj: [E, 2E]
   x[j]   = e[j]                                            j >= 1
3  for i in 0..4:                                           one residual stream, pre-norm
     xn   = input_layernorm_i(x)
     ao   = attention_i(xn)                                 below
     x    = x + ao
     xn   = post_attention_layernorm_i(x)
     x    = x + down_i( silu(gate_i(xn)) * up_i(xn) )       dense, width 9216, no clamp, no norm behind
4  z      = final_norm(x)                                   [6, E]
5  prev   = tok[0]
   for k in 0..4:                                           sequential in k, batched over sequences
     c    = [ hidden_norm(z[k+1]) | embed_norm(embed(prev)) ]            embed(prev) is the RAW embedding
     zc   = z[k+1] + down_c( silu(gate_c(c)) * up_c(c) )                 gate_c, up_c: [E, 2E], down_c: [E, E]
     logit= lm_head(zc)                                     trunk head, f32
     prev = draft[k] = argmax(logit)            (or a sample at the request's temperature)
```
Row 0 (the anchor) predicts nothing: `z[0]` is never read. Row `k+1` predicts the token **at its
own position** `a+k+1`. This is radiance's `block = depth + 1` convention (DFlash2's).

Attention of draft layer `i` (the sparse trunk layers' gated MLA, draft weights):
```
qr      = q_a_layernorm(q_a(xn))                 [1024]
q       = q_b(qr) -> [32, 256] = [nope 192 | rope 64], rope part rotated at the row's position
row     = [ kv_a_layernorm(kv_a(xn)[:512]) | rope(kv_a(xn)[512:]) ]      576, rotated at the row's position
STORE   row into the cache of trunk layer Lkv[i] at the row's position   (speculative slots)
keys(j) = Sel_{Lidx[i]}(src)  U  { a, .., a+j }                          row j of the block
s_t     = 256^-0.5 * ( q_nope . W_UK lat_t + q_rope . rope_t )           W_UK/W_UV = draft kv_b
P       = softmax over keys(j) with the draft layer's sink logit of the head appended, no value for it
ao      = post_proj_norm( o_proj( (sum_t P_t W_UV lat_t) * sigmoid(linear_gate(xn)) ) )
```
Key facts, all read from the code:
- History keys are the **trunk's** cache rows (written by trunk layer `Lkv[i]` with the trunk's
  `kv_a`); the block's own keys are the **draft's** rows for the same block, written into the same
  cache (`dfly_target_reuse`: the alias is dropped after allocation so the draft writes its query
  K/V). Draft layer `i` therefore must store its 6 rows before any of them attends.
- `Sel` is filtered to tokens `< a - local_window = a`: nothing is filtered, the snapshot only holds
  tokens `<= a-1`.
- The selection is the **source row's**, the same for all 6 rows of the block. The draft never
  scores groups.
- Rotary: theta 1e7, the sparse layers' (`target_rope_parameters`), text positions.
- Norm eps `rms_norm_eps` = 1e-5 everywhere in the draft; scale `256^-0.5`.

For the Flash trunk, `Sel_l(src)` (`expand_groups`) is: the recalled closed groups (at most 511,
all before the group of `a-1`), four tokens each, plus the tokens of the own group of `a-1` up to
`a-1`. With at most 511 closed groups every one is recalled, so the set is `[0, a-1]`: the dense
case of plan D3.

### 1.4 The key set as a page table

Sparse latent pages are groups of 4 (block 4). Let `g0 = (a-1) / 4`. For row `j` at `p = a + j`:

```
keys(j) = recalled pages (complete, all < g0)  U  tokens [4*g0, p]
```
The second part is contiguous: the own page of `a-1`, then `(p/4) - g0` further pages, the last
one partial. That is exactly the form `mla_attn_sel_sink` reads (`tokens [0, length)` of a per-row
table, page `bt[r][t/4]`, slot `t%4`): table = `[recalled.., page g0, page g0+1, ..]`,
`length = 4*nrec + (p - 4*g0) + 1`. **No new attention op is needed.** The table is at most
`(G - 1) + 1 + floor((B-1)/4) + 1 = G + 2` pages wide (`G` = `group_topk`; 514 at real size).

### 1.5 Does DFly2 have DFlash2's edge or tree structure? No.

DFlash2 emits, in one pass, K candidates and unary scores a position, and a **low-rank bilinear
edge** (`pred[x_{l-1}] . hp[l] . succ[c]`) links consecutive positions; its selector, coupled walk
and tree all run on that precomputed lattice without touching the model again.

DFly2 has no candidate lattice, no codebooks and no selector. Its backbone pass gives one hidden
row a position (`z[k+1]`), and the dependence on the predecessor is the **hidden-correction head**
of step 5: a SwiGLU MLP over `[norm(z[k+1]) | norm(embed(x_{k-1}))]` followed by the full lm_head
(the paper's "predecessor-conditioned autoregressive head", which it contrasts with DFlash's
token-only Markov bias). So:
- a position's law `Q_k = softmax(lm_head(zc_k(x_{k-1})) / T)` exists only **after** `x_{k-1}` is
  chosen, and costs one correction and one head evaluation: it is a serial head's *round*, a very
  cheap one (no attention, no cache), not a lattice lookup;
- the drive is BLOCK (one backbone pass, mask rows, no rounds for the engine to schedule), the
  chain is SERIAL in nature and runs inside the query pass;
- `dflash_select`, `dflash_couple_propose`, `dflash_tree` and flint-spec's `dflash_couple_tree`
  **do not apply** (they need `unary`, `hp`, `pred`, `succ`). The coupled draw is
  `mtp_couple_propose`, the serial head's op (§8.2);
- a tree is possible in principle (each tree level is one batched correction + head launch over
  that level's nodes, grown by the serial head's coupled grower), but verifying one needs a tree
  mask in the absorbed MLA kernels, a selection a node and tree rollback of the tail ring and the
  Engram history. Not planned (§8.4).

### 1.6 What the reference leaves open

- `confidence_head`: no forward anywhere (§8.3).
- Draft quantisation: vLLM serves the draft bf16 or blockwise FP8. Radiance: bf16 first (plan §4.3).
- vLLM draws drafts with Gumbel noise keyed by `(seed, position)` under `probabilistic`; verification
  is "standard". Radiance's exact options are §8.

---

## 2. Shapes

| | real | `gpu-mini` | `mini` / `mini4` | `mini6` |
|---|---|---|---|---|
| hidden E / vocab | 3072 / 120832 | 512 / 3584 | 128 / 320 | 128 / 320 |
| draft layers | 5 | 1 | 1 | 1 |
| `Lkv` = `Lidx` | 25, 29, 33, 37, 41 | **3 (sliding: invalid)** -> 2 | 2 | 5 |
| heads x (nope/rope/v), q / kv latent | 32 x 192/64/256, 1024 / 512 | same | 2 (4) x 24/8/32, 64 / 32 | as mini |
| draft MLP width | 9216 | 192 | 192 | 192 |
| fusion `eh_proj` | [3072, 6144] | [512, 1024] | [128, 256] | same |
| correction gate, up / down | [3072, 6144] x2 / [3072, 3072] | [512, 1024] x2 / [512, 512] | [128, 256] x2 / [128, 128] | same |
| block B / proposals | 6 / 5 | 6 / 5 | 6 / 5 | 6 / 5 |
| `G` (group_topk) / dense limit | 512 / 2048 | 512 / 2048 | 4 / 16 | 4 / 16 |
| table width W = G + 2 | 514 | 514 | 6 | 6 |
| mask token | 120009 | **120009 >= vocab** -> 3583 | **-> 319** | -> 319 |
| query pass rows M | 6 x drafting sequences | | | |

Per sequence: `e`, `x`, `z` are `[6, E]`; `q [6, 32, 256]`; cache rows `[6, 576]`;
`ao [6, 32*256]`; chain tensors `[1, E]` a step; draft logits `[1, V / world]`.
Draft weights at real size: 135.4 MiB attention + 162 MiB MLP a layer, 36 MiB fusion, 90 MiB
correction: **1.58 GiB bf16** in all (arithmetic from the shapes).

**Fixture defects to fix first (WP13.1), config only, no weight change:**
1. `ref/minicfg.py` copies `dfly_mask_token_id 120009` and `dfly_target_index_topk 2048` from the
   real config into every preset. Set `dfly_mask_token_id = V - 1` and
   `dfly_target_index_topk = index_topk` of the preset.
2. `dfly_target_*_layer = [n - 1]` picks layer 3 on `gpu-mini`, a sliding layer. Set it to 2.
3. Add a preset `mini6d2` (mini6 trunk, **2** draft layers, `Lkv = Lidx = [1, 5]`): the only way to
   exercise the residual stream across draft layers and two different trunk layers on the host.
   `tensor_specs` hard-codes draft layer `0.`; loop over the layer count.
The plugin must refuse (by name) a mask id `>= vocab`, a target layer that is not sparse or is 0,
duplicate `Lkv`, and `dfly_target_index_topk != group_topk * ratio`.

---

## 3. Mapping to radiance

### 3.1 Drive: a block drafter with no context pass of its own

```c
RadDrafterDecl d = { .name = "dfly2", .kind = RAD_DRAFT_BLOCK,
    .depth = ctx->max_spec,            /* 1..5; above 5 the drafter is not declared (untrained rows) */
    .proposal = dfly.tokens,           /* i32 [max_seqs, depth] */
    .proposal_pitch = depth,
    .mask_token = dfly_mask_token_id,
    .window = see 3.5,
    .block = depth + 1 };              /* the anchor predicts nothing */
```
The block driver (`engine_draft.cpp`) then runs, behind every trunk step and before the step issued
ahead: `ADV_ACCEPT` -> context pass (`draft_pass -1`, the trunk step's rows mirrored) -> `ADV_BLOCK`
-> query pass (`draft_pass 1`: per drafting sequence the anchor at the committed end and `B-1`
masks, positioned on the card) -> `ADV_BTAKE` (proposals into the next verify's token row).

What the plugin gets in the query pass and DFly needs:
- `token_ids`, `positions` `[M]` (device; the host mirrors are staged at the furthest position and
  must not be used for anything position-dependent),
- `anchor_rows [n_seq]` (ABI 36): the trunk step's row of the last committed position = `src`,
- every KV group's batch, so the sparse latent group's `slot_mapping [M]` and `block_table`
  for positions `a .. a+5`; `block_begin` reserved them (`ensure_from(next_hi + block)`).

DFly's **context pass issues nothing**: it has no cache of its own. Everything it conditions on is
left behind by the trunk step in persistent buffers (§5).

`rad_arch_probe`: `draft_depth = min(5, speculative_config.num_speculative_tokens)` once served.
A depth below 5 is the same forward with fewer mask rows; because the draft is causal, a shorter
block's rows are bit-for-bit the front of the long block's, so `rad_drafter_short_block(b, 4)`
(3 proposals) costs only the declaration (later, WP13.9).

### 3.2 Trunk step: three additions (only when the drafter is declared)

1. After the indexer of each distinct layer `l` in `Lidx` (`Indexer::step`):
   - selecting step (`p.select`): `cast` i32 `ix.selected [T, G]` -> `dfly.sel_snap.<l> [max_tok, G]`
     and `ix.nsel [T]` -> `dfly.nsel_snap.<l> [max_tok]`;
   - otherwise: `fill` i32 0 over `dfly.nsel_snap.<l>` rows `[0, T)` (the tap `op_tap_nsel0` does the
     same today). `nsel = 0` means "this row attended densely".
   Both `RAD_BUF_PERSIST`. `ix.selected` is shared by the 12 sparse layers and transient, hence the copy.
2. `final_hidden [max_tok, E]` is already `RAD_BUF_PERSIST` and written for all `T` rows. Nothing to add.
3. Nothing else. The draft's layer caches are the trunk's.

### 3.3 Query pass as ops

`n` = drafting sequences, `M = n*B`. Buffers of the sparse attention wire (`aw_sparse`), `tr.x`,
`tr.xn`, `tr.y` are reused (same geometry; the draft ops are declared after the trunk's, so the
planner sees disjoint uses). New buffers are named `dfly.*`.

| stage | ops (all existing unless bold) | output |
|---|---|---|
| A embed | `embed_lookup`(token_ids) -> `dfly.emb [M,E]`; `cast` copy -> `dfly.x` | raw embeddings kept for step F |
| B fusion | `iota`(start 0, step B) -> `dfly.row0 [n]`; `gather_rows`(`dfly.emb`, row0) -> `ea [n,E]`; `gather_rows`(`final_hidden`, `anchor_rows`) -> `th [n,E]`; `rmsnorm`(ea, `fusion_embed_norm`) and `rmsnorm`(th, `fusion_hidden_norm`), copied (`cast`) into the halves of `cat [n,2E]`; GEMM `dfly.fusion` -> `fu`; `mul_weight`(fu, `fusion_gate`); `add`(ea, fu); `scatter_rows`(-> `dfly.x`, row0) | `x[0]` fused |
| C tables | per distinct `l` in `Lidx`: **`dfly_sel`**(`sel_snap.<l>`, `nsel_snap.<l>`, `anchor_rows`, latent group `block_table`, `positions`, `cu_seqlens`) -> `dfly.sel.<l> [M,W]`, `dfly.len.<l> [M]` | per-row page tables |
| D layer i | `rmsnorm`(attn_norm) -> xn; GEMM `attn_q_a` -> `rmsnorm`(q_a_norm) -> qa; GEMM `attn_q_b` -> `q [M,H,256]`; two `cast` copies -> `q_nope [M,H*192]`, `q_rope [M,H*64]`; `rope`(gptj, 1e7) on `q_rope`; GEMM `attn_kv_a` -> `row [M,576]`; `rmsnorm`(kv_a_norm) on the first 512; `rope` on the last 64; `kv_store`(row, sparse group `slot_mapping`, `kv_cache(kv_sparse, Lkv[i])`); GEMM `attn_gate`; `mla_project_kv`(absorb, draft `kv_b`); `mla_attn_sel_sink`(blocks W, block_size 4, draft sinks, cache `kv_cache(kv_sparse, Lkv[i])`, table `dfly.sel.<Lidx[i]>`, `dfly.len.<Lidx[i]>`); `mla_project_kv`(expand); `gate_mul`; GEMM `attn_output`; `all_reduce` (TP > 1); `rmsnorm`(attn_post_norm) -> y; `add`(x, y); `rmsnorm`(ffn_norm) -> xn; GEMM `ffn_gate_up`; `silu_mul`; GEMM `ffn_down`; `all_reduce`; `add`(x, y) | `x` |
| E final | `rmsnorm`(`dfly.output_norm`) -> `dfly.z [M,E]` | |
| F chain, k = 0..depth-1 | `iota`(start k+1, step B) -> rowk; `gather_rows`(`dfly.z`, rowk) -> `zk [n,E]`; prev embedding: k = 0 `gather_rows`(`dfly.emb`, row0), else `embed_lookup`(`dfly.tokens[:, k-1]`); `rmsnorm`(zk, `correct_hidden_norm`) and `rmsnorm`(prev, `correct_embed_norm`) into `cat` (hidden first); GEMM `correct_gate` and GEMM `correct_up` into the halves of `gu [n, 2E]`; `silu_mul`; GEMM `correct_down`; `add`(zk, delta) -> `zc`; `logits_gemm`(zc, trunk head shard) -> `dfly.logits [max_seqs, V/world]` f32; `cast` f32->bf16; `row_topk`(R, `vocab_off`); TP > 1: `all_gather` + `row_topk_merge`; `cast` i32 of candidate column 0 -> `dfly.tokens[:, k]` | proposals |

Notes on the table:
- Order inside a layer matters once: `kv_store` of all `M` rows before `mla_attn_sel_sink` (the
  block's rows read each other causally through `len`). The trunk's `Attn::front`/`core` already
  has this order.
- The attention is the trunk's absorbed path only. No non-absorbed path, no `mla_attn_paged_sink`:
  `dfly_sel` emits the identity table for densely attended sources, and by the kernel note
  (`mla_sink_attn.md` §3) the two forms give the same bytes for the same pages in order.
- `logits_gemm` is declared exactly as the trunk's (same weight handle, same encoding rule: bf16 or
  the int8 head), with its own `M` range `[1, max_seqs]` and its own output buffer. The trunk's
  `logits` buffer is not touched: the sampler, logprobs and the step issued ahead own it.
- `row_topk` on libr4d serves bf16 planes only, hence the `cast`. Draft logits rounded to bf16 can
  flip a near-tie; drafts need not be exact. `R` = 8 (room for the coupled draw of §8.2; greedy
  reads column 0; ties go to the lower id in `row_topk` and in the merge, so every rank picks the
  same token).
- Reading a head-interleaved slice out of `q [M,H,256]` with `cast` (source operand with `offset`
  / `cols` inside a wider head) is the mirror of what WP8 added for the write side **(verify the
  read side on libr4d; if it does not resolve, add the row, it is the same kernel)**.
- Rounding points follow torch's bf16 op sequence (each op rounds once). Two known differences from
  vLLM, both at tolerance level and already accepted for the trunk: radiance's `rmsnorm` rounds
  once (vLLM: norm to bf16, then gain), and `silu_mul` rounds once (torch: silu, then product).

### 3.4 Weights: serve the manifest as it is declared

`hyv4flash.cpp:drafter()` declares every `dfly.*` tensor as the checkpoint holds it
(`RAD_ACCESS_RARE`, `RAD_SHARD_NONE`). DECISIONS U4 says "ready implies the final weight set", so the
served graph must **not** rename or re-fuse anything:
- `attn_q_b` stays one tensor `[H*256, 1024]` (the trunk split it into `q_nope`/`q_rope` row maps;
  the draft splits the activation instead, §3.3 stage D).
- `correct_gate` and `correct_up` stay two tensors (two GEMMs).
- Access class and sharding change at declare only. **(verify: neither is part of the stored
  container; `tests/hyv4flash_convert_test.cpp` must show an unchanged tensor list and bytes for
  `dfly.*` before and after WP13.3.)** If sharding turns out to be stored, the friend re-converts
  the 1.6 GiB of `dfly.*` only (`rad-convert` from an earlier container, GUIDE §3.2).
- Encodings: bf16 linears, f32 gains and sinks, as declared. A quantised `dfly.*` linear is refused
  by name until WP13.9.

### 3.5 Three core contact points (no ABI change intended; each needs a test, not an assumption)

1. **Draft floor after a prefix-cache hit.** `block_begin` computes
   `reach = lo <= 0 ? true : (window > 0 && next_lo - window + 1 >= lo)`; `lo` is the hit length for
   a sequence that never had a draft cache. With `window = 0` a sequence that started from a prefix
   hit would **never draft**. DFly has no draft cache, so reach is always true in fact. Without a
   core change: declare `window = 1` (then `reach` holds for every `next_lo >= lo`, and the
   "draft refill" ends a hit one token early, which is harmless). The clean alternative is an
   additive export in the spirit of `rad_drafter_shared_kv` for BLOCK drafters (no context pass, no
   floor, no refill); propose it upstream, do not block on it. **(verify by a scheduler test: prefix
   hit, then drafts from the first decoded token.)**
2. **An empty context pass** must be legal (a `block_pass` that issues no op). It costs one barrier
   round a step; the export of point 1 would remove it.
3. **`ADV_BLOCK` positions every paged group's slot mapping** for the query rows at the committed
   end, including the trunk's sparse group (DFlash2 only ever wrote its own group). **(verify by
   dumping the sparse group's `slot_mapping` in a query pass after a partial rejection.)**

Also to confirm while there: an op issued in a draft pass may name a trunk layer's cache
(`kv_cache(kv_sparse, Lkv[i])`); the builder's only check in this area is for
`rad_drafter_shared_kv`. No new `rad_bind_layer_kv`: the layer is bound by the trunk.

### 3.6 Tensor parallelism (WP13.6)

As the trunk's sparse layers (PLAN §1.5), which is also what vLLM requires ("matching TP"):

| part | sharding |
|---|---|
| `attn_q_b`, `attn_kv_b`, `attn_gate` rows, `attn_sinks`, `attn_output` columns | whole heads a rank (32 / 16 / 8), `RAD_SHARD_ROW` / `COL` |
| `ffn_gate_up` rows (two row parts), `ffn_down` columns | 9216 / 4608 / 2304 |
| `attn_q_a`, `attn_kv_a`, norms, fusion, correction, `dfly_sel`, embedding | replicated (D8: small; fusion + correction are 126 MiB) |
| draft logits | the trunk head's vocabulary shard; `row_topk` -> `all_gather` -> `row_topk_merge` (DFlash2's pattern) |

Two exact `all_reduce` a draft layer (`decl_all_reduce`, exact = 1). The latent cache is replicated,
so every rank stores the same 6 rows. Proposals are equal on all ranks because everything upstream
of the merge is replicated or exactly reduced and the merge is deterministic. Refuse by name when
the draft's heads or MLP width do not divide by the world (as `split_ranks` does).

---

## 4. New op: `dfly_sel`

The only new op. Integer only; libref, libavx and libr4d must agree byte for byte.

| op | params | operands |
|---|---|---|
| `dfly_sel` | `M`(range) `blocks_in` `blocks` `ratio` `seqs` | `sel` i32 `[R, blocks_in]`, `nsel` i32 `[R]`, `src` i32 `[S]`, `bt` i32 `[S, pitch]`, `pos` i32 `[M]`, `cu` i32 `[S+1]` -> `out` i32 `[M, blocks]`, `len` i32 `[M]` |

Constraint: `blocks >= blocks_in + 1 + (rows_per_seq - 1) / ratio` is the caller's to guarantee
(the op checks every write against `blocks` and fails the row, see below).

For sequence `s` with rows `r` in `[cu[s], cu[s+1])`:
```
a    = pos[cu[s]]                       the anchor's position (first row of the sequence)
g0   = a > 0 ? (a - 1) / ratio : 0
k    = src[s]
if k < 0:                               the card did not position this row
    out[r][*] = -1 ; len[r] = 0         (attention then yields 0; the proposal is verified anyway)
n    = nsel[k]
if n > 0:   nrec = n - 1 ;  out[r][c] = sel[k][c]      for c < nrec      (the selecting source)
else:       nrec = g0    ;  out[r][c] = bt[s][c]       for c < nrec      (the dense source: identity)
e    = pos[r] / ratio - g0                                   further pages, 0..(rows-1)/ratio + 1
out[r][nrec + c] = bt[s][g0 + c]        for c = 0..e
out[r][c] = -1                          for c > nrec + e
len[r] = nrec * ratio + (pos[r] - g0 * ratio) + 1
```
If `nrec + e + 1 > blocks` or `nrec > blocks_in`: `out[r][*] = -1`, `len[r] = 0`, and libref
returns `RAD_E_SHAPE` in debug builds. `sel[k][n-1]` (the source's own page) is not copied; it is
`bt[s][g0]` and the op reads it from `bt` (a tap can assert the two are equal).

Why the dense branch is exact: `nsel = 0` is written only on steps where every sequence ends at
`<= group_topk * ratio`, so `g0 <= G - 1` and all `g0` closed groups are the recalled set.

**libr4d kernel `dfly_sel_i32` (gfx1201).** Grid `(M, ceil(blocks / 64))`, one workgroup of 64
threads a (row, 64-column chunk). Each thread recomputes the row's six scalars
`(a, g0, k, n, nrec, e)` from `pos`/`cu`/`src`/`nsel` (5 loads) and writes its own column: no LDS,
no scratch, no barrier.
The row's sequence is found as in `qsa_select_i32` (binary search of `cu`, `seqs <= 4096`). `len` is
written by the thread with column 0. At `M = 384`, `blocks = 514`: 3456 workgroups, under 0.8 MiB
written. No tuning axis. libavx: two `memcpy` and a fill a row.

Validation: `tests/ref_hy4_ops_test.cpp` cases from §7.2 (c); `tests/r4d_emu/test_dfly_sel.cpp`
byte-equal to libref; `r4d_selftest` row; `rad-kbench` fixture row in `data/kernels_hy4.rkb`;
`avx_check`.

**Existing kernels at new parameters (no new code expected, check `--debug-selection` shows no
host fallback in a draft pass on a card):** `mla_attn_sel_sink_bf16` at `blocks` 514 (the schema
does not constrain `blocks`; `M <= 64` takes the 8-split path, above that the unsplit one: a
drafter needs no row-count invariance); bf16 `gemm_nt` at `K` 6144 (fusion, correction),
`N` 18432 / world, `N` 576; `row_topk` at `N = V / world` (a multiple of 8 at 1, 2, 4 ranks: 120832,
60416, 30208; 3584; 320); `cast` f32->bf16 at `n = V / world`; `mul_weight` with an f32 `[E]` gain;
`gather_rows` / `scatter_rows` bf16; `iota`.

Not designed here because they are optimisation, not function (§9): a shortlist head for the five
chain logits, and a fused fusion/correction row.

---

## 5. What the drafter owns

| object | kind | size | written | read |
|---|---|---|---|---|
| `dfly.sel_snap.<l>`, `dfly.nsel_snap.<l>` | PERSIST i32 `[max_tok, G]`, `[max_tok]` per distinct `Lidx` | 5 x 16 MiB at `max_tok` 8192, a rank | every trunk step, all its rows | query pass, row `anchor_rows[s]` |
| `final_hidden` | PERSIST bf16 `[max_tok, E]` (exists) | 48 MiB at 8192 | every trunk step | query pass, row `anchor_rows[s]` |
| `dfly.tokens` | PERSIST i32 `[max_seqs, depth]` | | query pass | `ADV_BTAKE` |
| `dfly.sel.<l>`, `dfly.len.<l>` | transient i32 `[max_seqs*B, W]` | 0.8 MiB each at 64 sequences | query pass | query pass |
| `dfly.emb`, `dfly.x`, `dfly.z` | transient bf16 `[max_seqs*B, E]` | | | |
| `dfly.logits` | transient f32 `[max_seqs, V/world]` | 29.5 MiB at 64 sequences, one rank | | |
| KV | **none of its own** | | rows `a..a+B-1` of trunk layers `Lkv` in the sparse latent group | the same rows, plus committed rows |

No KV group, no ring, no window, no linear state, nothing in the prefix cache, nothing to snapshot
or restore. A prefix hit, a host-tier restore or a checkpoint resume leaves the drafter ready: the
next trunk step rewrites all three snapshots before the next query pass reads them.

The snapshot can later be compacted to the rows that can be anchors (decode rows and each prefill
sequence's last row: `gather_rows` by `out_ids` into `[max_logit_rows, G]`, with an index map from
`anchor_rows`); 80 MiB a rank is acceptable for the first version and is stated in the startup note.

Interaction with WP14 (streamed sparse KV): if the sparse latent group streams, `dfly.sel.<l>` goes
through `kv_resolve` exactly like the trunk's table (the recalled pages were resident for the trunk
step just run, so this is a re-resolve of the same pages plus at most three new ones).

---

## 6. Rollback, precisely

### 6.1 Terms

A verify step for a sequence with committed length `c` (positions `0..c-1` final) runs rows at
`c .. c+k` (`k` drafts, `k <= max_spec`): row 0 carries the bonus token, rows `1..k` the drafts.
The sampler accepts a prefix: the step commits `n_acc` tokens (`1 <= n_acc <= k+1`,
`RadBatch::num_accepted`, one-based, handed to the **next** step). The new committed length is
`c' = c + n_acc`. Rows at positions `>= c'` were speculative and rejected.

Rollback in this model is a **position rewind**. Nothing is undone on the card; correctness rests on
two rules that every state below satisfies:
- (R1) position-keyed state is written by a step for every position the step carries, **before** any
  of that step's rows reads it;
- (R2) every reader bounds what it reads by positions `<=` its own.
Then state at positions `>= c'` is stale but unreachable until the step that carries those positions
again rewrites it. The one state that is not position-keyed (Engram history) is rolled back
explicitly.

### 6.2 State by state

| state | keyed by | a verify writes | after rejection (positions >= c') | why it is safe | drafter |
|---|---|---|---|---|---|
| sparse latent cache (FULL, block 4, 12 layers) | position | rows `c..c+k`, `kv_store` before attention | stale rows stay | R1; R2 via `seqused`/`cu` (paged), `len` (sel); the non-absorbed prefill reads no cache | writes `a..a+B-1` in layers `Lkv` only (§6.3) |
| sliding latent cache (WINDOW, 30 layers) | position | same | same | same; the window trims on commit only | untouched |
| indexer group keys (FULL, one pooled key a page) | group `g` | every group whose last token is at `<= c+k`, **including groups closed by speculative rows** | key of group `g` with `4g+3 >= c'` is stale | invisible: `qsa_select` takes candidates below `n = min(nc, pos/4)` and `nc[s] = (last position + 1)/4` is recomputed by `qsa_work` each step. Rewritten: `qsa_work` lists blocks `p0/4 .. (p1+1)/4 - 1` for a step over `p0..p1`; every stale `g` satisfies `g >= c'/4` (from `4g+3 >= c'`), so the step that completes it pools it again, and `idx_pool_gated` is issued before `idx_score` | untouched |
| indexer tail (LINEAR, `[ratio + max_spec, 128]` raw keys) | `p % ring` | rows `c..c+k` at `(c+j) % ring`, after `qsa_work` read the tail | rejected positions' rows are stale | the next step rewrites its own positions before use. The keys it needs from **before** its start `c'` are `q in [4*(c'/4), c')`, at most `c'-3..c'-1`. Those at `>= c` are accepted rows' keys (final). Those at `< c` were not overwritten by the verify: a collision needs `(c+j) - q = m*ring`, but `(c+j) - q <= k + 2 < ring = max_spec + 4` | untouched; a draft pass must **never** issue `qsa_work` / `qsa_tail_store` / `idx_pool_gated` |
| Engram history (CONV i32, the last 2 token ids, per Engram layer) | sequence | `engram_ids` advances it over the step's rows | not position-keyed | `engram_ids` on the decode half takes `num_accepted` and rolls the state back to the accepted prefix before advancing (`PleLayer::ids`' contract, WP1/WP3) | untouched; the draft has no Engram and a draft pass issues no `engram_ids` |
| `dfly.*_snap`, `final_hidden` | trunk row | all rows, every trunk step | n/a | read only at `anchor_rows[s]`, the row of position `c'-1`, an accepted row | read |
| `dfly.tokens` | sequence | | | `ADV_BTAKE` copies it behind the bonus token; the next verify decides | written |

**This is the prerequisite of any speculation on Hy4 (the levers note and PLAN §5 both name it
as open), so it is settled here and gated first (WP13.0), before any drafter code:**
- indexer tail: correct by the ring-size argument in the table; no operand needed;
- pooled group keys of speculative groups: stale keys are unreachable (`nc`, the per-query bound)
  and re-pooled when the group completes again;
- Engram CONV history: rolled back by `engram_ids`' `num_accepted` operand on the decode half.
These are arguments from the sources (`ref_qsa.cpp`, `hyv4flash_indexer.h`, `hyv4flash_engram.h`),
not measurements. WP13.0 turns them into tests that need no drafter: prompt lookup with a
deliberately wrong oracle file produces every rejection pattern.

The trunk half of this table is today's behaviour (WP8 measured verify rows bit-equal to plain
greedy on `gpu-mini`, TP1 and TP2, with `--spec-ngram 7 --debug-draft-oracle`). WP13 adds tests that
pin it for the cases the oracle run may not have reached: contexts past 2048, rejections that leave
a group closed by a rejected row, and reject-all (§7.4).

"`num_accepted` semantics must cover speculative groups" (PLAN §5) resolves to: **the indexer needs
no `num_accepted`.** Group keys and tails are position-keyed and re-derived; only Engram consumes it.

### 6.3 What the drafter adds, and why it cannot disturb the trunk

Timeline on each rank's stream (serial; the step issued ahead is queued behind the passes):
```
trunk step N (rows c..c+k)  ->  ADV_ACCEPT (c' known on the card)  ->  [context pass: empty]
  ->  ADV_BLOCK (query rows at a = c' .. c'+B-1)  ->  query pass  ->  ADV_BTAKE  ->  trunk step N+1 (rows c'..c'+k')
```
1. **Reads.** The query pass reads (i) latent rows `< a` of layers `Lkv`: committed (rows `c..c'-1`
   were written by step N's accepted rows, the same bytes a plain decode would have written, by the
   verify-equals-decode property); (ii) the snapshots at row `anchor_rows[s]`, the accepted row of
   position `a-1`; its selection lists groups all of whose tokens are `< 4*g0 <= a-1`, i.e.
   committed tokens, with pooled keys computed from accepted rows only; (iii) latent rows `a..a+j`
   that the query pass itself stored in this pass. It never reads a rejected row: rejected rows sit
   at positions `>= c' = a`, and each of those positions is overwritten by the draft layer's own
   `kv_store` before that layer's attention (R1) in the layers the draft reads (`Lkv`); other
   layers are not read.
2. **Writes.** Latent rows `a..a+B-1` of the five layers `Lkv`, positions `>= c'`, plus
   transients. By R1/R2 for the trunk these rows are unreachable for the trunk until a trunk step
   carries those positions, and that step stores its own rows first. If step N+1 carries fewer rows
   than `B` (adaptive window, a held lookup, the sequence ended) the leftover draft rows stay as
   ordinary stale rows past the committed end, the same class as rejected verify rows.
3. **Not written:** group keys, tails, Engram history, the sliding group, `nc`, any trunk buffer the
   step issued ahead or the sampler reads (`logits`, `logit_input`), the block tables.
4. **Pages.** `block_begin` reserves pages up to `next_hi + block` in every group before the pass;
   a page shared with a cached prefix that contains the committed end is already private, because
   verify rows write there too. **(verify with the prefix cache on.)**
5. **Tree verify** (`rad_drafter_tree`) moves K/V after acceptance and is not declared. `--kld-record`
   and calibration draft nothing.

Consequence: with the drafter on, the trunk executes exactly the row sequence it would execute
under prompt-lookup speculation with the same proposals, and its state transitions are those of
§6.2. Losslessness of the text follows from the verify rule (GUIDE §4.6) and the trunk's
verify-equals-decode property, which the drafter does not weaken. §7.4 tests exactly this.

### 6.4 Failure modes this design rules out (each has a test in §7.4)

| mistake | symptom |
|---|---|
| draft pass issues the indexer key side or `engram_ids` | fluent wrong text some steps later (a tail or history advanced by drafts) |
| snapshot row taken at the step's last row instead of `anchor_rows` | correct text, acceptance collapses after any rejection |
| `nsel_snap` not cleared on a non-selecting step | a short sequence drafts from another step's stale selection; correct text, low acceptance |
| draft rows stored after the attention | row `j` reads the previous step's rejected rows at `a..a+j`; correct text, low acceptance |
| host `positions_h` used to build the table | table built for the furthest position; wrong pages after a rejection |
| `block = depth` declared | proposals shifted by one; acceptance at chance (rad_builder.h warns about exactly this) |

All but the first are invisible in the output text, which is why acceptance against the reference
(§7.3, §7.5) is a gate and not a nice-to-have.

---

## 7. Validation without an HF reference

### 7.1 The reference: `ref/dfly_ref.py` (torch, CPU, f32 and bf16 modes)

A literal port of the vLLM forward of §1.3, one function per vLLM module, with the vLLM file and
class named above each. Plain `nn.functional.linear`; attention written the textbook
(non-absorbed) way: `k_t = [W_UK lat_t | rope_t]`, `v_t = W_UV lat_t`, sink logit appended to the
softmax. Rotary and RMSNorm taken from the HF port (`research/hf/modeling_c2ec8505.py`) so the
conventions are the trunk's reference's.

Inputs (one npz a draft): the draft weights from the dummy checkpoint
(`ref/mini_ckpt/<preset>`: `model.dfly_layers.*`, `model.embed_tokens.weight`, `lm_head.weight`),
and the **trunk state**: `t` (final hidden at `src`), the bonus token, `a`, for each layer in `Lkv`
the cache rows `[a, 576]`, for each layer in `Lidx` the selection at `src` as group indices
(or "dense"). Output: every tensor of §1.3 by name, and the 5 tokens.

Two ways to get the trunk state:
- **conditioned (the gate):** from radiance's own taps of the trunk step
  (`dbg.final_hidden`, `dbg.attn_kv_latent.<l>` plus a new tap `dbg.attn_kv_rope.<l>` for the 64
  rotated key columns, `dbg.index_sel.<l>` / `dbg.index_nsel.<l>`). This isolates the drafter: the
  trunk is already gated against HF (WP4, WP8).
- **end to end (informative):** from the HF reference run with two extra hooks (cache row and final
  hidden). Differences are the trunk's known floor plus the drafter's.

### 7.2 How the port itself is checked (it is the weakest link: vLLM's DFly needs CUDA and cannot run here)

(a) **Attention against HF ground truth.** Feed the port's attention function a *trunk* sparse
    layer's weights and that layer's dumped inputs; compare with the HF dump `attn_out` of the same
    layer (`ref/dumps/<preset>/f32`). The draft layer is the same module family (gated MLA, sink,
    post-norm), so this pins rotary convention, `kv_b` row order, scale, sink, gate and norm of the
    port. Gate: relL2 <= 1e-5 in f32.
(b) **Fusion and correction against vLLM's own classes.** `HYV4DFlyFusion` and
    `HYV4DFlyHiddenCorrection` are small torch modules; import them from the checkout on CPU with
    stub linears if the import chain allows, else compare against a second, independently written
    copy. Gate: bit-equal in f32.
(c) **Key set.** Port `_reference_reuse_indices` (vLLM's test) and `expand_groups` verbatim; for
    random group selections and anchors (all residues of `a mod 4`, `a` at 1, 2, 16, 17, 2047, 2048,
    2049, 2052, dense and selecting sources, `B` 2..6) the token set of `dfly_sel`'s table (pages
    expanded, cut at `len`) equals the port's token set for every row. These cases become libref's
    golden file (`tests/data/hy4/gen_hy4_golden.py`).
(d) **Independent read.** One reviewer agent that has not seen this note or the port's author's
    reasoning gets the vLLM files and the port and lists every semantic difference.
(e) **Real weights** (§7.5): a wrong forward gives chance acceptance. This is the only check of
    the trained conventions as a whole and it is the friend's.

### 7.3 Parity per tensor (host first, then cards)

Taps under the WP2 contract, written in the query pass (the engine already dumps after each block
pass, tagged `query`): `dbg.dfly.embed`, `dbg.dfly.fused` (anchor rows), per draft layer
`attn_in_norm`, `attn_q_resid`, `attn_kv_row`, `attn_out`, `resid_mid`, `ffn_in_norm`, `mlp_out`,
`resid`; `dbg.dfly.final`; per chain step `corrected.<k>`, `cand_idx.<k>`, `cand_val.<k>`;
`dbg.dfly.tokens`; `dbg.dfly.sel_ids.<l>` (the table as group indices: `dfly_sel` again over the
`index_sel` tap and an identity table, the trunk tap's trick) and `dbg.dfly.len.<l>`.

Method and tolerances are WP4's (`ref/README.md`), reused, not reinvented:
- floor = relL2 of the ref's bf16 run to its f32 run per tensor; gate
  `relL2(radiance, ref f32) <= max(1e-3, 2.0 x floor)`;
- tables and `len`: exact;
- chain: **teacher-forced** (the ref takes radiance's `prev` tokens) so one near-tie does not fail
  the later steps; token `k` must equal the ref's argmax unless the ref's margin to radiance's pick
  is below 2 bf16 ulps of the top logit (the `cast` to bf16); report the count of such ties;
- libavx vs libref and libr4d vs libref: WP4's `avx` profile.

Scenarios (each on `mini`, `mini6d2`, `gpu-mini`; host libref and libavx, then libr4d TP1 and TP2):
first draft after a prompt (anchor row is a prefill row, non-absorbed and absorbed), after an
all-accepted verify, after a partial and a total rejection, anchor at every `a mod 4`, dense source
with the block crossing the dense limit (`a` = limit - 2), selecting source (`mini`: context 17, 20,
33, 64; `gpu-mini`: 2049, 2052, 3000), two sequences of different lengths in one pass, a mixed
step.

### 7.4 Losslessness and rollback

Greedy, fixed prompts, text **and logprobs** compared byte for byte (WP8's `work/wp8/spec.py`
method) between:
1. drafter off (`num-speculative-tokens: 0`);
2. drafter on, depth 5, 3 and 1;
3. drafter on with **sabotaged proposals** (`RADIANCE_HY4_DFLY_SABOTAGE=1`: `dfly.tokens` replaced by
   `(token + 1) % vocab` before `ADV_BTAKE`): every draft rejected, the maximum of rollbacks;
4. `--spec-ngram 5 --debug-draft-oracle` (every draft accepted) with the drafter declared but
   declining.
On prompts that cross the window (512) and the dense limit (2048) during generation, that end on
every residue mod 4, with the prefix cache on (second turn of a chat), with 1 and 4 concurrent
sequences, TP1 and TP2 (host fake ranks, then cards). At temperature > 0 with a fixed seed, runs 1
and 2 are equal as well (a point-mass proposal is accepted iff the target's own draw equals it).

State probes for §6.2 (host, libref): after each step dump the tail ring, the group-key pages and
the Engram history of a sabotaged run and of the plain run at the same committed length; they must
be equal on every committed position / closed group (stale entries past the committed end are
masked by the comparison, and the test asserts they are never selected: `index_sel` equal).

### 7.5 Acceptance-rate sanity

Random draft weights accept at chance (1/vocab), so acceptance means nothing on the stock dummy
checkpoints. Two checks:
- **Distilled mini (here, host).** `ref/dfly_train.py` trains only `model.dfly_layers.*` of `mini`
  with the port (it is differentiable) to predict the dummy trunk's own greedy continuations,
  teacher-forced over the 5 positions; a random trunk's greedy output is strongly repetitive, so
  this is learnable in minutes on CPU. Write the result as checkpoint `mini-dfly` and convert it.
  Gates: (i) the port's offline acceptance on held-out prompts is at least 20x chance at position 1;
  (ii) radiance's measured acceptance per position (`radiance:draft_*` metrics,
  `RADIANCE_DEBUG_DRAFT`) agrees with the port's offline number within 2 points absolute over at
  least 2000 drafted positions; (iii) acceptance does not fall after rejections (catches the stale
  snapshot and shifted-block mistakes of §6.4). If training does not reach (i), report it and drop
  to the friend's run; do not weaken the gate.
- **Real weights (the friend, TP4).** Greedy and sampled, code-heavy prompts (the workload that
  matters here) and prose: tokens a step, acceptance by position (must fall monotonically from
  position 1), draft pass time against a verify row's time. vLLM's own acceptance on the same
  prompts if he can run it. A chance-level position 1 means the forward or a convention is wrong.

---

## 8. Sampling, coupling, confidence

### 8.1 First version: greedy proposals
`argmax` per chain step. Under the engine's standard verify a deterministic proposal is exact at
every temperature and keeps fixed-seed tokens (GUIDE §4.6). At temperature > 0 acceptance is the
target's probability of the proposal, so it is lower than vLLM's `probabilistic` mode.

### 8.2 Coupled proposals (WP13.8, wanted for sampled serving)
Upstream (and so the Hy4 branch) already couples a block drafter's **chain**: `step_state_init`
accepts `RAD_DRAFT_BLOCK` under `--draft-coupling`, finds the planes `mtp_couple_draws` and
`mtp_couple_state [rounds, rows, RAD_MTP_COUPLE_STATE_WORDS]` by name, stages the draws before the
query pass and reads the states back after it, "position l standing for round l + 1". The verifies
(`mtp_couple_verify`, `mtp_trav_verify`, the block marking) read bound states only.

DFly2 fills those states with the **serial** op, once a chain step: step `k` hands its top-R
shortlist (`row_topk` ids and values after the rank merge; R >= 8) to
`mtp_couple_propose(round = k+1, max_rounds = depth)`, which builds `Q_k` over the eight best,
draws `x_k ~ Q_k` at the staged draw and stores `Q_k` exactly in state row `(k, sequence)`.
`x_k` then feeds step `k+1`'s correction. `Q_k` is a function of the committed text and of
`x_0..x_{k-1}` only, which is all the maximal coupling needs. A greedy request gets status 0 and
the argmax (the op's mode 0), so greedy text is unchanged. No new kernel.
**(verify: the state row index the engine binds is the query pass's sequence order; the shortlist
values must be the logits the law is built from, so under coupling take them from the f32 logits,
i.e. give `row_topk` an f32 row on libr4d or accept bf16-rounded values on both host and card. Q
only has to be the law the draw was made from, so either is exact as long as the stored Q is the
one sampled.)** Temperature: the op's `q_temp` handling as the serial heads use it.

Tests: `tests/mtp_rrs_test.cpp`'s chi-square over the first three tokens, row-wise and traversal,
with the two negative controls, driven by DFly's chain.

### 8.3 Confidence head
`dfly.confidence.{weight [1, E], bias [1]}`. No forward exists in vLLM or SGLang. By the paper's
D-cut it is a per-token survival estimate `c_k` in (0, 1) with prefix survival `prod c_k`, used to
choose how many drafts to verify. The plausible form is `c_k = sigmoid(w . h_k + b)`; whether `h_k`
is `z[k+1]` or the corrected `zc_k` is **unknown**.

Decision: not served in WP13. It is not needed for correctness or for `draft-window: adaptive`
(which learns acceptance per position online). When wanted: compute both candidates
(`gemm_nt` with `N = 1` plus the bias on the host from a readback of 5 floats a sequence is enough
for an experiment), record them with the verify outcomes on real weights, and keep the input whose
`c_k` is calibrated (reliability curve, AUC); only then wire `prod c_k` into the verify depth as a
per-sequence stop. That is the friend's experiment plus a small package, listed as 13.10.

### 8.4 The Flash-Next / Flint speculation levers on DFly2, and what to port

Numbers quoted for these levers (`notes/flashnext-levers.md`: Flint's DFlash2 +69 % chain 3, +74 %
chain 7, +86 % tree 4 over no speculation; Flash-Next +28.7 % for coupling + traversal + adaptive
over an uncoupled chain) are Flint and Flash-Next measurements. Nothing is measured on Hy4; its
verify rows (top-16 of 512 experts, latent MoE) are expected to cost more than Flint's, which
favours short, coupled, adaptive chains.

| lever | where it lives | on DFly2 | to do in the Hy4 branch |
|---|---|---|---|
| coupling of the chain (`draft-coupling`) | upstream, in the Hy4 branch | applies, through `mtp_couple_propose` (§8.2), not `dflash_couple_propose` | plugin only (WP13.8) |
| traversal / block verify (`draft-verify`) | upstream | applies unchanged: both read bound states, and DFly's states have the chain's wire (round k+1, nkids 0) | nothing beyond coupling |
| `draft-window: adaptive` | upstream | applies as for any block drafter (first `min(D, stop)` rows) | nothing |
| `draft-short-from` | upstream | applies; cheap because DFly is causal (§3.1) | declaration (WP13.9) |
| **adaptive depth = the verified prefix** (`draft-depth adaptive` over a block) | **flint-spec only** (76649b6b: `core/sched/draft_depth.h` block rules, `block_depth_decide`, `couple_settle`, `issue_ahead` waiting for the walk's event) | applies exactly: after the query pass all laws `Q_1..Q_D` are known before the next verify is issued; draft j is verified iff j-1 is and (`j <= min` or `conf(j) >= stop`). `stop` and `min` stay, **`lag` drops** (no rounds to stop; the pass costs the same). `top` may read `Q_j`, `path` reads `Q_i(x_i)` for `i < j`; never `Q_{j+1}`, never the target | **port** (below) |
| **load-aware depth** (`draft-depth-load`) over a block | flint-spec only (same commit) | applies; statistic `top`; a block step counts as one round; no round price | port with the above |
| **suffix steps beside a block drafter** (`draft-suffix`) | flint-spec only (same commit); upstream refuses it on a block drafter | applies and is simpler than for DFlash2: DFly has no draft cache, so a suffix step leaves no hole; the snapshots are rewritten by every trunk step, suffix steps included; the window stays the block | port with the above |
| DFlash2 tree under DARTree (`dflash_couple_tree`, 12716103, e9be438e) | flint-spec only | **does not apply** (no lattice, §1.5); a DFly tree would be a serial-head tree and needs tree attention and tree rollback in the trunk | do not port |
| `draft-coupling-temp*`, `draft-calib online` | dartree only | optional later | do not port now |

**The port.** One commit, core only: 76649b6b (7 files: `core/engine.cpp`, `engine.h`,
`engine_bringup.cpp`, `engine_draft.cpp`, `core/sched/draft_depth.h`, `tests/mtp_rrs_test.cpp`,
`tests/sched_test.cpp`; +839 / -56). It sits on `flint-spec`, which is built on dartree, while the
Hy4 branch is on upstream main, so it will not apply cleanly and must not drag dartree in:
- cherry-pick `-x` onto a branch `hy4-blockspec` off the Hy4 integration branch, resolving by hand;
  drop every hunk that refers to dartree-only code (`draft-calib online`, DARTree's `note_grown`,
  `draft-tree-*`, the coupling-temperature autofit). What must survive: the block rules in
  `draft_depth.h` (`draft_depth_block_stop` and the block form of `draft_depth_select`),
  `block_depth_decide` and `couple_settle` in `engine_draft.cpp`, the bring-up changes that stop
  refusing `--draft-depth adaptive` and `--draft-suffix` on a block drafter, `issue_ahead` /
  `expect_draft` planning each row's prefix, and the block sections of the two tests (they use a
  fake block drafter and need no DFlash2);
- do **not** rebase or merge the Hy4 branch onto dartree (levers note §3). If upstream takes the
  flint-spec work first, take it by merging upstream instead and drop this package;
- the rules read only the coupled states, so the port is independent of which op wrote them: the
  ported tests pass with the fake drafter before DFly2 is wired, and DFly2 then needs nothing more
  than §8.2.
The exactness arguments and negative controls are the design doc's (§4, §5 there) and carry over
without change, because DFly's `Q_k` is a law at the node `x_1..x_{k-1}`, the property those
arguments use.

The confidence head (§8.3), once understood, is a second source for `conf(j)`; the stored `Q` is
enough to run the rule without it.

Two engine flags the levers note lists that a launch-heavy draft pass will want, outside this
design: `chain_pays` (after the `fix/sync-audit` kernel fixes) and `rad_kv_group_replicated`.

---

## 9. Cost, and what to optimise after correctness

Weight bytes read by one query pass at one rank, bf16, any `n` (arithmetic, not a measurement):
5 layers 1.45 GiB + fusion 36 MiB + 5 x correction 90 MiB + **5 x lm_head 708 MiB = 3.46 GiB**:
5.4 GiB, two thirds of it the head. Divide by the world for the sharded parts. A quantised trunk
step reads far less than its bf16 size, so an unoptimised bf16 draft pass can cost as much as a
trunk step; DFly only pays if this is brought down. In order of expected gain (each is a WP16-style
package, measured before it is kept):
1. **Head: int8.** If the recipe made the head int8 the chain reads it at half the bytes with no
   work (`logits_gemm` takes the stored encoding).
2. **Head: shortlist.** One full head pass over the 5 **uncorrected** rows `z[1..5]` (one weight
   read instead of five), top-64 a row, then each chain step scores `zc_k` against the 64 gathered
   head rows only. Changes proposals slightly (never text). Needs a gather of head rows by id
   (`logits_gemm_vocab`'s machinery or a small `gemm_rows` row); design when measured.
3. **Draft linears int8** (`dfly.*` under an RTN `i8` recipe rule, no calibration): halves 2 GiB.
   vLLM serves the draft in FP8, so 8-bit is within what the checkpoint was validated at.
4. **Short block under load** (`rad_drafter_short_block`), free here because the draft is causal.
5. An additive core export that drops the empty context pass (§3.5 point 1).
Launch count of a pass: about 26 ops a layer x 5, 12 for the fusion and tables, about 14 a chain
step x 5: roughly 210 launches, mostly dependent; chained launches (`rad_chain_on`) apply.

---

## 10. Work packages

Implementation by Opus, tests by Sonnet, each in its own worktree off
`claude/peaceful-wright-hhqnhi` (AGENT_RULES.md). Host first; a card only where stated, under the
GPU lock, live server stopped. A package is done when its criteria hold and its note is in
`notes/wp13-<n>.md`.

| WP | what | impl / tests | needs | GPU |
|---|---|---|---|---|
| 13.0 | **rollback gate**: trunk state under rejected rows (no drafter needed) | S (fixes O) | - | host, then short |
| 13.1 | reference port and fixtures | S / S, review O | - | no |
| 13.2 | `dfly_sel`: libref, schema, OPS.md | O / S | 13.1 (goldens) | no |
| 13.3 | plugin: snapshots, query pass, declaration, taps; TP1 host | O / S | 13.2 | no |
| 13.4 | host parity, losslessness, rollback probes | S (fixes O) | 13.1, 13.3 | no |
| 13.5 | `dfly_sel` on libavx and libr4d | S / S | 13.2 | emu no, card short |
| 13.6 | TP sharding of the draft | O / S | 13.3 | no (fake ranks) |
| 13.7 | cards: `gpu-mini` TP1, TP2 | O / S | 13.4, 13.5, 13.6 | yes |
| 13.8 | coupled proposals (`mtp_couple_propose` in the chain) | O / S | 13.7 | short |
| 13.8b | port of the block levers from flint-spec 76649b6b (adaptive prefix, load-aware, suffix) | O / S | - (core only; before or beside 13.8) | no |
| 13.9 | distilled-mini acceptance, short block, int8 draft, docs and hand-over | S, O for int8 | 13.4 | short |
| 13.10 | confidence head (after the friend's data) | O / S | real weights | - |

**13.0 Rollback gate (Sonnet; Opus fixes).** No drafter. `--spec-ngram 5` with
`--debug-draft-oracle` files built from the plain greedy run: (i) all right, (ii) all wrong,
(iii) right up to a random position per step, (iv) wrong exactly at the row that closes a group,
(v) wrong at the row after it. Prompts and generation lengths chosen so the committed end passes
the window (512) and the dense limit (2048) and lands on every residue mod 4; `mini` (limit 16)
and `gpu-mini` on the host, then `gpu-mini` on a card, TP1 and TP2; prefix cache off and on
(second turn), and one run that resumes from a checkpoint if the engine checkpoints the tail and
CONV groups (report if it does not: the levers note lists it as unchecked).
Accept: text and logprobs byte-equal to the plain run in every case; the §7.4 state probes (tail
ring on committed positions, pooled keys of closed groups, Engram history, `index_sel`) equal to
the plain run's at the same committed length; a ring declared one row too small
(`RADIANCE_HY4_TEST_RING=-1`, test only) is caught, which shows the probes can fail. If any case
differs, stop WP13 and fix the trunk first.

**13.1 Reference port and fixtures (Sonnet; an Opus reviewer for §7.2 d).**
`ref/dfly_ref.py`, `ref/dfly_state.py` (trunk state from radiance taps and from HF hooks),
`minicfg.py` fixes of §2, preset `mini6d2`, regenerated configs.
Accept: §7.2 (a) relL2 <= 1e-5, (b) bit-equal, (c) all cases pass in the port; checkpoint hashes
of `mini`, `mini4`, `mini6`, `gpu-mini` weights unchanged (config files only); the reviewer's
difference list is empty or each item is resolved in the note; `ref/README.md` has a DFly section.

**13.2 `dfly_sel` in libref (Opus; Sonnet tests).** `libref/ref_dfly.cpp`, schema rows in
`ref_registry.cpp` and `r4d_rows.cpp` (the row refuses to launch until 13.5), the OPS.md table row
and section.
Accept: `ctest -R ref_hy4_ops` with the §7.2 (c) goldens, `ctest -R ops_doc`; overflow and `src < 0`
cases yield `len 0`.

**13.3 Plugin (Opus; Sonnet writes `tests/hyv4flash_arch_test.cpp` additions).**
`arch/hyv4flash/hyv4flash_dfly.h` (+ the three trunk additions of §3.2, `step` dispatch on
`draft_pass`, `rad_declare_drafter`, probe, refusals of §2, the sabotage switch, taps).
Accept: declared tensor list and bytes of `dfly.*` unchanged (convert test); fake-builder lint of a
context pass (no op), a query pass at depth 5, 3, 1 and of the trunk step with snapshots; the op
list of a query pass contains none of `qsa_work`, `qsa_tail_store`, `idx_pool_gated`, `idx_score`,
`qsa_select`, `engram_ids`, `ihc_*`; refusals fire by name; `mini` serves with the drafter on the
host (libref and libavx) and the three contact points of §3.5 each have a passing test or a
documented fallback; existing hyv4flash tests and the baseline list unchanged.

**13.4 Host parity, losslessness, rollback (Sonnet; Opus fixes what it finds).**
`run_rad.py --dfly` (collect query-pass dumps, call the port conditioned), the §7.3 scenarios, the
§7.4 matrix and probes on `mini`, `mini6d2` (libref, libavx) and `gpu-mini` (libref, the long
contexts).
Accept: 0 FAIL in the §7.3 profile; tables exact; teacher-forced tokens equal up to counted ties;
§7.4 runs 1 to 4 byte-equal in text and logprobs; state probes equal; each row of §6.4 reproduced
once by a deliberate break (a test-only switch) and caught by the named check.

**13.5 `dfly_sel` kernels (Sonnet).** libavx row; `libr4d/r4d_dfly.hip` + rows + shape hook;
`tests/r4d_emu/test_dfly_sel.cpp`, `r4d_selftest` row, kbench fixture row, `avx_check`.
Accept: byte-equal to libref on every golden and on 1000 random cases in the emulator; selftest
green on a card; no scratch, no LDS.

**13.6 TP (Opus; Sonnet tests).** Sharding of §3.6, the two reductions a layer, the vocab-shard merge.
Accept: fake ranks TP2 (`mini`) and TP4 (`mini4`): proposals equal on all ranks; §7.3 against the
port in the `tp` profile; §7.4 run 2 vs run 1 byte-equal at TP2; refusals for indivisible heads /
widths.

**13.7 Cards (Opus; Sonnet runs the matrices).** `gpu-mini` TP1 then TP2 on libr4d.
Accept: `--debug-selection` shows no host kernel in a draft pass; §7.3 (libr4d vs port, WP4 `hf`
profile) 0 FAIL; §7.4 byte-equal at TP1 and TP2 including contexts 2049 and 3000 and the sabotaged
run; draft pass and verify row times recorded once each (plain observations, not claims);
`hy4flash_ready` still green.

**13.8 Coupling (Opus; Sonnet tests).** §8.2. Accept: chi-square on the first three tokens as
`mtp_rrs_test` does, row-wise and under `draft-verify traversal`, with the negative controls
failing; greedy text unchanged (byte-equal to 13.7's); startup line reports the coupled chain.

**13.8b Block levers (Opus; Sonnet tests).** §8.4 "The port": branch `hy4-blockspec`, cherry-pick
`-x` of flint-spec 76649b6b without its dartree-only hunks. Accept: the block sections of
`tests/mtp_rrs_test.cpp` and `tests/sched_test.cpp` pass with their fake block drafter, including
the two controls that must fail (reading `Q_{j+1}`, admitting `x_j` by `Q_j(x_j)`); every existing
serial-head test unchanged; `--draft-depth adaptive`, `--draft-depth-load on` and `--draft-suffix on`
start on a coupled block drafter and are still refused where the doc says (suffix window above the
block, an uncoupled chain for adaptive depth); with DFly2 coupled (13.8) on `gpu-mini`: greedy text
byte-equal, sampled chi-square passes with adaptive prefix on. Skip the package if upstream has
merged the flint-spec work by then (merge upstream instead).

**13.9 Acceptance and hand-over (Sonnet; Opus for the int8 draft).** §7.5 distilled mini;
`rad_drafter_short_block`; `dfly.*` int8 recipe rule and its parity floor; docs
(`ARCHITECTURES.md` hyv4flash sections, `HY4FLASH-QUANT.md` "not served" list, GUIDE §3.2 line);
`quant.md` hand-over paragraph for the friend's real-weight run (§7.5) and the §8.3 experiment.
Accept: §7.5 gates (i) to (iii) or the documented fallback; docs build (`ops_doc`), no stale
"declared (not served)" text.

**13.10 Confidence head.** Only after the friend's recording; §8.3.

Parallel sets: {13.0, 13.1, 13.8b} first (13.0 is a gate: nothing merges past it red), then
{13.2}, then {13.3, 13.5}, then {13.4, 13.6}, then 13.7 (GPU, serial), then {13.8, 13.9}.

---

## 11. Open points and risks

- **D-1 (user / friend): depth.** Declared depth must equal `max_spec`. Proposed: serve 1..5, refuse
  the drafter (prompt lookup remains) above 5. Confirm.
- **D-2 (user): draft floor.** `window = 1` workaround now, or an additive core export first (§3.5).
  Proposed: workaround now, export proposed upstream.
- **D-3 (friend): does the release's `dfly_layers` carry `post_proj_norm`** for all five layers, and
  are the sinks bf16 there (the dummy generator assumes both)? One `rad-inspect` of his container.
- **D-4 (user): port of the block levers.** Cherry-pick flint-spec 76649b6b into the Hy4 branch
  (13.8b) or wait for upstream. Proposed: port, core only, no dartree; the tree commits stay out.
- **R-0: trunk rollback is argued, not measured, past 2048 and across group-closing rejections.**
  13.0 gates everything.
- **R-1: the port is unverified against a running vLLM.** Mitigated by §7.2 (a) to (d); closed only
  by §7.5 on real weights. Until then the drafter is "implemented, text-safe, acceptance unproven".
- **R-2: cost.** §9: without the head shortlist or an int8 head, DFly may not pay at one or two
  ranks. Losslessness makes shipping it off-by-default-for-speed safe; measure before recommending.
- **R-3: the three contact points of §3.5** are read from the scheduler, not run. 13.3 tests them
  first, before the rest of the plugin work.
- **R-4: near-ties in the selection differ between host and card** (plan §6); the drafter inherits
  the trunk's table, so its parity is conditioned on radiance's own selection (§7.1) and unaffected.
- **R-5: snapshot memory** scales with `max_tok` (80 MiB a rank at 8192); compaction is designed
  (§5) and deferred.
