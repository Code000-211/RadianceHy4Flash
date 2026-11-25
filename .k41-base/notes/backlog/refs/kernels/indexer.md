# Compressed indexer: `idx_pool_gated`, `idx_score`, and the selection rule

Design note, not implemented, nothing measured. Reference: HF c2ec8505
`HYV4FlashCompressedIndexer` and `HYV4FlashAttention._mask_unrecalled_tokens`
(`research/hf/modeling_c2ec8505.py:344-436, 613-662`). Read first: `arch/common/rad_block_glm_indexer.h`,
`arch/common/rad_qsa.h`, `libr4d/r4d_dsa.hip`, `libref/ref_dsa.cpp`, `libref/ref_qsa.cpp`, docs/QSA.md.

## 1. Why a per-sequence dense rule equals HF

HF switches on `kv_length <= 2048` for the whole call. Inside the indexer a query at slot `p` sits
in group `g = p // 4`; groups `>= g` are masked to `-inf`, `topk(min(511, groups))` is taken, the
own group is appended, and the causal mask removes tokens after `p`.
- If `g <= 511` (that is `p <= 2047`), every closed group before the query is finite and there are
  at most 511 of them, so all are recalled; the remaining picks are `-inf` groups that lie at or
  after the query and are causally masked anyway. The result is dense attention.
- So for each query the outcome depends only on its own position, not on how the prompt was
  chunked. A sequence at ≤ 2048 after the step may run dense; a longer one runs the indexer for all
  its rows, and its early rows (if any are in the chunk) still come out dense through the table.
`qsa_select` implements exactly this bound (`n = min(nc, (pos+1)/ratio)`), the appended partial
page, and a deterministic tie-break (lower index; HF's `topk` is unspecified on ties).

When a query sits on a group boundary (`(p+1) % 4 == 0`) its own group is complete: HF still
excludes it from the candidates (`group_ids >= query_group_index`) and appends it. `qsa_select`
treats a full own block as a candidate of the top-k instead (`n = (pos+1)/ratio` includes it) and
appends no partial page. Then 511 picks compete among `g+1` groups including the own one, which HF
would always keep. **This is a semantic difference on one query in four.** WP1/WP3 must settle it
with a test against HF: either `qsa_select` gains an optional parameter (`own`/opt = 1: the block
holding `pos` is never a candidate and is always appended, complete or not), or the plugin calls it
with a position bound that excludes the own group and appends it itself. Prefer the optional
parameter (union rule; rows that do not know it must add a constraint refusing it).

## 2. Token keys and the cache

- `tk = rmsnorm(index_key(xa), index_key_norm, eps 1e-5)` → bf16 `[T, 128]`. Existing ops.
- Tail: `qsa_tail_store` keeps the raw keys of tokens whose group is still open; `qsa_work` emits,
  for the groups a step closes, `stage [W*4, 128]` (their four keys in slot order), `page [W]` and
  the group's first position.
- Rollback under speculation follows `qsa_tail_store`'s contract; a group is pooled only from
  committed plus current rows, and a rejected draft's pooled page is overwritten when the group
  closes again. Check this path in WP13.

## 3. `idx_pool_gated`

Operands: `k [W*ratio, n]` bf16; `wv`, `wg` f32 `[n, n]`; `ape` f32 `[ratio, n]`; `gain` `[n]`
(bf16 or f32 as stored, read as f32); `pos` i32 `[W]` or `[3, W]` (first token of each group);
`page?` i32 `[W]`, `cache?` bf16 `[pages, 1, 4, n/4]` inout (page pitch ≥ `2n` bytes); `out?` `[W, n]`.

For group `w`, slots `s = 0..ratio-1`, channel `c` (all f32, plain IEEE, no contraction):
```
x_s      = f32(k[w*ratio + s])
val_s[c] = sum_j wv[c][j] * x_s[j]                 j ascending
gat_s[c] = sum_j wg[c][j] * x_s[j] + ape[s][c]
mx[c]    = max_s gat_s[c]
e_s[c]   = exp(gat_s[c] - mx[c]) ;  p_s[c] = e_s[c] / (e_0 + e_1 + e_2 + e_3)   s ascending
y[c]     = sum_s val_s[c] * p_s[c]                 s ascending
r        = rsqrt( sum_c y[c]^2 / n + eps )         libref: the sum in f64
z[c]     = bf16( gain[c] * (y[c] * r) )
```
Then the rotary part on the last `rotary_dim` (64) channels of `z`, adjacent pairs, at `pos`:
```
a = f32(z[2i]) ; b = f32(z[2i+1]) ; ang = pos_c(i) * theta^(-2i/rotary_dim)
z[2i] = bf16(a*cos - b*sin) ; z[2i+1] = bf16(b*cos + a*sin)
```
(`mode gptj`: one position; `imrope_gptj` with `sections`: component per pair as in PLAN §2.2.)
Store `z` to `cache[page[w]]` and/or `out[w]`. A negative or out-of-range page is skipped.

HF detail kept: the value before the rotation is rounded to bf16 (`.to(token_keys.dtype)`), and
the rotated key is rounded again. HF detail not kept: cos/sin rounded to bf16 (plan D5).

libr4d `idx_pool_gated_f32` (n 128, ratio 4): one workgroup of 128 threads a group. The four `x_s`
are staged in LDS as f32 (2 KiB). Thread `c` reads rows `wv[c]` and `wg[c]` (2 x 512 B) and forms
its 8 dot products with `j` ascending, then its softmax and `y[c]`: these are byte-equal to libref
(same order, no FMA; `-ffp-contract=off` is set for every libr4d unit). The norm's sum of squares
is a workgroup reduction in f32 (lanes, then waves, fixed order) against libref's f64: last-bit
differences allowed there. Rotation and store by the threads of the last 64 channels in pairs.
Cost: 131 072 multiply-adds a group; a 4096-token prefill closes 1024 groups a layer.

## 4. `idx_score`

Operands: `q [M, heads, n]` bf16 (already rotated at the query position), `gate [M, heads]` bf16
(`index_head` output), `k` = the indexer cache `[pages, 1, 4, n/4]` bf16 (or flat `[G, n]`),
`bt [seqs, blocks]`, `cu [seqs+1]`, `nc [seqs]` → `score [M, blocks]` f32 (own row pitch, may be
narrower than `bt`).

```
w_h      = f32(gate[m][h]) * (heads * n)^-0.5
d_h      = sum_c f32(q[m][h][c]) * f32(key_b[c])        f32
score_b  = ( sum_{h<8} w_h * max(d_h, 0) ) + ( sum_{8<=h<16} w_h * max(d_h, 0) )     each ascending
```
The two-halves order is part of the reference so the device kernel can match it (for `heads` not
16: halves of `heads/2`). A block past `nc`, a page < 0 or out of range: `-inf`. Causality is
`qsa_select`'s business.

libr4d `idx_score_bf16` (n 128, heads 16): one wave = one query row against 16 keys. A = the 16
heads' query fragments (8 `v8bf`, loaded once a row), B = 16 keys loaded straight into registers in
fragment order (256 B a key, no LDS), 8 WMMA → `C[16 heads][16 keys]`. Then relu, times `w_h`, sum
over the head axis in the stated order (elements within the lane, then one lane swap), store 16
scores. Workgroup: 4 waves = 64 keys of one row; grid `(M, ceil(blocks/64))`. A row's bytes do not
depend on M or on the other rows. `M <= 65535`, `blocks <= 2^20`.
Cost at 64 K context: 16 384 groups → 256 workgroup-quarters a row a layer; key bytes read a row:
4 MiB. At 1 M context 64 MiB a row a layer, which is what the optional FP8 path (WP14) halves twice.

Dot products go through WMMA, so scores are not bit-equal to libref; near-ties can flip a pick
between host and card. Gate: Jaccard ≥ 0.999 of the recalled sets on `gpu-mini`, and the same
bytes on repeated card runs.

## 5. Query side

`iq = index_query(qr)` → `[T, 16, 128]`; `rope` (`gptj`, theta 1e7, `rotary_dim 64`, applied to the
trailing 64 of each head; check that `rope` can address a trailing slice of a 128-wide head through
`rotary_dim`, otherwise pass a pitched view of the last 64 columns). `iw = index_head(xa)`.

## 6. Validation

- libref unit cases: pooling against a double-precision formula; a group whose gate logits tie;
  rotation at position 0 (identity) and at a large position; score with negative `gate`.
- Harness: recalled sets against HF on `mini` (dense limit 16) at contexts 17, 20, 33, 64, and on
  `gpu-mini` at 2049, 2052, 3000; the group-boundary query of §1 is an explicit case.
- `tests/r4d_emu/test_idx.cpp`, `r4d_selftest`, `rad-kbench` fixture rows; `avx_check`.
