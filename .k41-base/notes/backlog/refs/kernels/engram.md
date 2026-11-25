# Engram: `engram_ids`, `embed_lookup_q` with a row scale, `engram_norm`, `engram_gate`

Design note, not implemented, nothing measured. Reference: HF c2ec8505 `HYV4FlashEngramLayer`,
`HYV4FlashEngramEmbedding`, `build_ngram_tokens`, `build_engram_inputs`
(`research/hf/modeling_c2ec8505.py:665-800, 1466-1486, 1882-1901`). Read first:
`arch/common/rad_block_ple.h`, OPS.md `embed_lookup_q` / `ngram_ids` / `ple_gate`,
`libref/ref_ngram.cpp`, `libavx/avx_ngram.cpp`, `libr4d/r4d_ngram_ids.hip`, `r4d_ple_gate_bf16.hip`.

Why not `ngram_ids` and `ple_gate`: `ngram_ids` hashes raw token ids and falls back to EOS across
an EOS; Hy4 hashes ids compressed through a token map and pads before the document start.
`ple_gate` has a convolution and a different gate; Hy4 norms the value, adds a branch term and
gates each stream separately.

## 1. `engram_ids`

Operands: `tok [M]` i32; `state` (the CONV group holding each sequence's last `ngram-1` token ids)
inout; `cu`; `pos [M]` i32 (position within the document); `token_map` i32 `[map_size]`;
`mult` i64 `[ngram]`; `vocab_sizes`, `offsets` i64 `[heads]`; `num_accepted?` → `ids [M, heads]`
i64 (or i32 pairs as `ngram_ids` writes them; follow that op), `mask [M]` i32.
Params: `heads` (16), `ngram` (3), `pad` (120002), `map_size` (120818), `image`/opt, `video`/opt
(token ids for the mask).

For row `m`, shift `k = 0..ngram-1` (`k = 0` is the row's own token):
```
t_k  = token k places back in the sequence;  pad if pos[m] < k
c_k  = (t_k < map_size) ? token_map[t_k] : token_map[pad]
x_k  = (int64) c_k * mult[k]                       wraps never: the multipliers are bounded
mix2 = x_0 ^ x_1 ;  mix3 = x_0 ^ x_1 ^ x_2
ids[m][i] = floor_mod(mix_{2 + i / (heads/(ngram-1))}, vocab_sizes[i]) + offsets[i]
mask[m]   = tok[m] != pad && tok[m] != image && tok[m] != video
```
`floor_mod` is Python's: non-negative for a positive modulus. The first `heads/(ngram-1)` heads
are the bigram's. `mult`, `vocab_sizes`, `offsets` come from `data/hy4flash_engram_tables.py`
(`blk.<l>.engram_mult` …); a container without them is refused at declare, by name, with the
command that adds them.
One issue a step serves all four Engram layers if the op is issued once per layer with that
layer's constants (4 issues); the history state is shared, so only the first issue of a step
advances it (as `PleLayer::ids` does once a step; follow its `num_accepted` and draft handling).
Runs on the host for a disk-placed table (ids must be host-readable) and on the card otherwise,
like `ngram_ids`.

## 2. `embed_lookup_q` with a scale a row

`x[m] = bf16( decode_e4m3(wte[id]) * f32(scale[id]) )` when `scale` has `n_vocab` entries.
One issue gathers `M*heads` rows of 96 (the ids buffer flattened) into `e_raw [M, heads*96]`.
Changes: `ref_embed_lookup_q`; libavx kernel and its row cache (a cached row stores its scale);
`avx_layout_ngram` and `r4d_rad_layout_embed_q` accept `block = 1 x *` scale planes; the device
gather reads `scale[id]`. Disk placement reads the scale plane too: keep it resident (165 M x 2 B
= 330 MB a layer) rather than a second O_DIRECT read a row; say so in the layout hook.
A plain bf16 table (no recipe) must also work: then the plugin applies the E4M3 round trip the
model was trained with (`engram_fp8_embedding`) as a small op after the gather, `fp8_roundtrip_rows`
(`scale = max(|row|, 1e-12) / 448`, `q = e4m3(row / scale)`, `row = q * scale`); with it, the
no-recipe container is the exact reference HANDOFF describes. Do the division in bf16 arithmetic as
the port does (`x / scale` on bf16 tensors), since this path exists only to be exact.

## 3. `engram_norm`

`e_raw [M, heads*n]` (n = 96), `gain [heads*n]` → `e [M, heads*n]`:
`e[m][i] = bf16( gain[i] * x[i] * rsqrt( mean_{head of i}(x^2) + eps ) )`, one rms a head.
`eps` is decision U1: 1.19e-7 (f32: HF in f32, vLLM, llama.cpp, the friend; the plugin default) or 0.0078125 (HF in bf16). A parameter of the op.

## 4. Projections

Existing GEMMs with bias on `e [M, 1536]`: `engram_key` → `ek [M, hc*n]` (the four `key_projs`
joined), `engram_value` → `ev [M, n]`. Replicated on every rank.

## 5. `engram_gate`

Operands: `key [M, hc*n]`, `val [M, n]` bf16; `mask [M]` i32; `w_key`, `w_query`, `w_branch`
`[hc*n]`, `w_val [n]` (`unified_pre_gamma`); `h [M, hc*n]` inout. Params `n`, `hc`, `eps` (U1).

```
main[i]  = w_val[i] * val[i] * rsqrt( mean(val^2) + eps )              no gain inside this norm
for stream c:
  kn[i]  = w_key[c][i]   * key_c[i] * rsqrt( mean(key_c^2) + eps )
  qn[i]  = w_query[c][i] * h_c[i]   * rsqrt( mean(h_c^2)   + eps )
  d      = ( sum_i kn[i] * qn[i] ) * n^-0.5                             libref: f64 sum
  g      = sigmoid( sign(d) * sqrt( max(|d|, 1e-6) ) )
  h_c[i] = bf16( f32(h_c[i]) + bf16( g * main[i] + w_branch[c][i] * main[i] ) )
```
Rows with `mask == 0` are left untouched. The inner `bf16(...)` mirrors HF, which forms the
residual in bf16 and then adds it to the bf16 stream.
HF computes every step of this in bf16 tensors (each intermediate rounded, the 3072-term dot
summed in bf16). The op computes in f32 and rounds where stated; this is a known difference from
the bf16 reference and is why Engram layers get their own row in the parity report.

libr4d `engram_gate_bf16`: one workgroup a token row, one wave a stream plus shared `main`:
three sums of squares and one dot per stream (wave reductions), then the elementwise update.
`engram_norm_bf16`: one wave a (row, head). `engram_ids_i64`: one thread a (row, head).
No LDS beyond the reductions, no scratch.

## 6. Validation

- libref unit: hash against values computed by the HF port for fixed tokens (table in the test,
  produced once by the harness); pad before document start; id ≥ map_size; masked row untouched;
  `sign(d) = 0`; both eps values.
- Harness: the Engram layer's output delta (stream after minus before) against a hook on
  `engram_layer`, for the bf16 table and for the E4M3-row container.
- `tests/r4d_emu/test_engram.cpp`, `r4d_selftest`, `rad-kbench` fixture rows, `avx_check`.
