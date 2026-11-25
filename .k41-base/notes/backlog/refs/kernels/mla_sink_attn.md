# Absorbed MLA with sink: `mla_attn_paged_sink`, `mla_attn_sel_sink`, `mla_project_kv`

Design note, not implemented, nothing measured. Base to read first: `libr4d/r4d_mla.hip`
(`attend_rows` at line 782, `attend_paged` at 300, `merge_paged` at 353, `paged_body` at 387,
`project_absorb_rows` / `project_expand_rows` at 28 / 48), `libref/ref_mla.cpp`, OPS.md
"GLM-5.3's indexer and latent attention". Idea sources: vLLM `rocm_aiter_mla_sparse.py`
(@820fce0d, Apache-2.0) for the sink-in-reduce rule. Write new code; name the source in the header.

## 1. Semantics (the reference, libref)

Cache row of a token: `[latent 512 bf16 | rotated key 64 bf16]`, 576 wide.
Inputs a query row `r`, head `h`: `q[r,h] [512]` (absorbed), `q_rope[r,h] [64]` (rotated), both bf16.

```
s_j   = scale * ( sum_c q[c]*lat_j[c]  +  sum_c q_rope[c]*rope_j[c] )      f32, scale = 256^-0.5
m     = max_j s_j
den   = sum_j exp(s_j - m) + exp(sink[h] - m)
out   = bf16( sum_j exp(s_j - m) * lat_j / den )                             [512]
```
- `sinks` is an f32 weight `[heads]` (the rank's slice). Natural-log domain. Applied once.
- No visible key: `out = 0` (also with a sink).
- libref order: keys ascending, latent columns ascending, then rope columns; f32 accumulation.

Visible keys:
- `mla_attn_paged_sink`: operands as `mla_attn_paged` (`block_table [seqs, blocks]`, `seqused`,
  `cu_seqlens`) plus `sinks`; `q_rope` required. Row `i` of sequence `s` sits at
  `p = seqused[s] - qlen + i` and sees `j` in `[max(0, p - window + 1), p]`; `window 0` means from 0.
  A page id < 0 or ≥ pages contributes nothing.
- `mla_attn_sel_sink`: operands as `mla_attn` (`bt [M, blocks]` a query, `length [M]`) plus
  `q_rope`, `sinks`. Tokens `t` in `[0, length)`, page `bt[r][t / block_size]`, slot `t % block_size`.

No gate operand: Hy4's gate acts after the expansion (`gate_mul`).

## 2. Kernel `mla_sink_rows<HG, FORM>` (gfx1201, wave32)

`attend_rows` with three changes: rope columns, a window lower bound, the sink at the end.

- Workgroup = one query row, `HG` groups of 16 heads, 4 waves a group (`HG*128` threads).
  `HG` 1 or 2. Head of a lane: `blockIdx.y*16*HG + group*16 + (lane & 15)`.
- Scores as `S^T = K Q^T`: tokens are the WMMA rows (16 a chunk), heads the columns, so a head's
  running max and denominator live in its lane.
- Wave `quarter` computes latent columns `[128*quarter, 128*quarter+128)` (8 WMMA, k-step 16) and
  rope columns `[16*quarter, 16*quarter+16)` (1 WMMA): 9 per chunk. Query fragments: 8 latent +
  1 rope `v8bf`, loaded once.
- The four partial dot products meet in LDS (`partial[HG*4][32*8]` f32) and every wave adds them in
  the order quarter 0, 1, 2, 3, so all four hold identical scores.
- Online softmax per chunk exactly as `attend_rows` (chunk max, `alpha`, `p[e]`, both lane halves
  add tokens 0-7 before 8-15).
- Value product: `value[mt] = value[mt]*alpha; value[mt] = mma(fragT(tile, cols), P_hi, value[mt]);
  … P_lo …` for the wave's 8 column tiles of 16 (latent only, never the rope columns). P as bf16
  high part plus bf16 of the remainder, as `attend_rows` does.
- Key staging: 16 tokens x 576 bf16 = 1152 pieces of 16 bytes a chunk. `HG 1`: 9 loads a thread
  (exact). `HG 2`: 5 loads a thread with a guard `flat < 1152`. Double buffer: the next chunk is
  fetched into registers while the current one is multiplied, stored after the value product, two
  barriers a chunk.
- Row pitch in LDS: 584 bf16 (576 + 8 so consecutive rows do not share an LDS bank pattern; any
  multiple of 8 ≥ 576 works).

LDS budget: `keys[2][16*584]` bf16 = 37 376 B; `partial` = 4096 B a group; `visible[2][16]` = 128 B.
Total 41.6 KiB (`HG 1`), 45.7 KiB (`HG 2`). Under 64 KiB. `HG 4` (53.9 KiB) is possible but needs
2.25 loads a thread; not planned.

VGPR estimate a lane: `value` 8 x 8 = 64, query fragments 9 x 4 = 36, staged loads 9 x 4 = 36,
scores and p 16 to 24, temporaries about 10: about 150 to 170 of 256. If the build spills, fetch
the next chunk in two halves (before and after the value product).

Head counts: 64 (TP1 sliding), 32, 16, 8 (TP4 sparse). `heads % 16 == 8`: lanes with
`(lane & 15) >= 8` of the last group carry a zero query and store nothing. Constraint row:
`heads` in {8, 16, 32, 64}, `latent_dim == 512`, `rope_dim == 64`, `block_size` in {4, 16} (state
what is compiled), `M <= 65535`.

Token enumeration (the only difference between the two forms):
- paged: `lo = max(0, p + 1 - window)` (or 0), `live = p + 1`; split `z` covers
  `[lo + (live-lo)*z/splits, lo + (live-lo)*(z+1)/splits)`; page from the sequence's table.
- sel: `[0, length)` split the same way; page from the row's table.

## 3. Splits and merge

- Split count is a function of the op and `window` only, never of the launch's rows or lengths:
  paged with `window > 0`: 2. Paged with `window 0` and sel: 8. Reason: a verify of k+1 rows must
  give the single-row decode at each position bit for bit.
- That holds for `M <= 64` (the scratch hook sizes for 64 rows: `64 * heads * splits * 514 * 4` B,
  at most 67 MiB at 64 heads x 8... so cap: scratch for `min(M, 64)` rows; at 64 heads and 8 splits
  that is 67 MiB, too much: sparse layers have at most 32 heads (33.7 MiB), sliding uses 2 splits
  (16.8 MiB). State these two sizes in the hook.) For `M > 64` (prefill through the absorbed path)
  the launch runs unsplit; prefill rows are not required to match decode rows.
- Partial of a (row, head, split): `value[512]`, `max`, `den` in f32 (`PART = 514`), relative to the
  split's own max.
- Merge kernel `merge_sink` (one wave a (row, head), as `merge_paged`): global max over the splits
  with `den > 0`; `den = sum den_s * exp(max_s - max)`; values likewise; then
  `den += exp(min(sink - max, 80))` if any key was visible; `out = bf16(value / den)`.
  The splits' maxima are read first, so no serial dependency between splits.
- Unsplit: the same sink step in the store of the main kernel.
- Equality of the two forms: with the same split count and a table that lists a sequence's pages in
  order, chunk boundaries and summation order coincide, so `mla_attn_paged_sink` (window 0) and
  `mla_attn_sel_sink` give the same bytes. This is an acceptance test; it lets a sequence cross
  2048 without a visible seam at decode.

## 4. Numerics

- Scores, softmax state, partials, sink: f32. Operands of the WMMA: bf16. Output rounded once.
- `expf` as in `r4d_mla.hip`. If `exp2f` is used instead, multiply scale and sink by `log2(e)`.
- Against libref: WMMA sums 16 products in the hardware's order, so not bit-equal; bound as the
  existing `mla_attn_rows_bf16` row states it in `rad-kbench`.
- Against HF: HF rounds each score to bf16 before the softmax; this kernel does not (plan D4).

## 5. `mla_project_kv_bf16`

`project_absorb_rows<R, HD>` / `project_expand_rows<R, HD>` with the head's block generalised:
weight rows of head `h` start at `h*(k_dim+v_dim)`; absorb reads the first `k_dim` rows, expand the
`v_dim` rows after them. Compiled for `(k_dim, v_dim, latent) = (192, 256, 512)`, up to 64 heads,
pitched tensors. Keep the property that a row's sum order does not depend on the group size `R`.
- absorb: `y[m,h,j] = sum_{c<192} a[m,h,c] * b[h*448 + c][j]`.
- expand: `y[m,h,c] = sum_{j<512} a[m,h,j] * b[h*448 + 192 + c][j]`.
libref: plain loops, f32, `c` / `j` ascending.

## 6. Validation

- `tests/r4d_emu/test_mla_sink.cpp`: both forms against `ref_host`, heads 8/16/32/64, lengths 0, 1,
  15, 16, 17, 511, 512, 513, 2048; window edge (key `p-511` visible, `p-512` not); sink far above
  and far below the scores; unmapped page; forced split counts; verify rows vs single rows (bytes);
  paged vs sel (bytes).
- `r4d_selftest`: smoke and numeric case a row; `rad-kbench` cases in `data/kernels_hy4.rkb`.
- libavx: `avx_check` cases for the same list.
