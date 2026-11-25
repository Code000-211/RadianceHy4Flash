# Non-absorbed prefill: `mla_kv_split`, `attn_kv_sink`

Design note, not implemented, nothing measured. Read first: `libr4d/r4d_attn_paged_window.hip`
(WMMA prefill and its fragment comments at 572-610, sink seeding at 737-742),
`libr4d/r4d_attn_prefill_split.h`, `libref/ref_attn.cpp` (`ref_attn_paged_sink` at 345),
`libr4d/r4d_mla.hip` `attend_rows` for the `K Q^T` form.

## 1. Where it sits

Prefill rows of a sliding layer, and of a sparse layer whose sequence stays ≤ 2048 after the step.
Per sequence the plugin builds a key range of `n_s = m_s + prefix_s` latent rows:
- sliding: `prefix_s = min(511, tokens cached before the chunk)`;
- sparse: `prefix_s` = all tokens cached before the chunk (≤ 2048 - m_s).

Steps:
1. Collect the latent rows `[N, 576]` (N = sum of `n_s`): the cached prefix rows copied out of the
   KV pages (a `gather_rows`-style copy through the block table; check whether an existing op reads
   paged rows into a flat buffer, e.g. what `attn_prefix_self` or the prefix cache uses, otherwise
   add a small `kv_rows_gather` op in WP1) followed by the chunk's new rows.
2. GEMM `kv_b` on the first 512 columns → `kvb [N, heads*448]` bf16 (the rank's heads).
3. `mla_kv_split` → `k [N, heads, 256]`, `v [N, heads, 256]`.
4. `attn_kv_sink` → `out [M, heads, 256]`.

Transient memory at real size, bf16: `kvb` + `k` + `v` = `N * heads * (448 + 512) * 2` B.
Sliding TP1, chunk 1024: N = 1535, 64 heads: 189 MiB. Sparse TP1, N = 2048, 32 heads: 126 MiB.
Halved a rank at TP2. If that is too much for the arena, process heads in groups of 16 (GEMM over
a row range of `kv_b`, then split and attend for those heads); design the plugin loop so a head
range is a parameter from the start.

## 2. `mla_kv_split`

```
k[i,h,0:192]   = kvb[i, h*448 : h*448+192]
k[i,h,192:256] = k_rope[i, 0:64]            (the same for every head)
v[i,h,0:256]   = kvb[i, h*448+192 : h*448+448]
```
Pure copy, bf16. libr4d: one thread per 16 bytes of output. No numerics.

## 3. `attn_kv_sink` semantics (libref)

Operands: `q [M, heads, 256]`, `k [N, heads, 256]`, `v [N, heads, 256]`, `cu_q [S+1]`, `cu_k [S+1]`
(absolute offsets), `sinks` f32 `[heads]`, → `out [M, heads, 256]`.
Sequence `s`: `m_s = cu_q[s+1]-cu_q[s]`, `n_s = cu_k[s+1]-cu_k[s]`, `n_s >= m_s` else `RAD_E_SHAPE`.
Query `i` at key index `p = n_s - m_s + i` sees `j` with `j <= p` and (`window == 0` or `p - j < window`).

```
s_j = scale * sum_c q[c] * k_j[c]            f32, c ascending, scale = 256^-0.5
m   = max_j s_j ;  den = sum_j exp(s_j - m) + exp(sink[h] - m)
out = bf16( sum_j exp(s_j - m) * v_j / den )
```
A query always sees its own key, so the empty case cannot occur; still define it as zero.

## 4. Kernel `attn_kv_sink_h256` (gfx1201, wave32)

- Workgroup: one head, 64 consecutive query rows of one sequence (4 tiles of 16), 8 waves
  (256 threads). Wave `(tile t, half c)`: half `c` owns columns `[128c, 128c+128)`.
- Grid: `x` = 64-row tiles over all sequences (host builds a small tile table `[tile] -> (seq,
  first row, rows)` in scratch, or the kernel derives the sequence from `cu_q` as
  `paged_seq` does), `y` = heads.
- Key range of a workgroup: `[max(0, p_first - window + 1), p_last]` of its sequence, walked in
  tiles of 32 keys. Tiles wholly outside every row's range are never loaded.
- Scores `S^T = K Q^T`: a key sub-tile of 16 as WMMA rows, the 16 query rows as columns; each wave
  does 8 WMMA (its 128 columns) per sub-tile, two sub-tiles per key tile. The two halves of a query
  tile exchange their partial sums through LDS (`partial[8][2][32*8]` f32 = 4 KiB... 16 KiB at
  8 waves x 2 sub-tiles; keep to one sub-tile at a time: 8 KiB) and add in the order half 0, half 1.
- Mask per (key, query): `j <= p` and inside the window, else `-inf`. Only tiles that straddle a
  row's bounds need the per-element test; the kernel can skip it for interior tiles (a flag from
  the tile's range).
- Online softmax per query row in its lane (max, den), as `attend_rows`.
- `P V`: each wave accumulates `O^T` for its 128 columns: 8 accumulator fragments (64 VGPR),
  `value[mt] = mma(fragT(V tile, cols), P, value[mt])`, P as bf16 high + low (same rule as the
  absorbed kernel, so the two paths stay close).
- Sink: at the store, `den += exp(sink[h] - max)`, `out = bf16(value / den)`.
- No split-K: a prefill launch has `M/64 * heads` workgroups.

LDS: K tile 32 x 256 bf16 = 16 KiB, V tile 16 KiB, partial dots 8 KiB: 40 KiB. The next tile is
fetched into registers during the current tile's products (32 KiB / 256 threads = 128 B a thread =
8 loads of 16 B) and stored after it; single LDS buffer, two barriers a tile.

VGPR estimate a lane: value 64, query fragments 8 x 4 = 32, staged 32, scores/p about 24: about
160. If it spills, halve the key tile to 16.

Constraint row: `head_dim == 256`, `heads <= 64`, `M <= 65535`, `N <= 2^24`.

Short tails: a last tile with fewer than 64 rows runs with the missing rows masked (zero query,
nothing stored). A chunk shorter than 64 rows (small prefills) therefore wastes up to 3 tiles of
work on one workgroup per head; acceptable, since such chunks are cheap anyway.

## 5. Numerics

f32 scores and softmax; bf16 operands; output rounded once. Not bit-equal to libref (WMMA order)
nor to the absorbed kernel (`(q W_UK) . c` against `q . (W_UK c)`): the plan's gate is 5e-3
relative between the two paths on the same inputs.

## 6. Validation

`tests/r4d_emu/test_attn_kv_sink.cpp`: against `ref_host` for (m, prefix) in
{(1,0), (16,0), (64,511), (100,37), (1024,511), (300,1748)}, windows 0 and 512, two sequences in
one launch, heads 16/32/64; a case where the sink dominates; causality (changing a future key does
not change an earlier row's output, bytes). `r4d_selftest` smoke; `rad-kbench` fixture rows;
`avx_check` for libavx.
