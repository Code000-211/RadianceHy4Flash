# Independent hyper-connections: `ihc_pre`, `ihc_post`

Design note, not implemented, nothing measured. Reference: HF c2ec8505 `HYV4FlashHyperConnection`,
`HYV4FlashHyperHead`, and the two residual lines of `HYV4FlashDecoderLayer.forward`
(`research/hf/modeling_c2ec8505.py:813-937`). Read first: OPS.md "Hyper-connections" (the `mhc_*`
family, which always computes a Sinkhorn mixing and therefore does not fit), `libref/ref_mhc.cpp`,
`libr4d/r4d_mhc.hip` (projection and finishing kernels), `tests/r4d_emu/test_mhc.cpp`.

## 1. `ihc_pre`

Operands: `h [M, hc*n]` bf16; `fn` f32 `[rows, hc*n]`; `scale` f32 `[2]` (or `[1]` for the head);
`base` f32 `[rows]`; → `x [M, n]` bf16; `post? [M, hc]` f32.
Params: `rows` = `2*hc` with `post`, `hc` without; `eps` (1e-5, the model's `rms_norm_eps`);
`hc_eps` (1e-6); `post_mag` (2.0, required when `post` is present).

```
f        = f32(h)                                   [hc*n]
r        = rsqrt( sum(f^2) / (hc*n) + eps )         libref: sum in f64, then f32
c[j]     = ( sum_i fn[j][i] * f[i] ) * r            j < rows, i ascending, f32
pre[j]   = sigmoid(c[j] * scale[0] + base[j]) + hc_eps                 j < hc
x[i]     = bf16( sum_{j<hc} pre[j] * f[j*n + i] )                      j ascending
post[j]  = post_mag * sigmoid(c[hc+j] * scale[1] + base[hc+j]) + hc_eps      (only with post)
```
Differences from `mhc_pre` that matter: no mixing logits and no Sinkhorn; `post` carries
`+ hc_eps`; the magnitude is a parameter; one scalar norm a token over all streams.

## 2. `ihc_post`

Operands: `y [M, n]` bf16, `post [M, hc]` f32, `h [M, hc*n]` bf16 inout.
```
h[j*n + i] = bf16( post[j] * f32(y[i]) + f32(h[j*n + i]) )
```
In place is safe: no stream reads another. The stream is never normalised.

## 3. libr4d

`ihc_pre_f32` (`hc` 2 to 8, `n` up to 16384, `M` up to 65535): one workgroup a token row, 8 waves.
- Each thread strides the row's `hc*n` columns (thread `t` takes columns `t, t+256, …`) and keeps
  `rows + 1` f32 partial sums (the projections and the sum of squares). At the real geometry:
  12288 columns, 48 a thread, 9 sums. `fn` rows are read as f32 (4 B), `h` as bf16.
- Reduction: lanes of a wave (`r4d_wave_sum`), then the 8 waves through LDS in wave order. libref's
  order is plain ascending, so the kernel matches it to rounding, not in bytes; state the bound
  (relative 1e-5 on `c`). A byte-exact scalar twin is not needed: unlike GLM there is no partial /
  finish pair to keep identical across ranks, only the requirement that every rank runs this same
  kernel on the same bytes.
- Finish in the same workgroup: one thread computes `r`, the gates and `post`; they go to LDS; all
  threads then form `x` for their columns (`n` columns, each a sum of `hc` products) and round.
- Weight traffic: `fn` is 8 x 12288 f32 = 393 KiB a connection, read once a token row. Two
  connections a layer: 786 KiB a token a layer, 33 MiB a token over 42 layers at decode. Small
  against the experts, but it is the reason to revisit sharing the projection between rows of a
  prefill (several rows a workgroup, as `r4d_mhc.hip`'s two-row form) in WP16.

`ihc_post_bf16`: elementwise, one thread per 8 columns of every stream, 16-byte stores.

LDS: under 1 KiB. No scratch.

## 4. TP

Both ops are replicated: every rank holds the whole stream and all of `fn`. `y` is the block
output after the all-reduce and the norm that follows it, so every rank adds the same bytes. A
fused all-reduce + norm + `ihc_post` row is a WP16 candidate.

## 5. Validation

- libref unit: against a double-precision formula; a case with `scale` and `base` chosen so `pre`
  and `post` are far from 0.5; the head form (`rows == hc`, `post` absent); `hc_eps` visible on a
  saturated-low gate (`post` equals `hc_eps`, not 0).
- Harness: stream parity after every layer covers both ops; an extra check compares `x` and `post`
  against a forward hook on `attn_hc` for one layer.
- `tests/r4d_emu/test_ihc.cpp` at `(hc, n)` = (4, 3072), (4, 512), (2, 128); `r4d_selftest`;
  `rad-kbench` fixture rows; `avx_check`.
