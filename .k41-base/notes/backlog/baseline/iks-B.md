# §2 baseline, iks-B lane

Re-measurement of `notes/backlog/optimisation-backlog.md` §2 on the second `iks` lane at the
baseline commit. Three runs of every end-to-end figure; the tables give the values and their
median, and each section ends with the delta to §2. This lane's numbers move much more than the
other two (see "Lane noise" below), so the range is as important as the median.

| | |
|---|---|
| commit | `dd4c2255b5e1cea93339262346f8c215a39a6dd5` ("Merge #3: … three GPU lanes") |
| lane | `iks` B, 2 x Radeon AI PRO R9700 (`gfx1201`, 32 GiB, 210 W cap), TP2, 48 host cores |
| model | `$RAD_MODELS/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad` (`~/models/…` on iks, read only) |
| build | iks toolchain via `"$RAD_CONFIGURE"` and `cmake --build build -j 48`, `gfx1201`, HIP on; same tree at the same commit as iks-A, `build/` reused |
| configs | `scripts/backlog/{nospec,nsv,coupled03,cpv,prof}.yaml` as committed; `lane.sh` rewrites the port and model path and sets `host-pool-mib 16000` with Engram on disk (`~/ngram`) for this lane |
| method | `RAD_IKS_NAME=RadianceHy4Flash-run-1654 scripts/backlog/lane.sh iks-B 'scripts/backlog/run.sh <tag> <cfg> <phases>'`, tags `-r1..-r3` (plus `ns-st-r4` to separate the cold first run), plus `sm-r1..r3` and `smrev-r1..r2` for the small prompts on the next tree (`RAD_IKS_NAME=RadianceHy4Flash-run-1673`, `ef147763`, notes only above the baseline commit); stamps `RADIANCE_AQL_STAMP=7`, profile `RADIANCE_PROFILE_EVERY=1`; raw data in `runs/iks-B/<tag>/` (copied back, not committed) |
| phases | `c1`, `c1b`, `c8`, `pf6` 6681-token prefill, `pf20` 24856-token prefill, `pf2` 2385-token prompt, `tail400` 159-token prompt, `stall1`/`stall4` at 1.5 s; the small-prompt runs add `tail742` 256 and `tail2560` 723 tokens |

One lane only: `lane-pair.sh` (both iks lanes at once) exceeds the 112 GiB of the machine, because
two 16 GiB host pools and the 60 GiB tmpfs Engram table do not fit together with the disk-backed
Engram page cache. The lanes were therefore measured sequentially.

## Lane noise

The B lane is the one with `ngram-placement disk`, i.e. the Engram table is read through the
page cache from the unpacked tree's disk. Its size is 59 GiB and the machine has 112 GiB, so the
table competes with the model and the OS cache; a run can be 10-100 % slower than the next one
without any kernel changing. Examples from this baseline:

- `ns-st` `c1`: r1 21.401 ms (first run, cold), r2 14.911, r3 18.786, r4 17.888.
- `ns-st` `c1b` (warm): 14.738 / 14.779 / 14.781 / 14.558 ms, 65.9-67.0 tok/s, stable.
- `ns-st` `c8`: 274.4 / 220.1 / 269.5 / 274.0 tok/s summed 287.5 / 229.2 / 283.2 / 287.3.
- `nsv` `pf6`: 2354 / 2473 / 4755 ms; `ns-st` `pf6`: 5160 / 2407 / 3831 / 5232 ms.
- `nsv` `pf20`: 7793 / 10042 / 7769 ms.

The stable figure is the warm `c1b`: 65.9-67.0 tok/s, 14.56-14.78 ms. Everything that makes a
first pass over the Engram table (the first `c1`, and the prefills) has a wide range. The
category and layer tables below use the *GPU busy* time, which is stable across the bad and the
good runs; the spread is host wait, not kernel time, and that is what this lane says.

## 2.1 One stream, no speculation

`nospec.yaml`, `c1` / `c1b` / `c8`; `ns-st` stamped, `nsv` unprofiled debug.

| run | `c1` unprofiled (nsv) | `c1` stamped (ns-st) | `c1b` stamped (ns-st) |
|---|---|---|---|
| r1 | 14.725 ms / 66.2 tok/s | 21.401 ms / 45.7 | 14.738 ms / 66.1 |
| r2 | 14.589 ms / 66.8 | 14.911 ms / 65.5 | 14.779 ms / 65.9 |
| r3 | 17.693 ms / 54.8 | 18.786 ms / 52.3 | 14.781 ms / 65.9 |
| r4 | — | 17.888 ms / 54.5 | 14.558 ms / 67.0 |
| **median** | **14.725 ms / 66.2** | **18.337 ms / 53.4** | **14.759 ms / 65.9** |
| §2 | 13.84 ms / 72.3 | 13.5-13.9 ms / 72-74 | 13.5-13.9 ms / 72-74 |
| delta | +6.4 % / -8.4 % | +32.5 % / -26.1 % | +6.6 % / -8.9 % (above the range) |

`c1b` is the lane's steady one-stream figure: 14.558-14.781 ms / 65.9-67.0 (median 14.759 /
65.9), 6.6 % slower than §2's 13.84 ms (9 % above the bottom of its range). The other `c1`
values carry the first Engram read.

The stamped per-pass figure (`r2`, pass kind 1322): 37 passes, 1314 dispatches, wall 15.577 ms,
busy 13.124 ms, hit 0.989, 8 missed units a step. The GPU busy is 13.124 ms, within 0.5 % of
local (13.184) and iks-A (13.154); the 2.45 ms of wall above it is host wait (the rank/collect
side, which the debug line puts at 0.3-0.5 ms of rank work for this lane), not kernel time. §2: busy 13.4-13.7.

Category table, `c1b` r2, pass kind 1322 (37 passes, 1314 dispatches, busy 13.124 ms):

| category | iks-B us | §2 us | delta |
|---|---|---|---|
| trunk linears IQR (`iqr_gemv` 251 x 9.7) | 2427 | 2558 | -5.1 % |
| trunk linears int8 (`gemm_i8_kpar` 125 x 16.2) | 2029 | 1959 | +3.6 % |
| junction (`ar_ihc_junction_2rank` 89 x 18.0) | 1601 | 1306 | +22.6 % |
| routed gate_up (`moe_iqr` 41 x 35.4) | 1452 | 1779 | -18.4 % |
| sink rows 42 x 7.5 + merge 42 x 15.5 | 964 | 996 | -3.2 % |
| routed down (`moe_down_dec` 41 x 21.3) | 872 | 1104 | -21.0 % |
| attention project_kv / expand / latent (92) | 854 | 760 | +12.4 % |
| router (82) | 746 | 766 | -2.6 % |
| activation quantisers (213) | 548 | 741 | -26.0 % |
| cache row, rope, copies (129) | 502 | 550 | -8.7 % |
| bf16 GEMMs (54) | 320 | 366 | -12.6 % |
| head + top-k + couple (1) | 301 | 302 | -0.3 % |
| indexer / selection (36) | 196 | 185 | +5.9 % |
| MoE sort/gather/scatter/swiglu (41) | 144 | 146 | -1.4 % |
| Engram (16) | 118 | 129 | -8.5 % |
| norms / silu (17) | 43 | 85 | -49.4 % |
| other (2) | 21 | — | |
| **busy** | **13124** | **13733** | **-4.4 %** |

The B split is the iks-A split, not the local one: junction +23 %, routed GEMMs -18 to -21 %,
quantisers -26 %. The GPU total is 0.2 % below A and 0.5 % below local; only the host wait
differs. This is a strong hint that the lane difference is a clock/power or memory-path
difference on the cards, not something in the B tree.

Per-layer tables at the `c1b` pass (`layer.py <csv> <json> c1b 1322 <LI>`). Only records of 4 us
or more are listed; the chained records below that are summed in the last row.

**Sliding layer, LI=2: 29 records, 289.2 us** (§2: 29 records, 287 us, +0.8 %)

| rec | kernel | us |
|---|---|---|
| 97 | `moe_down_dec` | 19.9 |
| 101 | `gemm_i8_kpar` | 6.4 |
| 102 | `iqr_gemv` | 7.8 |
| 104 | `gemm_i8_kpar` | 4.2 |
| 105 | `ar_ihc_junction_2rank` | 20.3 |
| 106 | `iqr_gemv` | 5.2 |
| 108 | `iqr_gemv` | 8.9 |
| 110 | `r4dchain_gemm_bf16_nt_m64` | 7.6 |
| 111 | `mla_row_front` | 7.2 |
| 113 | `iqr_gemv` | 27.2 |
| 115 | `mla_sink_rows` | 8.1 |
| 116 | `merge_sink` | 17.0 |
| 117 | `kv_expand_gate_quant` | 4.3 |
| 118 | `gemm_i8_kpar` | 42.6 |
| 119 | `ar_ihc_junction_2rank` | 14.2 |
| 120 | `iqr_gemv` | 10.2 |
| 121 | `gemm_nt_f32_tile` | 7.7 |
| 122 | `router_sigmoid_scatter` | 10.4 |
| 124 | `moe_iqr` | 33.2 |
| — | 10 chained records (<4 us) | 26.8 |

**Sparse layer, LI=4: 34 records, 259.1 us** (§2: 34 records, 258 us, +0.4 %)

| rec | kernel | us |
|---|---|---|
| 155 | `moe_down_dec` | 20.0 |
| 159 | `gemm_i8_kpar` | 6.3 |
| 160 | `iqr_gemv` | 7.8 |
| 163 | `ar_ihc_junction_2rank` | 14.9 |
| 164 | `iqr_gemv` | 4.8 |
| 166 | `iqr_gemv` | 6.0 |
| 168 | `r4dchain_gemm_bf16_nt_m64` | 5.5 |
| 171 | `iqr_gemv` | 15.5 |
| 175 | `idx_pool_gated` | 10.1 |
| 178 | `mla_sink_rows` | 11.5 |
| 179 | `merge_sink` | 6.4 |
| 180 | `kv_expand_gate_quant` | 9.6 |
| 181 | `gemm_i8_kpar` | 22.4 |
| 182 | `ar_ihc_junction_2rank` | 16.3 |
| 183 | `iqr_gemv` | 9.6 |
| 184 | `gemm_nt_f32_tile` | 8.3 |
| 185 | `router_sigmoid_scatter` | 10.4 |
| 187 | `moe_iqr` | 32.6 |
| — | 16 chained records (<4 us) | 41.1 |

## 2.2 Eight streams, no speculation

`ns-st` stamped, phase `c8` (8 x 2048 tokens, 255 steps); `nsv` unprofiled for comparison.

| run | stamped `c8` agg | summed tok/s | busy / wall |
|---|---|---|---|
| r1 | 274.4 tok/s | 287.5 | 24.077 / 28.549 ms |
| r2 | 220.1 | 229.2 | 24.026 / 29.341 |
| r3 | 269.5 | 283.2 | 24.155 / 29.024 |
| r4 | 274.0 | 287.3 | 24.125 / 29.005 |
| **median** | **271.8** | **285.3** | **24.101 / 29.015** |
| §2 | 306-317 | 315.7 | busy 24.9 / wall 25.5 |
| delta | -11 to -14 % | -9.6 % | -3.2 % busy / +13.8 % wall |

The GPU busy (24.0-24.2 ms) is the same as local (24.6-25.1) and A (24.4); wall is 28.5-29.3
because the stamped run waits 5 ms a step on the host Engram reads. The unprofiled `nsv` run is
the fair e2e figure: 275.7 / 266.8 / 274.7 agg, 289.6 / 279.0 / 288.0 summed (median 274.7 /
288.0, -10.5 % agg / -8.8 % summed against §2). Stamps cost this lane 4-5 ms a step against
2 ms on local, which is itself a symptom of the same host bottleneck.

Category table (`c8`, r2, pass kind 1322; `busy` 24.026 ms):

| category | iks-B us | §2 us | delta |
|---|---|---|---|
| routed gate_up (`moe_iqr` 41 x 176.6) | 7241 | 7723 | -6.2 % |
| routed down (`moe_down_dec` 41 x 108.2) | 4437 | 4562 | -2.7 % |
| trunk linears IQR (251 x 11.5) | 2892 | 2870 | +0.8 % |
| trunk linears int8 (125 x 17.7) | 2209 | 2241 | -1.4 % |
| junction (89 x 17.9) | 1595 | 1274 | +25.2 % |
| sink rows 42 x 12.7 + merge 42 x 15.2 | 1173 | 1196 | -1.9 % |
| attention project_kv / expand / latent (92) | 1142 | 1201 | -4.9 % |
| router (82) | 875 | 926 | -5.5 % |
| activation quantisers (213) | 627 | 859 | -27.0 % |
| cache row, rope, copies (129) | 478 | — | |
| bf16 GEMMs (54) | 350 | — | |
| head + top-k + couple (1) | 317 | — | |
| indexer / selection (36) | 242 | — | |
| MoE sort/gather/scatter/swiglu (41) | 223 | — | |
| Engram (16) | 136 | — | |
| norms / silu (17) | 47 | — | |
| other (2) | 22 | — | |
| **busy** | **24026** | **24894** | **-3.5 %** |

The same redistribution as 2.1: junction +25 %, routed GEMMs -3 to -6 %, quantisers -27 %. The
GPU busy is 3.5 % below §2, but the e2e throughput is 9-10 % below it because of the host wait.

## 2.3 DFly2 coupled, one and eight streams

`coupled03.yaml` (DFly2, coupled, `num-spec-tokens 5`, `draft-tree-depth 3`, 5 heads), `cp-st`
with stamps and `cpv` without.

| run | `c1` stamped | tok/step | summed tok/s | `c8` summed tok/s | `c8` tokens/step |
|---|---|---|---|---|---|
| r1 | 24.040 ms / 111.2 agg | 2.813 | 116.6 | 381.5 | 19.51 |
| r2 | 23.969 ms / 111.5 | 2.813 | 116.9 | 386.3 | 22.02 |
| r3 | 24.190 ms / 110.6 | 2.813 | 115.8 | 382.7 | 19.88 |
| **median** | **24.040 / 111.2** | **2.813** | **116.6** | **382.7** | **19.88** |
| §2 | 22.4-22.9 / 125-136 | 2.8-3.1 | 125-136 | 443 | 18.96 |
| delta | +5.2 % ms / -8.5 % | ~same | -8.5 % | -13.6 % | +4.9 % |

`cpv` (no stamps): `c1` 23.397 / 23.598 / 23.212 ms, 114.0 / 113.0 / 115.1 agg, 119.8 / 118.7 /
120.7 summed; `c8` 343.6 / 358.6 / 329.1 agg, 391.1 / 408.5 / 396.0 summed. `think` (132-token
prompt, 384 tokens): 22.937 / 22.862 / 22.850 ms, 3.122 tokens a step, 130.1 / 130.5 / 130.7 agg
(135.8-136.3 summed). §2.5: 21.3 ms, 2.67 tokens, 125.1; this lane is +7 % on the step and +4 %
on agg tok/s, with the same larger tree as the other lanes.

Pass costs at one stream (`cp-st` r1, `c1` block):

| pass | records | passes | wall | busy | §2 |
|---|---|---|---|---|---|
| verify | 1327 | 13 | 21.575 ms | 17.824 | 18.1-18.2 |
| draft | 303 | 13 | 4.033 ms | 3.778 | 4.0-4.6 |

The verify busy is 1.7 % below §2 and the draft 5.5 % below it, but the verify wall is 21.6 ms:
3.75 ms a step of host wait. That wait is the DFly2-specific cost of this lane: each verify pass
reads the Engram table for 1327 records from disk, and every extra draft/verify round adds
another read. The c1 step is 24.0 ms instead of 22.4-22.9 for that reason, not because a kernel
is slower.

At eight streams the main verify passes are 1297 records / 4 passes / 46.990 ms busy / 58.084 ms
wall and 1367 records / 3 passes / 52.534 ms busy / 63.834 ms wall; the draft is 298 records /
10 passes / 4.951 ms. §2's 56.5 ms verify and 6.3 ms draft compare with 47.0 and 5.0 busy, so
the kernels are ahead of §2 and the wall is 58-64 ms. The step carries 19.9-22.0 tokens
(§2 18.96) and sums 381.5-386.3 (median 382.7, -13.6 %).

## 2.4 Prefill profile, 6681-token prompt

`prof.yaml`, phase `pf6`, `RADIANCE_PROFILE_EVERY=1`; `profdiff.py <server.log> 0 150 50`.
Profiled TTFT: 2.783 / 2.810 / 2.658 s. The first full chunk is 1067.1 / 1091.1 /
942.0 ms with wait 656.4 / 681.8 / 529.8 ms: that is the cold Engram read (`embed_lookup_q`
is 4 calls in the third chunk below), not kernel time. The two later chunks are 588-594 ms with
40 ms of wait.

Third full chunk, 2048 rows at context 4096-6144 (dump 98 of 99):

| run | op time | wait |
|---|---|---|
| r1 | 592.7 ms | 41.1 ms |
| r2 | 590.8 ms | 39.9 ms |
| r3 | 591.0 ms | 39.9 ms |
| **median** | **591.0 ms** | **39.9 ms** |
| §2 | 564.7 ms | 20.5 ms |
| delta | +4.7 % | +19.4 ms |

Top ops, r2:

| op | calls | local ms | share | §2 ms |
|---|---|---|---|---|
| routed `moe_gemm_q` gate_up | 41 | 90.23 | 15.3 % | 87.2 |
| `mla_attn_paged_sink` | 30 | 84.51 | 14.3 % | 82.2 |
| `mla_attn_sel_sink` | 12 | 81.22 | 13.7 % | 79.9 |
| `ar_rmsnorm_ihc_junction` | 84 | 66.98 | 11.3 % | 62.3 |
| routed `moe_gemm_q` down | 41 | 45.72 | 7.7 % | 45.6 |
| `embed_lookup_q` (Engram, disk) | 4 | 26.38 | 4.5 % | — |
| `gemm_nt_q` attn_output | 42 | 21.91 | 3.7 % | 22.5 |
| `mla_expand_gate` | 42 | 21.27 | 3.6 % | 21.2 |
| `gemm_nt_q` attn_gate | 42 | 20.37 | 3.4 % | 20.5 |
| `gemm_nt_f32` router GEMM | 41 | 17.30 | 2.9 % | 17.3 |
| `moe_gather` | 41 | 14.10 | 2.4 % | 14.6 |
| `swiglu_clamped_had_quant_i8g` | 41 | 10.38 | 1.8 % | 10.4 |
| `mla_project_kv` | 42 | 8.52 | 1.4 % | 8.5 |
| `idx_score` | 12 | 6.27 | 1.1 % | 6.1 |
| `gemm_nt_q` attn_q_nope | 42 | 5.98 | 1.0 % | 6.4 |
| `had_quant_act_i8g` | 130 | 5.58 | 0.9 % | 5.9 |

Every kernel op matches the other lanes within 6 %, except the junction (+9 % against local,
the same as iks-A) and `embed_lookup_q`, which is the disk Engram read at 6.6 ms a call. The
latter is the whole +19 ms of wait and the reason the first full chunk is 1.8x slower.

Last chunk, 537 rows: 309.9 / 309.6 / 311.2 ms against §2's 338.3 (-8.4 %). Once the table is
warm the last chunk is the same as iks-A (302 ms) and faster than local (342 ms).

## 2.5 Unprofiled rates, small prompts, stalls, host phases

Unprofiled (`nsv`) and the §2 reference:

| figure | iks-B r1 / r2 / r3 | median | §2 | delta |
|---|---|---|---|---|
| one stream | 14.725 / 14.589 / 17.693 ms | 14.725 ms | 13.84 | +6.4 % |
| one stream, summed | 67.9 / 68.5 / 56.5 tok/s | 67.9 | 72.3 | -6.1 % |
| eight streams, agg | 275.7 / 266.8 / 274.7 | 274.7 | 306-317 | -10.5 % |
| eight streams, summed | 289.6 / 279.0 / 288.0 | 288.0 | 315.7 | -8.8 % |
| DFly2 one stream (`cpv`) | 23.397 / 23.598 / 23.212 ms | 23.397 | 21.9-22.4 | +4.5 % |
| DFly2 eight (`cpv`) | 391.1 / 408.5 / 396.0 summed | 396.0 | 446.1 | -11.2 % |
| `pf2` 2385 tokens, TTFT | 1423 / 980 / 958 ms | 980 | 707 | +38.6 % |
| `tail400` 159 tokens, TTFT | 428 / 181 / 182 ms | 182 | — | |
| `pf6` 6681 tokens, TTFT | 2354 / 2473 / 4755 ms | 2473 | 1.92-1.99 s | +24 % / +139 % |
| `pf20` 24856 tokens, TTFT | 7793 / 10042 / 7769 ms | 7793 | 7.03 s | +10.9 % / +42.8 % |
| `stall1` arrival | 4.58 / 4.36 / 4.22 s | 4.36 | 3.08 | +41.6 % |
| `stall4` arrival | 3.81 / 3.85 / 3.96 s | 3.85 | 2.82 | +36.5 % |

Small prompts against §2.6 (`nsv`; tags `sm-r1..r3` for phases `c1,tail742,tail2560,tail2600` and
`smrev-r1..r2` for `c1,tail2600,tail2560`, i.e. the same two big prompts in the other order):

| prompt tokens | position in the run | iks-B TTFT | median | §2.6 | delta |
|---|---|---|---|---|---|
| 73 (`c1`) | first phase | 112 / 107 / 104 ms | 107 | 78-88 ms | +22 % over 88 |
| 256 (`tail742`) | second phase | 232 / 228 / 221 ms | 228 | 223 ms | +2.2 % |
| 723 (`tail2560`) / 734 (`tail2600`) | the first ~720-row prefill of a run | 425 / 441 / 413 ms (723) and 561 / 550 ms (734, `smrev`) | 441 | — | |
| 723 / 734 | the next ~720-row prefill of the same run | 270 / 269 / 271 ms (734) and 266 / 263 ms (723, `smrev`) | 269 | 255 ms (at 717) | +5.5 % |

The 73-token figure is the `c1` phase's own TTFT: §2.6 lists that point but the tables above never
reported it. The 723-token point is six tokens over §2.6's 717 (a `tail` cut lands on characters,
not tokens).

The ~720-row prefill costs whatever position it is in, not whatever size it is: the prompt that
follows `tail742` costs 413-441 ms when it is the 723-token one and 550-561 ms when it is the
734-token one, and the prompt that follows *that* costs 263-271 ms either way. This lane reads its
Engram table from disk (`~/ngram`), so the first big prefill of a run faults it in and the next one
hits it; §2.6's 255 ms is the second, warm figure, which this lane reaches at 269 ms. It is the
same effect that makes `pf6` 24 % slower here (see below). The `tail` phases' single decode step is
also much cheaper on this lane (2.2-2.9 ms against 13.5-14 ms on the other two), which does not
touch the TTFTs reported here. The same runs' `c1` gave 14.67 / 14.59 / 14.66 ms a step, next to the
14.725 ms above.

The prefill TTFT is the figure this lane cannot reproduce: the best runs are 2354-2473 ms
(18-29 % above §2's 1.92-1.99 s), one of the three `nsv` runs and three of the four `ns-st`
runs are 3.8-5.2 s, and the profiled TTFT is 2.66-2.81 s. The 537-row last chunk is not slower per row (309.9 ms, -8.4 %); the full
chunks pay the Engram read-in as they go, and the first one is 1.8x. The warm third chunk is
+4.7 %, and 20 ms of its 26 ms excess is wait.

Host phases from the debug `step decode:` lines (`nsv` r1, mean/128):

| class | n | total | ranks | collect | host work |
|---|---|---|---|---|---|
| one (<18 ms) | 4 | 14.88 | 0.33 | 14.43 | 0.01 |
| eight (18-50 ms) | 8 | 24.68 | 4.02 | 18.18 | 0.02 |
| stall (>=50 ms) | 1 | 115.54 | 81.39 | 25.11 | 9.04 |

`r2` gives 14.72 / 0.47 / 14.48, 23.28 / 3.27 / 20.38 and stall 129.32 / 96.70 / 23.61; `r3`
gives 14.75 / 0.23 / 14.42, 23.07 / 4.31 / 19.23 and 129.29 / 96.58 / 23.75. The `ranks` column
is 3.3-4.3 ms on eight streams against 0.4-0.7 ms on local and iks-A: that is the disk Engram
gather, and it is the whole difference between B's 279-290 summed tok/s and the other lanes'
318-325. On the stall step it is 81-97 ms of the 115-129 ms total.

## Delta summary

| §2 figure | iks-B | delta |
|---|---|---|
| 2.1 one stream, warm `c1b` | 14.759 ms / 65.9 | +6.6 % / -8.9 %, above §2's range |
| 2.1 layer sliding / sparse | 289.2 / 259.1 us | +0.8 % / +0.4 % |
| 2.2 eight streams, unprofiled | 274.7 agg / 288.0 summed | -10.5 % / -8.8 % |
| 2.3 DFly2 one | 24.040 ms, 2.813 tok/step | +5.2 % ms |
| 2.3 DFly2 eight | 382.7 summed | -13.6 % |
| 2.4 third chunk | 591.0 ms | +4.7 % |
| 2.5 one / eight unprofiled | 14.725 / 274.7 | +6.4 % / -10.5 % |
| 2.5 small prompts, 73 / 256 / ~720 warm | 107 / 228 / 269 ms | +22 % / +2.2 % / +5.5 % |
| 2.5 `pf6` / `pf20` TTFT | 2473 / 7793 ms | +24 % / +10.9 % |
| 2.5 stall1 / stall4 | 4.36 / 3.85 s | +41.6 % / +36.5 % |

The B lane reproduces the GPU work of the other two lanes (busy 13.124 ms one stream, 24.026 ms
eight, layer totals 289.2/259.1 us) and misses §2's end-to-end rates by 9-14 % because every
step waits on the disk-backed Engram table (3-4 ms a step on eight streams, 3.75 ms on a DFly2
verify). Prefill TTFT and the stall probe are the worst affected. The lane is usable for
kernel-time comparisons, but its e2e throughput is not a stable target until the Engram
placement changes. The small prompts show the same hand: the first ~720-row prefill of a run
pays 413-561 ms to fault the table in, the second 263-271 ms, and only the second is within
5.5 % of §2.6's 255 ms.

End of the iks-B baseline.
