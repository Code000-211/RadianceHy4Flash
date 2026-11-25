# §2 baseline, iks-A lane

Re-measurement of `notes/backlog/optimisation-backlog.md` §2 on the first `iks` lane at the
baseline commit. Three runs of every end-to-end figure; the tables give the three values and
their median, and each section ends with the delta to §2.

| | |
|---|---|
| commit | `dd4c2255b5e1cea93339262346f8c215a39a6dd5` ("Merge #3: … three GPU lanes") |
| lane | `iks` A, 2 x Radeon AI PRO R9700 (`gfx1201`, 32 GiB, 210 W cap), TP2, 48 host cores |
| model | `$RAD_MODELS/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad` (`~/models/…` on iks, read only) |
| build | iks toolchain via `"$RAD_CONFIGURE"` and `cmake --build build -j 48`, `gfx1201`, HIP on; the tree is unpacked fresh from the same commit, `build/` survives between calls |
| configs | `scripts/backlog/{nospec,nsv,coupled03,cpv,prof}.yaml` as committed; `lane.sh` rewrites the port and model path and sets `host-pool-mib 16000` with Engram in `/dev/shm/radiance-ngram` (ram) for this lane |
| method | `RAD_IKS_NAME=RadianceHy4Flash-run-1654 scripts/backlog/lane.sh iks-A 'scripts/backlog/run.sh <tag> <cfg> <phases>'`, tags `-r1..-r3`, plus `sm-r1..r3` for the small prompts on the next tree (`RAD_IKS_NAME=RadianceHy4Flash-run-1673`, `ef147763`, notes only above the baseline commit); stamps `RADIANCE_AQL_STAMP=7`, profile `RADIANCE_PROFILE_EVERY=1`; raw data in `runs/iks-A/<tag>/` (copied back, not committed) |
| phases | `c1`, `c1b`, `c8`, `pf6` 6681-token prefill, `pf20` 24856-token prefill, `pf2` 2385-token prompt, `tail400` 159-token prompt, `stall1`/`stall4` at 1.5 s; the small-prompt runs add `tail742` 256 and `tail2560` 723 tokens |

One lane only: `lane-pair.sh` (both iks lanes at once) exceeds the 112 GiB of the machine, because
two 16 GiB host pools and the 60 GiB tmpfs Engram table do not fit together with the disk-backed
Engram page cache. The two iks lanes were therefore measured sequentially with `lane.sh`.

## 2.1 One stream, no speculation

`nospec.yaml`, `c1` / `c1b` / `c8`; `ns-st` stamped, `nsv` unprofiled debug.

| run | `c1` unprofiled (nsv) | `c1` stamped (ns-st) | `c1b` stamped (ns-st) |
|---|---|---|---|
| r1 | 13.607 ms / 72.2 tok/s | 13.505 ms / 72.7 | 13.664 ms / 71.9 |
| r2 | 13.537 ms / 72.6 | 13.538 ms / 72.7 | 13.482 ms / 72.9 |
| r3 | 13.583 ms / 72.4 | 13.547 ms / 72.6 | 13.497 ms / 72.8 |
| **median** | **13.583 ms / 72.4** | **13.538 ms / 72.7** | **13.497 ms / 72.8** |
| §2 | 13.84 ms / 72.3 | 13.5-13.9 ms / 72-74 | 13.5-13.9 ms / 72-74 |
| delta | -1.9 % / +0.1 % | inside the range | at the low end |

`c1b` r1 is 13.664 ms / 71.9, the only outlier; r2/r3 are 13.48-13.50 ms / 72.8-72.9. The stamped
per-pass figures for `c1b` (`r1`, pass kind 1330): 37 passes, 1318 dispatches, wall 13.288 ms,
busy 13.154 ms, hit 0.988, 8 missed units a step. The pass is 2-4 % faster than local (wall 13.43,
busy 13.18); the difference is in the category split, below.

Category table, `c1b` pass kind 1330 (37 passes, 1318 dispatches, busy 13.154 ms):

| category | iks-A us | §2 us | delta |
|---|---|---|---|
| trunk linears IQR (`iqr_gemv` 251 x 9.7) | 2440 | 2558 | -4.6 % |
| trunk linears int8 (`gemm_i8_kpar` 125 x 16.3) | 2041 | 1959 | +4.2 % |
| junction (`ar_ihc_junction_2rank` 89 x 17.9) | 1594 | 1306 | +22.0 % |
| routed gate_up (`moe_iqr` 41 x 35.9) | 1473 | 1779 | -17.2 % |
| sink rows 42 x 7.4 + merge 42 x 15.3 | 954 | 996 | -4.2 % |
| routed down (`moe_down_dec` 41 x 21.7) | 889 | 1104 | -19.5 % |
| attention project_kv / expand / latent (92) | 883 | 760 | +16.2 % |
| router (82) | 739 | 766 | -3.5 % |
| activation quantisers (213) | 523 | 741 | -29.4 % |
| cache row, rope, copies (129) | 501 | 550 | -8.9 % |
| bf16 GEMMs (54) | 318 | 366 | -13.1 % |
| head + top-k + couple (1) | 301 | 302 | -0.3 % |
| indexer / selection (36) | 193 | 185 | +4.3 % |
| MoE sort/gather/scatter/swiglu (41) | 145 | 146 | -0.7 % |
| Engram (20) | 135 | 129 | +4.7 % |
| norms / silu (17) | 42 | 85 | -50.6 % |
| other (2) | 21 | — | |
| **busy** | **13154** | **13733** | **-4.2 %** |

The total is 4.2 % below §2 but the split differs from local: routed GEMMs are 17-20 % faster,
the junction is 22 % slower and quantisers 29 % faster. The junction is the two-rank all-reduce
and the biggest sign change between the lanes (+22 % here, -6 % on local); it is worth watching
in a junction-focused measurement, but the end-to-end step is not affected.

Per-layer tables at the `c1b` pass (`layer.py <csv> <json> c1b 1330 <LI>`). Only records of 4 us
or more are listed; the chained records below that are summed in the last row.

**Sliding layer, LI=2: 29 records, 286.0 us** (§2: 29 records, 287 us, -0.3 %)

| rec | kernel | us |
|---|---|---|
| 99 | `moe_down_dec` | 20.1 |
| 103 | `gemm_i8_kpar` | 6.6 |
| 104 | `iqr_gemv` | 8.0 |
| 106 | `gemm_i8_kpar` | 4.1 |
| 107 | `ar_ihc_junction_2rank` | 15.8 |
| 108 | `iqr_gemv` | 5.3 |
| 110 | `iqr_gemv` | 8.9 |
| 112 | `r4dchain_gemm_bf16_nt_m64` | 7.7 |
| 113 | `mla_row_front` | 7.3 |
| 115 | `iqr_gemv` | 27.3 |
| 117 | `mla_sink_rows` | 8.2 |
| 118 | `merge_sink` | 17.2 |
| 119 | `kv_expand_gate_quant` | 4.4 |
| 120 | `gemm_i8_kpar` | 43.0 |
| 121 | `ar_ihc_junction_2rank` | 14.4 |
| 122 | `iqr_gemv` | 10.4 |
| 123 | `gemm_nt_f32_tile` | 7.5 |
| 124 | `router_sigmoid_scatter` | 10.4 |
| 126 | `moe_iqr` | 32.9 |
| — | 10 chained records (<4 us) | 26.5 |

**Sparse layer, LI=4: 34 records, 259.7 us** (§2: 34 records, 258 us, +0.7 %)

| rec | kernel | us |
|---|---|---|
| 157 | `moe_down_dec` | 20.2 |
| 161 | `gemm_i8_kpar` | 6.5 |
| 162 | `iqr_gemv` | 7.7 |
| 164 | `gemm_i8_kpar` | 4.0 |
| 165 | `ar_ihc_junction_2rank` | 14.0 |
| 166 | `iqr_gemv` | 4.8 |
| 168 | `iqr_gemv` | 6.2 |
| 170 | `r4dchain_gemm_bf16_nt_m64` | 5.5 |
| 173 | `iqr_gemv` | 15.2 |
| 177 | `idx_pool_gated` | 10.0 |
| 180 | `mla_sink_rows` | 11.7 |
| 181 | `merge_sink` | 6.4 |
| 182 | `kv_expand_gate_quant` | 9.6 |
| 183 | `gemm_i8_kpar` | 22.4 |
| 184 | `ar_ihc_junction_2rank` | 17.1 |
| 185 | `iqr_gemv` | 9.8 |
| 186 | `gemm_nt_f32_tile` | 8.2 |
| 187 | `router_sigmoid_scatter` | 10.4 |
| 189 | `moe_iqr` | 32.7 |
| — | 15 chained records (<4 us) | 37.3 |

## 2.2 Eight streams, no speculation

`ns-st` with `RADIANCE_AQL_STAMP=7`, phase `c8` (8 x 2048 tokens, 255 steps).

| run | stamped `c8` | summed tok/s | busy / wall |
|---|---|---|---|
| r1 | 313.0 tok/s (agg) | 324.4 | 24.401 / 24.771 ms |
| r2 | 312.8 | 323.5 | 24.378 / 24.796 ms |
| r3 | 312.4 | 323.3 | 24.361 / 24.763 ms |
| **median** | **312.8** | **323.5** | **24.378 / 24.771** |
| §2 | 306-317 | 315.7 | busy 24.9 / wall 25.5 |
| delta | inside 306-317 | +2.5 % | -2.1 % busy / -2.9 % wall |

Pass kind 1330, 36 passes, 1318 dispatches, hit 0.992, 34 missed units a step.

Category table (`c8`, pass kind 1330; `busy` 24.401 ms):

| category | iks-A us | §2 us | delta |
|---|---|---|---|
| routed gate_up (`moe_iqr` 41 x 180.2) | 7387 | 7723 | -4.4 % |
| routed down (`moe_down_dec` 41 x 108.8) | 4462 | 4562 | -2.2 % |
| trunk linears IQR (251 x 11.4) | 2863 | 2870 | -0.2 % |
| trunk linears int8 (125 x 17.9) | 2236 | 2241 | -0.2 % |
| junction (89 x 18.9) | 1680 | 1274 | +31.9 % |
| attention project_kv / expand / latent (92) | 1151 | 1201 | -4.2 % |
| sink rows 42 x 13.5 + merge 42 x 15.1 | 1199 | 1196 | +0.3 % |
| router (82) | 892 | 926 | -3.7 % |
| activation quantisers (213) | 610 | 859 | -29.0 % |
| cache row, rope, copies (129) | 482 | — | |
| bf16 GEMMs (54) | 353 | — | |
| head + top-k + couple (1) | 318 | — | |
| indexer / selection (36) | 253 | — | |
| MoE sort/gather/scatter/swiglu (41) | 229 | — | |
| Engram (20) | 158 | — | |
| norms / silu (17) | 46 | — | |
| other (2) | 25 | — | |
| **busy** | **24401** | **24894** | **-2.0 %** |

The junction is +32 % against both §2 and local (89 x 18.9 us a call here against 14.3 us
on local). The
routed GEMMs, quantisers and cache path give the 2 % back. Eight streams scale routed gate_up
x5.0, down x5.0, IQR x1.17, int8 x1.10, junction x1.05, quantisers x1.17.

## 2.3 DFly2 coupled, one and eight streams

`coupled03.yaml` (DFly2, coupled, `num-spec-tokens 5`, `draft-tree-depth 3`, 5 heads), `cp-st`
with stamps and `cpv` without.

| run | `c1` stamped | tok/step | summed tok/s | `c8` summed tok/s | `c8` tokens/step |
|---|---|---|---|---|---|
| r1 | 21.435 ms / 126.3 agg | 2.813 | 130.7 | 477.0 | 20.48 |
| r2 | 21.481 ms / 126.0 | 2.813 | 130.5 | 460.3 | 21.56 |
| r3 | 21.488 ms / 126.0 | 2.813 | 130.4 | 456.9 | 22.76 |
| **median** | **21.481 / 126.0** | **2.813** | **130.5** | **460.3** | **21.56** |
| §2 | 22.4-22.9 / 125-136 | 2.8-3.1 | 125-136 | 443 | 18.96 |
| delta | -4.6 % ms | ~same | low end of 125-136 | +3.9 % | +13.7 % |

`cpv` (no stamps): `c1` 21.201 / 21.261 / 21.228 ms, 127.4 / 127.1 / 127.3 agg, 132.2 / 131.8 /
132.0 summed; `c8` 389.6 / 376.7 / 360.5 agg, 443.5 / 450.6 / 432.4 summed. `think` (132-token
prompt, 384 tokens): 21.131 / 21.091 / 21.119 ms, 3.122 tokens a step, 141.7 / 142.0 / 141.8 agg
(147.4-147.6 summed). §2.5: 21.3 ms, 2.67 tokens, 125.1; HEAD accepts more tokens and is +13 %
faster in agg tok/s at the same step time.

Pass costs at one stream (`cp-st` r1, `c1` block):

| pass | records | passes | wall | busy | §2 |
|---|---|---|---|---|---|
| verify | 1335 | 13 | 18.057 ms | 17.846 | 18.1-18.2 |
| draft | 303 | 13 | 4.102 ms | 3.873 | 4.0-4.6 |

At eight streams the main verify pass is 1305 records / 6 passes / 48.222 ms busy, the draft is
298 records / 9 passes / 5.086 ms, and §2's 1375-record pass appears once. The draft is 5.1 ms
against §2's 6.3 (-19 %), the verify 48.2 against 56.5 (-15 %); the step carries 21.6 tokens
against 18.96, giving 460.3 summed against 443 (+3.9 %).

## 2.4 Prefill profile, 6681-token prompt

`prof.yaml`, phase `pf6`, `RADIANCE_PROFILE_EVERY=1`; `profdiff.py <server.log> 0 150 50`.
Profiled TTFT: 2.111 / 2.107 / 2.107 s (unprofiled 1.918-1.930 s, see 2.5).

Third full chunk, 2048 rows at context 4096-6144 (dump 98 of 99):

| run | op time | wait |
|---|---|---|
| r1 | 570.1 ms | 12.5 ms |
| r2 | 570.4 ms | 12.3 ms |
| r3 | 569.4 ms | 12.4 ms |
| **median** | **570.1 ms** | **12.4 ms** |
| §2 | 564.7 ms | 20.5 ms |
| delta | +1.0 % | -8.1 ms |

The chunk before it is 566.2 / 566.3 / 564.9 ms; the two are within 1 %.

Top ops, r2:

| op | calls | local ms | share | §2 ms |
|---|---|---|---|---|
| routed `moe_gemm_q` gate_up | 41 | 90.95 | 15.9 % | 87.2 |
| `mla_attn_paged_sink` | 30 | 85.87 | 15.1 % | 82.2 |
| `mla_attn_sel_sink` | 12 | 81.47 | 14.3 % | 79.9 |
| `ar_rmsnorm_ihc_junction` | 84 | 66.16 | 11.6 % | 62.3 |
| routed `moe_gemm_q` down | 41 | 46.15 | 8.1 % | 45.6 |
| `gemm_nt_q` attn_output | 42 | 22.03 | 3.9 % | 22.5 |
| `mla_expand_gate` | 42 | 21.48 | 3.8 % | 21.2 |
| `gemm_nt_q` attn_gate | 42 | 20.75 | 3.6 % | 20.5 |
| `gemm_nt_f32` router GEMM | 41 | 17.43 | 3.1 % | 17.3 |
| `moe_gather` | 41 | 13.90 | 2.4 % | 14.6 |
| `swiglu_clamped_had_quant_i8g` | 41 | 10.26 | 1.8 % | 10.4 |
| `mla_project_kv` (absorb) | 42 | 8.69 | 1.5 % | 8.5 |
| `idx_score` | 12 | 6.41 | 1.1 % | 6.1 |
| `gemm_nt_q` attn_q_nope | 42 | 6.24 | 1.1 % | 6.4 |
| `had_quant_act_i8g` | 130 | 5.62 | 1.0 % | 5.9 |

Every op is within 6 % of §2. The junction line is 66.2 ms here but 61.2 ms in the local
profile; that is the decode junction difference seen in 2.1/2.2, not a prefill difference.

Last chunk, 537 rows: 306.8 / 301.9 / 301.8 ms against §2's 338.3 (-10.8 %). The 36 ms is in
the routed gate_up + down pair: A runs it at 179.0 ms (104.66 + 74.36) against §2's 213.6 ms
(63 % of §2's total). The whole last chunk sits on the unstaged read-in-place path. This is
the one prefill figure that differs materially from local (341 ms); the first three chunks are
within 1 %.

## 2.5 Unprofiled rates, small prompts, stalls, host phases

Unprofiled (`nsv`) and the §2 reference:

| figure | iks-A r1 / r2 / r3 | median | §2 | delta |
|---|---|---|---|---|
| one stream | 13.607 / 13.537 / 13.583 ms | 13.583 ms | 13.84 | -1.9 % |
| one stream, summed | 73.9 / 73.6 / 73.6 tok/s | 73.6 | 72.3 | +1.8 % |
| eight streams, agg | 311.6 / 313.9 / 311.6 | 311.6 | 306-317 | +1.9 % |
| eight streams, summed | 322.6 / 325.0 / 322.7 | 322.7 | 315.7 | +2.2 % |
| DFly2 one stream (`cpv`) | 21.201 / 21.261 / 21.228 ms | 21.228 | 21.9-22.4 | -3.1 % |
| DFly2 eight (`cpv`) | 443.5 / 450.6 / 432.4 summed | 443.5 | 446.1 | -0.6 % |
| `pf2` 2385 tokens, TTFT | 680 / 682 / 678 ms | 680 | 707 | -3.8 % |
| `tail400` 159 tokens, TTFT | 166 / 163 / 165 ms | 165 | — | |
| `pf6` 6681 tokens, TTFT | 1925 / 1927 / 1926 ms | 1926 | 1.92-1.99 s | -0.5 % |
| `pf20` 24856 tokens, TTFT | 7172 / 7171 / 7174 ms | 7172 | 7.03 s | +2.0 % |
| `stall1` arrival | 2.96 / 2.97 / 2.95 s | 2.96 | 3.08 | -3.9 % |
| `stall4` arrival | 2.84 / 2.83 / 2.83 s | 2.83 | 2.82 | +0.4 % |

Small prompts against §2.6 (`nsv`, tags `sm-r1..r3`, phases `c1,tail742,tail2560,tail2600`; the
phases are all new prompts in a run, and `tail2600` shares 2560 characters of prefix with
`tail2560` yet is no cheaper, so the prefix cache does not resume between phases here and each
`tail` meets a cold prefix):

| prompt tokens | iks-A sm r1 / r2 / r3, TTFT | median | §2.6 | delta |
|---|---|---|---|---|
| 73 (`c1`) | 74 / 74 / 74 ms | 74 | 78-88 ms | -5 % at the bottom |
| 256 (`tail742`) | 135 / 136 / 135 ms | 135 | 223 ms | **-39 %** |
| 723 (`tail2560`) | 242 / 244 / 243 ms | 243 | 255 ms (at 717) | -4.7 % |
| 734 (`tail2600`) | 239 / 239 / 237 ms | 239 | — | |

The 73-token figure is the `c1` phase's own TTFT: §2.6 lists that point but the tables above never
reported it. The 723-token point is six tokens over §2.6's 717 (a `tail` cut lands on characters,
not tokens); six rows of prefill are 1-2 ms, so the comparison holds, and `tail2600` (734 tokens,
239 ms) shows the same slope from the other side. The same three runs' `c1` gave 13.51 / 13.51 /
13.52 ms a step, next to the 13.583 ms above.

As on `local`, the 256-token step is far cheaper than §2.6's 223 ms while the 73- and 723-token
points agree within a few per cent, so the §2.6 row for 256 tokens is the outlier of the four;
`iks-B` measures 228 ms at that point, so the two fast lanes are the difference, not §2.6.

Host phases from the debug `step decode:` lines (`nsv` r1, mean/128):

| class | n | total | ranks | collect | host work |
|---|---|---|---|---|---|
| one (<18 ms) | 7 | 13.98 | 0.21 | 13.76 | 0.01 |
| eight (18-50 ms) | 5 | 22.22 | 0.36 | 21.82 | 0.03 |
| stall (>=50 ms) | 1 | 98.28 | 13.66 | 33.67 | 50.95 |

`r2` gives 13.94 / 0.21 / 13.74 / 0.00, 22.25 / 0.46 / 21.76 / 0.03 and stall 98.04 / 13.79 /
33.29 / 50.96. Host work is off the critical path for decode (0.01-0.03 ms); the stall line is
the arrival prefill, 51.0 ms of host plan/build/save for 5881 prompt tokens (local: 46.8 ms).

## Delta summary

| §2 figure | iks-A | delta |
|---|---|---|
| 2.1 one stream, stamped | 13.538 ms / 72.7 | inside 13.5-13.9 / 72-74 |
| 2.1 `c1b` | 13.497 ms / 72.8 | at the low end |
| 2.1 layer sliding / sparse | 286.0 / 259.7 us | -0.3 % / +0.7 % |
| 2.2 eight streams | 312.8 agg / 323.5 summed | +2.5 % (summed) |
| 2.3 DFly2 one | 21.481 ms, 2.813 tok/step | -4.6 % ms, low end of tok/s |
| 2.3 DFly2 eight | 460.3 summed | +3.9 % |
| 2.4 third chunk | 570.1 ms | +1.0 % |
| 2.4 last chunk | 301.9 ms | -10.8 % |
| 2.5 one / eight unprofiled | 13.583 / 311.6 | -1.9 % / +1.9 % |
| 2.5 small prompts, 73 / 256 / 723 tokens | 74 / 135 / 243 ms | -5 % / -39 % / -4.7 % |
| 2.5 `pf6` / `pf20` TTFT | 1926 / 7172 ms | -0.5 % / +2.0 % |
| 2.5 stall1 / stall4 | 2.96 / 2.83 s | -3.9 % / +0.4 % |

The lane reproduces §2 within a few percent. It is 1-3 % faster than local on decode and DFly2
one stream, the prefill first three chunks are the same, and the 537-row last chunk is 10 %
faster. The category split differs (junction +22-32 %, routed GEMMs and quantisers lower) at a
total within 2-4 % of both §2 and local. The small-prompt steps are 5 % faster at 73 tokens and
4.7 % faster at 723, but 39 % faster at 256, the same outlier `local` shows (`iks-B` measures
228 ms there, so the outlier is the small-step path of these two lanes).

End of the iks-A baseline.
