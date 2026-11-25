# §2 baseline, local lane

Re-measurement of `notes/backlog/optimisation-backlog.md` §2 on the local GPU lane at the
baseline commit. Three runs of every end-to-end figure; the tables give the three values and
their median, and each section ends with the delta to §2.

| | |
|---|---|
| commit | `dd4c2255b5e1cea93339262346f8c215a39a6dd5` ("Merge #3: … three GPU lanes") |
| lane | local, 2 x Radeon AI PRO R9700 (`gfx1201`, 32 GiB, 210 W cap), TP2, 48 host cores, 170 GiB RAM |
| model | `/srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad` (read only) |
| build | `cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201 && cmake --build build --target radiance -j 32`; plugin targets `radiance_home/{kernels/libr4d.so,kernels/libavx.so,kernels/libref.so,kernels/spechub.so,quantizers/libquant.so,architectures/hyv4flash.so}` |
| configs | `scripts/backlog/{nospec,nsv,coupled03,cpv,prof}.yaml` as committed; `lane.sh` only rewrites port, model path, host pool / ngram placement and checkpoint interval |
| method | `scripts/backlog/lane.sh local "scripts/backlog/run.sh <tag> <cfg> <phases>"`, three tags `-r1..-r3` for §2 and `sm-r1..r3` for the small prompts (those on `ef147763`, which is notes only above the baseline commit); stamps `RADIANCE_AQL_STAMP=7`, profile `RADIANCE_PROFILE_EVERY=1`; raw data in `runs/local/<tag>/` (not committed) |
| phases | `c1` 73-token prompt decode, `c1b` same after a warm pass, `c8` eight streams, `pf6` 6681-token prefill, `pf20` 24856-token prefill, `pf2` 2385-token prompt, `tail400` 159-token prompt, `stall1`/`stall4` 1/4 background streams arriving at 1.5 s; the small-prompt runs add `tail742` 256 and `tail2560` 723 tokens |

`lane.sh` takes `gpu2` (both cards) through the card scheduler. No other GPU job of this
worktree ran at the same time; other Moebius runs may have shared the host, which is what the
three runs measure.

## 2.1 One stream, no speculation

`nospec.yaml`, `c1` / `c1b` / `c8`; `ns-st` has `RADIANCE_AQL_STAMP=7`, `nsv` has debug logs and
no stamps.

| run | `c1` unprofiled (nsv) | `c1` stamped (ns-st) | `c1b` stamped (ns-st) |
|---|---|---|---|
| r1 | 13.710 ms / 71.6 tok/s | 13.844 ms / 70.8 | 13.363 ms / 73.3 |
| r2 | 13.738 ms / 71.5 | 13.762 ms / 71.4 | 13.411 ms / 73.2 |
| r3 | 13.704 ms / 71.6 | 13.773 ms / 71.4 | 13.403 ms / 73.2 |
| **median** | **13.710 ms / 71.6** | **13.773 ms / 71.4** | **13.403 ms / 73.2** |
| §2 | 13.84 ms / 72.3 | 13.84 ms / 72.3 | 13.5-13.9 ms / 72-74 |
| delta | -0.9 % / -1.0 % | -0.5 % / -1.3 % | below the range |

The stamped per-pass figures for `c1` (`r1`, pass kind 1330): 36 passes, 1318 dispatches, wall
13.742 ms, busy 13.500 ms, hit 0.984, 11 missed units a step. §2: 1330 dispatches, wall 13.5-13.9, busy
13.4-13.7.

Category table, `c1b` pass kind 1330 (37 passes, 1318 dispatches, busy 13.184 ms; §2 13.733):

| category | local us | §2 us | delta |
|---|---|---|---|
| trunk linears IQR (`iqr_gemv` 251 x 9.9) | 2486 | 2558 | -2.8 % |
| trunk linears int8 (`gemm_i8_kpar` 125 x 15.6) | 1956 | 1959 | -0.2 % |
| routed gate_up (`moe_iqr` 41 x 39.1) | 1604 | 1779 | -9.8 % |
| junction (`ar_ihc_junction_2rank` 89 x 13.8) | 1227 | 1306 | -6.0 % |
| routed down (`moe_down_dec` 41 x 24.7) | 1014 | 1104 | -8.2 % |
| sink rows 42 x 7.6 + merge 42 x 16.0 | 992 | 996 | -0.4 % |
| router (`gemm_nt_f32` + select) | 769 | 766 | +0.4 % |
| attention project_kv / expand / latent | 758 | 760 | -0.3 % |
| activation quantisers (213) | 688 | 741 | -7.2 % |
| cache row, rope, copies (129) | 517 | 550 | -6.0 % |
| bf16 GEMMs (54) | 361 | 366 | -1.4 % |
| head + top-k + couple (1) | 301 | 302 | -0.3 % |
| indexer / selection (36) | 184 | 185 | -0.5 % |
| MoE sort/gather/scatter/swiglu (41) | 143 | 146 | -2.1 % |
| Engram (20) | 126 | 129 | -2.3 % |
| norms / silu (17) | 53 | 85 | -37.6 % |
| other (2) | 18 | — | |
| **busy** | **13184** | **13733** | **-4.0 %** |

The 4 % lower busy is spread over the routed GEMMs (-8 to -10 %), junction (-6 %), quantisers
(-7 %) and cache (-6 %); IQR, int8, sink/merge, router and head are within 1 %. It matches the
-0.5 % wall delta once the stamp overhead is counted. Nothing in the table points at one kernel.

Per-layer tables at the `c1b` pass (`scripts/backlog/layer.py <csv> <json> c1b 1330 <LI>`; `LI`
is the record distance, the same sliding/sparse split as §2.1). Only records of 4 us or more are
listed; the chained records below that are summed in the last row.

**Sliding layer, LI=2: 29 records, 283.7 us** (§2: 29 records, 287 us, -1.2 %)

| rec | kernel | us |
|---|---|---|
| 99 | `moe_down_dec` | 19.9 |
| 103 | `gemm_i8_kpar` | 5.1 |
| 104 | `iqr_gemv` | 9.4 |
| 107 | `ar_ihc_junction_2rank` | 14.2 |
| 108 | `iqr_gemv` | 5.4 |
| 110 | `iqr_gemv` | 8.0 |
| 111 | `iqr_gemv` | 4.0 |
| 112 | `r4dchain_gemm_bf16_nt_m64` | 8.1 |
| 113 | `mla_row_front` | 6.4 |
| 115 | `iqr_gemv` | 28.0 |
| 117 | `mla_sink_rows` | 7.0 |
| 118 | `merge_sink` | 19.6 |
| 119 | `kv_expand_gate_quant` | 4.4 |
| 120 | `gemm_i8_kpar` | 42.6 |
| 121 | `ar_ihc_junction_2rank` | 10.1 |
| 122 | `iqr_gemv` | 9.3 |
| 123 | `gemm_nt_f32_tile` | 8.2 |
| 124 | `router_sigmoid_scatter` | 10.5 |
| 126 | `moe_iqr` | 32.6 |
| — | 10 chained records (<4 us) | 30.9 |

**Sparse layer, LI=4: 34 records, 252.1 us** (§2: 34 records, 258 us, -2.3 %)

| rec | kernel | us |
|---|---|---|
| 157 | `moe_down_dec` | 20.4 |
| 161 | `gemm_i8_kpar` | 5.0 |
| 162 | `iqr_gemv` | 8.8 |
| 165 | `ar_ihc_junction_2rank` | 12.2 |
| 166 | `iqr_gemv` | 5.0 |
| 168 | `iqr_gemv` | 4.6 |
| 170 | `r4dchain_gemm_bf16_nt_m64` | 6.6 |
| 171 | `mla_row_front` | 5.0 |
| 173 | `iqr_gemv` | 10.3 |
| 177 | `idx_pool_gated` | 9.5 |
| 180 | `mla_sink_rows` | 11.1 |
| 181 | `merge_sink` | 6.2 |
| 182 | `kv_expand_gate_quant` | 9.5 |
| 183 | `gemm_i8_kpar` | 22.2 |
| 184 | `ar_ihc_junction_2rank` | 11.0 |
| 185 | `iqr_gemv` | 9.3 |
| 186 | `gemm_nt_f32_tile` | 8.7 |
| 187 | `router_sigmoid_scatter` | 10.6 |
| 189 | `moe_iqr` | 32.2 |
| — | 15 chained records (<4 us) | 43.9 |

## 2.2 Eight streams, no speculation

`ns-st` with `RADIANCE_AQL_STAMP=7`, phase `c8` (8 x 2048 tokens, 255 steps).

| run | stamped `c8` | summed tok/s | busy / wall |
|---|---|---|---|
| r1 | 308.8 tok/s (agg) | 320.8 | 24.589 / 25.208 ms |
| r2 | 309.3 | 321.0 | 24.824 / 25.428 ms |
| r3 | 306.2 | 317.5 | 25.068 / 25.703 ms |
| **median** | **308.8** | **320.8** | **24.824 / 25.428** |
| §2 | 306-317 | 315.7 | busy 24.9 / wall 25.5 |
| delta | inside 306-317 | +1.6 % | -0.3 % busy / -0.3 % wall |

Pass kind 1330, 36 passes, 1318 dispatches, hit 0.992, 33 missed units a step (4105 routed units).

Category table (`c8`, r1, pass kind 1330; `busy` 24.589 ms):

| category | local us | §2 us | delta |
|---|---|---|---|
| routed gate_up (`moe_iqr` 41 x 183.4) | 7520 | 7723 | -2.6 % |
| routed down (`moe_down_dec` 41 x 111.1) | 4557 | 4562 | -0.1 % |
| trunk linears IQR (251 x 11.3) | 2839 | 2870 | -1.1 % |
| trunk linears int8 (125 x 17.9) | 2243 | 2241 | +0.1 % |
| junction (89 x 14.3) | 1276 | 1274 | +0.2 % |
| attention project_kv / expand / latent (92) | 1187 | 1201 | -1.2 % |
| sink rows 42 x 12.3 + merge 42 x 16.2 | 1194 | 1196 | -0.2 % |
| router (82) | 906 | 926 | -2.2 % |
| activation quantisers (213) | 806 | 859 | -6.2 % |
| cache row, rope, copies (129) | 631 | — | |
| bf16 GEMMs (54) | 379 | — | |
| head + top-k + couple (1) | 317 | — | |
| indexer / selection (36) | 250 | — | |
| MoE sort/gather/scatter/swiglu (41) | 224 | — | |
| Engram (20) | 166 | — | |
| norms / silu (17) | 65 | — | |
| other (2) | 26 | — | |
| **busy** | **24589** | **24894** | **-1.2 %** |

Eight streams scale as in §2: routed gate_up x4.7, down x4.5, IQR x1.14, int8 x1.15, junction
x1.04, router x1.18, quantisers x1.17 against the one-stream table.

## 2.3 DFly2 coupled, one and eight streams

`coupled03.yaml` (DFly2, coupled, `num-spec-tokens 5`, `draft-tree-depth 3`, 5 heads), `cp-st`
with `RADIANCE_AQL_STAMP=7`; `cpv` is the same config with debug logs and no stamps.

| run | `c1` stamped | tok/step | summed tok/s | `c8` summed tok/s | `c8` tokens/step |
|---|---|---|---|---|---|
| r1 | 22.742 ms / 118.7 agg | 2.813 | 123.2 | 427.8 | 20.08 |
| r2 | 23.111 ms / 116.7 | 2.813 | 121.2 | 416.2 | 21.56 |
| r3 | 22.392 ms / 120.4 | 2.813 | 125.1 | 425.3 | 20.08 |
| **median** | **22.742 / 118.7** | **2.813** | **123.2** | **425.3** | **20.08** |
| §2 | 22.4-22.9 / 125-136 | 2.8-3.1 | 125-136 | 443 | 18.96 |
| delta | +0.2 % ms / -5.5 % sum | ~same | -5.5 % | -4.0 % | +5.9 % |

`cpv` (no stamps, §2.5's source): `c1` 21.586 / 21.981 / 21.700 ms, 124.5 / 122.3 / 123.9 agg,
129.8 / 127.5 / 129.1 summed; `c8` 385.4 / 385.5 / 355.8 agg, 453.1 / 437.0 / 427.6 summed.
`think` (132-token prompt, 384 tokens): 21.117 / 21.116 / 21.036 ms, 3.122 tokens a step,
140.9 / 140.7 / 141.4 agg (145.4-148.0 summed). §2.5: 21.3 ms, 2.67 tokens, 125.1; HEAD accepts
more tokens a step (+17 %) and is +12 % faster in agg tok/s on the same step time.

Pass costs at one stream (`cp-st` r1, `c1` block):

| pass | records | passes | wall | busy | §2 |
|---|---|---|---|---|---|
| verify | 1335 | 13 | 18.178 ms | 17.937 | 18.1-18.2 |
| draft | 303 | 13 | 4.692 ms | 4.331 | 4.0-4.6 |

At eight streams the pass kinds split (298 records: 11 passes, 6.323 ms busy, draft; 1305: 4
passes, 48.743 ms, the main verify; 1335: 2 passes, 25.422; 303: 2 passes, 4.007). The main
verify is 48.7 ms against §2's 56.5 ms for its 1375-record pass, and the draft is 6.3 ms, the
same as §2's 6.3. The step is 20.08 tokens against 18.96, so per-token time is lower; summed throughput is
425.3 against 443 (-4.0 %).

## 2.4 Prefill profile, 6681-token prompt

`prof.yaml`, phase `pf6`, `RADIANCE_PROFILE_EVERY=1`; `profdiff.py <server.log> 0 150 50`
(rank 0, op time per step). The prompt is 2048 + 2048 + 2048 + 537 rows. Profiling inflates
the TTFT: 2273 / 2261 / 2261 ms in the three runs (unprofiled 1952-1968 ms, see 2.5).

Third full chunk, 2048 rows at context 4096-6144 (dump 98 of 99):

| run | op time | wait |
|---|---|---|
| r1 | 571.9 ms | 22.5 ms |
| r2 | 567.8 ms | 22.0 ms |
| r3 | 568.1 ms | 22.5 ms |
| **median** | **568.1 ms** | **22.5 ms** |
| §2 | 564.7 ms | 20.5 ms |
| delta | +0.6 % | +2.0 ms |

The chunk before it is 570.4 / 564.7 / 566.9 ms, so the two full chunks at context 2048-4096
and 4096-6144 are within noise of each other; §2's 564.7 is inside this range.

Top ops, r2 (51 op kinds; §2 in the last column):

| op | calls | local ms | share | §2 ms |
|---|---|---|---|---|
| routed `moe_gemm_q` gate_up | 41 | 89.91 | 15.8 % | 87.2 |
| `mla_attn_paged_sink` (30 sliding) | 30 | 82.92 | 14.6 % | 82.2 |
| `mla_attn_sel_sink` (12 sparse) | 12 | 80.08 | 14.1 % | 79.9 |
| `ar_rmsnorm_ihc_junction` | 84 | 61.20 | 10.8 % | 62.3 |
| routed `moe_gemm_q` down | 41 | 46.39 | 8.2 % | 45.6 |
| `gemm_nt_q` attn_output | 42 | 22.10 | 3.9 % | 22.5 |
| `mla_expand_gate` | 42 | 20.63 | 3.6 % | 21.2 |
| `gemm_nt_q` attn_gate | 42 | 20.34 | 3.6 % | 20.5 |
| `gemm_nt_f32` router GEMM | 41 | 18.14 | 3.2 % | 17.3 |
| `moe_gather` | 41 | 14.65 | 2.6 % | 14.6 |
| `swiglu_clamped_had_quant_i8g` | 41 | 10.35 | 1.8 % | 10.4 |
| `mla_project_kv` (absorb) | 42 | 8.30 | 1.5 % | 8.5 |
| `gemm_nt_q` attn_q_nope | 42 | 6.29 | 1.1 % | 6.4 |
| `idx_score` | 12 | 6.04 | 1.1 % | 6.1 |
| `had_quant_act_i8g` | 130 | 5.92 | 1.0 % | 5.9 |

The routed gate_up + down + junction are 197.5 ms (34.8 %), the two MLA sinks 163.0 ms
(28.7 %), and every listed op is within 5 % of §2; the shape is the same as the reference.

Last chunk, 537 rows: 339.5 / 343.2 / 341.9 ms against §2's 338.3 (+0.4 to +1.5 %). §2's routed
gate_up + down are 213.6 ms of its 338.3 (63.1 %); local's pair is 221.7 ms of 343.2, the same
unstaged read-in-place cost.

## 2.5 Unprofiled rates, small prompts, stalls, host phases

Unprofiled (`nsv`, debug logs, no stamps) and the §2 reference:

| figure | local r1 / r2 / r3 | median | §2 | delta |
|---|---|---|---|---|
| one stream | 13.710 / 13.738 / 13.704 ms | 13.710 ms | 13.84 | -0.9 % |
| one stream, summed | 73.8 / 73.2 / 73.2 tok/s | 73.2 | 72.3 | +1.2 % |
| eight streams, agg | 310.1 / 309.1 / 305.7 | 309.1 | 306-317 | +1.1 % |
| eight streams, summed | 322.2 / 320.8 / 317.4 | 320.8 | 315.7 | +1.6 % |
| DFly2 one stream (`cpv`) | 21.586 / 21.981 / 21.700 ms | 21.700 | 21.9-22.4 | -2.0 % |
| DFly2 eight (`cpv`) | 453.1 / 437.0 / 427.6 summed | 437.0 | 446.1 | -2.0 % |
| `pf2` 2385 tokens, TTFT | 716 / 706 / 730 ms | 716 | 707 | +1.3 % |
| `tail400` 159 tokens, TTFT | 192 / 186 / 206 ms | 192 | — | |
| `pf6` 6681 tokens, TTFT | 2049 / 1954 / 1968 ms | 1968 | 1.92-1.99 s | +0.4 % |
| `pf20` 24856 tokens, TTFT | 7173 / 7141 / 7179 ms | 7173 | 7.03 s | +2.0 % |
| `stall1` arrival | 3.02 / 3.05 / 3.04 s | 3.04 | 3.08 | -1.3 % |
| `stall4` arrival | 2.84 / 2.87 / 2.89 s | 2.87 | 2.82 | +1.8 % |

Small prompts against §2.6 (`nsv`, tags `sm-r1..r3`, phases `c1,tail742,tail2560,tail2600`; the
phases are all new prompts in a run, and `tail2600` shares 2560 characters of prefix with
`tail2560` yet is no cheaper, so the prefix cache does not resume between phases here and each
`tail` meets a cold prefix):

| prompt tokens | local sm r1 / r2 / r3, TTFT | median | §2.6 | delta |
|---|---|---|---|---|
| 73 (`c1`) | 78 / 78 / 78 ms | 78 | 78-88 ms | at the low end |
| 256 (`tail742`) | 152 / 149 / 151 ms | 151 | 223 ms | **-32 %** |
| 723 (`tail2560`) | 269 / 264 / 269 ms | 269 | 255 ms (at 717) | +5.5 % |
| 734 (`tail2600`) | 265 / 257 / 261 ms | 261 | — | |

The 73-token figure is the `c1` phase's own TTFT: §2.6 lists that point but the tables above never
reported it. The 723-token point is six tokens over §2.6's 717 (a `tail` cut lands on characters,
not tokens); six rows of prefill are 1-2 ms, so the comparison holds, and `tail2600` (734 tokens,
261 ms) shows the same slope from the other side.

The local lane's small steps are cheaper than §2.6's at 73 and 256 rows (78 vs 78-88 ms, 151 vs
223 ms) and dearer at 717 rows (269 vs 255 ms), so the gap is not one fixed cost. §2.6's 256-token
row is described as a lone cold prefill as well, so the -32 % is a difference between the two
measurements rather than a definition; `iks-B` reaches §2.6 here (228 ms), so the low figure is
this lane's small-step path rather than a mistake in §2.6.
The same three runs' `c1` gave 13.65 / 13.68 / 13.64 ms a step, within noise of the 13.710 ms above.

Host phases from the debug `step decode:` lines (`nsv` r1, mean/128):

| class | n | total | ranks | collect | host work |
|---|---|---|---|---|---|
| one (<18 ms) | 7 | 13.98 | 0.37 | 13.59 | 0.02 |
| eight (18-50 ms) | 4 | 23.37 | 0.67 | 22.65 | 0.04 |
| stall (>=50 ms) | 1 | 99.91 | 20.43 | 32.68 | 46.80 |

`r2` gives 13.93 / 0.45 / 13.46 / 0.02, 23.43 / 0.74 / 22.57 / 0.04 and stall 98.63 / 19.83 /
31.94 / 46.85. Host work is off the critical path for decode (0.02-0.04 ms); the stall line is
the arrival prefill, whose host side is 46.8 ms of plan/build/save for 5881 prompt tokens.

## Delta summary

| §2 figure | local | delta |
|---|---|---|
| 2.1 one stream, stamped | 13.773 ms / 71.4 | -0.5 % / -1.3 % |
| 2.1 `c1b` | 13.403 ms / 73.2 | below §2's 13.5-13.9 / 72-74 |
| 2.1 layer sliding / sparse | 283.7 / 252.1 us | -1.2 % / -2.3 % |
| 2.2 eight streams | 308.8 agg / 320.8 summed | +1.6 % (summed) |
| 2.3 DFly2 one | 22.742 ms, 2.813 tok/step | +0.2 % ms |
| 2.3 DFly2 eight | 425.3 summed | -4.0 % |
| 2.4 third chunk | 568.1 ms | +0.6 % |
| 2.5 one / eight unprofiled | 13.710 / 309.1 | -0.9 % / +1.1 % |
| 2.5 small prompts, 73 / 256 / 723 tokens | 78 / 151 / 269 ms | at the range / -32 % / +5.5 % |
| 2.5 `pf6` / `pf20` TTFT | 1968 / 7173 ms | +0.4 % / +2.0 % |
| 2.5 stall1 / stall4 | 3.04 / 2.87 s | -1.3 % / +1.8 % |

The local lane reproduces §2: one stream and the layer path within 1-2 %, eight streams +1.6 %,
DFly2 c1 +0.2 % a step, prefill +0.6 %. The one material difference is DFly2 eight streams
-4.0 % in summed tok/s at +5.9 % tokens a step, i.e. HEAD spends slightly more time per step
for a larger accepted tree; §1's own DFly2 one-stream range (125-139 tok/s) also puts this
run's 123.2 summed at the low end.
