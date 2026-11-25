# idxint8 notes: RADIANCE_HY4_INDEXER=int8 (branch hy4-idxint8, wt/idxint8)

`RADIANCE_HY4_INDEXER` is now `exact` (default) | `fp8` (unchanged) | `int8`. int8 = fp8's wiring on the same
packed 136-byte page (128 codes + f32 scale at byte 128) with symmetric int8: scale = max(amax,1e-4)/127 (f32, per
key / per query head row), codes rint(v * (1/scale)) clamped to +-127, after the same f32 key and FWHT-128 * 128^-0.5
(no bf16 rounding). Ops: `idx_pool_gated_i8`, `dsa_query_i8` (new, libref/libavx/libr4d), `dsa_score` with
`dtype int8` (libref/libavx generic; libr4d rows `dsa_score_i8_h16`/`dsa_score_i8`: v_wmma_i32_16x16x16_iu8 via
r4d_fp8::mma8<true>, exact int32 sums, relu, then qscale*gate and kscale*(heads*n)^-0.5 as fp8).
Scale rule: host study (/root/work/claude-scratch/qstudy.py, 2048 keys x 16 heads, top-511 Jaccard vs exact scores
on FWHT'd heavy-tailed data): fp8 pow2 0.954 (relL2 3.1e-2), int8 absmax/127 0.987 (6.8e-3), int8 pow2 0.981 (1.1e-2)
-> absmax. (An int8 scale is not a power of two, so the f32 key's 1-ulp libm differences show in the scale on card
vs libref: codes and page bytes are exact, scales agree to 2 ulp; the selftest checks that.)

## Tests
host (ROCR_VISIBLE_DEVICES=99): ref_hy4_ops_test (new: pool_i8 open-coded absmax, query_i8 + int8 score), ref_test,
hyv4flash_arch_test (exact/fp8/int8 op sets, page layout, lint, bad modes) all pass; rad-avx-check idx_pool_gated_i8,
dsa_query_i8, dsa_score (incl. int8 cases) PASS. Card (R9700, under lock): r4d_selftest --case idx_pool_gated_i8
(gptj/imrope, 3 and 37 groups), dsa_query_i8, dsa_score_i8_h16, dsa_score_i8: ok; fp8 cases unchanged ok.

## gpu-mini comparison (work/idxint8/gpu.sh, cmp.py, cmp.out; oneshot, taps index_sel/index_nsel only, HF = free f32)
p2049/p2052 (1 / 4 selecting rows, every closed group recalled): identical in all modes, KL(exact||x)=0.
p3000 (952 selecting rows per sparse layer; Jaccard of recalled sets vs exact):
| | fp8 | int8 |
| L0 Jaccard mean / min / rows different (TP1) | 0.9882 / 0.965 / 92.6 % | 0.9967 / 0.984 / 61.4 % |
| L2 (inputs already moved) TP1 | 0.9715 / 0.762 / 95.5 % | 0.9867 / 0.806 / 84.3 % |
| L0 / L2 TP2 | 0.9882 / 0.9714 (92.6 / 95.8 %) | 0.9967 / 0.9859 (61.4 / 85.9 %) |
| logits KL(exact||x) mean / max, TP1 | 1.20e-2 / 0.60 | 5.75e-3 / 0.52 |
| same, TP2 | 1.22e-2 / 0.61 | 6.48e-3 / 0.52 |
| top-1 agreement with exact TP1 / TP2 | 94.5 / 94.7 % | 97.3 / 97.1 % |
| KL(HF free || x) TP1 (exact 1.02e-2) | 1.74e-2 | 1.20e-2 |
| KL(HF free || x) TP2 (exact 1.11e-2) | 1.81e-2 | 1.31e-2 |
| top-1 vs HF free TP1 (exact 0.9476) | 0.9263 | 0.9460 |
(fp8 numbers reproduce WP14's.) int8 roughly halves the selection error and the logit shift against exact; it sits
within ~0.002 KL of exact against free HF. Random weights, many near-ties: pessimistic.

## Indexer time (r4d_selftest --perfidx CTX M, 16 heads, one sequence, warmed so L2-resident; us per launch)
| ctx | M | exact idx_score | fp8 query+score | int8 query+score | qsa_select |
| 64k (16384 groups) | 1 | 6.1 | 3.2+10.8 = 14.0 | 3.3+10.7 = 13.95 | 153.8 |
| 64k | 4 | 9.6 | 32.1 | 32.4 | 156.5 |
| 200k (50000 groups) | 1 | 11.6 | 27.8 | 27.4 | 304.7 |
| 200k | 4 | 22.8 | 75.8 | 76.3 | 313.8 |
int8 and fp8 cost the same (same WMMA rate); both are SLOWER than the exact bf16 idx_score at these sizes (the scorer's
tile is not tuned for 16 heads / 1 row and the exact row is a plain bf16 kernel), and qsa_select (150-310 us) dominates
all three. The quantised page is 136 vs 256 B a group (less DRAM traffic on a cold cache; the warmed loop does not show
it) and saves cache memory. Pooling (once per 4 tokens) is not in these numbers.
