# Issue #324: W2 on, 16 streams on TP4, about 20 % slower — measurements

**Verdict:** the loss is W2's form on passes above the routed GEMMs' decode band, which then ran every routed
layer twice. With the form kept inside the band, the TP4 c16 aggregate is back at the W2-off level (+25.7 %
against the base), c1 and c8 are level, #309's own DFly2 c8 case is unchanged, and greedy output is
byte-identical.

- **before** = `e3834473` (default branch, W2 on by default), **after** = `922518af` (this fix). The two arms of
  each pair run the same config and differ only in the build.
- Measured by adamczhang's agent on RED-LINUX, 2026-11-27, 10:33–13:10 CT.

## 1. Setup

| | |
|---|---|
| GPUs | 4 × AMD Radeon AI PRO R9700 (gfx1201), 32 GiB, HIP 0–3 (PCI 03, 06, 09, 20:00.0). Root ports: x8 on 03/09, x4 on 06/20 (Gen5 links). Power cap 270 W, read back from `power1_cap` on all four. |
| Host | Ryzen 9 9950X3D, 121 GiB RAM, kernel 7.0.0-44-generic, ROCm core 10.1.0 (no container), in-tree amdgpu |
| Build | `cmake -G Ninja -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201`, ROCm 10.1 hipcc / clang, RelWithDebInfo; before and after built the same way |
| Model | `hy4-flash-iqr-v4-u4.rad` (keptn), TP4; #309's case: `hy4-flash-iqr-v4-u3.rad`, TP2 on HIP 0,2 (both x8), with HIP 2,3 (x8 + x4) as a side check |
| Serving | `max-model-len` 524288, DFly2 `num-speculative-tokens` 5 (`draft-coupling on`, `draft-verify traversal`, `draft-depth adaptive`, `draft-depth-load on`, `draft-suffix on`), `ngram-placement: ram`, `placement: expert_tiered`, `host-pool-mib` 24000, `prefix-cache-mib` 2048, `checkpoint-slots` 128 |
| Issue shape | `max-num-batched-tokens` 8192, `checkpoint-interval` 4096, `max-num-seqs` 16 |
| Owner shape | `max-num-batched-tokens` 2048, `checkpoint-interval` 1984, `max-num-seqs` 16 |
| Load | BetterBench 0.6.1, `--concurrency`, levels 1 / 8 / 16, 96 requests a level, temperature 0.9, top-p 0.95, top-k 20, 3 warm-up, unique nonce (cold prefix). Fresh server per arm. Nothing else on the cards. |

**Why a host pool.** Without one, every u4 expert fits on the four cards and the server prints `W2 off: no expert
sits in a pinned pool`. W2 never turns on, and the loss cannot show; that is why an earlier c16 run on this machine
at 775 tok/s did not show it. A 24000 MiB pool (5.9 GiB a rank) puts experts in the pool, and the server prints the
issue's line: `W2 on: resident experts first on passes of 16 rows and up, at most 8 whole units a routed layer by
DMA (budget at 50 / 100 / 200 routed experts: 1 / 3 / 7)`. The prefix cache is cut to 2048 MiB so that the pool
fits in RAM beside the 60 GiB pinned n-gram table.

## 2. Reproduction (before build, `RADIANCE_W2=0` against the default)

Aggregate tok/s, every row 96/96 ok, run in the order shown:

| arm | start (CT) | W2 | c1 | c8 | c16 |
|---|---|---|---|---|---|
| off1 | 10:33 | `RADIANCE_W2=0` | 200.4 | 598.9 | 767.2 |
| on1 | 10:41 | on | 205.3 | 616.5 | **615.2** |
| on2 | 10:47 | on | 206.0 | 619.3 | **623.9** |
| off2 | 10:53 | `RADIANCE_W2=0` | 205.6 | 619.2 | 773.3 |
| own-on (owner shape) | 11:41 | on | 203.7 | 614.1 | **617.2** |
| own-off (owner shape) | 11:48 | `RADIANCE_W2=0` | 206.1 | 626.5 | 792.9 |

At c16, W2 on is 19.8 % below off in the issue's shape and 22.2 % below in the owner's. c1 and c8 are level, as in
the issue.

W2's own counters (rank 0, on1) rule out the copy budget: `112627 routed layers planned, 30 misses, 30 units posted,
0 read from the ring and 15 in place by the tail`. The heat engine keeps the hot experts on the card at this pool
size, so the copies carry nearly nothing.

## 3. Cause

- The routed GEMMs read a W2 operand only on their (tile, run) decode grid. libr4d takes that grid while
  `2 T <= 3 n_expert`, with T the pass's sorted rows (`r4d_moe_decode_band`).
- Above that, gate_up (`moe_iqr`) and down (`moe_gemm_q`) take a prefill form that ignores the operand and runs
  **every expert on both** the first launch and the tail. The quantiser between them runs again too.
- The MoE block asked for the form at every decode pass of 16 rows and up, with no upper bound.

Hy4-flash routes 512 experts at top-16, so the band ends at T = 768, which is **48 pass rows**:
- A DFly2 c8 verify pass is at most 8 × 6 = 48 rows, inside the band.
- At c16 the verify pass is 64–96 rows at depth 3–5. In the step log of the c16 diagnosis run (below), 198 of the
  354 sixteen-sequence decode steps were over 48 rows.

**AQL stamps.** Card 0, every 25th pass (`RADIANCE_AQL_STAMP=25`), BetterBench c16 only, 48 requests, with
`RADIANCE_LOG_STEPS=1`. Sums per pass of the routed kernels, then the pass's wall time (median), by tape kind:

| build | pass kind (tape records) | passes | gate_up launches / µs | down launches / µs | wall ms |
|---|---|---|---|---|---|
| before | W2, above the band (1561) | 6 | 82 / 24,282 | 82 `moe_gemm_q` / 12,718 | **59.43** |
| after | the same height, no W2 (1356) | 6 | 41 / 12,232 | 41 `moe_gemm_q` / 6,386 | **40.05** |
| before | W2, in the band (1439) | 8 | 82 / 13,300 | 82 `moe_down_dec` / 6,351 | 36.85 |
| after | W2, in the band (1439) | 2 | 82 / 13,361 | 82 `moe_down_dec` / 6,234 | 35.73 |

An out-of-band c16 verify pass spent about 18 ms (+48 %) running its routed layers a second time. This
diagnosis run's aggregate (stamps and the step log on, 48 requests) was 623.5 before and 731.9 after. The clean
A/B below is the measurement.

## 4. Before / after A/B (TP4, issue shape, W2 on by default in both)

Interleaved ABBAAB, fresh server per arm, 11:58–12:33 CT. Raw BetterBench lines (level, ok/req, aggregate tok/s,
TTFT p50 ms, TTFT p99 ms†, per-request decode median tok/s):

```
before1 11:58:49 | 1 | 96/96 | 204.4 | 51.3 | 69.4† | 225.1 |
before1 11:58:49 | 8 | 96/96 | 616.9 | 92.0 | 226.0† | 89.3 |
before1 11:58:49 | 16 | 96/96 | 619.3 | 114.3 | 310.4† | 43.9 |
after1 12:04:42 | 1 | 96/96 | 205.9 | 50.8 | 71.1† | 223.1 |
after1 12:04:42 | 8 | 96/96 | 621.6 | 91.7 | 197.2† | 90.0 |
after1 12:04:42 | 16 | 96/96 | 786.7 | 121.7 | 317.6† | 58.5 |
after2 12:10:18 | 1 | 96/96 | 204.3 | 50.7 | 94.5† | 222.4 |
after2 12:10:18 | 8 | 96/96 | 615.2 | 92.5 | 225.2† | 87.5 |
after2 12:10:18 | 16 | 96/96 | 782.1 | 119.0 | 314.2† | 60.4 |
before2 12:15:58 | 1 | 96/96 | 204.0 | 51.4 | 69.7† | 223.8 |
before2 12:15:58 | 8 | 96/96 | 615.0 | 93.0 | 197.7† | 91.3 |
before2 12:15:58 | 16 | 96/96 | 618.4 | 122.3 | 314.2† | 43.5 |
before3 12:21:49 | 1 | 96/96 | 202.7 | 51.1 | 69.4† | 225.0 |
before3 12:21:49 | 8 | 96/96 | 617.6 | 91.3 | 198.6† | 86.6 |
before3 12:21:49 | 16 | 96/96 | 626.4 | 127.8 | 304.4† | 43.9 |
after3 12:27:40 | 1 | 96/96 | 205.3 | 51.4 | 69.6† | 226.7 |
after3 12:27:40 | 8 | 96/96 | 610.2 | 91.7 | 197.2† | 88.5 |
after3 12:27:40 | 16 | 96/96 | 773.8 | 124.9 | 318.2† | 59.5 |
```

† BetterBench marks a p99 from 96 samples as under-sampled. Read it as roughly the maximum.

| level | before (3 runs) | after (3 runs) | mean before → after | change |
|---|---|---|---|---|
| c1 | 204.4 / 204.0 / 202.7 | 205.9 / 204.3 / 205.3 | 203.7 → 205.2 | +0.7 % (level) |
| c8 | 616.9 / 615.0 / 617.6 | 621.6 / 615.2 / 610.2 | 616.5 → 615.7 | −0.1 % (level) |
| c16 | 619.3 / 618.4 / 626.4 | 786.7 / 782.1 / 773.8 | 621.4 → 780.9 | **+25.7 %** |

Per-request decode at c16 goes from 43.5–43.9 to 58.5–60.4 tok/s. After the fix, c16 (780.9) matches the
same-build W2-off runs of §2 (767.2 / 773.3), so the fix leaves no remainder against W2 off.

W2 counters on rank 0: before planned 112,750 / 114,636 / 111,684 routed layers, after 81,221 / 81,426 / 83,517.
The roughly 31,000 a run that are gone are the out-of-band passes' layers, which now issue the switch-off step.

## 5. #309's measured case (TP2, u3, DFly2 c8), one pair a lane

The setup:
- u3, `host-pool-mib` 30000 (#309 measured at 39000; 30000 is what RAM holds beside the n-gram table);
- `max-num-seqs` 8, 2048 / 1984, `checkpoint-slots` 64, otherwise as §1;
- BetterBench c1 and c8, 96 requests, AQL stamps every 25th pass;
- the same before and after builds as §4, before then after on each lane.

Here W2 is fully at work: every arm had about 75,000 routed layers planned, 263,000–275,000 misses, and 93–95 % of
posted units read from the ring.

**The x8 lane: HIP 0,2** (PCI 03:00.0 and 09:00.0, each behind an x8 Gen5 root port), 12:56–13:10 CT. Rank 0's
counters: before 74,743 layers, 274,995 misses, 249,841 units posted, 95.4 % from the ring; after 75,440 / 274,556 /
250,603 / 95.5 %.

```
b309 (before) 12:56:42 | 1 | 96/96 | 143.9 | 75.3 | 106.6† | 157.8 |
b309 (before) 12:56:42 | 8 | 96/96 | 403.3 | 177.6 | 320.7† | 58.5 |
a309 (after)  13:03:29 | 1 | 96/96 | 144.0 | 76.1 | 100.1† | 156.9 |
a309 (after)  13:03:29 | 8 | 96/96 | 405.9 | 184.8 | 384.9† | 56.5 |
```

The c8 verify pass (tape of 1314 records), from `W2-tools/w2stamps.py <stamps> 1314 --on`, p50 a layer:

| row | before | after |
|---|---|---|
| passes stamped | 66 | 52 |
| gu first / dn first (µs) | 372.7 / 204.3 | 375.6 / 205.6 |
| gu tail / dn tail (µs) | 33.2 / 20.4 | 33.6 / 20.5 |
| routed span, router end → last dn end (µs) | 656.8 | 659.2 |
| pass wall (ms, median) | 41.51 | 41.32 |

**Side check, the x8 + x4 lane: HIP 2,3** (09:00.0 x8, 20:00.0 x4), 12:33–12:47 CT. Rank 0's counters: before
77,736 layers, 263,440 misses, 240,991 units posted, 93.3 % from the ring; after 77,203 / 268,232 / 245,114 / 93.8 %.

```
b309 (before) 12:33:20 | 1 | 96/96 | 138.7 | 103.2 | 147.3† | 153.8 |
b309 (before) 12:33:20 | 8 | 96/96 | 357.6 | 233.2 | 540.1† | 50.7 |
a309 (after)  12:40:24 | 1 | 96/96 | 136.9 | 103.5 | 150.5† | 152.2 |
a309 (after)  12:40:24 | 8 | 96/96 | 357.3 | 239.8 | 512.3† | 49.8 |
```

| row (c8 verify pass, p50 a layer) | before | after |
|---|---|---|
| passes stamped | 68 | 65 |
| gu first / dn first (µs) | 366.1 / 200.6 | 363.9 / 199.2 |
| gu tail / dn tail (µs) | 31.6 / 19.6 | 31.2 / 19.3 |
| routed span (µs) | 645.7 | 642.4 |
| pass wall (ms, median) | 43.28 | 43.44 |

On both lanes the launches are the same (82 gate_up, 82 down and 124 quantiser launches a pass, before and after),
and every row is within noise. Every pass of this shape lies inside the band, so the fix does not touch it. The two
lanes differ from each other (c8 403–406 tok/s on x8 + x8 against 357 on x8 + x4), but before and after do not
differ on either.

## 6. Output parity

Greedy (temperature 0), two fixed prompts of 200 tokens each (Python Fibonacci function; why the sky is blue),
SHA-1 over reasoning plus content. TP4, issue shape, 11:55–11:58 CT:

| build | setting | prompt 0 | prompt 1 |
|---|---|---|---|
| after | `RADIANCE_W2_MIN_ROWS=1` (W2 at every height) | df62699fb96f | d3f8497c0572 |
| after | `RADIANCE_W2=0` | df62699fb96f | d3f8497c0572 |
| before | `RADIANCE_W2=0` | df62699fb96f | d3f8497c0572 |
| before | `RADIANCE_W2_MIN_ROWS=1` | df62699fb96f | d3f8497c0572 |

With `MIN_ROWS=1` the single-stream passes take the form, with misses moving: rank 0 had 45 misses, 28 read from
the ring and 14 in place (after); 46 / 29 / 14 (before). In both TP2 pairs every arm's greedy probe gave
adc8034a374c / 0078cb982726, before and after. Above the band the after build issues the switch-off step, which
`hyv4flash_arch_test` checks issue for issue.

## 7. Link and AER

All four cards stayed at 32 GT/s x16 with correctable / nonfatal / fatal AER 0 / 0 / 0, before and after every
script. The cards' own links read 32 GT/s x16; the root ports behind them are x8 and x4 (§1). NVMe temperatures peaked at 73 °C.

## 8. Merge bar (bug fix)

| clause | result |
|---|---|
| output byte-identical | holds (§6) |
| the targeted step faster (stamps) | out-of-band c16 verify pass 59.43 → 40.05 ms, routed launches 82 → 41 a kind (§3) |
| end-to-end not worse | c16 +25.7 %, c8 −0.1 %, c1 +0.7 % (3 + 3 interleaved, §4); #309's c8 case level on the x8 lane and the x8 + x4 lane (§5) |
