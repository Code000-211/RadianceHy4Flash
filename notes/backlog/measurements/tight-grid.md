# #315: tight run-tile grid for `moe_iqr` (gate_up); `moe_down_dec` keeps its bound

Issue #315 (part of #307), from `W2-proposals.md` §5.3 / proposal 1. Lossless: the same tile is computed by
the same per-tile block code, only possibly by another block, so every byte is the one-tile-a-block launch's.

**Result.** The decode-band grid of `moe_gemm_iqr` now holds `r4d_moe_run_grid(T, bound) = min(bound, T/2 + 64)`
tile blocks plus the tail block, and a launch with more tiles has block `x` serve tiles `x, x + G, ...` (the
`WR` form, taken only where the grid is below its bound, i.e. from 129 rows). At the DFly2 c8 verify pass
(T = 512) that is 321 instead of 513 blocks a column group. **DFly2 c8 verify pass: −0.75 ms routed block,
−0.71 ms wall (miss-normalised, 5 runs against 7), the gate_up segment −1.03 ms with no overlap between the
arms' runs; DFly2 c1, no-spec c1/c8 and prefill/TTFT level; greedy output identical in every run.** The same
grid on `moe_down_dec` was built, measured and dropped: it made the down launch 0.2-0.5 ms a pass slower.

Machine: local lane (the Moebius host, 2x R9700 gfx1201, cap 210 W read back by `lane.sh` before every run),
u3 container, `coupled03.yaml` / `nospec.yaml` as served, `RADIANCE_AQL_STAMP=1`, `RAD_SRC` pinned to the base
tree for every arm. Compiler per CARDS.md §1.1 (`/opt/rocm-repeatable`, `RAD_HIP_VOPD_REPEATABLE=ON`), both arms.

| arm | tree / commit | what |
|---|---|---|
| `off2` | `fbdf3f90e` (default branch with the #314 VRAM-queue fix) | base |
| `gu3` | `b8ad5d5ad` (= `79348ed22` + the merge of `fbdf3f90e`) | **the merged form**: tight grid for `moe_iqr`, `moe_down_dec` at its bound |
| `tg2` | `83f56b2e5` (`674603861` + merge of `fbdf3f90e`) | tight grid for both kernels, `WR` form only below the bound |
| `off`, `tg` | `49934f24c` / `15ba72ed8` | first rounds (a-c): base before #314, tight grid for both with the walk always compiled in |

Rounds interleaved, arm order alternated (`d`-`k`; `g`-`i` three arms). Every start's server log shows
`a card at 35.36-35.37 MiB` for the hardware queues (no 175 MiB start among the `off2`/`tg2`/`gu3` runs;
`bns-off` in the first rounds had one and is left out). Runs: `runs/local/<round><set>-<arm>/` in each tree.

## 1. Tile counts (why T/2 + 64)

Counted per launch on the served model (temporary instrumentation on a side branch, `tight-grid-count`, not
merged: block (0,0) of every decode-grid `moe_iqr` launch adds (T, tiles) into a host-mapped histogram).
`DFly2` = `coupled03.yaml greedy,c1,c8`, `ns` = `nospec.yaml c1,c8,pf6`; launches per rank.

| run | T | launches | bound | new grid | min | p50 | max |
|---|---|---|---|---|---|---|---|
| DFly2 | 16 | 8 | 16 | 16 | 16 | 16 | 16 |
| DFly2 | 48 | 697 | 48 | 48 | 20 | 32 | 48 |
| DFly2 (c1 verify) | 96 | 12792 | 96 | 96 | 23 | 57 | 92 |
| DFly2 | 128 | 82 | 128 | 128 | 56 | 82 | 111 |
| DFly2 | 256 | 287 | 256 | 192 | 105 | 143 | 196 |
| DFly2 | 384 | 41 | 384 | 256 | 122 | 166 | 238 |
| DFly2 | 448 | 164 | 448 | 288 | 76 | 171 | 282 |
| DFly2 (c8 verify) | 512 | 1025 | 512 | 320 | 120 | 210 | 305 |
| DFly2 | 560 | 287 | 515 | 344 | 140 | 217 | 306 |
| DFly2 | 640 | 205 | 520 | 384 | 166 | 238 | 317 |
| DFly2 | 720 | 41 | 525 | 424 | 188 | 258 | 335 |
| ns (c1) | 16 | 14355 | 16 | 16 | 16 | 16 | 16 |
| ns (c8) | 128 | 10456 | 128 | 128 | 25 | 97 | 120 |

`T/2 + 64` is above the maximum at every T but 256 (196 against 192: a few launches walk one tile). Below 129 rows
the bound is smaller, so c1, no-spec c8 and prefill (which is the run map, above the band) launch exactly as
before. The rule is a function of T alone: recorded passes keep one launch, no device-to-host sync.

## 2. DFly2 c8 verify pass (the 1191-record pass)

`w2arms.py` (W2-tools, with a fixed slope of 0.0547 ms a miss from W2-proposals' eight `off` runs, since seven
runs over 114-134 misses do not fit their own slope; the intercept from the `off2` runs). `kc8-off2` is left
out: its gate_up took 26.0 ms (+3.6 ms, every other base run 21.8-22.1; the lane was shared with another agent's
card jobs that night). With it the `gu3` distance is −1.27 ms.

```
off: 7 runs, misses a step 114-134; routed block = 28.51 + 0.0547 x misses ms (residual sd 0.49), wall = 42.34 + 0.0547 x misses (sd 0.50)
arm        runs  routed block vs off      wall vs off  misses router ->gu end ->dn end  tail
off2          7         -0.00 (0.49)     -0.00 (0.50)   124.0   1.27    22.38    11.15  0.00
tg2           6         -0.31 (0.68)     -0.28 (0.69)   121.1   1.28    21.68    11.37  0.00
gu3           5         -0.75 (0.66)     -0.71 (0.73)   117.8   1.28    21.35    11.09  0.00
```

Raw (`passes.py`, kernel sums a pass, not normalised; then the yardstick-normalised pass wall; greedy sha256
prefixes of the three fixed prompts and the 5.7K arrival):

```
dc8-off2 | gu 21.842 dn 10.332 wall_norm 48.872 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
ec8-off2 | gu 21.775 dn 10.414 wall_norm 49.013 | a card at 35.37 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
fc8-off2 | gu 21.922 dn 10.437 wall_norm 49.217 | a card at 35.37 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
gc8-off2 | gu 21.843 dn 10.422 wall_norm 49.087 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
hc8-off2 | gu 21.942 dn 10.426 wall_norm 49.022 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
ic8-off2 | gu 21.810 dn 10.427 wall_norm 49.381 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
jc8-off2 | gu 22.083 dn 10.559 wall_norm 49.240 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
kc8-off2 | gu 26.173 dn 10.540 wall_norm 52.632 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420   (left out)
gc8-gu3  | gu 20.739 dn 10.395 wall_norm 47.548 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
hc8-gu3  | gu 20.971 dn 10.461 wall_norm 48.141 | a card at 35.37 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
ic8-gu3  | gu 21.283 dn 10.551 wall_norm 48.310 | a card at 35.37 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
jc8-gu3  | gu 20.971 dn 10.526 wall_norm 48.611 | a card at 35.37 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
kc8-gu3  | gu 21.262 dn 10.556 wall_norm 47.737 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
dc8-tg2  | gu 20.551 dn 10.391 wall_norm 47.637 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
ec8-tg2  | gu 21.352 dn 11.049 wall_norm 49.239 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
fc8-tg2  | gu 21.806 dn 10.815 wall_norm 48.380 | a card at 35.37 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
gc8-tg2  | gu 22.166 dn 10.884 wall_norm 48.947 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
hc8-tg2  | gu 22.078 dn 10.937 wall_norm 48.955 | a card at 35.37 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
ic8-tg2  | gu 21.483 dn 10.916 wall_norm 48.936 | a card at 35.36 MiB | 5b7a3975 f55fe020 0075e702 8f60e420
```

The first rounds (old base `49934f24c`, arms `off` / `tg`: tight grid on both kernels, the walk always compiled
in) read the same way: `tg` −1.17 ms routed block over 3 runs against 2, gate_up segment −0.8 ms, down
segment +0.4..0.8 ms.

**Why `moe_down_dec` loses.** Its grid is as empty as gate_up's (545 blocks a column group for ~210 tiles at T =
512), but it has 16 column groups to gate_up's 24, and its empty blocks were never its cost: in W2-proposals §5.3
512 more of them a row did not move the down launch. The walk, however, changes its code: the `WR`
instantiations take 7-15 more VGPRs (97-128 → 112-135) and 107 SGPRs (73-81 before), and the down launch is
0.2-0.5 ms a pass slower with it (`tg2` against `gu3`, same gate_up). Tile order and locality are not the cause:
below the grid, block `x` still serves tile `x` and dispatch order is unchanged; no launch at T = 512 walks.

**Why `WR` is a template flag.** With the walk always compiled in (`tg`, the first rounds), `moe_iqr` on
Hy4-flash's 3-bit form went from 138 to 114 VGPRs and 59 to 107 SGPRs, and the DFly2 c1 verify pass's gate_up
(T = 96, where the grid is the bound and nothing else changes) was ~1.7 % slower: 5.112 → 5.199 ms a pass over
three runs each. With the flag, a grid at its bound launches the base's code again (the same VGPR/SGPR counts
in 46 of 50 instantiations, +2 SGPRs in four chained ones), and c1 is level (§3).

## 3. The rows that must not get worse (`off2` / `tg2`; below 129 rows `tg2`'s kernels are `gu3`'s)

DFly2 c1 (`coupled03.yaml c1`, the 1137-record verify pass, ms a pass):

```
dc1-off2  | c1 tps  129.0 miss    35.4 ttft 0.083 | rec1137 n102 wall 18.596 norm 20.845 gu 4.979 dn 3.045
ec1-off2  | c1 tps  126.0 miss    35.4 ttft 0.085 | rec1137 n103 wall 18.763 norm 21.004 gu 5.086 dn 3.095
fc1-off2  | c1 tps  126.4 miss    35.4 ttft 0.082 | rec1137 n103 wall 18.724 norm 20.942 gu 5.066 dn 3.089
dc1-tg2   | c1 tps  128.2 miss    35.4 ttft 0.082 | rec1137 n102 wall 18.533 norm 20.723 gu 4.988 dn 3.030
ec1-tg2   | c1 tps  128.3 miss    35.4 ttft 0.084 | rec1137 n102 wall 18.691 norm 20.912 gu 5.032 dn 3.075
fc1-tg2   | c1 tps  128.0 miss    35.4 ttft 0.083 | rec1137 n102 wall 18.708 norm 20.918 gu 5.047 dn 3.077
off2 n=3 | c1 tps 127.1 | rec1137 norm 20.930 (sd 0.080) gu 5.044 dn 3.076
tg2  n=3 | c1 tps 128.1 | rec1137 norm 20.851 (sd 0.111) gu 5.022 dn 3.061
```

No-spec (`nospec.yaml c1,c8,pf6,pf20`; 1052 records = c1 step, 1136 = c8 step):

```
dns-off2  | c1 70.7 | c8 305.8 | pf6 ttft 1.789 | pf20 ttft 6.406 | rec1052 norm 32.866 | rec1136 norm 26.153
ens-off2  | c1 71.5 | c8 301.2 | pf6 ttft 1.776 | pf20 ttft 6.437 | rec1052 norm 32.753 | rec1136 norm 26.498
fns-off2  | c1 71.1 | c8 306.0 | pf6 ttft 1.782 | pf20 ttft 6.449 | rec1052 norm 32.905 | rec1136 norm 25.973
dns-tg2   | c1 71.4 | c8 302.9 | pf6 ttft 1.762 | pf20 ttft 6.383 | rec1052 norm 32.784 | rec1136 norm 26.393
ens-tg2   | c1 70.6 | c8 294.8 | pf6 ttft 1.972 | pf20 ttft 6.726 | rec1052 norm 33.289 | rec1136 norm 26.243
fns-tg2   | c1 71.2 | c8 306.3 | pf6 ttft 1.777 | pf20 ttft 6.432 | rec1052 norm 35.250 | rec1136 norm 26.102
```

Level: c1 71.1 / 71.1 tok/s, c8 304.3 / 301.3 (302.9 and 306.3 without `ens-tg2`), TTFT pf6 1.78 / 1.77 s and
pf20 6.43 / 6.41 s without `ens-tg2`. `ens-tg2` (every phase slower, its c8 TTFT 0.476 s against 0.28) and the
c1 step of `fns-tg2` (15.0 ms against 13.7-13.8 in every other run) are disturbances of the shared lane; the
kernels these rows run are the base's code (no grid below 129 rows changes, and prefill is the run map).

## 4. Byte identity and tests

- `r4d_selftest --case moe_gemm_iqr` (W2 on and `RADIANCE_W2=0`): 0 failures, 347 ok, 99 grid checks: every
  decode shape at 1, 2, 3, tiles − 1, tiles and tiles + 1 tile blocks (`r4d_moe_gemm_pin_run_grid`) byte for
  byte against the rule's launch, plus a verify shape with every slot dropped (0 tiles), and W2's split cases
  (first launch with the walk at T 480 / 768, tail) against the plain launch.
- `--case moe_gemm_w4a8_mod4`: 0 failures (skips only, as on the base).
- `r4d_selftest --perfmoeiqr` (synthetic routing, back to back): the outputs' FNV hashes identical base vs new at
  M = 6, 8, 32; times level at M = 6/8 (±0.3 %); at M = 32 within ±2 %, no gain (a queue of back-to-back
  launches hides the empty blocks; the engine's stamps are the measure).
- `rad-kbench --no-ref --strict`: `kernels_moe.rkb --op moe` 80/80 passed; `kernels_hy4.rkb` (all 1747 cases)
  1729 passed, the 18 failures are `rmsnorm_ihc_junction` (host and device, untouched here).
- `r4d_emu_moefuse`, `r4d_emu_hy4`, `hyv4flash_arch_test`, `w2_test`, `runtime_test` pass.
- Greedy sha256 identical in all 24 DFly2 c8 runs of this note and in the merged tree's W2 checks (§5).

## 5. After the W2 merge (`ce861c423`)

W2's first gate_up launch takes the tight grid (and the walk where the grid is below its bound): a block whose
tile is a miss returns `ntiles` behind its search and moves on to its next tile. The tail stays on
`min(grid, rad_w2_tail_tiles(T) + 1)` blocks; `T/2 + 64` always exceeds `16 + T/16`, so the tail never walks.
`moe_down_dec` and its W2 form are the default branch's. Greedy checks on the merged tree (DFly2 `greedy,c8`):
one run each on `db6e9e902` (the merge), greedy sha256 prefixes, c8 tok/s and misses a step, then the verify
pass (1314 records with W2's tail form, 1191 without):

```
mw2on-c8   5b7a3975 f55fe020 0075e702 8f60e420  c8 376.6 tok/s, 129.9 misses | rec 1314: wall 47.428 gu 19.699 dn 10.571 wall_norm 47.284
mw2min1-c8 5b7a3975 f55fe020 0075e702 8f60e420  c8 363.6 tok/s, 117.1 misses | rec 1314: wall 45.954 gu 18.975 dn 10.241 wall_norm 46.630
mw2off-c8  5b7a3975 f55fe020 0075e702 8f60e420  c8 376.8 tok/s, 122.5 misses | rec 1191: wall 49.649 gu 22.076 dn 10.664 wall_norm 49.036
```

The bytes are the base's in all three. These are single runs for the sha check, not an A/B: the W2-off run's
gate_up sum (22.08 ms) is at the top of the `gu3` arm's spread and inside the base's, so the tight grid's gain
under the merged default (W2 on by default, P-16, #282) is not re-measured here; §2's interleaved A/B is the
measurement.

## 6. Tools

`passes.py` (per record count: pass wall, `moe_iqr` / `moe_down_dec` kernel sums, the trunk-linear yardstick),
`summary.py` and the driver are in the session's scratchpad; `w2arms.py` / `w2budget.py` are W2-tools' with a
`--slope` / `--ref` option added for this note.
