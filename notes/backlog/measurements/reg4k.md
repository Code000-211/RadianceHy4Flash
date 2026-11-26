# reg4k: the host pool on 4 KiB pages instead of THP (raw lines)

Issue #310 (part of #307). Question: `pcie-h2d.md` measured `reg4k` (4 KiB pages) at +20-25 % SDMA H2D throughput with
both TP2 cards copying at once, against THP-backed buffers. Does the engine's host pool (`host_pool`, `RAD_MEM_HOST_MAPPED`,
`pin_pieces` in `core/device/aql.cpp`) gain end to end when it sits on 4 KiB pages? **Answer: no. Pass wall and tok/s are
unchanged within the run-to-run spread, output is byte-identical, startup is ~5 s slower. The switch stays, default off.**

## Change

`RADIANCE_HOST_POOL_PAGES=4k` (also `4096`, `small`; default unset = THP as before) makes `pin_pieces` call
`MADV_NOHUGEPAGE` instead of `MADV_HUGEPAGE` for the host pool's one allocation, and only that one
(`core/mem/pools.cpp` sets `rad::g_pin_small_pages` around `host_.init`, `core/device/device.h`). Everything else keeps THP:
the Engram table copies (CPU gathers, `table_copy.h`: TLB walks), the relayout region (`engine_bringup.cpp`), the file-backed
path (tmpfs, `shmem_enabled` rules it). Unit test: `mem_test` `host_pool_page_size_switch_parses_and_stays_off_by_default`.

Pages in practice (`/proc/<pid>/smaps` of a running server, `AnonHugePages`): arm A, the two host-pool ranges (20.7 GB each)
show 20692992 kB of 20692992 kB huge; arm B the same two ranges show 0 kB. The four 15 GB Engram copies stay 100 % huge
in both arms. Server log of arm B: `host pool: 4 KiB pages (RADIANCE_HOST_POOL_PAGES=4k)`.

## Setup

Local lane (inference host, 2 x R9700 gfx1201, 210 W read back on both cards by `lane.sh` before every run), `u3` container,
`ngram-placement: ram`, host pool 39000 MiB, prefix cache 16384 MiB, EPYC 7402P. Tree = base `0b100d6d2` + this change, built
per CARDS.md §1.1 with `/opt/rocm-repeatable` (VOPD repeatable). **One binary, both arms**: A = environment unset, B =
`RADIANCE_HOST_POOL_PAGES=4k`. Runs: `scripts/backlog/lane.sh local 'scripts/backlog/run.sh <tag> <cfg> c1,c8 RADIANCE_AQL_STAMP=1 [RADIANCE_HOST_POOL_PAGES=4k]'`,
configs `coupled03.yaml` (DFly2) and `nospec.yaml`, three repeats of each arm interleaved (A,B / B,A / A,B per config, then
the other config), all `server-exit=0`. Parity: `greedy` phase (temperature 0, 3 prompts + a 5.7K arrival) once per arm and config.
Pass wall = `RADIANCE_AQL_STAMP` rank-0 stamps, passes with >= 1000 records inside the phase window, p50 of first-start to last-end.
The host also carried other work during the runs (other agents compile; no other server held the cards, the lane lock).

## A/B (mean [min-max] of 3 runs)

| config | phase | arm | agg tok/s | decode tok/s | tok/step | pass wall p50 us | TTFT ms |
|---|---|---|---|---|---|---|---|
| coupled03 | c1 | A (THP) | 126.0 [122.3-129.4] | 131.0 [127.2-134.5] | 3.3 [3.3-3.4] | 18686.6 [18451.6-18974.1] | 84.7 [82.0-87.0] |
| coupled03 | c1 | B (4 KiB) | 129.3 [128.5-130.5] | 134.4 [133.6-135.6] | 3.4 [3.4-3.4] | 18463.7 [18315.8-18610.6] | 81.3 [79.0-83.0] |
| coupled03 | c8 | A (THP) | 359.5 [312.8-394.7] | 415.5 [353.2-455.1] | 21.4 [20.7-22.8] | 48134.6 [47199.3-48723.5] | 310.0 [302.5-324.0] |
| coupled03 | c8 | B (4 KiB) | 385.3 [374.9-397.8] | 441.0 [433.8-454.1] | 22.0 [21.1-22.8] | 48588.3 [47960.8-49596.6] | 302.3 [296.5-309.5] |
| nospec | c1 | A (THP) | 71.4 [71.3-71.6] | 72.6 [72.5-72.7] | 1.0 [1.0-1.0] | 13504.9 [13457.3-13540.3] | 71.0 [70.0-72.0] |
| nospec | c1 | B (4 KiB) | 71.3 [71.0-71.6] | 72.5 [72.2-72.8] | 1.0 [1.0-1.0] | 13550.0 [13527.6-13566.5] | 71.3 [71.0-72.0] |
| nospec | c8 | A (THP) | 304.8 [303.5-306.7] | 315.8 [314.6-317.6] | 8.0 [8.0-8.0] | 24844.5 [24643.0-24970.4] | 275.7 [271.5-284.0] |
| nospec | c8 | B (4 KiB) | 305.3 [304.0-306.9] | 316.2 [315.0-317.9] | 8.0 [8.0-8.0] | 24788.5 [24742.6-24820.2] | 274.8 [272.5-278.5] |

Per pass wall: DFly2 c8 48135 (A) vs 48588 us (B), +0.9 %, inside the A spread (47199-48724); DFly2 c1 18687 vs 18464 (-1.2 %);
nospec c1 13505 vs 13550 (+0.3 %); nospec c8 24845 vs 24789 (-0.2 %). The DFly2 c8 tok/s means differ (359 vs 385), but A's spread
(313-395) is wider than the difference and it follows tok/step (20.7-22.8, the speculative acceptance of a sampled run), not the
pass: A rep 1 had 20.69 tok/step and 313 tok/s at a pass wall of 47199 us, the fastest pass of all six. Nospec has a
fixed 1.00 / 8.03 tok/step and shows the pass directly: 71.4 vs 71.3 (c1), 304.8 vs 305.3 (c8) tok/s.

Per run (tok/s agg, decode tok/s, tok/step, pass wall us, passes counted, tokens):
```
('coupled03', 'c1', 'A') 3 | tps 126.0 [122.3-129.4] | dec 131.0 [127.2-134.5] | tok/step 3.3 [3.3-3.4] | passwall us 18686.6 [18451.6-18974.1] | ttft 84.7 [82.0-87.0]
   rep 1 126.3 131.4 3.32 18634 76 256
   rep 2 129.4 134.5 3.37 18452 75 256
   rep 3 122.3 127.2 3.32 18974 76 256
('coupled03', 'c1', 'B') 3 | tps 129.3 [128.5-130.5] | dec 134.4 [133.6-135.6] | tok/step 3.4 [3.4-3.4] | passwall us 18463.7 [18315.8-18610.6] | ttft 81.3 [79.0-83.0]
   rep 1 130.5 135.6 3.37 18316 75 256
   rep 2 128.5 133.6 3.37 18611 75 256
   rep 3 129.0 134.1 3.37 18465 75 256
('coupled03', 'c8', 'A') 3 | tps 359.5 [312.8-394.7] | dec 415.5 [353.2-455.1] | tok/step 21.4 [20.7-22.8] | passwall us 48134.6 [47199.3-48723.5] | ttft 310.0 [302.5-324.0]
   rep 1 312.8 353.2 20.69 47199 55 2048
   rep 2 394.7 455.1 22.76 48724 42 2048
   rep 3 371.0 438.1 20.9 48481 48 2048
('coupled03', 'c8', 'B') 3 | tps 385.3 [374.9-397.8] | dec 441.0 [433.8-454.1] | tok/step 22.0 [21.1-22.8] | passwall us 48588.3 [47960.8-49596.6] | ttft 302.3 [296.5-309.5]
   rep 1 397.8 454.1 22.76 48207 48 2048
   rep 2 383.1 433.8 22.02 47961 48 2048
   rep 3 374.9 435.2 21.11 49597 44 2048
('nospec', 'c1', 'A') 3 | tps 71.4 [71.3-71.6] | dec 72.6 [72.5-72.7] | tok/step 1.0 [1.0-1.0] | passwall us 13504.9 [13457.3-13540.3] | ttft 71.0 [70.0-72.0]
   rep 1 71.3 72.5 1.0 13540 255 256
   rep 2 71.5 72.6 1.0 13457 255 256
   rep 3 71.6 72.7 1.0 13517 255 256
('nospec', 'c1', 'B') 3 | tps 71.3 [71.0-71.6] | dec 72.5 [72.2-72.8] | tok/step 1.0 [1.0-1.0] | passwall us 13550.0 [13527.6-13566.5] | ttft 71.3 [71.0-72.0]
   rep 1 71.6 72.8 1.0 13528 255 256
   rep 2 71.0 72.2 1.0 13556 255 256
   rep 3 71.4 72.5 1.0 13566 255 256
('nospec', 'c8', 'A') 3 | tps 304.8 [303.5-306.7] | dec 315.8 [314.6-317.6] | tok/step 8.0 [8.0-8.0] | passwall us 24844.5 [24643.0-24970.4] | ttft 275.7 [271.5-284.0]
   rep 1 303.5 314.6 8.03 24920 253 2048
   rep 2 306.7 317.6 8.03 24643 253 2048
   rep 3 304.3 315.0 8.03 24970 253 2048
('nospec', 'c8', 'B') 3 | tps 305.3 [304.0-306.9] | dec 316.2 [315.0-317.9] | tok/step 8.0 [8.0-8.0] | passwall us 24788.5 [24742.6-24820.2] | ttft 274.8 [272.5-278.5]
   rep 1 306.9 317.9 8.03 24743 253 2048
   rep 2 304.0 315.0 8.03 24803 253 2048
   rep 3 304.9 315.7 8.03 24820 253 2048
```

## Output

`greedy` phase, sha256 (first 12 hex) of each text, A and B identical in every case:
```
coupled03 A/B  5b7a3975defa:256 f55fe02071d8:256 0075e702dfdf:103 8f60e42028d1:64
nospec    A/B  d6da4fb33eed:256 0699709c4be2:256 e7a23c5d9a2f:102 8f60e42028d1:64
```
Runs `r4kg-coupled03-{A,B}`, `r4kg-nospec-{A,B}`.

## Startup

Wall from `run.sh` start (config render, process start, bring-up, pin, prefault) to `/health`, seconds:
```
coupled03  A 76 74 76 | B 86 80 79
nospec     A 76 75 170* | B 79 81 170*
```
(* a run that waited ~95 s on something else on the host; both arms hit one.) Excluding those, A ~75.5 s, B ~80.8 s:
**+5 s (+7 %)** for 4 KiB pages, from the 39000 MiB pool faulting 4 KiB at a time (10 M faults instead of 19 k) and the lock walking
that many more pages. Not separated further.

## Reading

The SDMA gain of `pcie-h2d.md` (28 vs ~22 GB/s per card for a long copy) is not in the pass: at DFly2 c8 ~110 units miss per
step (~1.5 MB each, about 80 MB a card in a 48 ms pass), so the H2D path is a small share of the pass, (not verified separately: which share of the pass is link-bound was not measured here). Merge bar (CARDS.md §2) not met on DFly2 c8
(no measurable gain); c1 and nospec not worse. Default stays THP, the switch stays for hosts with a different DIMM layout
(the effect is about channel interleave, `pcie-h2d.md`) and for a later A/B once M-20 moves more bytes per pass.
