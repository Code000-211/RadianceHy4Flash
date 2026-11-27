# Issue #147: §2 re-measured at HEAD (1476e50b) after the tier-1 cards, on the rig's one lane

The card: repeat `SETUP-BASE` per lane at HEAD, tabulate against the recorded baselines
(`baseline/<lane>.md`, all at `dd4c2255`) and the §6 ceilings, and say what is left. This is what
ran, where, and what it read. Numbers here were produced on this machine in this session; nothing
is copied from a doc except the baseline and ceiling references, which are marked as such.

## 0. Lane record

| | |
|---|---|
| lane | `local-u4` — this host's two cards, TP2. Not one of CARDS §1's three lanes; see §5 for what of them could not be run and why |
| machine | `ubuntu-MS-7E56`, AMD Ryzen 9 9950X (16C/32T, 121 GiB `MemTotal`, 127,001,880 kB), kernel `7.0.0-44-generic` |
| GPUs | 2 x AMD Radeon AI PRO R9700 (`gfx1201`, 34,208,743,424 B VRAM, `/sys/.../mem_info_vram_total`), TP2. `cap 300 W` read back before every run (`power1_cap` 300000000, card1/card2); the §1 lane's 210 W is not settable here (no root, `sudo` wants a password), so **this lane is not the §1 power cap** |
| ROCm / HIP | `libamdhip64-7` 7.1.0-0ubuntu2, `hipcc` 7.1.1+dfsg, device code built with `/usr/lib/llvm-21/bin/clang++` (Ubuntu clang 21.1.8); `RAD_HIP_VOPD_REPEATABLE=OFF` (that flag is for the image's patched LLVM 24; `/opt/rocm-repeatable` does not exist on this host) |
| container | keptn `hy4-flash-iqr-v4-u4.rad`, `/home/ubuntu/models/radiance/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u4.rad`, 150,444,707,840 B, mtime 2026-11-23 10:26:56 -0500, first 64 MiB sha256 `b7b762ae0795613b0c49fad7215877fcce5749f1e74223e0aed926337fc23b53`. **u4, not the u3 of the recorded baselines** — this host has no u3 container (`/srv/models` does not exist; the only Hy4-flash container here is u4) |
| Engram | `ngram-placement: disk` (the container's own bytes read through the page cache). `ram` (the §1 `local` lane's placement) does not fit: 121 GiB of RAM against a 150 GB container plus the 39,000 MiB host pool and the 16,384 MiB host tier |
| configs | `scripts/backlog/nospec.yaml` and `coupled03.yaml` **as committed at each arm's commit**, plus `ngram-placement: disk` and the model path (the only two keys changed). Both configs are quoted in §2 |
| server | port 8041, engine boot under `flock -w 7200 /tmp/hy4-lane.lock`; one server at a time, both cards free (the house engine `hy4-u4` on :8180 stopped for the session and restored after, §6) |
| host load | `load 1.2 .. 18.5` at boot (another worker's `-j30` build ran part of the session); every run's log carries `load` and `MemAvailable` at its start |
| harness | the backlog's own `scripts/backlog/run.sh` steps (engine, `/health` wait, `probe.py`, stop), driven by a local `run-lane.sh` because `lane.sh` here has no `gpu2` card scheduler and hardcodes `local`'s `ngram-placement: ram`; `probe.py`, the phase names and the prompts are unmodified (§2.1) |
| build | both arms: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_C_COMPILER=/usr/bin/cc -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201`, then `cmake --build build -j 32 --target radiance libr4d libavx libref spechub libquant hyv4flash rad-kbench r4d_selftest` (this is the house `CMakeCache.txt`'s recipe: gcc 15 host compiler, LLVM 21 HIP compiler) |

Commits: **before** `dd4c2255b5e1cea93339262346f8c215a39a6dd5` — the commit every
`notes/backlog/baseline/*.md` was taken at, rebuilt here as a second worktree, so "before" is
that commit's code *and its own configs*. **after** `1476e50b6c003c1d7a897bc0fc7d04421fd92ac1` —
`origin/claude/peaceful-wright-hhqnhi` at measurement time, i.e. every tier-1 card of §6 merged
(S-01 #5, P-16 #7, X-01 #8, K-08 #28, K-11 #30, K-12 #32, K-14 #34, K-16 #36, K-44 #54, K-45 #55,
K-47 #208, D-01 #212, P-01 #225; M-03a/b and M-05 closed without code). `git log --oneline
origin/claude/peaceful-wright-hhqnhi | grep -E 'S-01|P-16|…'` shows the merges. The commit that
adds this file sits on top of `1476e50b` and changes nothing else.

Builds differ only in the tree: `build/bin/radiance` sha256 (16) `c85236465602033d` (before) against
`d77ed019516216d6` (after); `libr4d.so` `a89b7c9d62c79b67` / `cef7d968919cfc0d`; `libavx.so`
`fa1f807b24cec9c0` / `0e51f07741a21ed7`; `spechub.so` `2ab3c9cc0736bde4` / `6862187f9b114ee5`;
`hyv4flash.so` `45baec72aaebc106` / `6085e2e96e9803e7`.

## 1. The table

Medians of three interleaved runs an arm (`b a b a b a`), `probe.py` phases
`c1,c1b,c8,pf6,pf20,pf2,tail400,tail742,tail2560,stall1,stall4` for the no-spec arm and
`c1,c8,think` for the coupled arm. `baseline/local.md` is the recorded baseline (u3, 210 W, the
Moebius host); it is quoted, not re-run. The §6 ceiling is `optimisation-backlog.md` §6 end, read
as a percentage of **this rig's before** row.

| row | `baseline/local.md` (u3, 210 W) | before `dd4c2255` here | **after `1476e50b` here** | after vs before (same rig) | §6 ceiling | left to the ceiling |
|---|---|---|---|---|---|---|
| `c1` no-spec | 13.710 ms / 71.6 tok/s | 14.523 ms / 68.9 tok/s | **14.236 ms / 70.2 tok/s** | **-2.0 % ms, +2.0 % tok/s** | -2.3..-2.9 ms (17-21 %) ⇒ 11.3-11.9 ms, 84-88 tok/s | 2.3-2.9 ms (16-20 %); no-spec c1 kernels (K-05, K-07, K-09/K-10, K-01..K-06, K-13) are all still open |
| `c1b` no-spec | 13.403 ms / 73.2 | 14.260 ms / 70.1 *(bimodal, §1.1)* | **16.687 ms / 59.9** *(bimodal)* | not attributable (§1.1) | §2's own band was 13.5-13.9 ms / 72-74 | — |
| `c8` no-spec | 308.8 agg / 320.8 summed (25.4 ms step) | 221.3 agg / 232.8 summed (33.95 ms step) | **227.9 agg / 240.7 summed (32.95 ms step)** | **+3.0 % agg, +3.4 % summed, -2.9 % step** | -3.5 ms of a step (14 %) ⇒ 29.5 ms, ~277 agg | 3.5 ms (11 %); M-01, M-09.., K-07, K-09, D-10 open |
| `pf2` 2,385 tok | 716 ms | 1,086 ms @2,385 | **1,017 ms @2,357** | -6.4 % raw, -5.2 % a token | prefill kernels; M-03 closed (§5) | see `pf6` |
| `pf6` 6,681 tok | 1,968 ms | 2,279 ms @6,681 | **2,253 ms @7,886** | -1.1 % raw, **-16 % a token** (0.341 → 0.286 ms) | -120..-130 ms of a 2048-row chunk (+27-30 %) ⇒ ~0.216 ms/token | ~0.065 ms/token (23 %); sink K-40/K-41/K-42, K-46, K-43/P-10, K-50..K-58, P-03 |
| `pf20` 24,856 tok | 7,173 ms | 7,477 ms @24,856 | **6,471 ms @23,044** | -13.5 % raw, -6.8 % a token | as `pf6` | as `pf6` |
| `tail400` 159 tok | 192 ms | 373 ms | **344 ms** | -7.8 % | small-step path, M-03 (§4) | none available: M-03 closed without code |
| `tail742` 256 tok | 151 ms | 344 ms | **305 ms** | -11.3 % | as above | as above |
| `tail2560` 723 tok | 269 ms | 523 ms | **488 ms** | -6.7 % | as above | as above |
| `stall1` 5,698 tok arrival | 3.04 s | 4.366 s @5,698 | **2.538 s @7,433** | -41.9 % raw, **-55 % a token** (0.766 → 0.341 ms) | 0.5-0.8 s a chunk until S-02 | the arrival still pays ~0.63 s a chunk here; only S-02 (open, tier 3) changes the order |
| `stall4` 5,881 tok arrival | 2.87 s | 4.162 s @5,881 | **2.650 s @6,934** | -36.3 % raw, -46 % a token (0.708 → 0.382 ms) | as above | as above |
| DFly2 `c1` | 22.742 ms, 2.813 tok/step, 123.2 summed | 29.273 ms, 2.909, 98.99 | **29.177 ms, 2.586, 88.28** | -0.3 % ms, **-10.8 % tok/s** (2.909 → 2.586 tokens a step) | +18-25 % tok/s ⇒ 117-124 | **-22 % from HEAD's own code best (§3.2): `draft-depth-load-from: 1` costs what the code gains** |
| DFly2 `c8` | 425.3 summed | 276.6 summed (19.3 tok/step, 69.8 ms step) | **266.6 summed (21.1 tok/step, 79.2 ms step)** | -3.6 % summed, +9 % tokens a step | 25-30 % of a step (the backlog's -12..-15 ms of 49 ms) ⇒ 55-59 ms here at 21.1 tokens a step, ~355-380 summed | ~30 %; D-02..D-04, D-06/D-07, D-10, K-18 |
| DFly2 `think` | 21.700 ms, 3.122 tok/step, 123.9 | 24.212 ms, 2.400, 98.86 | **23.987 ms, 2.560, 106.45** | -0.9 % ms, **+7.7 % tok/s** | as DFly2 `c1` | — |

Read the "recorded baseline" column as a reference only: it is another host, the u3 container and
210 W. §4 gives the measured reason the cross-rig gap exists (4.6x the expert misses a step at `c8`
with the same routed work), so a row that is 25 % behind `baseline/local.md` is not a HEAD
regression.

### 1.1 `c1b` is bimodal in both arms

`c1b` is the 84-token prompt decoded a second time. On this rig it has two modes, ~3.75 s of wall
(14.0-14.3 ms a step) and ~4.45 s (16.7-17.2 ms), and both arms show both: before
14.26 / 14.01 / 16.90 against the three runs, after 16.69 / 14.10 / 16.86 (and in the earlier
before/after pair that used the same config on both arms, 14.32 / 14.42 / 17.20 and
14.04 / 16.91 / 13.98). The probe's `miss_units_per_step` and `hit` are identical in both modes
(21.2 units, 0.9697), so it is not expert residency; the phase's whole wall time is what moves.
Three runs an arm cannot attribute a 2-of-3 against a 1-of-3 split, so `c1b` is reported and not
read; `c1` is the decode row.

## 2. The arms, field by field

### 2.1 The setup, as it ran

```sh
# one server, one probe, then stopped; the card reservation is the flock, taken by the batch
flock -w 7200 /tmp/hy4-lane.lock -c \
  'REPO=<tree> /tmp/r147/run-lane.sh <tag> <cfg> <phases>'
```

`run-lane.sh` (the only local file; `scripts/backlog/*` unmodified): reads `server.port` from the
config, boots `$REPO/build/bin/radiance --config <cfg>` with `RADIANCE_HOME=$REPO/build/radiance_home`,
waits on `/health` (11-16 s), runs `python3 scripts/backlog/probe.py 8041 <tag>.json <phases>`,
stops the server, and writes `server.log`, `probe.out`, `<tag>.json` per run under
`/tmp/r147/runs/<tag>/`. The phase list is the baseline's `SETUP-BASE` list plus the two small
prompts of §2.5 (`tail742`, `tail2560`); the prompts are read from the tree the probe sits in, so
the two arms sent **different text** on the corpus-derived phases — the token counts are in every
row above and every TTFT is given a per-token companion for that reason.

Configs (the only two keys changed from the commit are `model` and `ngram-placement`):

```
before (dd4c2255 scripts/backlog/nospec.yaml + disk/u4)     after (1476e50b + disk/u4)
server: port 8041, kernels libr4d,libavx                    server: port 8041, kernels libr4d,libavx
  gpu-headroom-mib: 128                                       gpu-headroom-mib: 128
  tp: 2, expert_tiered, host-pool-mib: 39000                  tp: 2, expert_tiered, host-pool-mib: 39000
  ngram-placement: disk                                       ngram-placement: disk
  max-model-len: 65536, max-num-batched-tokens: 2048          max-model-len: 65536, max-num-batched-tokens: 2048
  max-num-seqs: 8, mm-max-patches: 0                          checkpoint-interval: 1984
  prefix-resume: nearest                                      max-num-seqs: 8, mm-max-patches: 0
  busy-prefill-chunk: auto                                    prefix-resume: nearest, prefix-cache-mib: 16384
                                                              busy-prefill-chunk: auto
```

`coupled03.yaml` adds on both sides `num-speculative-tokens: 5`, `draft-window: fixed`,
`draft-coupling: on`, `draft-verify: traversal`, `draft-depth: adaptive`, `draft-depth-stop: 0.3`,
`draft-depth-load: on`, `draft-suffix: on` and `kernels libr4d,libavx,spechub`; the after side also
carries `draft-depth-load-from: 1` (P-16). §3.2 measures that key on its own.

### 2.2 No-spec, per run

| run | `c1` ms / tok/s | `c1b` ms / tok/s | `c8` agg / summed / step ms | `pf2` ms @tok | `pf6` ms @tok | `pf20` ms @tok | `tail400` | `tail742` | `tail2560` | `stall1` s @tok | `stall4` s @tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| before r1 | 14.610 / 68.45 | 14.260 / 70.12 | 244.63 / 259.10 / 30.63 | 1,086 @2,385 | 2,427 @6,680 | 7,480 @24,856 | 0.373 | 0.345 | 0.523 | 4.366 @5,698 | 4.162 @5,881 |
| before r2 | 14.259 / 70.13 | 14.014 / 71.36 | 220.91 / 232.76 / 34.12 | 1,086 @2,385 | 2,279 @6,681 | 7.400 @24,856 | 0.373 | 0.342 | 0.511 | 5.068 @5,698 | 4.467 @5,881 |
| before r3 | 14.523 / 68.86 | 16.897 / 59.18 | 221.26 / 232.81 / 33.95 | 1,038 @2,385 | 2.273 @6,681 | 7.477 @24,856 | 0.375 | 0.344 | 0.525 | 4.336 @5,698 | 4.116 @5,881 |
| **before med** | **14.523 / 68.86** | **14.260** | **221.26 / 232.81 / 33.95** | **1,086** | **2.279** | **7.477** | **0.373** | **0.344** | **0.523** | **4.366** | **4.162** |
| after r1 | 14.363 / 69.62 | 16.687 / 59.93 | 227.91 / 240.71 / 32.95 | 992 @2,357 | 2.226 @7,886 | 6.523 @23,044 | 0.353 | 0.315 | 0.493 | 2.538 @7,433 | 2.556 @6,934 |
| after r2 | 14.236 / 70.24 | 14.099 / 70.93 | 229.49 / 241.26 / 32.85 | 1,174 @2,357 | 2.253 @7,886 | 6.411 @23,045 | 0.344 | 0.305 | 0.447 | 2.583 @7,433 | 2.957 @6,934 |
| after r3 | 14.214 / 70.35 | 16.858 / 59.32 | 227.39 / 238.95 / 33.16 | 1,017 @2,357 | 2.256 @7,886 | 6.471 @23,044 | 0.342 | 0.275 | 0.488 | 2.522 @7,433 | 2.650 @6,934 |
| **after med** | **14.236 / 70.24** | **16.687** | **227.91 / 240.71 / 32.95** | **1,017** | **2.253** | **6.471** | **0.344** | **0.305** | **0.488** | **2.538** | **2.650** |

(`tok/s` is the probe's `sum_req_dec_tps`; `c8`'s `step ms` is
`(all_decoding_s[1] - all_decoding_s[0]) / decode_steps`, the only place the probe does not print
`ms_step`, which is null for a multi-client phase.)

### 2.3 Coupled (DFly2), per run

| run | `c1` ms / tok a step / tok/s | `c8` summed / tok a step | `think` ms / tok a step / tok/s |
|---|---|---|---|
| before r1 | 27.092 / 2.909 / 106.96 | 276.61 / 18.789 | 25.313 / 2.400 / 94.57 |
| before r2 | 29.286 / 2.909 / 98.95 | 277.35 / 19.321 | 23.668 / 2.400 / 101.14 |
| before r3 | 29.273 / 2.909 / 98.99 | 248.29 / 22.505 | 24.212 / 2.400 / 98.86 |
| **before med** | **29.273 / 2.909 / 98.99** | **276.61 / 19.321** | **24.212 / 2.400 / 98.86** |
| after r1 | 27.398 / 2.586 / 94.01 | 262.23 / 20.078 | 24.449 / 2.803 / 114.34 |
| after r2 | 29.866 / 2.612 / 87.12 | 288.44 / 21.113 | 23.987 / 2.560 / 106.45 |
| after r3 | 29.177 / 2.586 / 88.28 | 266.59 / 21.558 | 23.868 / 2.385 / 99.67 |
| **after med** | **29.177 / 2.586 / 88.28** | **266.59 / 21.113** | **23.987 / 2.560 / 106.45** |

## 3. What the tier-1 cards did on this rig

### 3.1 No-spec: +2 % at `c1`, +3 % at `c8`, and the arrival stall

`c1` 14.523 → 14.236 ms (two runs of each arm overlap: before 14.259-14.610, after
14.214-14.363, so a 0.29 ms median gap is ~2x the within-arm spread). `c8` 33.95 → 32.95 ms a step,
summed 232.8 → 240.7 tok/s. That is less than the sum of the tier-1 rows' own claims
(K-11 1-1.5 %, K-12 1.5-2.5 %, K-14 2-3 %, P-01 1-2 % at `c1`), and the reason is in §4: this
container/rig combination spends its time where those items do not reach.

The arrival rows are the big move: `stall1` 4.366 → 2.538 s on a *larger* arrival (5,698 → 7,433
tokens), 0.766 → 0.341 ms a token; `stall4` 4.162 → 2.650 s (5,881 → 6,934 tokens). Both arms'
arrivals enter at `arrival_ttft` with 1 / 4 decoders running. That is S-01 (`checkpoint-interval:
1984`: no remainder step sharing a step with decode rows) plus X-01 (the host tier, so the arrival's
loan is there); S-01's own note claims -27 % at one decoder and -14 % at four on the Moebius lane,
and this lane moves further because its before arm had neither key.

Prefill: at `pf20` the after arm is 13.5 % faster in wall time on 7 % fewer tokens (-6.8 % a
token); `pf6` is level in wall time on 18 % more tokens (-16 % a token). The kernel side of the
§6 prefill ceiling (K-44, K-45, K-47 merged) is inside that, but the profile (`prof.yaml`,
`profdiff.py`) was not re-run, so no chunk-level attribution is claimed (§5).

### 3.2 P-16's `draft-depth-load-from: 1` costs 22 % of DFly2 `c1` on this rig

`draft-depth-load-from: 1` is a config-only tier-1 item, so it is the one key that separates the
after arm from the before arm at DFly2 `c1` (one sequence). It was measured on its own: HEAD's code
with the same `coupled03.yaml` minus that one line, three interleaved-consecutive runs:

| config | `c1` ms | tokens a step | `c1` tok/s | `c8` summed |
|---|---|---|---|---|
| before, pre-P-16 (no key) | 29.273 | 2.909 | 98.99 | 276.61 |
| after, pre-P-16 (key removed) | 28.362 | **3.241** | **113.81** | 274.61 |
| after, served (key present) | 29.177 | 2.586 | 88.28 | 266.59 |

So HEAD's code, on this rig's `c1`, drafts deeper and reaches 3.241 tokens a step and 113.8 tok/s
(+15 % over the before arm) — inside the §6 DFly2 `c1` ceiling (117-124 tok/s from the before row) —
and the served config then declines to draft at one sequence and gives back 22 %
(113.8 → 88.3). It is not noise: every run of each config keeps its own tokens-a-step to three
decimals (3.241 / 3.241 / 3.241 against 2.586 / 2.612 / 2.586), and `think` (reasoning text) moves
the other way, +7.7 % tok/s with the key on, which is the text dependence P-16's own note
describes. P-16 measured +8 % at `c1` on the Moebius lane; on this rig's online calibration the
rule prices a draft round higher and stops at 2.6 tokens a step. **That is a losing trade here and
the largest single gap in the table.**

## 4. Why this lane is not `baseline/local.md`, measured

Same routed work, 4.6x the misses: at `c8` the probe reads `routed_per_step` 4,068-4,128 (before
and after, both arms) against the recorded baseline's 4,105 — but `miss_units_per_step` 144-152
against its 33. The extra ~118 unit reads a step are the 33.95 ms against 25.4 ms (`c8`), and the
`c1` misses are 31.8 against 11. The container is 150.4 GB against the u3's 138.1 GB, on a host
with 121 GiB of RAM, so the same 39,000 MiB host pool holds a smaller share of the experts: this is
the container and the rig, not HEAD (both arms show it). It is also why no row here is compared
with the `baseline/*.md` files as if the two were the same lane: the recorded numbers belong to a host whose
pool fits the u3.

The small-prefill rows show the rig's second divergence: `tail742` 344 ms here against 151 ms in
`baseline/local.md`, `tail2560` 523 against 269. Both arms pay it, so it is not a HEAD effect; it is
the unstaged small-step path (M-03's subject) reading experts in place on `disk` Engram.

## 5. What is left

**Lanes not run, and why.**

- `iks-A`, `iks-B`: no route from this host — `ssh iks` fails to resolve (`Could not resolve
  hostname iks`), so neither the tree nor a probe reaches them. Their recorded baselines
  (`baseline/iks-A.md`, `baseline/iks-B.md`, both `dd4c2255`, u3, 210 W) stand as the last values:
  `c1` 72.6 / 67.0 tok/s, `c8` 313.0 / 274.4 agg, `pf6` 1.91 / 2.35 s, `stall1` 2.98 / 4.62 s.
- CARDS §1's `local` lane **verbatim**: not reproducible on this host. No u3 container
  (`/srv/models` absent), `ngram-placement: ram` does not fit in 128 GiB beside the 150 GB u4
  container, `power1_cap` is 300 W and not settable without root, and there is no `gpu2` card
  scheduler. What ran is `local-u4` (§0), which is the lane every number above belongs to; it is
  not comparable with `baseline/local.md` (§4), only with itself.

**Phases not re-measured** (each is in the §2 baseline and would be a later card's row):
`RADIANCE_AQL_STAMP` category and per-layer tables (§2.1/2.2/2.3 — `agg.py`, `layer.py`), the
`prof.yaml` prefill op table of §2.4 (`profdiff.py`, so no chunk-level prefill attribution), the
`idle` / `long8` rows of X-01's tier check, `longctx`/`lctxN` (T-03's long-context decode), `stress`
(#251), and `think` beyond the coupled arm's one row.

**Gaps to the §6 ceiling, by row** (all lossless, tier 2/3 in §6):

- `c1` no-spec, 2.3-2.9 ms left: K-13 junction, K-05 static weight loads, K-09/K-10 router select,
  K-01..K-04/K-06 dual-weight launches, K-07 routed head. The tier-1 half of that list (K-11, K-12,
  K-14, P-01) is in and is what the +2 % is.
- `c8` no-spec, 3.5 ms left: M-01 (mirror pool), M-09..M-12 (VRAM receipt), K-07, K-09, D-10 —
  and, on this rig, the miss count of §4, which no kernel item in the backlog addresses.
- DFly2 `c1`, 22 % left **of the served config**: this is P-16's key on this rig (§3.2). Either the
  rule's online price is wrong for a 32-thread host with disk Engram, or the key wants to be
  per-host. Nothing in the backlog covers it; it wants a card.
- DFly2 `c8`, ~20 % left: D-02..D-04, D-06/D-07, D-10, K-18.
- Prefill, ~23 % a token left: the sink-attention wave (K-40/K-41/K-42), K-46, K-43/P-10,
  K-50..K-58, P-03. **M-03 is closed without code** (`closed/M-03a.md`, `closed/M-03b.md`), so the
  small-step half of the §6 prefill ceiling (~170 ms a prompt) has no owner: `tail742`/`tail2560`
  are the rows that would show it.
- Arrival stall: still ~0.63 s a chunk here; only S-02 (tier 3, L, design note first) changes its
  order of magnitude, and `closed/S-02-exec.md` records why the cheap variant was not taken.

**One rig caveat that would change these rows**: this host was shared during the session (load
1.2-18.5 at boot, another worker's `-j30` build), and `c1b` is bimodal in both arms (§1.1). The
decode rows (`c1`, `c8`) were stable to 1-3 % across runs; the coupled arm's `c8` summed spans
248-288 within one arm.

## 6. The house engine, restored

`hy4-u4` (`:8180`, `/home/ubuntu/r9700-serving/hy4flash/run-hy4-u4-disk.sh`) was stopped before the
first boot and restarted after the last one; `curl :8180/v1/models` answers both ids,
`qwen3.8-27b` and `qwen3.8-27b-local`, and the engine is `active (running)`.

## 7. Raw lines

The probe's own lines: `c1` and `c8` for all twelve runs (`b0ns-*` before, `ans-*` after, `b0cp-*` /
`acp-*` the coupled pair, `hnl-*` HEAD's code with `draft-depth-load-from` removed, §3.2), `stall1`
for the six no-spec runs. `stats` and the stall `stage` sampler blob are elided (`"..."`); every
number in the tables above is copied from these lines. `stall4`, the corpus-phase TTFTs
(`pf2/pf6/pf20/tail*`) and the `c1b` rows are the same shape and their per-run values are in §2.2.

#### b0ns-r1
{"name": "c1", "clients": 1, "wall": 3.892, "gen": 256, "prompt": [73], "ttft": [0.166], "agg_tps": 65.782, "sum_req_dec_tps": 68.448, "decode_steps": 255.0, "ms_step": 14.61, "tok_per_step": 1.004, "mono_ns": [165960448329323, 165964339970262], "all_decoding_s": [0.1661205291748047, 3.891566276550293], "hit": 0.954, "routed_per_step": 697.392, "miss_units_per_step": 31.835}
{"name": "c8", "clients": 8, "wall": 8.372, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.091, 0.552, 0.552, 0.552, 0.551, 0.551, 0.55, 0.55], "agg_tps": 244.626, "sum_req_dec_tps": 259.096, "decode_steps": 255.0, "ms_step": null, "tok_per_step": 8.031, "mono_ns": [165972161246587, 165980533201873], "all_decoding_s": [0.5523371696472168, 8.364190340042114], "hit": 0.963, "routed_per_step": 4083.678, "miss_units_per_step": 152.09}
{"name": "stall1", "bg": 1, "arrival_prompt": 5698, "arrival_ttft": 4.366, "arrive_at": 1.5, "first_at": 5.87, "typical_gap_ms": 14.4, "worst_gaps_ms": [[49.7, 50.4, 59.1, 62.7, 76.9, 541.0, 563.2, 603.5]], "clients": 1, "mono_ns": [0, 0]}

#### b0ns-r2
{"name": "c1", "clients": 1, "wall": 3.803, "gen": 256, "prompt": [73], "ttft": [0.167], "agg_tps": 67.307, "sum_req_dec_tps": 70.131, "decode_steps": 255.0, "ms_step": 14.259, "tok_per_step": 1.004, "mono_ns": [166133430818432, 166137234291144], "all_decoding_s": [0.16736078262329102, 3.803408622741699], "hit": 0.954, "routed_per_step": 697.392, "miss_units_per_step": 31.765}
{"name": "c8", "clients": 8, "wall": 9.271, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.09, 0.561, 0.56, 0.56, 0.56, 0.56, 0.56, 0.559], "agg_tps": 220.913, "sum_req_dec_tps": 232.757, "decode_steps": 255.0, "ms_step": null, "tok_per_step": 8.031, "mono_ns": [166144963474947, 166154234102949], "all_decoding_s": [0.5611722469329834, 9.262404918670654], "hit": 0.963, "routed_per_step": 4079.635, "miss_units_per_step": 150.796}
{"name": "stall1", "bg": 1, "arrival_prompt": 5698, "arrival_ttft": 5.068, "arrive_at": 1.5, "first_at": 6.57, "typical_gap_ms": 14.1, "worst_gaps_ms": [[47.6, 51.5, 58.9, 62.9, 74.0, 535.9, 573.4, 598.3]], "clients": 1, "mono_ns": [0, 0]}

#### b0ns-r3
{"name": "c1", "clients": 1, "wall": 3.886, "gen": 256, "prompt": [73], "ttft": [0.182], "agg_tps": 65.875, "sum_req_dec_tps": 68.856, "decode_steps": 255.0, "ms_step": 14.523, "tok_per_step": 1.004, "mono_ns": [166309614376834, 166313500526778], "all_decoding_s": [0.18268990516662598, 3.886077880859375], "hit": 0.954, "routed_per_step": 697.392, "miss_units_per_step": 31.765}
{"name": "c8", "clients": 8, "wall": 9.256, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.18, 0.589, 0.18, 0.589, 0.589, 0.589, 0.589, 0.589], "agg_tps": 221.263, "sum_req_dec_tps": 232.805, "decode_steps": 255.0, "ms_step": null, "tok_per_step": 8.031, "mono_ns": [166321995670306, 166331251621185], "all_decoding_s": [0.5901944637298584, 9.247245073318481], "hit": 0.963, "routed_per_step": 4078.471, "miss_units_per_step": 151.624}
{"name": "stall1", "bg": 1, "arrival_prompt": 5698, "arrival_ttft": 4.336, "arrive_at": 1.5, "first_at": 5.84, "typical_gap_ms": 14.1, "worst_gaps_ms": [[52.0, 53.3, 55.7, 71.0, 76.1, 540.4, 574.8, 604.3]], "clients": 1, "mono_ns": [0, 0]}

#### ans-r1
{"name": "c1", "clients": 1, "wall": 3.842, "gen": 256, "prompt": [73], "ttft": [0.18], "agg_tps": 66.624, "sum_req_dec_tps": 69.624, "decode_steps": 255.0, "ms_step": 14.363, "tok_per_step": 1.004, "mono_ns": [166048066418433, 166051908878459], "all_decoding_s": [0.17984747886657715, 3.842400074005127], "stage_layers": 0, "hit": 0.956, "routed_per_step": 696.435, "miss_units_per_step": 30.953, "steps_sliced": 0, "substeps": 0}
{"name": "c8", "clients": 8, "wall": 8.986, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.556, 0.148, 0.556, 0.556, 0.556, 0.556, 0.555, 0.555], "agg_tps": 227.913, "sum_req_dec_tps": 240.708, "decode_steps": 255.0, "ms_step": null, "tok_per_step": 8.031, "mono_ns": [166060343674366, 166069329552916], "all_decoding_s": [0.5566215515136719, 8.958050727844238], "stage_layers": 0, "hit": 0.963, "routed_per_step": 4128.169, "miss_units_per_step": 152.416, "steps_sliced": 0, "substeps": 0}
{"name": "stall1", "bg": 1, "arrival_prompt": 7433, "arrival_ttft": 2.538, "arrive_at": 1.5, "first_at": 4.04, "typical_gap_ms": 13.4, "worst_gaps_ms": [[37.9, 48.0, 48.8, 181.4, 367.6, 521.4, 575.1, 595.0]], "clients": 1, "mono_ns": [0, 0], "stage": "...", "steps_sliced": 0, "substeps": 0}

#### ans-r2
{"name": "c1", "clients": 1, "wall": 3.811, "gen": 256, "prompt": [73], "ttft": [0.18], "agg_tps": 67.18, "sum_req_dec_tps": 70.243, "decode_steps": 255.0, "ms_step": 14.236, "tok_per_step": 1.004, "mono_ns": [166222798638206, 166226609306454], "all_decoding_s": [0.18032503128051758, 3.8106045722961426], "stage_layers": 0, "hit": 0.956, "routed_per_step": 696.435, "miss_units_per_step": 30.953, "steps_sliced": 0, "substeps": 0}
{"name": "c8", "clients": 8, "wall": 8.924, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.087, 0.516, 0.516, 0.515, 0.516, 0.515, 0.515, 0.515], "agg_tps": 229.487, "sum_req_dec_tps": 241.263, "decode_steps": 255.0, "ms_step": null, "tok_per_step": 8.031, "mono_ns": [166235039726242, 166243963985509], "all_decoding_s": [0.5168037414550781, 8.894295930862427], "stage_layers": 0, "hit": 0.965, "routed_per_step": 4074.769, "miss_units_per_step": 144.62, "steps_sliced": 0, "substeps": 0}
{"name": "stall1", "bg": 1, "arrival_prompt": 7433, "arrival_ttft": 2.583, "arrive_at": 1.5, "first_at": 4.08, "typical_gap_ms": 13.4, "worst_gaps_ms": [[45.9, 50.5, 52.2, 134.3, 359.4, 524.3, 573.0, 574.1]], "clients": 1, "mono_ns": [0, 0], "stage": "...", "steps_sliced": 0, "substeps": 0}

#### ans-r3
{"name": "c1", "clients": 1, "wall": 3.805, "gen": 256, "prompt": [73], "ttft": [0.18], "agg_tps": 67.281, "sum_req_dec_tps": 70.352, "decode_steps": 255.0, "ms_step": 14.214, "tok_per_step": 1.004, "mono_ns": [166398579608221, 166402384540123], "all_decoding_s": [0.1802082061767578, 3.8048593997955322], "stage_layers": 0, "hit": 0.955, "routed_per_step": 696.435, "miss_units_per_step": 30.996, "steps_sliced": 0, "substeps": 0}
{"name": "c8", "clients": 8, "wall": 9.007, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.087, 0.516, 0.516, 0.516, 0.516, 0.515, 0.515, 0.515], "agg_tps": 227.39, "sum_req_dec_tps": 238.953, "decode_steps": 255.0, "ms_step": null, "tok_per_step": 8.031, "mono_ns": [166410862430092, 166419868965706], "all_decoding_s": [0.5167651176452637, 8.971850633621216], "stage_layers": 0, "hit": 0.963, "routed_per_step": 4069.349, "miss_units_per_step": 151.945, "steps_sliced": 0, "substeps": 0}
{"name": "stall1", "bg": 1, "arrival_prompt": 7433, "arrival_ttft": 2.522, "arrive_at": 1.5, "first_at": 4.02, "typical_gap_ms": 13.8, "worst_gaps_ms": [[47.6, 47.8, 48.1, 174.7, 369.2, 531.2, 569.0, 609.3]], "clients": 1, "mono_ns": [0, 0], "stage": "...", "steps_sliced": 0, "substeps": 0}

#### b0cp-r1
{"name": "c1", "clients": 1, "wall": 2.573, "gen": 256, "prompt": [73], "ttft": [0.189], "agg_tps": 99.482, "sum_req_dec_tps": 106.959, "decode_steps": 88.0, "ms_step": 27.092, "tok_per_step": 2.909, "mono_ns": [166484240305206, 166486813640971], "all_decoding_s": [0.18920493125915527, 2.5732908248901367], "hit": 0.954, "routed_per_step": 1987.136, "miss_units_per_step": 90.92}
{"name": "c8", "clients": 8, "wall": 8.862, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.107, 0.64, 0.639, 0.639, 0.639, 0.639, 0.638, 0.638], "agg_tps": 231.093, "sum_req_dec_tps": 276.611, "decode_steps": 109.0, "ms_step": null, "tok_per_step": 18.789, "mono_ns": [166488817694161, 166497679930414], "all_decoding_s": [0.6399846076965332, 7.0762951374053955], "hit": 0.935, "routed_per_step": 7180.853, "miss_units_per_step": 470.266}

#### b0cp-r2
{"name": "c1", "clients": 1, "wall": 2.744, "gen": 256, "prompt": [73], "ttft": [0.167], "agg_tps": 93.291, "sum_req_dec_tps": 98.945, "decode_steps": 88.0, "ms_step": 29.286, "tok_per_step": 2.909, "mono_ns": [166558468704041, 166561212801384], "all_decoding_s": [0.16683340072631836, 2.744013786315918], "hit": 0.954, "routed_per_step": 1987.136, "miss_units_per_step": 90.818}
{"name": "c8", "clients": 8, "wall": 8.826, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.106, 0.577, 0.577, 0.577, 0.577, 0.576, 0.576, 0.576], "agg_tps": 232.041, "sum_req_dec_tps": 277.354, "decode_steps": 106.0, "ms_step": null, "tok_per_step": 19.321, "mono_ns": [166563217329924, 166572043374181], "all_decoding_s": [0.5773317813873291, 6.752923965454102], "hit": 0.936, "routed_per_step": 7406.009, "miss_units_per_step": 475.292}

#### b0cp-r3
{"name": "c1", "clients": 1, "wall": 2.766, "gen": 256, "prompt": [73], "ttft": [0.19], "agg_tps": 92.558, "sum_req_dec_tps": 98.991, "decode_steps": 88.0, "ms_step": 29.273, "tok_per_step": 2.909, "mono_ns": [166632180670110, 166634946506517], "all_decoding_s": [0.1897735595703125, 2.7657663822174072], "hit": 0.954, "routed_per_step": 1987.136, "miss_units_per_step": 90.92}
{"name": "c8", "clients": 8, "wall": 9.214, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.106, 0.64, 0.64, 0.64, 0.64, 0.64, 0.639, 0.639], "agg_tps": 222.282, "sum_req_dec_tps": 248.288, "decode_steps": 91.0, "ms_step": null, "tok_per_step": 22.505, "mono_ns": [166636951578396, 166646165121395], "all_decoding_s": [0.640676736831665, 7.883296966552734], "hit": 0.933, "routed_per_step": 8589.099, "miss_units_per_step": 572.231}

#### acp-r1
{"name": "c1", "clients": 1, "wall": 2.886, "gen": 256, "prompt": [73], "ttft": [0.173], "agg_tps": 88.709, "sum_req_dec_tps": 94.013, "decode_steps": 99.0, "ms_step": 27.398, "tok_per_step": 2.586, "mono_ns": [166521955452839, 166524841307472], "all_decoding_s": [0.17335844039916992, 2.885754108428955], "stage_layers": 0, "hit": 0.957, "routed_per_step": 2101.081, "miss_units_per_step": 89.848, "steps_sliced": 0, "substeps": 0}
{"name": "c8", "clients": 8, "wall": 9.074, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.11, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58], "agg_tps": 225.712, "sum_req_dec_tps": 262.226, "decode_steps": 102.0, "ms_step": null, "tok_per_step": 20.078, "mono_ns": [166526846009751, 166535919536226], "all_decoding_s": [0.581096887588501, 7.092290639877319], "stage_layers": 0, "hit": 0.937, "routed_per_step": 7761.539, "miss_units_per_step": 491.637, "steps_sliced": 0, "substeps": 0}

#### acp-r2
{"name": "c1", "clients": 1, "wall": 3.113, "gen": 256, "prompt": [73], "ttft": [0.185], "agg_tps": 82.246, "sum_req_dec_tps": 87.122, "decode_steps": 98.0, "ms_step": 29.866, "tok_per_step": 2.612, "mono_ns": [166595673984698, 166598786613486], "all_decoding_s": [0.185638427734375, 3.112553358078003], "stage_layers": 0, "hit": 0.958, "routed_per_step": 2149.918, "miss_units_per_step": 89.602, "steps_sliced": 0, "substeps": 0}
{"name": "c8", "clients": 8, "wall": 8.236, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.108, 0.566, 0.566, 0.565, 0.566, 0.566, 0.565, 0.565], "agg_tps": 248.665, "sum_req_dec_tps": 288.436, "decode_steps": 97.0, "ms_step": null, "tok_per_step": 21.113, "mono_ns": [166600790877575, 166609026846705], "all_decoding_s": [0.5664505958557129, 6.7280189990997314], "stage_layers": 0, "hit": 0.938, "routed_per_step": 8346.082, "miss_units_per_step": 520.361, "steps_sliced": 0, "substeps": 0}

#### acp-r3
{"name": "c1", "clients": 1, "wall": 3.074, "gen": 256, "prompt": [73], "ttft": [0.185], "agg_tps": 83.27, "sum_req_dec_tps": 88.28, "decode_steps": 99.0, "ms_step": 29.177, "tok_per_step": 2.586, "mono_ns": [166670332454958, 166673406808578], "all_decoding_s": [0.1857457160949707, 3.0742733478546143], "stage_layers": 0, "hit": 0.959, "routed_per_step": 2135.02, "miss_units_per_step": 88.03, "steps_sliced": 0, "substeps": 0}
{"name": "c8", "clients": 8, "wall": 9.374, "gen": 2048, "prompt": [73, 76, 79, 73, 135, 84, 79, 89], "ttft": [0.108, 1.22, 1.22, 1.22, 1.22, 1.22, 1.22, 1.219], "agg_tps": 218.475, "sum_req_dec_tps": 266.593, "decode_steps": 95.0, "ms_step": null, "tok_per_step": 21.558, "mono_ns": [166675410985643, 166684785036608], "all_decoding_s": [1.2210264205932617, 7.768750905990601], "stage_layers": 0, "hit": 0.936, "routed_per_step": 8407.811, "miss_units_per_step": 541.895, "steps_sliced": 0, "substeps": 0}

#### hnl-r1
{"name": "c1", "clients": 1, "wall": 2.609, "gen": 256, "prompt": [73], "ttft": [0.187], "agg_tps": 98.135, "sum_req_dec_tps": 105.306, "decode_steps": 79.0, "ms_step": 30.652, "tok_per_step": 3.241, "mono_ns": [166816302563373, 166818911220956], "all_decoding_s": [0.1870415210723877, 2.6085667610168457], "stage_layers": 0, "hit": 0.952, "routed_per_step": 2099.089, "miss_units_per_step": 100.962, "steps_sliced": 0, "substeps": 0}

#### hnl-r2
{"name": "c1", "clients": 1, "wall": 2.428, "gen": 256, "prompt": [73], "ttft": [0.187], "agg_tps": 105.437, "sum_req_dec_tps": 113.809, "decode_steps": 79.0, "ms_step": 28.362, "tok_per_step": 3.241, "mono_ns": [166847580100828, 166850008104807], "all_decoding_s": [0.18731164932250977, 2.427915573120117], "stage_layers": 0, "hit": 0.952, "routed_per_step": 2099.089, "miss_units_per_step": 100.962, "steps_sliced": 0, "substeps": 0}

#### hnl-r3
{"name": "c1", "clients": 1, "wall": 2.376, "gen": 256, "prompt": [73], "ttft": [0.185], "agg_tps": 107.754, "sum_req_dec_tps": 116.394, "decode_steps": 79.0, "ms_step": 27.732, "tok_per_step": 3.241, "mono_ns": [166878111375178, 166880487156844], "all_decoding_s": [0.1848616600036621, 2.375696897506714], "stage_layers": 0, "hit": 0.952, "routed_per_step": 2099.089, "miss_units_per_step": 100.962, "steps_sliced": 0, "substeps": 0}
