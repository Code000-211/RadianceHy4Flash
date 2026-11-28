# #337: the replay harness, and what the fault report's arguments say

The issue asks for a reproduction first — "a script that replays the prompt lengths above … and
faults 2/2 times is enough to work against" — and then for a measurement of the paths that were
healthy (single-stream decode tok/s and TTFT on a ~100k prefix-cache hit, 3 runs a side).

**The fault did not reproduce on this machine.** Two sessions of that workload ran to the end,
growing past both lengths that faulted on the reporter's box (112,174 and 123,191), with 0 fault
lines, on a tree that carries `e3834473`. So this record has no before/after pair for a fix: no run
here can justify one. What it has is the three things that were actually missing from the card:

1. the reason no arm of the harness had ever run on a measurement box (a `curl` wait on a box
   without curl), and the fix that lets it run at all (section 1);
2. the arm that did run, with its numbers and its raw lines (section 2);
3. the reading of the reporter's own queue dump, which answers the issue's question 2 — the
   argument block in the report does not belong to the kernel the packet names (section 3).

## 1. No arm could run: the wait for the server asked with `curl`

`scripts/repro-337.sh` waited for a live server with `curl -sf -m 2 http://127.0.0.1:$PORT/health`.
On iks, which has no `curl`, the wait ran its 900 sleeps with the server up and then printed "the
server did not come up" beside a server log that was already serving. That is exactly what the
delivery of the previous round's harness did there: the arm was announced and produced nothing,
which is why this issue had no measurement. The probe is now
`python3 scripts/agentreplay.py --probe URL` — the replay needs python3 on every box it can run on,
so the wait needs nothing new — and it exits 0 on a 2xx and 1 on anything else (a refused
connection, a timeout, the 503 of a model that is not loaded yet).
`tests/repro_337_script_test.sh` covers it: it starts a real HTTP server, requires the probe to call
it up, requires it to call a closed port down, and fails if any non-comment line of the harness calls
curl again. `sh tests/repro_337_script_test.sh` passes; the C++ wrapper the build already registers
(`tests/repro_337_script_test.cpp`, a `*_test.cpp`) puts it under ctest.

The harness also carries the reported configuration now, which it lacked: `pin: true` (a fleet key —
without it the server may park and refill the pageable ranges the fault report decodes) and
`num-speculative-tokens` / `draft-coupling` as `SPEC` / `COUPLING`, defaulting to the issue's 5 and
`on`, so the default arm is the reporter's configuration and the issue's bench question (b) is one
environment variable away.

## 2. The arm that ran

- **Tree:** `82b82ce8`, the branch's merge base, carrying this PR's harness (the harness fix is what
  makes a run possible on this machine); `e3834473`, the reporter's faulting build, is an ancestor
  of it. `git log` of `libr4d/r4d_gemm_fp8a8.hip` is `e9e38917` in both.
- **Machine:** iks, 4 × AMD Radeon AI PRO R9700 (gfx1201), **TP 2**, 48 cores, 112 GiB RAM.
- **Power cap:** 210 W on both cards, read back from the server's own device line (default 300 W, max
  330 W), voltage offset 0 mV (the reporter's box has -88 mV), runtime pm active.
- **Toolchain:** `~/moebius-gpu/configure.sh` on iks — host clang 23, device code with radiance's own
  HIP compiler (ROCm 10.1's clang 24 `d6f6cb69` over the VOPD-repeatable libLLVM,
  `RAD_HIP_VOPD_REPEATABLE=ON`). It is not the reporter's `-DRAD_HIP_VOPD_REPEATABLE=OFF` ROCm 7.2.3
  host build; see section 5.
- **Container:** `hy4-flash-iqr-v4-u3.rad` (67.30 GiB read from it), `kernels: libr4d,libavx,spechub`,
  `pin: true`, `ngram-placement: disk`, `kv-cache-dtype: bf16`, `max-model-len: 262144`,
  `max-num-batched-tokens: 2048`, `max-num-seqs: 4`, `num-speculative-tokens: 5`,
  `draft-coupling: on`, `draft-verify: traversal`, `draft-depth: adaptive`, `draft-suffix: on`,
  `placement: expert_tiered`, `host-pool-mib: 20000`, `prefix-cache-mib: 4096`,
  `prefix-resume: nearest`, `checkpoint-interval: 1984`, `checkpoint-slots: 128`,
  `control-vector: on`, `gpu-headroom-mib: 128`, W2 on.
- **Workload:** `scripts/agentreplay.py --parents 1 --system 95000 --context 130000 --turn
  1800,2200 --reply 2048 --sub-every 3 --sub-turns 3 --tools 12`, temperature 0: one parent session
  growing 95k → 130k in ~2k steps with 12 tools and a subagent every third turn, the prefix cache hot.
- **Session:** 32 requests in 290 s, largest prompt **128,036** tokens, 31 of 32 with a prefix-cache
  hit ≥ 90k (`replay.log`):
  ```
  turn                    turns   prompt tok   cached tok  cached  re-pref ttft p50 s
  parent                     12      1324425      1202128   90.8%        0       1.18
  parent after subagent       5       563335       556528   98.8%        0       0.95
  subagent first turn         5       482399       477952   99.1%        0       0.60
  subagent later turn        10       999313       977968   97.9%        0       1.17
  ```
- **Decode and TTFT**, the 31 requests whose prompt cache hit was ≥ 90k tokens (median, min–max):
  decode **112.5 tok/s** (84.4–175.8), TTFT **1.06 s** (0.15–2.71). The reporter's healthy numbers
  were 120–138 tok/s and 0.4–1.8 s; this is the same band, on a 4-card box at TP2 with a 210 W cap.
- **Faults:** 0. `bash scripts/repro-337.sh --summarise runs/iks/repro-337-base-a` prints
  ```
  repro-337: 32 requests, largest prompt 128036 tokens, 31 with a prefix-cache hit
  repro-337: 0 fault line(s) in runs/iks/repro-337-base-a/server.log
  ```
  and `grep -c '^E ' server.log` is 0: no fault, no stall, no OOM, no driver reset. Raw lines from
  the log, verbatim:
  ```
  I req chatcmpl-6783491cb6a9dd96bb1fdf47 done from 127.0.0.1:39218 finish=tool_calls prompt=96370 (cache 0) gen=329 reason=298 ttft=28.53s 87.5 tok/s
  I req chatcmpl-2e42b68dbfdcdbb7e90d836e done from 127.0.0.1:56178 finish=tool_calls prompt=98619 (cache 96368) gen=272 reason=259 ttft=1.06s 84.4 tok/s
  I req chatcmpl-d07ffe49630e313738f219b7 done from 127.0.0.1:42070 finish=tool_calls prompt=96555 (cache 95232) gen=89 reason=76 ttft=0.85s 136.5 tok/s
  I       power  cap 210 W (default 300 W, max 330 W) | voltage offset 0 mV | sclk 500-2350 MHz | mclk 96-1258 MHz | runtime pm active (control on) | iommu group identity
  I activation arena: 256 tokens use 250.94 MiB, 1024 tokens use 901.52 MiB of 1.73 GiB (the fixed region is 32.86 MiB); the rest is lent to the expert slab where the plan allows
  ```
  (The first line is the cold prefill of the session; the two after it are cache hits, which is what
  the decode and TTFT rows are read from.)

The 0-fault result also holds for the previous round's two iks sessions (`RADIANCE_W2` default,
`num-speculative-tokens: 5`, different prompt shapes): three sessions on this machine, none faulted,
the largest prompt above both of the reporter's fault lengths. Nothing about the fault can be
concluded from this machine either way.

### The head arm, through the fixed harness

The same command on the head tree (`19b993f2`; this note's commit sits on top of it and changes
nothing but this file) runs the harness end to end for
the first time on this machine: the wait finds the server up, the script echoes the configuration it
started, and it sums its own run up. Two sessions, one after the other:

| arm | tree | requests | largest prompt | cache ≥ 90k | decode tok/s (median, min–max) | TTFT s (median, min–max) | fault lines |
|---|---|---|---|---|---|---|---|
| base-a | `82b82ce8` | 32 | 128,036 | 31 | 112.5 (84.4–175.8) | 1.06 (0.15–2.71) | 0 |
| head-a | `19b993f2` | 36 | 128,800 | 35 | 106.5 (85.5–141.2) | 1.09 (0.18–2.69) | 0 |
| head-b | `19b993f2` | 30 | 130,211 | 29 | 111.1 (81.7–149.6) | 1.05 (0.17–2.53) | 0 |

The three arms are within each other's spread, so there is no regression beyond noise on the decode
and TTFT rows (the number to beat was 120–138 tok/s and 0.4–1.8 s on the reporter's box, whose
prompts and cache hits are not this workload's). Raw lines, head-a and head-b:
```
repro-337: W2 on, 2 card(s), 5 speculative token(s), draft-coupling on, host pool 20000 MiB, max-model-len 262144, server up
repro-337: 36 requests, largest prompt 128800 tokens, 35 with a prefix-cache hit
repro-337: 0 fault line(s) in …/runs/iks/repro-337-head-a/server.log
repro-337: 30 requests, largest prompt 130211 tokens, 29 with a prefix-cache hit
repro-337: 0 fault line(s) in …/runs/iks/repro-337-head-b/server.log
HEAD-A rc=0
HEAD-B rc=0
```
The same head tree's own startup line names its activation arena at **2.24 GiB**, the size of the
range the reporter's `+16/+24/+32` pointers sit in: that range is the arena, as section 3 says.

The issue's bench questions (a) `RADIANCE_W2=0`, (b) `num-speculative-tokens: 0` /
`draft-coupling: off`, (c) TP1 and TP4 are **not** answered here: with the fault not reproducing at
all on this box, a run with a knob off says nothing (0 faults is the result either way). The
harness has all three as knobs — `RADIANCE_W2=0 bash scripts/repro-337.sh <tag>`,
`SPEC=0 COUPLING=off …`, `TP=1` / `TP=4` on iks — so they are one command each on a box where the
fault repeats, and that box is the reporter's.

## 3. Question 2: the argument block and the kernel do not agree

The issue asks: "Is `gemm_i8_kpar_kernel` with `r4dchain` expected to be issued against a 2.24 GiB
activation arena at all, or is there a bound on `M` that this session crossed?"

**Yes, it is expected there, and no `M` bound was crossed.** The reading below is from the source and
the reporter's dump, not from a run.

What the tile is. `libr4d/r4d_gemm_fp8a8.hip:1185-1194` picks the instantiation from `K` and `M`
alone: with `i8 == 2` (the int8 weights here) the group size is 256, `ngrp = K / 256`, and
`KW = 4` only when `ngrp >= 4`, i.e. **K ≥ 1024**; `NT` from `M` — `M > 48` → 4, `> 32` → 3,
`> 16` → 2, else 1 (`:1216-1217`), and the narrow tile covers `mw * 16` rows with `mw` up to 16
(`narrow_mw`, `:985`). So the bounds of the shape are: `K % 256 == 0`, `K ≥ 1024` for
the 4-wide form, `M ≤ 16` for `NT = 1` (and ≤ 32 / 48 for 2 / 3), `sa_ld = K / 256`, and `N` in
16-row tiles, guarded where it is not a whole number of them. A pass is chunked by
`max-num-batched-tokens` (2048 here) and the tile splits `M` over the grid, so **no prompt length
grows `M` past 16 for `NT = 1`**: a 112k-token session sends many 16-row tiles, not a taller one.

What the report's arguments are. The stopped packet's name is
`_ZN12_GLOBAL__N_119gemm_i8_kpar_kernelILj4ELj256ELj1E8r4dchainEEvPKhPKtPKaPKfPtjjjjj`, i.e.
`<KW=4, sb=256, NT=1, r4dchain>`, which the launcher emits only for K ≥ 1024 and M ≤ 16. The report
prints the kernarg slot's 8-byte words that look like pointers, so at `e3834473` the layout shows as
pairs: `+40 = 0x100000000c00` is `(N, K) = (3072, 256)`, `+48 = 0x100000000006` is `(M, a_ld) =
(6, 256)`, and `sa_ld` at `+56` (= 1) is not printed. `M = 6` agrees with `NT = 1`; `K = 256` does
not agree with `KW = 4` — it selects `<1, 256, 1>`. The five pointers (+0, +8 in one sealed 24.37 GiB
pageable range; +16, +24, +32 in one 2.24 GiB range) are a self-consistent set for that other launch:
`a_ld = 256 = K` and `sa_ld = 1 = K / 256`.

So the kernel and its arguments in that packet come from two different launches of the same entry
point: the packet names the K ≥ 1024 instantiation while its argument block holds a K = 256 dispatch's
values. That is not a shape the session crossed; it is a launch that ran with arguments that did not
belong to it, and an argument read as a size or a pointer from the wrong launch is what an address
"in no range this process holds" looks like (run 2's fault address `0xebbcff295000`, page-not-present)
far better than any tile bound is. On the activation arena: the `+16/+24/+32` pointers sit in the
pass's activation arena, whose **whole size** the report prints (2.24 GiB on the reporter's box,
1.73 GiB on iks for this config) — that is the arena, not the tile's buffer, so "issued against a
2.24 GiB arena" is expected and says nothing about the tile's bounds.

What to hunt next is therefore the argument path, not the shapes: a packet whose `kernarg_address`
holds another dispatch's arguments means a slot reused across dispatches, or a packet written against
a block that was already rewritten, in the chained-launch path of `core/device/aql.cpp` (`chain_record`,
the per-play argument copy at a play, and whatever frees or reuses a tape's argument block). With the
report change of this tree the same fault prints each argument's parameter name and type, so the next
report names the field that was wrong instead of leaving it to be decoded.

## 4. Question 1: types in the fault report

Answered by the report change carried in this tree (`core/device/karg_report.{h,cpp}`,
`core/device/aql.cpp`): each argument line now prints the parameter's name, its offload-metadata
`value_kind`, and — for a kind that is not an address — its numeric value, so an integer argument is
no longer printed as "in no range this process holds". In the dump above that would have shown
`+40 cols=256 (by_value)`, which is the line that decides section 3's reading in one look.

## 5. What is owed to close the issue

- **A reproduction on a box where it repeats** (the reporter's, or one with the same `-88 mV` offset
  and ROCm 7.2.3 host build): three runs of `bash scripts/repro-337.sh <tag>` on `e3834473` that
  fault, then the same three on the fixing head. Until that exists, no fix for this fault can be
  measured or merged on evidence.
- **The issue's three bench questions**, one command each with the knobs above, on that box.
- **The decode/TTFT pair** (3 + 3 runs on a ~100k prefix-cache hit) for whichever change fixes it.
  This record's row (section 2) is the same workload at the merge base, for comparison.
