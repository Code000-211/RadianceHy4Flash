# Issue #251: TP2 hang under load from a stale chain record (PR #252)

A bug report with a proposed fix, not a backlog item: the bar the review set was "the hang is there before, gone
after, greedy text unchanged, the cost known". Measured by the reviewing maintainer; the change and the diagnosis are
krisf's. **Outcome: the hang does not occur on the default branch, so the scalar-cache change is not taken**
(`../closed/issue-251.md`); what merges is the stall report's chain line, its device test and the `stress` phase.

## Commits, machine

- **before** = `cfd79b13625ffc4798311e8f3f217549a8a834c3`, the default branch when the PR was taken up.
- **after** = `f0e43395b227f652fbd697f6fcbec13771a07f9b`, the PR head `806188f2` merged with that base: krisf's
  `42befd90` (`s_dcache_inv` at the entry of `r4d_chain_wait_impl()`; `gemm_i8_kpar`, `hc_read_fused` and
  `ar_rmsnorm_quant_i8_sp` from `r4d_chain_wait_vec()` to `r4d_chain_wait()`; the audit's `inv > 0` guard) and
  `806188f2` (the chain line in the stall report).
- **final** = `80fb0b39`, what the PR merges: `42befd90` taken out again (`aa849bcf`), the chain line as
  `chain_line()` with its device test (`6944745b`), `probe.py`'s `stress` (`80fb0b39`).
- Both arms ran `scripts/backlog/probe.py` with the `stress` phase of `80fb0b39` (its only difference to the base's
  tool; the same file in both arms; the step counters were added to the phase after the first four runs).
- Machine: iks (`ki01-gpu`), 4 x AMD Radeon AI PRO R9700 (gfx1201), Linux 7.0.14-26-pve, ROCm SDK clang 23.0.0git
  (`_rocm_sdk_devel`), `-DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201`, each arm its own tree and build under
  `~/moebius-gpu/`. Lane **iks-A** for every before/after row (TP2, a pair from `gpu2`, Engram
  `ram=/dev/shm/radiance-ngram`, host pool 16000 MiB, prefix cache 8192 MiB), through `scripts/backlog/lane.sh iks-A
  'scripts/backlog/run.sh <tag> <cfg> <phases>'`. Power cap read back 210 W on every card before every run.
  Container `hy4-flash-iqr-v4-u3.rad`. 2026-11-25.
- The device tests ran on the `local` lane's pair (`gpu2 build/tests/device_test <filter>`), kernel 7.0.2-6-pve.

## 1. The hang: not there before

`stress` (new, `probe.py`): 40 waves of eight requests at once, each ~12K tokens of context (39,500 characters of
tree source, 11.1K-13.5K tokens) decoded for 96 tokens; a wave's prompts are fresh (a cold prefill beside the
others) and the wave after it sends the same eight again. 320 requests a run, the shape the issue and the PR name
("8-way concurrent, ~12K-token context, 40 waves", "320 ok / 0 err after, hung before"). A watchdog exit would be
`server-exit=70` with the queue dump in `server.log`.

| run | arm | config | waves | ok / err | server at the end | `^E` lines in `server.log` | engine steps |
|---|---|---|---|---|---|---|---|
| st-b1 | before | `nospec.yaml` | 40 / 40 | 320 / 0 | serving, exit 0 | 0 | not recorded |
| st-a2 | after | `nospec.yaml` | 40 / 40 | 320 / 0 | serving, exit 0 | 0 | not recorded |
| stc-b1 | before | `coupled03.yaml` | 40 / 40 | 320 / 0 | serving, exit 0 | 0 | not recorded |
| stc-b2 | before | `coupled03.yaml` | 40 / 40 | 320 / 0 | serving, exit 0 | 0 | 7,941 (810 prefill, 341 mixed, 6,790 decode) |
| stc-a1 | after | `coupled03.yaml` | 40 / 40 | 320 / 0 | serving, exit 0 | 0 | 8,061 (762 prefill, 386 mixed, 6,913 decode) |
| st-b2r | before | `nospec.yaml` | 40 / 40 | 320 / 0 | serving, exit 0 | 0 | 17,172 (704 prefill, 437 mixed, 16,031 decode) |
| std-b1r | before | `nospec.yaml`, decode-heavy: `RAD_STRESS_CHARS=2000 RAD_STRESS_NTOK=1024 RAD_STRESS_WAVES=20` | 20 / 20 | 160 / 0 | serving, exit 0 | 0 | 21,439 (20 prefill, 58 mixed, 21,361 decode) |
| fin-st | final | `nospec.yaml` | 40 / 40 | 320 / 0 | serving, exit 0 | 0 | 17,332 (694 prefill, 466 mixed, 16,172 decode) |

Before: five runs of five that started, 1,440 requests, no hang and no error. After: two of two. Final: one of one.

Raw lines (`probe.py`'s summary of the phase):

```
st-b1  {"name": "stress", "waves": 40, "of": 40, "clients": 8, "chars": 39500, "ntok": 96, "ok": 320, "err": 0, "slow": 97, "server_dead": false, "wall": 1511.1}
st-a2  {"name": "stress", "waves": 40, "of": 40, "clients": 8, "chars": 39500, "ntok": 96, "ok": 320, "err": 0, "slow": 46, "server_dead": false, "wall": 859.4}
stc-b1 {"name": "stress", "waves": 40, "of": 40, "clients": 8, "chars": 39500, "ntok": 96, "ok": 320, "err": 0, "slow": 36, "server_dead": false, "wall": 776.5}
stc-b2 {"name": "stress", "waves": 40, "of": 40, "clients": 8, "chars": 39500, "ntok": 96, "ok": 320, "err": 0, "slow": 38, "server_dead": false, "wall": 776.7, "steps": {"engine_steps_total": 7941.0, "engine_steps_prefill_total": 810.0, "engine_steps_mixed_total": 341.0, "engine_steps_decode_total": 6790.0}}
stc-a1 {"name": "stress", "waves": 40, "of": 40, "clients": 8, "chars": 39500, "ntok": 96, "ok": 320, "err": 0, "slow": 39, "server_dead": false, "wall": 790.2, "steps": {"engine_steps_total": 8061.0, "engine_steps_prefill_total": 762.0, "engine_steps_mixed_total": 386.0, "engine_steps_decode_total": 6913.0}}
st-b2r  {"name": "stress", "waves": 40, "of": 40, "clients": 8, "chars": 39500, "ntok": 96, "ok": 320, "err": 0, "slow": 46, "server_dead": false, "wall": 842.3, "steps": {"engine_steps_total": 17172.0, "engine_steps_prefill_total": 704.0, "engine_steps_mixed_total": 437.0, ...}}
std-b1r {"name": "stress", "waves": 20, "of": 20, "clients": 8, "chars": 2000, "ntok": 1024, "ok": 160, "err": 0, "slow": 0, "server_dead": false, "wall": 528.0, "steps": {"engine_steps_total": 21439.0, "engine_steps_prefill_total": 20.0, "engine_steps_mixed_total": 58.0, "engine_steps_decode_total": 21361.0}}
fin-st  {"name": "stress", "waves": 40, "of": 40, "clients": 8, "chars": 39500, "ntok": 96, "ok": 320, "err": 0, "slow": 46, "server_dead": false, "wall": 859.9, "steps": {"engine_steps_total": 17332.0, "engine_steps_prefill_total": 694.0, "engine_steps_mixed_total": 466.0, "engine_steps_decode_total": 16172.0}}
```

(`slow` counts requests whose first token took over 30 s: eight 12K prompts prefilled behind one another. st-b1
ran while a second lane's stress run shared the host; its wall is twice the others'.)

What that load is, against the diagnosis: a run is about 8,000 engine steps with the drafter, 17,000 without and 21,000 in the decode-heavy one, a
decode pass 1,067-1,194 dispatches a card (the stamped rows below), so 9 to 18 million launches and 2,000 to
4,000 laps of each card's 4096-slot argument ring. The failure the issue describes is not a rare wrong value: a chained kernel that reads a record it does not
accept runs without waiting and without counting, the queue's completion counter stays short for good, and the
next slot wait or flush never ends. So one such read in a run is a hang of that run, and no run hung.

The same on the cards directly, on the default branch (`cfd79b13`), three times each, all passing:
`device_test chain` (4 cases, 4,219 checks), `device_test ring` (3 cases, 1,528 checks) and `device_test record`
(2 cases, 16,915 checks). `a_copy_issued_after_a_pause_finds_the_record_written_for_it` is the issue's case to the
letter: a ring lap of launches with chaining off leaves every slot with a record no kernel accepts, then 6,000
chained launches go out with their packets unacquired (`FENCE_SCOPE_NONE`), each storing whether it found its
record, on both cards: 0 without one. What makes the read fresh is already in `core/device/aql.cpp`: the system-scope
acquire the ring asks for every half lap (`acquire_scope`), which a bare chained packet keeps, and the flush of the
host's write before the doorbell (`host_writes_out`).

### Runs that gave no result, and why

| run | what happened | cause |
|---|---|---|
| st-b2, before, `nospec.yaml` | the server never came up: `E --ngram-placement ram: the n-gram table (60.24 GiB) cannot be held in memory … 88.52 GiB of memory is available, 25.29 GiB more is still to be allocated`, `startup stopped: out of memory`, `server-exit=1`; 0 requests | host memory on iks: two other jobs' servers were up beside the lane |
| std-b1 (`nospec.yaml`) and stdc-b1 (`coupled03.yaml`), before, `RAD_STRESS_CHARS=2000 RAD_STRESS_NTOK=1024 RAD_STRESS_WAVES=20` | killed while loading the model (`Killed` from the shell, `server-exit=137`); the last log line is `checkpoint slots: the host cannot pin … 7.51 GiB of memory available` (7.40 on the second); 0 requests, no stall line | the same: SIGKILL during load with 7.5 GiB of host memory left (the kernel log is not readable for the lane's user, so "the OOM killer" is inferred from the signal, the moment and the memory line) |
| st-cp-b1 (before, `coupled03.yaml`), st-cp-a1 (after, `coupled03.yaml`), stB-b1 (before, `nospec.yaml`), all on **iks-B** | the engine stopped in wave 20 of 40 in all three (160 ok, then errors): `E kernel avx_embed_lookup_q (libavx …) refused op 'embed_lookup_q' at M=32768 …: io error`, `step … failed: io error -- the engine cannot continue`, `server-exit=0`; in stB-b1 a `W step: nothing has been logged for 20 s` precedes it | the disk Engram reader (`disk=~/ngram`, `libavx/avx_ngram.cpp`, an `io_uring` read that failed or came back short), with 3 GiB of host memory available at the time. Not a chain wait: the engine reports the error and stops, the queues are not stuck, and the arm with the PR's change fails the same way in the same wave. It is a failure of its own and is not fixed here |

None of them is a watchdog exit or a short chain counter: three never served a request (host out of memory at
startup or while pinning the checkpoint slots), and the three on iks-B stop on a host kernel's I/O error with and
without the change. They are in neither arm's count above. st-b2 and std-b1 were queued again once the host had
memory and are the rows st-b2r and std-b1r above; stdc-b1's second attempt was stopped before it started (the
owner called the measurement complete), so the decode-heavy load has no `coupled03.yaml` row.

## 2. What the change costs (iks-A, interleaved b1 a1 a2 b2 b3 a3)

`run.sh <tag> <cfg> greedy,c1,c8 RADIANCE_AQL_STAMP=1`, one server a run; the kernel rows are `agg.py --top 60` over
the run's stamps (the full-model pass kind: 1,079 records at `c1` and 1,163 at `c8` on `nospec.yaml`, 1,164 and
1,206 on `coupled03.yaml`).

### `nospec.yaml`

| run | greedy sha256 (3 prompts, 5.7K arrival) | c1 tok/s | c1 ms/step | tok/step | c1 pass wall / busy µs | c8 agg tok/s | c8 pass wall / busy µs |
|---|---|---|---|---|---|---|---|
| pa-b1 (before) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f 466e53a2921d | 73.77 | 13.282 | 1.00 | 12969 / 12476 | 294.6 | 23953 / 23220 |
| pa-a1 (after) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f 466e53a2921d | 73.51 | 13.334 | 1.00 | 13018 / 12537 | 321.1 | 23993 / 23284 |
| pa-a2 (after) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f 466e53a2921d | 73.51 | 13.328 | 1.00 | 13013 / 12538 | 318.1 | 24172 / 23484 |
| pa-b2 (before) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f 466e53a2921d | 73.99 | 13.240 | 1.00 | 12922 / 12433 | 322.4 | 23848 / 23151 |
| pa-b3 (before) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f 466e53a2921d | 73.98 | 13.244 | 1.00 | 12928 / 12449 | 320.2 | 24138 / 23389 |
| pa-a3 (after) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f 466e53a2921d | 73.56 | 13.322 | 1.00 | 13007 / 12517 | 319.1 | 24159 / 23429 |

| row | before, median (range) | after, median (range) | delta |
|---|---|---|---|
| c1 tok/s | 73.978 (73.766..73.990) | 73.513 (73.506..73.555) | -0.63 % |
| c1 ms/step | 13.244 (13.240..13.282) | 13.328 (13.322..13.334) | +0.63 % |
| c1 pass wall µs | 12928 (12922..12969) | 13013 (13007..13018) | +0.66 % |
| c1 pass busy µs | 12449 (12433..12476) | 12537 (12517..12538) | +0.71 % |
| c8 agg tok/s | 320.172 (294.568..322.444) | 319.084 (318.121..321.078) | -0.34 % |
| c8 pass wall µs | 23953 (23848..24138) | 24159 (23993..24172) | +0.86 % |
| c8 pass busy µs | 23220 (23151..23389) | 23429 (23284..23484) | +0.90 % |

| kernel, µs a launch (launches a pass) | pa-b1 | pa-a1 | pa-a2 | pa-b2 | pa-b3 | pa-a3 | before median | after median | delta |
|---|---|---|---|---|---|---|---|---|---|
| c1 `gemm_i8_kpar_kernel` (125) | 16.8 | 17.1 | 17.1 | 16.8 | 16.8 | 17.1 | 16.8 | 17.1 | +1.8 % |
| c1 `ar_ihc_junction_2rank_kernel` (84) | 14.8 | 15.1 | 15.1 | 14.7 | 14.8 | 14.9 | 14.8 | 15.1 | +2.0 % |
| c1 `moe_iqr_kernel` (41) | 37.2 | 37.3 | 37.4 | 37.0 | 37.0 | 37.3 | 37.0 | 37.3 | +0.8 % |
| c1 `r4dchain_gemm_bf16_nt_m64` (54) | 8.1 | 7.9 | 7.9 | 8.0 | 8.0 | 8.1 | 8.0 | 7.9 | -1.2 % |
| c1 `r4dchain_rmsnorm_quant` (42) | 2.7 | 2.7 | 2.6 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 | +0.0 % |
| c8 `gemm_i8_kpar_kernel` (125) | 17.8 | 18.1 | 18.2 | 17.8 | 17.9 | 18.2 | 17.8 | 18.2 | +2.2 % |
| c8 `ar_ihc_junction_2rank_kernel` (84) | 17.3 | 17.3 | 17.3 | 17.1 | 17.1 | 17.1 | 17.1 | 17.3 | +1.2 % |
| c8 `moe_iqr_kernel` (41) | 177.9 | 178.5 | 181.7 | 178.2 | 180.6 | 181.1 | 178.2 | 181.1 | +1.6 % |
| c8 `r4dchain_gemm_bf16_nt_m64` (54) | 6.9 | 7.3 | 7.3 | 6.9 | 6.9 | 7.3 | 6.9 | 7.3 | +5.8 % |
| c8 `r4dchain_rmsnorm_quant` (42) | 3.1 | 3.2 | 3.1 | 3.0 | 3.0 | 3.3 | 3.0 | 3.2 | +6.7 % |

Greedy text is byte for byte the same in all six. The step is 0.63 % slower at one decoder (13.244 -> 13.328
ms/step, the two arms' ranges apart: 13.240..13.282 against 13.322..13.334) and the pass 0.86 % at eight (wall
23,953 -> 24,159 us; `c8` tok/s itself is inside its run-to-run range). `gemm_i8_kpar` is the one kernel of the
three in this model's decode pass (125 launches): 16.8 -> 17.1 us a launch (+1.8 %), 17.8 -> 18.2 at eight (+2.2
%): its wait now empties the scalar cache behind the weight tile and scale it loaded ahead of the wait and refetches
its arguments. The entry invalidate is on every chained kernel, which is where the other rows' 1-2 % come from.

### `coupled03.yaml` (DFly2)

| run | greedy sha256 (3 prompts, 5.7K arrival) | c1 tok/s | c1 ms/step | tok/step | c1 pass wall / busy µs | c8 agg tok/s | c8 pass wall / busy µs |
|---|---|---|---|---|---|---|---|
| pc-b1 (before) | 5b7a3975defa f55fe02071d8 0075e702dfdf 717c3b789eee | 123.07 | 23.946 | 3.08 | 18367 / 17673 | 393.1 | 49724 / 48905 |
| pc-a1 (after) | 5b7a3975defa f55fe02071d8 0075e702dfdf 717c3b789eee | 124.47 | 23.667 | 3.08 | 18312 / 17647 | 398.2 | 48496 / 47298 |
| pc-a2 (after) | 5b7a3975defa f55fe02071d8 0075e702dfdf 717c3b789eee | 124.24 | 24.006 | 3.12 | 18462 / 17795 | 393.8 | 48595 / 47479 |
| pc-b2 (before) | 5b7a3975defa f55fe02071d8 0075e702dfdf 717c3b789eee | 123.54 | 23.567 | 3.05 | 18442 / 17779 | 407.3 | 48451 / 47241 |
| pc-b3 (before) | 5b7a3975defa f55fe02071d8 0075e702dfdf 717c3b789eee | 129.78 | 24.103 | 3.28 | 18434 / 17757 | 416.4 | 48747 / 47826 |
| pc-a3 (after) | 5b7a3975defa f55fe02071d8 0075e702dfdf 717c3b789eee | 123.86 | 23.786 | 3.08 | 18472 / 17865 | 384.7 | 48579 / 47571 |

| row | before, median (range) | after, median (range) | delta |
|---|---|---|---|
| c1 tok/s | 123.537 (123.069..129.779) | 124.238 (123.859..124.469) | +0.57 % |
| c1 ms/step | 23.946 (23.567..24.103) | 23.786 (23.667..24.006) | -0.67 % |
| c1 pass wall µs | 18434 (18367..18442) | 18462 (18312..18472) | +0.15 % |
| c1 pass busy µs | 17757 (17673..17779) | 17795 (17647..17865) | +0.21 % |
| c8 agg tok/s | 407.280 (393.133..416.416) | 393.768 (384.673..398.241) | -3.32 % |
| c8 pass wall µs | 48747 (48451..49724) | 48579 (48496..48595) | -0.34 % |
| c8 pass busy µs | 47826 (47241..48905) | 47479 (47298..47571) | -0.73 % |

| kernel, µs a launch (launches a pass) | pc-b1 | pc-a1 | pc-a2 | pc-b2 | pc-b3 | pc-a3 | before median | after median | delta |
|---|---|---|---|---|---|---|---|---|---|
| c1 `gemm_i8_kpar_kernel` (125) | 17.2 | 17.5 | 17.5 | 17.3 | 17.2 | 17.5 | 17.2 | 17.5 | +1.7 % |
| c1 `ar_ihc_junction_2rank_kernel` (84) | 16.7 | 16.9 | 16.7 | 16.5 | 16.5 | 16.7 | 16.5 | 16.7 | +1.2 % |
| c1 `moe_iqr_kernel` (41) | 111.3 | 110.6 | 112.1 | 110.2 | 112.7 | 112.4 | 111.3 | 112.1 | +0.7 % |
| c1 `r4dchain_gemm_bf16_nt_m64` (54) | 6.7 | 7.0 | 7.0 | 6.8 | 6.7 | 6.9 | 6.7 | 7.0 | +4.5 % |
| c1 `r4dchain_rmsnorm_quant` (42) | 2.6 | 2.6 | 2.5 | 2.6 | 2.6 | 2.6 | 2.6 | 2.6 | +0.0 % |
| c8 `gemm_i8_kpar_kernel` (125) | 18.8 | 18.7 | 18.9 | 18.6 | 18.9 | 18.7 | 18.8 | 18.7 | -0.5 % |
| c8 `ar_ihc_junction_2rank_kernel` (84) | 29.8 | 29.2 | 29.2 | 29.6 | 29.5 | 29.1 | 29.6 | 29.2 | -1.4 % |
| c8 `moe_iqr_kernel` (41) | 519.2 | 503.4 | 504.1 | 504.8 | 513.7 | 500.5 | 513.7 | 503.4 | -2.0 % |
| c8 `r4dchain_gemm_bf16_nt_m64` (54) | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 7.1 | 6.9 | 6.9 | +0.0 % |
| c8 `r4dchain_rmsnorm_quant` (42) | 2.5 | 2.8 | 2.7 | 2.5 | 2.5 | 2.7 | 2.5 | 2.7 | +8.0 % |

Greedy text again the same in all six. With the drafter the step and the pass are inside the run-to-run range
(tok/step moves 3.05..3.28 with the sampled text); `gemm_i8_kpar` at one decoder shows the same +1.7 % a launch.

## 3. The audit, before and after

`J=8 python3 tests/chain_scalar_audit.py build build/chain_scalar_audit --check` (the build's
`compile_commands.json`, every HIP unit with a chained kernel compiled to gfx1201 assembly):

| tree | result |
|---|---|
| before `cfd79b13` | `chained kernels 2310: keep s_dcache_inv 1512, skip it 798, of those with a scalar load behind the acquire 0`, rc 0 |
| after `f0e43395` | `chained kernels 2310: keep s_dcache_inv 1574, skip it 736, of those with a scalar load behind the acquire 0`, rc 0 |
| final `80fb0b39` | `chained kernels 2310: keep s_dcache_inv 1512, skip it 798, of those with a scalar load behind the acquire 0`, rc 0 |

On the base the three kernels the change moves to `r4d_chain_wait()` are listed as
`ok  skips s_dcache_inv: … scalar loads behind the acquire: 0` in every instantiation (`gemm_i8_kpar_kernel<…,
r4dchain>`, `hc_read_fused_kernel<…, r4dchain>`, `r4dchain_ar_rmsnorm_quant_i8_sp<…>`): the `Sa` scale of
`gemm_i8_kpar` is a per-lane load issued ahead of the wait on purpose (a static weight), and the scalar loads
behind the acquire in the other two are from the kernel argument segment. So the audit covers today's kernels
(K-52, K-43, D-04, P-09 paths included: it compiles every unit) and finds no wait_vec kernel that reads a scale
through a scalar load behind its wait.

## 4. The final tree

| check | result |
|---|---|
| `fin-ns` (`nospec.yaml`, iks-A) | greedy `d6da4fb33eed 0699709c4be2 e7a23c5d9a2f 466e53a2921d`, c1 74.02 tok/s, 13.237 ms/step, pass wall 12,927 us, `gemm_i8_kpar` 16.8 us; c8 323.3 tok/s, pass wall 23,831 us |
| `fin-cp` (`coupled03.yaml`, iks-A) | greedy `5b7a3975defa f55fe02071d8 0075e702dfdf 717c3b789eee`, c1 124.30 tok/s, 23.701 ms/step; c8 388.7 tok/s |
| `fin-st` (`nospec.yaml` `stress`, iks-A) | 40 / 40 waves, 320 ok / 0 err, serving at the end, exit 0, no `E` line; 17,332 engine steps (694 prefill, 466 mixed, 16,172 decode) |
| `device_test accounting` (new case, `local` pair, 3 runs) | `ok the_stall_report_gives_a_queues_chain_accounting`, 37 checks; skipped without a card |
| `device_test chain` / `record` / `ring` (`local` pair, 2 runs) | 4,256 checks over 5 cases / 16,915 over 2 / 1,528 over 3, all ok |
| card-less | `hyv4flash_arch_test` 31,947 checks ok; `sched_test` ok; `r4d_emu_hy4` 1,362 cases 0 failures; `r4d_emu_norm`, `r4d_emu_pick` (80), `r4d_emu_escale` (32), `r4d_emu_row` 0 failures; `tests/backlog_stamps_test.py` 10 tests ok |

## Not measured

- The hang on the reporter's own machine: no server log or queue dump of it was posted, and the reporter's base
  (`b56a27f4`) already had the half-lap acquire, the host-write flush and their device tests.
- The entry invalidate and the three wait switches separately: the change is not taken, so its cost was measured
  as a whole.
- iks-B's Engram I/O error with the machine to itself (no other server, memory free), which would say whether it
  needs memory pressure.
