# Issue #2: no link hold for lane-1 collectives (PR #205)

Bug fix, not a backlog item: the bar is "the hang is gone, greedy text unchanged, end to end not worse", and the
rows below are served runs (`scripts/backlog/probe.py`), not kernel stamps. Measured by the reviewing maintainer;
the change is Pearcenator's.

- before = merge base `165daed7cf3a0803e1dfd08b888589f61b435605`; after = code commit
  `ade793ad8e95db01ff4f975eb4eb785f7fd54fb4`. The two engines differ in one condition
  (`core/runtime/issue.cpp`, `&& lane_ == 0` on `hold`). On iks the "before" engine is the `ade793a` tree with that
  condition taken out again, built in the same build directory.
- Container `hy4-flash-iqr-v4-u3.rad` on both machines. Power cap read back 210 W on every card before each run.
- TP4: iks, 4 x Radeon AI PRO R9700 (gfx1201), all four through `gpu4`, inside `lane.sh iks-A` (Engram
  `ram=/dev/shm/radiance-ngram`, host pool 16000 MiB, prefix cache 8192 MiB), ROCm SDK clang from
  `_rocm_sdk_devel`, `-DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201`.
- TP2: the `local` lane (2 x R9700, `ngram-placement: ram`, host pool 39000 MiB), kernel 7.0.2-6-pve, AMD clang
  24.0.0git, through `lane.sh local` / `run.sh`.
- Config, both: `scripts/backlog/nospec.yaml` with `max-num-batched-tokens: 8192`, `checkpoint-interval: 4096`,
  `prefill-span: 2`, `placement: expert_tiered`; TP4 adds `tp: 4`, `tp-wire: exact`, `max-model-len: 200000`,
  `max-num-seqs: 16`, `kv-cache-dtype: bf16`. `RADIANCE_HY4_PIPE=1` unless a row says off.
- Phases `greedy,c1,c8,pf6,pf20,idle` with `RAD_IDLE_CHARS=150000`: `idle` sends a 44,596-token conversation, the
  same again (resume) and a 48,419-token cold prompt, the prompts that reach the first 8192-row step.
- Runs interleaved before / after. TP4 has three before and two after: the third after-run was stopped by a restart
  of iks, not by the engine.

## TP4, iks: the hang (before) and its absence (after)

| run | greedy sha256 (3 prompts, 5.7K arrival) | c1 tok/s (ms/step) | c8 agg tok/s | pf6 ttft s | pf20 ttft s | `idle` (44.6K conv / 48.4K cold ttft s) |
|---|---|---|---|---|---|---|
| base-1 | 178d3ddc9ae97371 cdceabcc75992436 d3d2aa2a1091868e fa3ba84b31fba866 | 107.224 (9.186) | 499.821 | 1.151 | 4.568 | **hang**, server exit 70 |
| head-1 | 178d3ddc9ae97371 cdceabcc75992436 d3d2aa2a1091868e fa3ba84b31fba866 | 107.016 (9.194) | 499.061 | 1.155 | 4.578 | 8.412 / 8.523, resume 0.053 |
| base-2 | 178d3ddc9ae97371 cdceabcc75992436 d3d2aa2a1091868e fa3ba84b31fba866 | 106.795 (9.213) | 494.907 | 1.154 | 4.597 | **hang**, server exit 70 |
| head-2 | 178d3ddc9ae97371 cdceabcc75992436 d3d2aa2a1091868e fa3ba84b31fba866 | 106.808 (9.215) | 493.527 | 1.159 | 4.601 | 8.446 / 8.561, resume 0.055 |
| base-3 | 178d3ddc9ae97371 cdceabcc75992436 d3d2aa2a1091868e fa3ba84b31fba866 | 106.760 (9.214) | 499.673 | 1.163 | 4.607 | **hang**, server exit 70 |

Before hangs on the 44,596-token conversation in 3 of 3 runs (4 of 4 with the logging run below) and the process
exits on the stall watchdog; after serves it, its resume and the 48.4K cold prompt in 2 of 2. Greedy text is byte
for byte the same in all five; c1, c8, pf6 and pf20 are level (inside 1%).

The stall dump of a before run (base-2; base-1 is the same to the digit on cards 1-3):

```
E   device 0 queue ...: written 1780729, read 1777647 -> kernel r4dchain_ar_gather_wide<4>
E   device 1 queue ...: written 1775053, read 1775053 -> idle        (cards 2, 3 the same)
E   device 0 queue ...: written 137018, read 136150 -> kernel r4d_ar_twoshot_coded_kernel<4>
E   device 1 queue ...: written 136146, read 136146 -> idle          (cards 2, 3 the same)
E   device 0 DMA lane: ... 213 lists handed over, 173 finished; link gate 4611686018427383401 (held), last hold issued off
E   device 1 DMA lane: ... 173 lists handed over, 172 finished; link gate 4611686018427383403 (held), last hold issued off
E   device 2 DMA lane: ... 173 lists handed over, 172 finished; link gate 4611686018427383403 (held), last hold issued off
E   device 3 DMA lane: ... 173 lists handed over, 172 finished; link gate 4611686018427383403 (held), last hold issued off
```

Cards 1 to 3 have nothing in flight, the host has released, and the gate is odd: a list waits on it for good, and
each of those ranks' hosts waits on that list in a synchronisation. Rank 0 ran on and its collectives wait for peers
that never launch.

## Which holds there are (TP4, before, one run with a line printed per hold)

```
 640 r0 lane 0 ar_gather_4rank              ret 1 (RAD_LINK_ENDED) gated 1
1312 r0 lane 1 ar_twoshot_4rank_exact_lane  ret 1 gated 1
 561 r1 lane 0 ar_gather_4rank              ret 1 gated 1          (r2, r3 the same)
1154 r1 lane 1 ar_twoshot_4rank_exact_lane  ret 1 gated 1
```

Rank 1's last holds before the stall, in issue order (lane, row, numel):

```
1 ar_twoshot_4rank_exact_lane 6291456     <- 2048-row pieces, as in every step before
0 ar_gather_4rank 262144
1 ar_twoshot_4rank_exact_lane 6291456
1 ar_twoshot_4rank_exact_lane 12582912    x6: the first 8192-row step, 4096-row pieces
0 ar_gather_4rank 524288                  <- last hold issued
```

So a staged pipelined pass holds the link from both lanes: lane 1 for each piece's junction reduce, lane 0 for the
router projection's all_gather in the arm that runs beside it, and both rows end their hold in the kernel. The gate
is one word whose parity is the hold. Two holds that overlap in time lose it: lane 0's hold barrier makes an odd
word even, the gather reads it even and does not decrement, the reduce decrements, and the word is odd with nothing
holding. With 2048-row pieces the reduce has landed before the gather starts; in the first 8192-row step it has not.
The overlap on the card is inferred from this order and the gate's state, not stamped; the absence
of the hang with lane 1's hold gone (above) is the check. With the change only lane 0 holds, on one stream, where
issue order is execution order.

## TP2, local: no hang either way; what the lane-1 hold was worth

| run | greedy sha256 (3 prompts) | c1 tok/s (ms/step) | c8 agg tok/s | pf6 ttft s | pf20 ttft s | idle conv / cold ttft s | resume s |
|---|---|---|---|---|---|---|---|
| pipe-b1 (before) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f | 68.5 (14.261) | 284.4 | 1.690 | 6.250 | 10.914 / 12.077 | 0.075 |
| pipe-a1 (after)  | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f | 68.8 (14.224) | 284.4 | 1.771 | 7.031 | 11.455 / 12.607 | 0.076 |
| pipe-b2 (before) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f | 67.6 (14.420) | 281.1 | 1.717 | 6.368 | 10.880 / 11.996 | 0.075 |
| pipe-a2 (after)  | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f | 68.9 (14.183) | 284.7 | 1.787 | 6.470 | 11.353 / 12.485 | 0.076 |
| pipe-b3 (before) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f | 67.5 (14.435) | 280.7 | 1.709 | 6.384 | 10.914 / 11.999 | 0.074 |
| pipe-a3 (after)  | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f | 68.0 (14.361) | 282.6 | 1.810 | 6.660 | 11.365 / 12.555 | 0.074 |
| off-b (before, pipe off) | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f, arrival 717c3b789eee | 70.6 (13.850) | 283.4 | 1.679 | 6.715 | 11.607 / 12.658 | 0.079 |
| off-a (after, pipe off)  | d6da4fb33eed 0699709c4be2 e7a23c5d9a2f, arrival 717c3b789eee | 70.8 (13.808) | 285.8 | 1.674 | 6.713 | 11.598 / 12.646 | 0.075 |

- Two ranks do not hang before or after: the two-rank lane reduce and gather release by barrier, so overlapping
  holds always count the word down an even number of times.
- Pipe off (the default): the two engines are the same path and the numbers and all four greedy texts agree.
- Pipe on: decode is level. The staged prefill is slower without the lane-1 hold, because the reduce now shares the
  link with the stager's lists: median pf6 1.709 -> 1.787 s (+4.6%), pf20 6.368 -> 6.660 s (+4.6%), 44.6K
  conversation 10.914 -> 11.365 s (+4.1%), 48.4K cold 11.999 -> 12.555 s (+4.6%). It stays ahead of pipe off on the
  long prompts (11.6 / 12.65 s).
- With the pipe on, the 5.7K arrival's 64-token text differs from run to run in both arms on this lane (before
  a388b80cb976 / 0a45e03f2ce3 / 80de148914d8, after 321173f56b50 / 420b4016fde1 / 110b678c8e28), so it says nothing
  about the change; the three fixed prompts and, with the pipe off, the arrival agree.

## Not measured

- A third TP4 after-run, and TP4 TTFT before / after on a staged long prompt (before does not finish one).
- Kernel stamps of the lane-1 reduce with and without the hold.
- A gate that survives overlapping holds (a count, or barrier-only release in a two-lane pass), which would give
  lane 1 its 4.6% back at two ranks.
