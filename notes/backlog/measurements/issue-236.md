# Issue #236: a `prefix-resume: exact` hit answers as the cold run does

Bug fix, not a backlog item. The bar, from the review: on one machine, the issue's resumed-against-cold greedy
comparison is byte-identical after and not before, with whole-chunk hits at 12,288 and 20,480 tokens and drafting
on; and the resumed turn's time to first token is not worse. Served runs, not kernel stamps. Measured by the
change's author (adamczhang). For the merge, another measurer posts the `## Measurement` comment.

- before = `c31cd19cf0ed4f9067f2755fb2c6cd87bde22ad5`, the upstream parent of this branch's merge (#323 fixed);
  after = code head `02054f6b1a7e1635ca23e1e524b760e8e2880b78`. Each was built in its own tree, with no local changes.
- One Radeon AI PRO R9700 (gfx1201, 32 GB), HIP 2 (PCI `09:00.0`), TP1, power cap 270 W. Ryzen 9 9950X3D, 121 GiB.
  ROCm core 10.1.0 (`/opt/rocm/core-10.1`), kernel 7.0.0-44. Link 32 GT/s x16 throughout, and AER stayed at 0
  correctable, 0 nonfatal, 0 fatal before and after every run.
- Model `hy4-flash-iqr-v4-u3.rad`. The config is the reporter's settings, where TP1 allows them:
  - `placement: expert_tiered`, `host-pool-mib: 52000`;
  - `max-model-len: 32768`, `max-num-seqs: 4`, `max-num-batched-tokens: 8192`;
  - `prefill-span: 2`, `checkpoint-interval: 4096`, `kv-cache-dtype: bf16`, `prefix-cache-mib: 8192`;
  - drafting on: `num-speculative-tokens: 5`, `draft-depth: adaptive`, `draft-verify: traversal`, `kernels: libr4d,libavx,spechub`;
  - `prefix-resume` at its default, `exact`.
- The harness has six conversations a set. Each has:
  - a long first user turn (source files of the before tree, a fixed seed per conversation), an assistant reply, a
    second user turn, another reply, and a new question;
  - requests at temperature 0 with `ignore_eos`, `reasoning_effort: none` and `top_logprobs: 20`.
  - The resumed reply has 24 tokens. `hash` is the sha256 of the text and logprobs.

  The two sets:
  - **Set A**: first turns of 10,980 to 14,451 tokens, whole conversations of 19,533 to 24,026. The resumed turn
    hits 12,288 (c0, c2, c3) or 4,096 (c1, c4, c5).
  - **Set B**: first turns of 19,857 to 26,208 tokens, whole conversations of 24,684 to 31,461. The resumed turn hits
    20,480 (c0, c2, c3, c5) or 12,288 (c1, c4).
- Modes, one fresh server a run:
  - *cold*: the whole conversation alone;
  - *resumed*: each conversation's first turn (8 tokens), then the whole conversation, which hits the first turn's
    cache.
- Order:
  - cold A-after, B-after, A-before, B-before;
  - then resumed, interleaved, A B B A A B (A = before, B = after), with set A and set B back to back in each slot;
  - then a TTFT-only interleave on set B, the same order, with the resumed request at `max_tokens: 1`.

## Byte-identity: resumed against cold, the same build

| set | build | run | same as cold | hits (first turn -> cached tokens) |
|---|---|---|---|---|
| A | before | 3 runs | 1/6 each (c0 only) | c0 14451->12288, c1 10980->4096, c2 12962->12288, c3 13877->12288, c4 11736->4096, c5 12202->4096 |
| A | after | 3 runs | **6/6 each** | the same hits |
| B | before | 3 runs | 1/6 each (c0 only) | c0 25594->20480, c1 20141->12288, c2 24539->20480, c3 26208->20480, c4 19857->12288, c5 23120->20480 |
| B | after | 3 runs | **6/6 each** | the same hits |

- **Before,** only the first conversation a server sees resumes exactly, at 4,096, 12,288 and 20,480 alike. That
  is the issue's "only some of the resumed turns matched". The rest differ from the cold run from the first
  logprob on.
- **After,** every resumed reply is the cold reply, byte for byte, at every hit length, in all 3 runs a side.
- **Cold is unchanged:** each set's cold run hashes the same before and after (6/6, both sets). Set A's
  cold hashes are also those of this morning's runs at `0ce4a139`.
- Hit lengths are the same before and after: the fix changes which blocks a hit may adopt, not how deep a
  conversation's own next turn resumes.
- Resumed runs are deterministic on both builds: the 3 runs a side hash the same.

## Resume-turn time to first token

`ttft` is the server's own figure per request, summed over the six resumed (hit) turns of a run.

| set | resumed reply | before (3 runs) | after (3 runs) | after / before |
|---|---|---|---|---|
| A | 24 tokens | 36.560 36.610 36.590 (mean 36.587) | 36.540 36.540 36.540 (mean 36.540) | 0.9987 |
| B | 24 tokens | 31.530 31.520 31.520 (mean 31.523) | 31.980 31.970 31.980 (mean 31.977) | 1.0144 |
| B | 1 token | 31.120 31.120 31.110 (mean 31.117) | 30.920 30.910 30.910 (mean 30.913) | 0.9935 |

Per conversation, the mean of 3 runs, in seconds:

```
set A, 24 tokens   before  c0 5.080  c1 6.650  c2 5.137  c3 5.147  c4 6.667  c5 7.907
                   after   c0 5.073  c1 6.650  c2 5.127  c3 5.130  c4 6.667  c5 7.893
set B, 24 tokens   before  c0 5.010  c1 6.840  c2 4.930  c3 5.120  c4 6.060  c5 3.563
                   after   c0 5.010  c1 7.067  c2 4.920  c3 5.120  c4 6.300  c5 3.560
set B, 1 token     before  c0 4.87   c1 6.82   c2 5.00   c3 4.90   c4 6.09   c5 3.43
                   after   c0 4.87   c1 6.83   c2 5.00   c3 4.91   c4 6.10   c5 3.21
```

- **The resume itself is not slower.** Set A is level. At one token, set B's resumed turns take the same time
  after as before, or less: c1/c4 +0.01 s, c5 -0.22 s. The step plan of each resumed turn is the same on both
  builds (`RADIANCE_LOG_STEPS`): the same chunks from the same hit, with `ck=` on the first.
- **Set B's +1.4% at 24 tokens is c1 and c4 alone** (+0.23 s each). Those are the two turns whose reply before
  the fix is not the cold run's. TTFT is stamped when the first token reaches the reply's sink. With drafting on,
  that point depends on the draft and verify steps that follow, which depend on the logits:
  - before: 13 decode steps for c1's 24 tokens; after (the cold reply): 11;
  - c1's whole request time is level, 8.44 s before and 8.46 s after.

  With one token there is nothing after the first, and the gap goes.
- First turns (no hit, the same cold prefill on both builds): set A 35.693 -> 35.670 s, set B 59.943 -> 59.977 s
  (means of 3), level.

## Verdict

The issue's own comparison is now byte-identical at every hit length it reported (12,288 and 20,480), and at
4,096, with drafting on: 6/6 in 3 of 3 runs, against 1/6 before. Cold output is unchanged. The resumed turn's
prefill costs the same: level in set A, 0.9935 at one token in set B. Set B's 24-token TTFT moves only where the
before build answered differently from a cold run.

Raw: `out/issue-236/runs/r2-*` (JSON replies, server logs with step lines), `sum236.out`, harness `rep236.py`,
`arm.sh`, `gpu2.sh`, `gpu3.sh`, `gpu4.sh` (in the take's working folder, not in this tree).
