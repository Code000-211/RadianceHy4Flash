# Issue #282: the pipelined prefill's junction pieces write the G-3b codes

Bug fix, not a backlog item. The bar is: with `RADIANCE_HY4_PIPE=1`, long-prompt greedy output is the same from run to
run and on topic; pipe on stays close to pipe off; the pipe's prefill gain survives; end to end is not worse. The rows
below are served runs, not kernel stamps. Measured by the change's author (adamczhang). For the merge, another
measurer posts the `## Measurement` comment.

- before = merge base `7b5bd87e068fcdb6e47a4383339ded7f87669801`; after = code commit
  `6c53abae86a283c658fae0f55d85110b16cba11a`. Each was built in its own tree (no local changes, `diff:` empty in every
  header line below).
- 4 x Radeon AI PRO R9700 (gfx1201, 32 GB), TP4, power cap read back 270 W on every card before every run. ROCm
  core 10.1.0 (`/opt/rocm/core-10.1`, its hipcc and clang, `-DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201`), kernel
  7.0.0-44. Container `hy4-flash-iqr-v4-u4.rad`, `ngram-placement: ram`.
- Config: `scripts/backlog/nospec.yaml` at TP4 with `max-num-batched-tokens: 8192`, `checkpoint-interval: 4096`,
  `prefill-span: 2`, `max-model-len: 65536`, `max-num-seqs: 8`, `prefix-resume: nearest`, `busy-prefill-chunk: auto`,
  `kernels: libr4d,libavx`, the whole u4 expert plane resident. Row D is the same with
  `max-num-batched-tokens: 2048`, `checkpoint-interval: 1984`, `prefix-cache-mib: 16384` (no `prefill-span`).
- One fresh server a run, order A1 B1 B2 A2 A3 B3 (A = before, B = after, both `RADIANCE_HY4_PIPE=1`), then C1-C3
  (after, `RADIANCE_HY4_PIPE=0`) and D1-D2 (after, pipe on, at the 1984 config). Then a decomposition at
  `RADIANCE_HY4_FUSE_JUNCTION_Q=0`, E1 F1 G1 F2 E2 G2 (E = before pipe on, F = after pipe on, G = after pipe off).
- Each run, in this order:
  - greedy with logprobs (T 0, `ignore_eos`, 64 tokens, `top_logprobs 5`): prompts of 3450 / 14544 / 51970 tokens
    (L3k / L12k / L58k: source files of the before tree, a fixed seed, and a question), then a 38-token one (S).
    `hash` is the sha256 of the text and logprobs.
  - The same server again for S, L12k, L3k: a prefix-cache resume.
  - `pf`: 4 cold ~11K-token prompts (9564-11632 tokens, distinct seeds, nothing cached), 1 output token, streamed;
    tok/s = prompt tokens / time to the first token. The first is a warm-up; the last three count.
  - `c1`: `scripts/backlog/probe.py` phase `c1`.
- AER and link of card 4 and its switch logged before and after every run (`pre` / `post`). No fatal or nonfatal
  error, and the link stayed at 32 GT/s throughout. The switch's corrected count rose by about 2 a run.

## Determinism and parity (greedy hash per prompt)

| run | L3k | L12k | L58k | S | answers |
|---|---|---|---|---|---|
| A1 A2 A3 (before, pipe on) | fbe59496f830 | fbe1c2f57c64 | f9bb843eb52f | f04dbb1c43eb | off topic: `retches the first time I used the word "a"`, `#  \|  \|  \|` |
| B1 B2 B3 (after, pipe on) | 87bbab84c152 | e6e2ce6e461b | 0ce8dae3268a | f04dbb1c43eb | on topic, the same opening as C |
| C1 C2 C3 (after, pipe off) | cbe18cde8d5b | e30ebffdf5d7 | a19f5573a1f5 | f04dbb1c43eb | on topic |
| D1 D2 (after, pipe on, 1984 config) | 3a37aadc8bb2 | 52740cfe8add | 9e95e45e5fac | f04dbb1c43eb | on topic, no cut |

- **Determinism:** B is byte for byte the same in 3 of 3 runs, and so is the resumed L3k. A was also the same in 3 of 3 here,
  because every run sent the same requests in the same order. The before tree's pipe-on answer depends on the
  requests that came before: the resumed L3k / L12k differ from the fresh ones (`df11636a9631` / `133b9f0b4d49`).
  In earlier runs on the before tree at the 1984 config (2 fresh servers a side, S, L12k, L3k in that order), the
  pipe-on answers differed from token 0 from those in the usual order, and pipe off did not. That is the "different on
  every run" of the issue.
- **Pipe on against pipe off (B against C):** not byte-identical, as the README says of the cut ("prefill with the
  cut, bit for bit: not the step run whole"). The first logprob differs in its digits. The greedy tokens are the same
  for the first 37 (L3k) and 14 (L12k) tokens, and all 64 of L58k, and the answers open with the same sentence. S
  (no cut) is identical. B's hashes are exactly what the before tree gives with the pipe on and
  `RADIANCE_HY4_FUSE_JUNCTION_Q=0` (no codes anywhere): the remaining difference is the cut's, not the codes'.
- **The gate (D):** at 1984 rows the step no longer cuts (no `pipelined prefill` line in the server log). L3k, L12k
  and S are byte-identical to the before tree with the pipe off at the same config. Those hashes, from 5 earlier
  fresh servers of the before tree, are L3k `3a37aadc8bb2`, L12k `52740cfe8add` and S `f04dbb1c43eb`.

## Prefill (pf, tok/s, 9 measured prompts a side) and c1

| arm | pf mean | pf min / max | c1 agg tok/s (3 runs) |
|---|--:|--:|---|
| A before, pipe on (wrong output) | 5582.7 | 5554.1 / 5616.6 | 119.125 118.878 118.876 |
| B after, pipe on | 5290.0 | 5260.0 / 5316.0 | 118.823 118.924 118.824 |
| C after, pipe off | 4849.1 | 4840.2 / 4861.7 | 118.863 118.912 118.906 |
| E before, pipe on, codes off (6) | 5304.9 | 5258.2 / 5340.7 | |
| F after, pipe on, codes off (6) | 5300.3 | 5260.6 / 5329.7 | |
| G after, pipe off, codes off (6) | 4844.4 | 4834.7 / 4857.9 | |

- The pipe keeps its gain: B is +9.1% over C, and F +9.4% over G.
- A's higher number came from the bug. With the codes off on both trees, before and after are level (E against F,
  -0.1%): naming the junctions' buffers concurrent costs nothing. Writing the codes in the pieces costs nothing
  either (B against F, inside noise). Arena at 4096 tokens: 3.77 GiB before, 3.81 GiB after, pipe on.
- c1 is level (118.96 against 118.86 mean, -0.1%).

## Verdict per row

- Long-prompt greedy output, pipe on: before is wrong and depends on history; after is deterministic (3 of 3) and on
  topic. **Fixed.**
- Pipe on against pipe off: close (same greedy prefix, same opening sentence), not bit for bit, as documented of the
  cut. **Holds.**
- Prefill with the pipe: +9.1% over pipe off after the fix. The before number was the bug's. **Holds.**
- End to end c1: level. **Not worse.**

## Raw lines

[A1]
== A1 (1/11) 06:51:20 7b5bd87e P-12: 64- / 128-row arena level diff: cfg=tp4-c4096.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=88 fatal=0 nonfatal=0
L3k prompt=3450 hash=fbe59496f830 secs=1.85 | retches the first time I used the word "a" in th
L12k prompt=14544 hash=fbe1c2f57c64 secs=3.16 |  retches the "bag bag" (in the sense of the bag)
L58k prompt=51970 hash=f9bb843eb52f secs=10.31 | #  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=133b9f0b4d49 secs=1.01 |  (= "invariance-extension" in the sense of "inva
L3k prompt=3450 hash=df11636a9631 secs=1.11 |  retan (and thus in particular) a "included" in 
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.0538, 'tps': 5663.6}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.0404, 'tps': 5616.6}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.0497, 'tps': 5609.0}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.7106, 'tps': 5591.1}
c1 agg_tps=119.125 sum_req_dec_tps=121.196 ttft=[0.045] gen=256
hyv4flash: pipelined prefill, 1728 + 1722 rows
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=90 fatal=0 nonfatal=0
[B1]
== B1 (2/11) 06:52:19 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c4096.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=90 fatal=0 nonfatal=0
L3k prompt=3450 hash=87bbab84c152 secs=1.43 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=e6e2ce6e461b secs=3.33 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=0ce8dae3268a secs=10.72 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=c4a2cb37eca0 secs=1.06 | Neither file defines a pipelined prefill; `abi/r
L3k prompt=3450 hash=87bbab84c152 secs=1.16 | None of the shown code defines pipelined prefill
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.173, 'tps': 5353.0}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.1557, 'tps': 5316.0}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.1656, 'tps': 5308.9}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.8158, 'tps': 5267.0}
c1 agg_tps=118.823 sum_req_dec_tps=120.928 ttft=[0.046] gen=256
hyv4flash: pipelined prefill, 1728 + 1722 rows
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=97 fatal=0 nonfatal=0
[B2]
== B2 (3/11) 06:53:19 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c4096.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=97 fatal=0 nonfatal=0
L3k prompt=3450 hash=87bbab84c152 secs=1.23 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=e6e2ce6e461b secs=3.34 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=0ce8dae3268a secs=10.73 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=c4a2cb37eca0 secs=1.06 | Neither file defines a pipelined prefill; `abi/r
L3k prompt=3450 hash=87bbab84c152 secs=1.16 | None of the shown code defines pipelined prefill
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.1768, 'tps': 5343.5}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.1638, 'tps': 5296.3}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.1675, 'tps': 5304.2}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.8183, 'tps': 5260.0}
c1 agg_tps=118.924 sum_req_dec_tps=121.044 ttft=[0.046] gen=256
hyv4flash: pipelined prefill, 1728 + 1722 rows
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=103 fatal=0 nonfatal=0
[A2]
== A2 (4/11) 06:54:18 7b5bd87e P-12: 64- / 128-row arena level diff: cfg=tp4-c4096.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=103 fatal=0 nonfatal=0
L3k prompt=3450 hash=fbe59496f830 secs=1.16 | retches the first time I used the word "a" in th
L12k prompt=14544 hash=fbe1c2f57c64 secs=3.21 |  retches the "bag bag" (in the sense of the bag)
L58k prompt=51970 hash=f9bb843eb52f secs=10.38 | #  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=133b9f0b4d49 secs=1.02 |  (= "invariance-extension" in the sense of "inva
L3k prompt=3450 hash=df11636a9631 secs=1.12 |  retan (and thus in particular) a "included" in 
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.0671, 'tps': 5627.1}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.0547, 'tps': 5577.3}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.0611, 'tps': 5578.0}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.722, 'tps': 5554.1}
c1 agg_tps=118.878 sum_req_dec_tps=120.970 ttft=[0.045] gen=256
hyv4flash: pipelined prefill, 1728 + 1722 rows
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=104 fatal=0 nonfatal=0
[A3]
== A3 (5/11) 06:55:19 7b5bd87e P-12: 64- / 128-row arena level diff: cfg=tp4-c4096.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=104 fatal=0 nonfatal=0
L3k prompt=3450 hash=fbe59496f830 secs=1.3 | retches the first time I used the word "a" in th
L12k prompt=14544 hash=fbe1c2f57c64 secs=3.18 |  retches the "bag bag" (in the sense of the bag)
L58k prompt=51970 hash=f9bb843eb52f secs=10.38 | #  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=133b9f0b4d49 secs=1.02 |  (= "invariance-extension" in the sense of "inva
L3k prompt=3450 hash=df11636a9631 secs=1.12 |  retan (and thus in particular) a "included" in 
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.0684, 'tps': 5623.7}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.0529, 'tps': 5582.3}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.0614, 'tps': 5577.4}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.7205, 'tps': 5558.9}
c1 agg_tps=118.876 sum_req_dec_tps=120.923 ttft=[0.044] gen=256
hyv4flash: pipelined prefill, 1728 + 1722 rows
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=109 fatal=0 nonfatal=0
[B3]
== B3 (6/11) 06:56:18 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c4096.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=109 fatal=0 nonfatal=0
L3k prompt=3450 hash=87bbab84c152 secs=1.4 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=e6e2ce6e461b secs=3.36 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=0ce8dae3268a secs=10.74 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=c4a2cb37eca0 secs=1.06 | Neither file defines a pipelined prefill; `abi/r
L3k prompt=3450 hash=87bbab84c152 secs=1.16 | None of the shown code defines pipelined prefill
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.1779, 'tps': 5340.9}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.1641, 'tps': 5295.5}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.1704, 'tps': 5297.3}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.8165, 'tps': 5265.2}
c1 agg_tps=118.824 sum_req_dec_tps=120.908 ttft=[0.045] gen=256
hyv4flash: pipelined prefill, 1728 + 1722 rows
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=111 fatal=0 nonfatal=0
[C1]
== C1 (7/11) 06:57:17 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c4096.yaml PIPE=0 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=111 fatal=0 nonfatal=0
L3k prompt=3450 hash=cbe18cde8d5b secs=1.28 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=e30ebffdf5d7 secs=3.59 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=a19f5573a1f5 secs=11.87 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=7b42a21ee4d9 secs=1.09 | Neither file defines a pipelined prefill. The sh
L3k prompt=3450 hash=cbe18cde8d5b secs=1.23 | None of the shown code defines pipelined prefill
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.3852, 'tps': 4876.8}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.3639, 'tps': 4847.9}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.3732, 'tps': 4844.5}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.9689, 'tps': 4857.5}
c1 agg_tps=118.863 sum_req_dec_tps=120.922 ttft=[0.045] gen=256
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=118 fatal=0 nonfatal=0
[C2]
== C2 (8/11) 06:58:19 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c4096.yaml PIPE=0 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=118 fatal=0 nonfatal=0
L3k prompt=3450 hash=cbe18cde8d5b secs=1.27 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=e30ebffdf5d7 secs=3.6 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=a19f5573a1f5 secs=11.71 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=7b42a21ee4d9 secs=1.09 | Neither file defines a pipelined prefill. The sh
L3k prompt=3450 hash=cbe18cde8d5b secs=1.24 | None of the shown code defines pipelined prefill
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.3825, 'tps': 4882.2}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.3638, 'tps': 4848.1}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.3753, 'tps': 4840.2}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.9672, 'tps': 4861.7}
c1 agg_tps=118.912 sum_req_dec_tps=121.009 ttft=[0.045] gen=256
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=125 fatal=0 nonfatal=0
[C3]
== C3 (9/11) 06:59:21 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c4096.yaml PIPE=0 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=125 fatal=0 nonfatal=0
L3k prompt=3450 hash=cbe18cde8d5b secs=1.28 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=e30ebffdf5d7 secs=3.59 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=a19f5573a1f5 secs=11.7 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.56 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=7b42a21ee4d9 secs=1.09 | Neither file defines a pipelined prefill. The sh
L3k prompt=3450 hash=cbe18cde8d5b secs=1.24 | None of the shown code defines pipelined prefill
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.3826, 'tps': 4882.1}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.367, 'tps': 4841.6}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.3749, 'tps': 4841.1}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.9681, 'tps': 4859.4}
c1 agg_tps=118.906 sum_req_dec_tps=121.023 ttft=[0.046] gen=256
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=134 fatal=0 nonfatal=0
[D1]
== D1 (10/11) 07:00:23 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c1984.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=134 fatal=0 nonfatal=0
L3k prompt=3450 hash=3a37aadc8bb2 secs=1.3 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=52740cfe8add secs=3.79 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=9e95e45e5fac secs=12.47 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.58 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=b36d4debac72 secs=0.75 | Neither file defines a pipelined prefill. The sh
L3k prompt=3450 hash=3a37aadc8bb2 secs=0.56 | None of the shown code defines pipelined prefill
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=139 fatal=0 nonfatal=0
[D2]
== D2 (11/11) 07:01:15 6c53abae hyv4flash: the pipelined prefill's junction pieces  diff: cfg=tp4-c1984.yaml PIPE=1 caps 270W 270W 270W 270W 
pre aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=139 fatal=0 nonfatal=0
L3k prompt=3450 hash=3a37aadc8bb2 secs=1.3 | None of the shown code defines pipelined prefill
L12k prompt=14544 hash=52740cfe8add secs=3.79 | Neither file defines a pipelined prefill. The sh
L58k prompt=51970 hash=9e95e45e5fac secs=12.47 | The file defining the pipelined prefill is `libr
S prompt=38 hash=f04dbb1c43eb secs=0.58 | A KV cache stores the key and value vectors comp
S prompt=38 hash=f04dbb1c43eb secs=0.55 | A KV cache stores the key and value vectors comp
L12k prompt=14544 hash=b36d4debac72 secs=0.75 | Neither file defines a pipelined prefill. The sh
L3k prompt=3450 hash=3a37aadc8bb2 secs=0.56 | None of the shown code defines pipelined prefill
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=146 fatal=0 nonfatal=0
[E1]
== E1 07:02:55 /run/media/adam-zhang/LAB/Projects/RadianceHy4Flash/.worktrees/base-7b5bd87e/build PIPE=1 JQ=0 pre aer co
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.1988, 'tps': 5290.2}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.1458, 'tps': 5340.7}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.1581, 'tps': 5327.3}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.8098, 'tps': 5284.4}
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=148 fatal=0 nonfatal=0
[F1]
== F1 07:03:31 /run/media/adam-zhang/LAB/Projects/RadianceHy4Flash/.worktrees/issue-282/build PIPE=1 JQ=0 pre aer cor=0 
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.2007, 'tps': 5285.6}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.1502, 'tps': 5329.7}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.1615, 'tps': 5319.0}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.8134, 'tps': 5274.0}
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=154 fatal=0 nonfatal=0
[G1]
== G1 07:04:08 /run/media/adam-zhang/LAB/Projects/RadianceHy4Flash/.worktrees/issue-282/build PIPE=0 JQ=0 pre aer cor=0 
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.4164, 'tps': 4813.8}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.3617, 'tps': 4852.3}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.3773, 'tps': 4836.1}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.9688, 'tps': 4857.9}
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=160 fatal=0 nonfatal=0
[F2]
== F2 07:04:44 /run/media/adam-zhang/LAB/Projects/RadianceHy4Flash/.worktrees/issue-282/build PIPE=1 JQ=0 pre aer cor=0 
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.2071, 'tps': 5270.2}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.1558, 'tps': 5315.8}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.168, 'tps': 5302.9}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.8181, 'tps': 5260.6}
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=162 fatal=0 nonfatal=0
[E2]
== E2 07:05:20 /run/media/adam-zhang/LAB/Projects/RadianceHy4Flash/.worktrees/base-7b5bd87e/build PIPE=1 JQ=0 pre aer co
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.208, 'tps': 5268.2}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.1542, 'tps': 5319.9}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.1697, 'tps': 5298.8}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.8189, 'tps': 5258.2}
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=164 fatal=0 nonfatal=0
[G2]
== G2 07:05:55 /run/media/adam-zhang/LAB/Projects/RadianceHy4Flash/.worktrees/issue-282/build PIPE=0 JQ=0 pre aer cor=0 
pf {'seed': 0, 'prompt_tokens': 11632, 'ttft': 2.4167, 'tps': 4813.2}
pf {'seed': 1, 'prompt_tokens': 11460, 'ttft': 2.3692, 'tps': 4837.1}
pf {'seed': 2, 'prompt_tokens': 11497, 'ttft': 2.378, 'tps': 4834.7}
pf {'seed': 3, 'prompt_tokens': 9564, 'ttft': 1.9728, 'tps': 4848.0}
post aer cor=0 fatal=0 nonfatal=0 link="32.0 GT/s PCIe" x4 card-link="32.0 GT/s PCIe" x16 sw1e cor=168 fatal=0 nonfatal=0
