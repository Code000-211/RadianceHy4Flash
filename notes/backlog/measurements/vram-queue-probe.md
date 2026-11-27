# #314: the VRAM budget's hardware-queue probe read 175.37 instead of 35.37 MiB in ~1 start of 8

Not a backlog item: issue #314, a bug in `Engine::plan` (`core/engine_bringup.cpp`). Found in the W2 work
(`W2-proposals.md` §4 on `w2-explore`).

- Before: `49934f24c` (the base branch). After: the fix commit on `vram-queue-budget` (this note's parent).
- Machine: the local lane (inference host, TP2 pair of R9700 gfx1201, cap 210 W read back on both cards),
  u3 container, `nospec.yaml` / `coupled03.yaml` as the lane renders them, 2026-11-26.
- Build: CARDS.md §1.1 (ROCm clang++, `/opt/rocm-repeatable` HIP compiler, VOPD repeatable), targets `radiance
  libr4d libref libavx libquant spechub hyv4flash`.

## 1. What went wrong

The probe opened two streams and took the drop in free VRAM over the second. The first stream normally takes
a queue the kernel inits kept (AQL backend spares), so exactly one fresh `hsa_queue_create` was measured.

A VRAM sampler beside the server (every change of `mem_info_vram_used`, 0.1 ms) shows what a card's first
fresh queue costs: +29.34 MiB (the queue), +140.00 MiB (a one-time allocation of the runtime), +29.34 MiB (a
second queue of the runtime's), +6.00 MiB (the engine's argument ring); every later queue +29.34 +6.00 =
35.34 MiB. The 140 MiB is made with the first queue that **finds room** for it. When the engine opened its
first queues while the card was still nearly full, it was skipped and landed on the next fresh queue, which
was the probe's:

```
good  probe: a=32317.84 b=32282.52 MiB free  -> 35.32 MiB a queue
bad   probe: a=32457.84 b=32282.52 MiB free  -> 175.37 MiB a queue  (b, and `already held`, equal to a good start)
```

The card is full at our start when the lane hands it on before the previous process's VRAM has been
released: a run that took the lock right after another agent's server read 32,252 MiB used at its first
reading, and measured 175.37 MiB (`HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0` in that run: neither it nor
`HSA_SCRATCH_SINGLE_LIMIT_ASYNC` / `HSA_SCRATCH_MEM` changes the 140 MiB, so it is not queue scratch). Over
all lane trees of 11-25/26: 27 of 228 starts (`runs/local/*/server.log` of every worktree), clustered when
the lane was busiest.

## 2. Reproduction

Without the trigger, 0 of 49 back-to-back starts were bad (`nospec.yaml` 24, `coupled03.yaml` 25). With it --
a helper holding 31.5 GB on both cards and killed (SIGKILL) `d` ms after the server starts, so the server's
first queues open into a full card -- on the base:

| d | 300 ms | 800 ms | 1300 ms | 1800 ms |
|---|---|---|---|---|
| queue measured (base) | 35.37, 35.37 | 35.37, 35.36 | **175.37, 175.38** | **175.28** |
| slab slots (base) | 12,823 | 12,823 | 12,490 | 12,490 |

## 3. Fix

The probe takes the kept queues first, then opens fresh queues until two have been measured and takes the
smaller drop (`vram_queue_cost` in `core/mem/vram_budget.cpp`, test
`the_queue_probe_takes_the_smaller_fresh_drop` in `tests/mem_test.cpp`). The one-time allocation can land on
at most one of them. The probe used to leave two kept queues inside the `free` reading; the one more it can
leave now is one of the engine's own streams' queues (they take kept queues first, `n_queues` charges them),
so it is added back once and a normal start's split is unchanged.

## 4. Verification (after)

- 30 plain `coupled03.yaml` starts (server up to the budget and mover lines, then stopped): 30 of 30 at
  35.36-35.37 MiB a queue, 12,823 slab slots in every one (`already held` 341.54-341.61 MiB), as a good start
  of the base.
- The trigger (§2) on the fix: d = 1300 ms 6 of 6 at 35.37 MiB, 12,823 slots; d = 1800 ms 4 of 4 at 35.37 MiB,
  12,780 slots (`already held` 400.47 MiB: the runtime opened two more queues of its own into the full card,
  memory that is really on the card, read correctly).
- Full runs (`run.sh`), greedy byte-identical to the base's runs on the same commit:

| run | greedy sha256 (4 texts) | c1 ms/step | c8 tok/s | c8 hit | c8 misses/step | slots |
|---|---|---|---|---|---|---|
| `qb-ns1` (fix, `nospec.yaml greedy,c1`) | d6da4fb3 0699709c e7a23c5d 8f60e420 | 13.128 | | | | 13,183 |
| `w2/p1-off` `p2-off` `p3-off` (base) | d6da4fb3 0699709c e7a23c5d 8f60e420 | 13.092 / 13.116 / 13.136 | | | | 13,183 |
| `qb-c81` (fix, `coupled03.yaml greedy,c8`) | 5b7a3975 f55fe020 0075e702 8f60e420 | | 381.1 | 0.985 | 118.5 | 12,823 |
| `grid-base/ac8-off` `bc8-off` (base) | 5b7a3975 f55fe020 0075e702 8f60e420 | | 378.2 / 368.1 | 0.986 / 0.986 | 114.3 / 113.5 | 12,823 |

```
qb-ns1 {"name": "greedy", "sha256": ["d6da4fb33eed29de", "0699709c4be29115", "e7a23c5d9a2f3a45", "8f60e42028d1d7db"], "gen": [256, 256, 102, 64]}
qb-ns1 {"name": "c1", "clients": 1, "gen": 256, "agg_tps": 74.256, "decode_steps": 255.0, "ms_step": 13.128, "hit": 0.99, "miss_units_per_step": 6.757}
qb-c81 {"name": "greedy", "sha256": ["5b7a3975defae9d6", "f55fe02071d85330", "0075e702dfdf7353", "8f60e42028d1d7db"], "gen": [256, 256, 103, 64]}
qb-c81 {"name": "c8", "clients": 8, "gen": 2048, "agg_tps": 381.125, "decode_steps": 100.0, "ms_step": null, "hit": 0.985, "miss_units_per_step": 118.53}
```

Effect of a bad start, for A/Bs read before this fix (`W2-proposals.md` §4): +1.8 % a no-spec c1 step,
~25 % more misses at DFly2 c8. An A/B arm whose `server.log` says `a card at 175.3x MiB` is such a start.
