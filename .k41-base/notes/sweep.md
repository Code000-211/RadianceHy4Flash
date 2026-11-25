# Hy4-flash scheduler sweep vs. arrival stalls (2026-11-22)

Snapshot binary, u3 container, TP2, 210 W, base work/u3/coupled03.yaml; BetterBench --quick --complete --workload --prefill
--workload-streams 2 4 --workload-duration 90. Results: bbench/res/sweep-<tag>.{json,html}, logs work/sweep/bb-<tag>.log.
Table and recommendation: docs/ARCHITECTURES.md Hy4 "TP2 recipe" (stalls while a prompt arrives).
Tags: auto b1024 b512 b256 i0 i2 ia2 b1024ia2 m300 m150, repeats b512r m300r (repeat within 10 ms of stall, 1 s of TTFT).
Single runs per depth are 1-2 arrivals; treat TTFT per depth as coarse. Cold prefill identical in all runs.
Best: b512 (balanced), m300 (no warning), HTML: /root/work/claude-scratch/bbench/res/sweep-b512.html

## m300 / m150 re-taken after the stager floor (2026-11-23, S-05)

The learned cap of `mixed-step-target-ms` used to push its chunks under the 1024-row stager step (224-368 rows in `m150`),
where a chunk reads its routed experts synchronously. The scheduler now bottoms that cap out at the stager floor, so the two
mixed-step tags above cannot be re-run as they were: BetterBench's `work/sweep` is not in this tree. The clamp's two rows were
therefore re-taken with the repository's own lane, `local` (TP2, u3, 210 W, `scripts/backlog/nospec.yaml` plus the one line),
phases `c1,c8,pf6,pf20,stall1,stall4` under `RADIANCE_LOG_STEPS=1`, one to three runs a side, alternating, one server a run, on
the tree the change lands on (`e714c086`, after the mixed step is issued ahead) -- see docs/ARCHITECTURES.md `TP2 recipe` (the
re-taken table). In short, on that lane's 5.7K arrival: `mixed-step-target-ms: 150` cut the arrival into 20-28 chunks of 64-256
rows before the clamp and 5 of 704-1984 after, and its TTFT went 7.25 -> 3.44 s (7.10 -> 3.83 s with four decoders) with the
decoders' worst stall 0.55 -> 0.46 s at one and 0.19 -> 0.41 s at four; `300 ms` went 4.70 -> 3.41 s and 5.13 -> 3.87 s. Both
settings now land on the 1024-row chunk, the row the explicit `busy-prefill-chunk: 1024` gives (3.70 / 3.80 s and 0.38 / 0.39 s
in its two runs), where before they were under the explicit `busy-prefill-chunk: 512` one (5.24 / 5.83 s and 0.25-0.38 s), and
`auto` (flag off) is 2.40-2.85 s at 0.58-0.78 s of stall. The coupled base config above
shows the same chunk sizes, but its drafting varies ~20 % a run on a contended host, so its TTFT rows are not quoted.
