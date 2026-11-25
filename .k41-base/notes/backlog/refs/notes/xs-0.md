# WP-X0 (expert streaming, diagnostics) — notes

Branch `hy4-xstream` (off claude/peaceful-wright-hhqnhi), measured merged into `hy4-perf4`
(= hy4-perf + hy4-xstream, wt/perf4). Data: `work/xs1/` (run.sh, xs.py, cmp.py, summ.py,
stlay.py, kdiff.py, dranges.py; *.json / *.out / *.server.log / *.trace / *-st.csv).
Real container, TP2, 210 W, Engram in RAM, 256 greedy tokens a request.

## What exists

- `RADIANCE_ROUTE_TRACE=FILE` (core/place/rtrace.h, written in Engine::tiering_dispatch, rank 0):
  header (layers, experts, unit bytes, movable), `T` tier snapshots every 256 dispatches, `R` one
  credited histogram (dispatch, hist_step, layer, reps, live_missed, sparse expert/count),
  `D` cumulative counters per dispatch. ~10 MB a minute at 8 clients.
- `bin/rad-heat-replay TRACE [--lat N] [--decay/--hysteresis/--min-gain/--moves/--credit-cap]
  [--presence] [--resident N] [--from D --to D] [--oracle]`: heat.cpp unchanged, one slab class,
  swaps published `--lat` passes after the dispatch that proposed them (2 = today's order).
  `work/xs1/dranges.py TRACE` finds the dispatch ranges of the probe phases.
- `RADIANCE_HEAT_{DECAY,HYSTERESIS,MIN_GAIN,MOVES,CREDIT_CAP}` (engine_bringup, logged in the
  heat-engine startup line).
- `/stats` experts: `layer_routed`, `layer_missed` (per layer, cumulative), `early_published`,
  `paced_units`, `paced_marks`, `paced_late`.

## Replay validation (live = the trace's own counters, same dispatches)

| trace / range | replay lat 2 missed/dispatch | live missed | replay promotions | live promotions |
|---|---|---|---|---|
| d0-B, c8 distinct (520–807) | 186.5 | 188.4 | 38.9 | 39.8 |
| d0-B, c8 distinct-2 (1540–1834) | 201.5 | 201.6 | 42.2 | 42.0 |
| d0-B, whole run (257–2088) | 100.0 | 99.5 | 23.5 | 23.8 |
| d5-B (DFly2), 257–831 | 327.5 | 313.9 | 46.1 | 47.7 |

No-spec within 1 % of misses and 2.5 % of promotions (hit within 0.1 point): accepted. DFly2
4 % off (lent-slot fills are not modelled); usable for relative statements.

## Replay: what the levers could give (misses a dispatch, c8 distinct)

| variant | d0 520–807 | d0 1540–1834 | d5 257–831 |
|---|---|---|---|
| lat 2 (today) | 186.5 | 201.5 | 327.5 |
| lat 1 (X1 a, if every copy had landed) | 182.7 (−2.0 %) | 196.7 (−2.4 %) | 316.7 (−3.3 %) |
| lat 0 (X3 bound: visible in the pass after the decision) | 172.0* (−5 %) | 191.6* | 302.1 (−7.8 %) |
| oracle: every miss admitted for the next pass, free | 148.3* (−18 %) | 165.8* (−18 %) | 233.3 (−29 %) |
| moves 128 (lat 1) | 178.2* | 197.1* | 311.4 |
| min-gain 0.5 (lat 1) | 180.0*, 60 promotions | 199.5* | — |
| presence credit | 182.1* | 203.7* | — |

(* from the d0-P trace, same phases; the B trace gives the same within 2 %.)
Readmits 66 % of promotions in the c8 phases, 89 % over the whole run.

Reading: the latency half of the design (§2.4, "at least 45 % of the misses are latency misses")
does not hold on the real traces: one step less latency removes 2–3 % of the misses, two steps
5–8 %. Even admitting every miss for free removes only 18 % (no spec) / 29 % (DFly2): most misses
at 8 distinct streams are first-time or long-gap. X2 (admission) and X3 (early publish) can only
reach part of the oracle's 18–29 %; X4 (router-ahead prediction) is the only lever for the rest.
Run the X2 variants on these traces before building it.

## H0 (the friend's 196 / 361 tok/s)

- `bench_throughput` is in neither his repository nor radiance/BetterBench; his README / BENCH /
  DESIGN only say "a throughput bench against the HTTP API, 256 generated tokens a request";
  `deploy/run-hy4-bench.sh` + `example-bench.yaml` show an external harness that addresses the
  model as `tencent/Hy4-flash` on port 8000. Which prompts it sends is not documented: ask him.
- Our probe (B arm, 210 W): distinct vs one prompt for all clients:

| | 1 client | 4 distinct | 4 one prompt | 8 distinct | 8 one prompt |
|---|---|---|---|---|---|
| no spec, tok/s | 60.1 | 148–150 | 166 | 203–205 | 291–292 |
| no spec, routed units / missed units a step | 705 / 22 | 2381 / 100 | 2044 / 65 | 4038 / 193 | 2365 / 66 |
| DFly2 5 adaptive, tok/s | 101–105 | 186–193 | 215–218 | 243–245 | 358–359 |
| friend, 330 W, no spec / DFly2 | 57.7 / 109.8 | 196.2 / 341.3 | | 360.6 / 425–603 | |

  One prompt on eight clients gives +43 % without speculation and +47 % with DFly2, although our
  eight texts still diverge (8 distinct hashes; the union of routed experts is 2365 a step instead
  of one row's 705). His 8-client figure is above our "one prompt" figure by the margin 330 W vs
  210 W and fully row-identical streams would explain; his 4-client DFly2 341 vs our 4-distinct
  186–193 is not reachable with distinct prompts on the same MoE path. H0 is consistent with every
  number but not proven: it needs his prompt set (or running his stack under xs.py, not done).
