# hy4-perf: real container of the friend, TP2, 210 W (branch hy4-perf, wt/perf; work/perf/)

Branch hy4-perf = claude/peaceful-wright-hhqnhi + hy4-adopt + hy4-compat (merge 2793e615; conflicts: engram.h -> adopt's
transient host-domain ids/rows + ids_ahead, compat's model-level hash constants; attn.h -> compat q_b declared after q_a,
fused q-residual codes (rmsnorm_had_quant_i8g) feed q_b too). 6cb3da9e adds RADIANCE_HY4_SHAPE_PROBE=0 (A/B switch).
Not pushed, not merged.

- Arena lending visible: "activation arena: 256 tokens use 187.87 MiB, 1024 tokens use 714.08 MiB of 1.38 GiB ... lent".
  Lending (+prefill stager, armed only with levels): decode +2..7 %, prefill +30 % (ram: 1645 -> 2110 tok/s).
- First request after start hung 5x early on with --ngram-placement disk (rank barrier, queues idle, 60 s stuck-exit
  watchdog kills the server). NOT lending-specific (also with SHAPE_PROBE=0 after a page-cache drop). Later runs with disk
  worked (incl. lending on + cache drop + scheduler.stuck-exit 0). Root cause open; intermittent, looks cache-state
  related. Workaround used: scheduler: stuck-exit: 0.
- Greedy texts identical across all runs (DFly2 on/off, disk/ram, lending on/off, Flash-Next flags) and == compat-only.
- Flash-Next flags: prefill-span 4 + 8192 batched crashes at ~19.5K prompt (swiglu_clamped_bf16 refuses M=131072);
  busy-prefill-chunk 1088 below the 4097-token stager step (core warns) -> prefill -26 %; draft-suffix needs
  draft-window fixed; draft-verify traversal refused (no plugin declares mtp_trav_verify).
- Expert VRAM hit (heat engine routed_res/routed): ~0.95 c1, 0.92-0.95 c4, 0.90-0.94 c8; streamed bytes 12-26 GB/s.
Scripts: bench.py, run.sh, queue*.sh, summ.py; results res/*.json, *.out, *.server.log.
