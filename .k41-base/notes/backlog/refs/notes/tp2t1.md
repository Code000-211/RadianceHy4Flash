# tp2t1: T1/T2/T5 of kernels/levers-tp2.md (branch hy4-tp2-t1, merged into claude/peaceful-wright-hhqnhi @ 4d1f957c)
Data: work/tp2t1/ (res/*.json, summ.py tables, id.py/cmp2.py identity, parity/ logs). Runs: a-* = shape_probe only (off/on via RADIANCE_HY4_SHAPE_PROBE),
b-* = +T2 config, c-* = +host-pool 61000, m-* = whole hy4-adopt merged. Concurrent greedy c8 is NOT reproducible even off-vs-off; use sequential identity.
T1 on: prefill +15..25 %, c8 +5 % (no spec) / +12 % (coupled). T2: only >=16K, decode -2..5 %: not recommended. T5: no-op (pool slots unchanged: 7848).
Merged adopt: eps f32 changes outputs (deliberate); with RADIANCE_HY4_ENGRAM_EPS=bf16 the u3 greedy text+logprobs are byte-identical to the pre-merge build.
