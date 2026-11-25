# u3-perf: hy4-flash-iqr-v4-u3.rad, TP2, 2x R9700, 210 W, Engram RAM, src@28e35f5d
Sampled only (T 0.9, top_p 1.0; user instruction), 256 tok/request, code prompts. Data: work/u3/res/*.json, *.out, run.sh, bench.py.
Configs: work/u3/nospec.yaml, coupled.yaml (DFly2 depth 5, window fixed, coupling on, traversal verify, depth adaptive, depth-load on, suffix on, kernels libr4d,libavx,spechub).
VRAM per card: weights+experts 22.44 GiB, KV 7.33 GiB (8x65536 fully backed), arena 1.39 GiB; host pool 39000 MiB unchanged.
Side: coupled-t07.json (T 0.7, 1 rep) = same picture as T 0.9; 8-client anomaly (150 tok/s, accept 0.48) does not reproduce.
Full table: see final report.
