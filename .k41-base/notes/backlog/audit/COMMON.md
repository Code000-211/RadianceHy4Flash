# Common brief for the optimisation-backlog code audits (read fully before starting)

You are one of five read-only auditors. Goal: find every concrete, ticket-sized performance optimisation in your
assigned part of the Hy4-flash serving path of radiance. ANALYSIS ONLY: do not modify the source tree, do not build,
do not run anything on the GPUs (no radiance, no rad-kbench on cards, no r4d_selftest). Reading code and docs only.
Write your findings to the output file named in your task (Markdown), then return a 15-line summary.

Code: /root/work/claude-scratch/hy4flash/src (git branch claude/peaceful-wright-hhqnhi @ c655f58c). Do not touch
/root/src/radiance. Docs: src/docs/ARCHITECTURES.md (Hy4 section, "TP2 recipe"), GUIDE.md, OPS.md, KERNELS.md, libr4d/README.md.

Setting: Hy4-flash (tencent hy_v4_flash): 42 layers (30 sliding-window-512 + 12 sparse with indexer), MLA attention with
sinks (latent 512 + rope 64, q 1024), Engram n-gram lookup on 4 layers, iHC (hyper-connection pre/post, 8 rows... see code),
latent MoE 512 experts top-16 (latent 1536, ff 1536) + shared expert, sigmoid router, n=3072, vocab ~120.8K.
Served at TP2 on 2x Radeon R9700 (gfx1201, RDNA4, 32 GB, 64 CUs, WMMA, ~550-600 GB/s HBM), P2P between cards, PCIe 16 GT/s
x16 uplink (27.6 GB/s). Container formats: trunk linears IQR u5 (iqr_gemv / iqr_m16 kernels) and int8 (gemm_i8_kpar), routed
gate_up u3 trellis (moe_iqr), routed down w4nl64 (moe_down_dec), head int8 (logits_gemm), kv_b bf16, router f32, DFly2
drafter (5 layers + correction chain + lm_head) IQ8R/int8. Activations are quantised to rotated int8 (had_quant) before
quantised GEMMs. Engine: steps are recorded as tapes and replayed as chained AQL launches.

Measured cost structure (TP2, 210 W, u3 container; before today's remeasurement):
- decode, 1 client, no spec: ~13.4 ms/step, ~1330 dispatches/pass, ~258 us/layer, ~32 dispatches a layer, about half of
  them at the chained-launch floor (~3.2 us of card time each; BUT fusing a launch away measured only ~0.3-0.5 us gain each
  because chained launches overlap their predecessor: G-3 fusions removed 161 launches for +1.2 %). ~8 ms of the pass is
  weight bytes (trunk linears ~4.3 ms, routed experts ~2.1 ms at 1 row, head 0.3 ms). Two-rank junction 84x ~13 us.
- decode, 8 clients: ~25.6 ms/step; routed gate_up+down ~12.3 ms (2.2 us per routed unit of 1.46 MiB a rank; bytes-bound),
  expert misses (non-resident units read in place over PCIe) 66 us a unit.
- DFly2 step 1 client: verify pass ~14-15 ms + draft pass 3.9 ms (5x logits_gemm head 1.5 ms, IQ8R linears 1.05 ms,
  94 small launches 0.48 ms, attention 0.45, rest 0.42); a verify row costs 1.0 ms (0.95 routed GEMMs).
- prefill 2048-token chunk ~520-640 ms: routed MoE GEMMs 38 % (half link reads of non-resident experts), attention
  20.5 % (mla_attn_paged_sink 1.75-2.5 ms/call, mla_attn_sel_sink 3.6-5.2 ms/call, expand_gate, project_kv, attn_kv_sink),
  trunk GEMMs 13.6 %, junction 9.2 % (0.45-0.67 ms/call; arithmetic 0.29 ms), quantisers 3.4 %, router 2.4 %,
  moe_gather+cast+swiglu 5.2 %, rest 7.4 %.
- host: step loop phases ~0.03-0.14 ms; rank threads replay the tape in 0.45-0.63 ms (1 client), 1.23 ms (8), overlapped.
- arrival stall: decoders stall ~0.4-0.8 s while a prefill chunk step runs; a prefill step has a floor ~0.2-0.3 s.

PRIOR WORK (must read what concerns your area; do NOT re-propose closed items unless you have NEW evidence from the code;
if you mention one, mark it CLOSED and cite the note):
  /root/work/claude-scratch/hy4flash/kernels/{perf.md,levers-general.md,levers-tp2.md,experts-streaming.md,
  mla_sink_attn.md,dfly2.md,attn_kv_sink.md,engram.md,ihc.md,indexer.md}
  /root/work/claude-scratch/hy4flash/notes/{g3.md,tp2t1.md,tp2t3.md,t4.md,g1g5.md,flashnext-port.md,xs-0.md,xs-1.md,
  hangfix.md,perf.md,u3-perf.md,adopt.md,wp16-*.md}
Closed/decided already (do not re-propose): chaining all rows (done); shared/fused quantisers of hy4-adopt (done); G-3
fusions a/b/c (done: swiglu+quant, junction writes codes, expand+gate+quant); arena lending + prefill stager (done);
draft-depth-stop 0.3 + load-aware depth from 1 seq (done); q_temp (no winner); expert prefetch at TP2 (T3 NO-GO);
two-piece pipelined prefill (T4 NO-GO); prefill-span/span-pass (only >=16K, not recommended); host mirror of the pool
(no gain alone); faster junction wire / fewer reductions / re-sharding / pipeline parallelism / router on one rank
(closed in levers-tp2 §3); host-CPU MoE; expert-pin-top/profile; heat damping; wht6 wire (lossy); MOE_VERIFY_BUDGET
(lossy); adaptive-depth host wait (not a lever); sampler chain vs greedy (0.2-0.5 ms); shortlist head for the drafter
(rejected by the user). Anything lossy or changing a quantisation/storage format is NOT ours to decide (a friend of the
user owns formats): list such ideas separately under "format / lossy (friend's decision)".
Open known items you may refine with code-level detail (file:line, register/LDS facts, exact change): G-3 d..h
(q_nope+q_rope dual launch, row_front+rope, router gemm+sigmoid, moe_down+gather(+copy16), sink_rows+merge), G-4 /
16.4b / 16.4c / 16.6 prefill attention row tiles, `cast` column copy removal, FP-2 in-step tail snapshot, FP-3
decode-headroom, FP-4 persistent prefix store, FP-8 packed map check, T6 fused reduction twins in the drafter, L5b chain
stop on card, L5c short block, X2 admission, X3 early publish, X5 shared arm beside routed GEMMs.

What a finding must contain (one block each, be concrete and terse):
  - title; area (kernel/plugin/core/scheduler/server/drafter/memory)
  - exact location: file:line(s) (verify the line numbers by reading the file)
  - what exactly to change (mechanism), in 2-6 lines
  - why it should pay: what cost it removes (launches, bytes read, copies, syncs, host us, VRAM), with whatever numbers the
    code gives you (sizes, loop counts, tile shapes, thread counts, LDS bytes, VGPR notes in comments, etc.)
  - lossless? (byte-identical / moves bits within stated bound / lossy-format)
  - how sure you are that the code really does what you describe (high/medium/low), effort S/M/L, dependencies
  - how to verify (existing test names / kbench fixture / selftest you found)
Also list "checked and found already optimal / not a lever" items briefly (so nobody re-derives them).
Completeness matters more than brevity: dozens of findings, including small ones (a redundant copy, a sync, an avoidable
launch, a host malloc per step, a mutex, a per-token JSON allocation) are wanted. But do NOT invent: if you have not read
the code that proves a claim, say "unverified". Quote code only where the exact text is load-bearing.
