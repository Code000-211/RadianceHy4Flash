# Flash-Next serving path -> Hy4-flash: what to copy, what is already there (research and plan only)

Status 2026-11-22. Nothing built, nothing measured on a card (the GPU lock is held by the busy-prefill-chunk
and long-context sweeps; the Flash-Next model files are deleted, so every Flash-Next figure below is quoted from
`docs/GUIDE.md` / `docs/FLINT.md` / commit messages / memory notes, not re-measured). Hy4 figures are from
`kernels/levers-general.md`, `levers-tp2.md`, `experts-streaming.md`, `notes/*.md` and the running sweep's BetterBench
logs (`work/sweep/bb-*.log`). This note supersedes the table in `notes/flashnext-levers.md` (2026-11-21), which was
right on the split but did not check the Hy4 tree option by option.

## 0. Result in one page

1. **The user's remark holds, and it is stronger than the first survey said.** Checked option by option against
   `claude/peaceful-wright-hhqnhi` @ 55a794dc (grep of the option, then a diff-identifier match of the dartree commit
   against our tree, then a `git merge-tree` probe): every Flash-Next decode and prefill lever that is lossless and
   generic is already in our tree, either as upstream code or as our own port, including `chain_pays = RAD_CHAIN_ALWAYS`
   and `rad_kv_group_replicated` for the three paged groups (first survey listed both as open: they are done, 
   `hyv4flash.cpp:1015-1017, :1232`). Upstream `origin/main` has moved only in libr3d since our base (27 commits), so
   there is nothing more to merge from there.
2. **What exists only on dartree (or the bridge branches) and is not in our tree**: draft trees (`draft-tree*`,
   `draft-adaptive`), `draft-coupling-temp*` / `draft-calib online`, the global suffix cache
   (`draft-suffix-global-mib`) with the suffix-index take-over across turns, the persistent prefix store
   (`prefix-cache-dir*`, LRFU, admission, write budget; upstream *removed* the disk tier, our `config.cpp:895` refuses
   the flag), `--decode-headroom` (PR #411), `RADIANCE_MOE_VERIFY_BUDGET` (lossy, excluded), the opt-in negative results
   (`RADIANCE_Q4_HCFOLD`, `RADIANCE_CKPT_ASYNC`), the per-project file index (`bridge-files`, needs a harness).
3. **Ranked candidates (details in section 3 and 4)**:
   1. size the arrival stall and the tail step first (FP-1, S, GPU 1 h): everything about the stall hangs on one
      number, the fixed cost of a small prefill step on Hy4;
   2. in-step tail snapshot for the Hy4 plugin (FP-2, M): removes the extra "tail" step of every nearest-resume
      prompt, i.e. one stall and a few hundred ms of TTFT per agentic turn, *if FP-1 shows the step is expensive*;
   3. persistent prefix store (FP-4, M, low conflict): after a restart a 64K history costs ~0.7 s instead of ~17 s at
      Hy4's 3.7K tok/s; capacity and LRFU for many projects; lossless;
   4. `decode-headroom` (FP-3, S-M): prefill under load +10..22 % on Flash-Next; only after FP-1 shows the remainder
      steps exist in our schedule;
   5. row-tile staging for `mla_attn_sel_sink` / `_paged_sink` with Flash-Next's `attn_usel` as the pattern (FP-7, M-L,
      this is G-4 / 16.4b with a precedent): prefill +3..5 %;
   6. DARTree for DFly2: **defer**. The expected gain is 0..3 % at one client and negative from four clients, for an
      L-sized change (tree mask in three MLA kernels, tree indexer selection, Engram/ring rollback). A one-day offline
      sizing (FP-6) from logged laws decides; do not start the implementation before it.
   7. global suffix cache (FP-5): +0..3 %, M, last.
4. **Closed, do not re-propose**: `tp-wire wht6` (norms need the exact sum), MOE_VERIFY_BUDGET (lossy), hc-fold, async
   checkpoint copy, tail-in-pieces `span-pass: step` (measured here: only >= 16K, decode -2..5 %), expert prefetch at
   TP2 (T3), split pipelined prefill (T4), host mirror (T5), heat damping, q_temp, launch fusions beyond G-3, the lean
   project base / bridge (user decision 2026-11-21).

## 1. Where Flash-Next and Hy4 differ in ways that decide the port

| | Flash-Next (qwen4exp) | Hy4-flash (hyv4flash) | consequence |
|---|---|---|---|
| attention | GDN linear + gated attention, QSA sparse | MLA absorbed, 30 sliding (window 512) + 12 sparse layers with sinks, Engram, iHC | no GDN state: snapshots are small (indexer ring 27 KiB + Engram CONV + 17.4 MiB window a rank) but there is a windowed KV group |
| routed MoE | 512 x top-10 | 512 x top-16 latent MoE, 29.3 GiB a rank, hit 0.90-0.99 at TP2 | each extra verify row ~1.0 ms (390 more routed units), same as Flash-Next's 0.9-1.0 ms; a small prefill step still reads nearly every expert |
| drafter | MTP head (serial, 2-bit candidate head, top-8) | DFly2: backbone pass for a block + per-position correction head (serial chain in the query pass), no lattice | `dflash_couple_tree`/`dflash_tree` do not apply (kernels/dfly2.md 1.5); coupling is `mtp_couple_propose` (done) |
| step | 12.5-13 ms, 930 launches | one client 13.4-14.1 ms (no spec) / 19.4-20.8 ms (DFly2, 2.0-2.9 tok), 1479 launches after adopt | launch floor share higher |
| prefill | 6.5-7.7K tok/s (BB, 210 W) | 3.0-4.0K tok/s (BB 3964 at 1.5K ... 2964 at 47K) | cold prefill twice as long: every prefill-avoiding lever is worth more in seconds |
| residual | gated residual + fused all-reduce writes | fused `ar_rmsnorm_ihc_junction` (coded wire from 512 KiB), exact sum | gated-residual AR fusion has an in-tree equivalent already |

## 2. Per-feature verdict (a: where it lives, b: applies, c: lossless, d: gain, e: port)

Legend for (a): **in** = in our tree (evidence in brackets); **dart** = only on dartree / another local branch.
Gains for Hy4 are estimates unless marked **M**.

| # | Flash-Next feature | (a) | (b) applies to Hy4 | (c) lossless | (d) expected Hy4 gain | (e) port |
|---|---|---|---|---|---|---|
| 1 | MTP coupling, traversal/block verify | in (`b759ca01` DFly2 couples via `mtp_couple_propose`; `mtp_trav_verify` in spechub) | yes, done | yes (chi-square tests) | in the measured 150 tok/s | done |
| 2 | adaptive depth (stop/min), load-aware depth, one-sequence rule | in (`e71cb41e`, `38b1dc17`; `draft-depth-stop 0.3`) | yes, as block prefix | yes | done: +11 % code / +22 % thinking at c1 (**M**) | done |
| 3 | per-request suffix selector beside a block | in (`54e38ffb`; window = block of 5+1 rows) | yes | yes | small, workload dependent | done |
| 4 | chained launches (`chain_pays`), chain forms of every row | in (`hyv4flash.cpp:1232`, `cc123aae`) | yes | yes | done: 463 -> 5 unchained | done |
| 5 | `rad_kv_group_replicated` | in (`hyv4flash.cpp:1015-1017`) | yes | yes | host tier / prefix store hold one copy | done |
| 6 | prefill-span, `span-pass: step`, `span-first`, span-multi-prefill, tail-snapshot floor | in (config keys present, `016a19f3`) | yes | yes (not bit-equal to 2048 chunks, deterministic) | **M**: only >= 16K, decode -2..5 %: not recommended (tp2t1.md T2) | closed |
| 7 | busy-prefill-chunk, `prefill-interleave auto:1` | in | yes | yes | sweep running (section 5) | tuning only |
| 8 | prefix cache VRAM+host, `prefix-resume nearest`, checkpoints (ring, CONV, window) | in (wp13-0: gate passes) | yes | yes | in use | done |
| 9 | **in-step tail snapshot** (`64bf299e`: `ckpt_cut`, `ckpt_snap_ok`, split operand in MoeFP8) | engine+ABI+MoeFP8 **in**; plugin side missing: `hyv4flash` does not set `ckpt_snap_ok` (qwen4exp does, `qwen4exp_fp8.cpp:2079`) | yes, needs plugin work | yes (snapshot bytes identical to the tail step's) | Flash-Next: -5..-9 % TTFT on 5-7K prompts (the tail step is ~60 ms there). Hy4: unknown, section 4 FP-1/FP-2 | M, plugin |
| 10 | persistent prefix store + LRFU + admission + write budget (`a8400909`, `bab7361b`) | **dart** (upstream removed the disk tier on purpose, `config.cpp:895`) | yes; supports paged, linear and window snapshots (Flint precedent) | yes (64-bit checksum, key from binary/container/env/geometry) | Flash-Next TP2: 64K turn 7.64 s -> 0.63 s after restart, 120K 14.55 -> 1.30 s. Hy4 cold prefill is ~2x slower, so ~18 s -> ~0.7 s at 64K. 14.7 KiB a token like Flash-Next | M, conflicts small (5 files) |
| 11 | global suffix cache (`46c77e35`), index take-over (`9e883a39`), head-acceptance gate | **dart** | yes (DFly2 states are not touched; window stays the block) | yes | Flash-Next: neutral at T 1.0, +3.2 % on greedy retries. Hy4 sampled T 0.9 code: +0..2 % | M, conflicts in scheduler |
| 12 | `--decode-headroom` (`15434c6b`, PR #411) | **dart** (not in our `config.cpp`) | yes | yes (greedy bit-identical; moves which rows share a step) | Flash-Next: prefill under load 3664 -> 4455 tok/s (+22 %), TTFT 17.3 -> 14.2 s, decoders 12.3 -> 9.5 tok/s each (interleave gives it back) | S-M, 9 conflict files |
| 13 | draft trees: serial-head trees, coupled RRS siblings, pruning, dynamic rows, exact rows, adaptive nodes (`296066fb`, `5d36c49d`, ...) | **dart** | DFly2 only as a *serial-head* tree (no lattice) | yes | Flash-Next: **no gain over the coupled chain** (row 0.9-1.0 ms; memory note). Flint DFlash2 tree 4: +9 % at c1 over chain 3, +2.6 % at c8, but rows cost 0.85 ms there | L, see section 4 FP-6 |
| 14 | `draft-coupling-temp*`, `draft-calib online` | **dart** | optional | yes | Hy4 q_temp sweep had no winner (levers-general 5a, closed) | skip |
| 15 | MoE prefill GEMMs (`8a441abd`: packed 128-row run map, 64-row w4nl64 down) | in (32/34 identifiers) | yes if Hy4's prefill GEMM shapes take the packed path (K=1536, u3/w4nl64) | yes (byte-identical) | Flash-Next +0.9..3.6 % prefill. Hy4: unverified | S check in kbench |
| 16 | tiled QSA prefill attention (`afa63933`, `R4D_ATTN_USEL=1`) | the `attn_usel` file is in; Hy4's `mla_attn_sel_sink` / `_paged_sink` do not use it | pattern only (different kernel) | scores bit-equal, softmax key order differs: 1-ulp prefill rows, like 16.4c | Flash-Next chunk 2.8 -> 1.5-2.0 ms, prefill +4.4..5.7 % from 8K. Hy4: sel+paged are 15.6 % of prefill (381 of 2436 ms): -25..-35 % = +3..5 % | M-L (= G-4 / 16.4b) |
| 17 | hc_read prefill up product, 48-row tile (`afa63933`, `c2c88f7f`) | tile form in (`ca382868`); prefill product is for Qwen's hc_read | no: Hy4's iHC pre is inside the junction / `ihc_pre` | yes | +1.35 % Flash-Next only | N |
| 18 | hc-fold, async checkpoint copy (`RADIANCE_Q4_HCFOLD`, `_CKPT_ASYNC`) | dart | n/a | yes | measured 1 % slower / 20 % slower on Flash-Next | N |
| 19 | coded / staged / register-form 2-rank all-reduce writes | in (`7b256cc4`, `d6e6a94c`, `c30dbdff`; junction coded from 86 rows) | yes, done for the junction | yes | closed by T4 arithmetic (< 3 % of prefill) | done |
| 20 | gated-residual all-reduce fusion | n/a (no gated residual); equivalent in `ar_rmsnorm_ihc_junction` | n/a | - | - | done |
| 21 | `tp-wire wht6` | in core | **no**: norms and calibration read the exact sum | lossy | - | N |
| 22 | `RADIANCE_MOE_VERIFY_BUDGET` | dart | would cut the 65 % routed share of a verify pass | **lossy** | - | excluded |
| 23 | sync-audit (8 commits) and spec-audit (8 commits) fixes | in by diff-identifier and by reading (`chain_kernel.h:48` `s_wait_kmcnt`; the magic-static race of `bb85d00a` does not exist here: our `ar_coded_stages` has no tuning statics) | yes | - | hang protection | verified, nothing to port |
| 24 | haz_note, prefetch acquire, staging ring 16, queue scratch, `expert-pin-top` fractional fix | in (`planner.cpp:1149` identical to `08c0c44c`) | yes | yes | - | done |
| 25 | expert placement / heat tuning | in (`expert_tiered`, pin-top, profile) | yes | yes | heat damping, pin-top, profile: no gain (experts-streaming.md); T3 prefetch NO-GO | closed |
| 26 | prefix-store key: binary hash in the key | dart | - | - | every rebuild discards the store: useful on a deployed build only | see FP-4 |
| 27 | per-project file index (`bridge-files`, `b2db692c`) | dart (38 files, +4300 lines) | yes, needs a client that syncs files; window = 6 rows | yes | Flash-Next: a file rewritten whole +19 % (6.2 -> 7.97 tok/step); files already in the prompt are covered by the per-request suffix | L, optional, last |

## 3. Ranked table (gain x confidence / effort)

| rank | package | what | Hy4 gain (est.) | lossless | effort | conflicts / base | depends on |
|---|---|---|---|---|---|---|---|
| 1 | FP-1 | size the arrival stall: cost of a 16/64/256/1024-row prefill step, tail step, chunk interplay | decides FP-2/3; no gain by itself | n/a | S (GPU ~1 h, one server start) | none | GPU lock (sweeps hold it) |
| 2 | FP-2 | in-step tail snapshot in the Hy4 plugin | per nearest-resume prompt: one step less (its size from FP-1); TTFT -5..-13 % on 2-6K prompts if the step is 0.1-0.25 s | yes | M | engine side in; plugin: ring cut, Engram CONV cut, windowed group in a step (engine check) | FP-1 |
| 3 | FP-4 | persistent prefix store (+ suffix file) | restart: 64K history 17 s -> ~0.7 s; capacity | yes | M | `git merge-tree`: 5 conflict files of 26 (`config.cpp`, `sched/metrics.h`, `scheduler.cpp/.h`, `GUIDE.md`) | none; value rises with FP-2 stable |
| 4 | FP-3 | `--decode-headroom` | prefill under load +10..22 %, TTFT -15 % (Flash-Next numbers) | yes | S-M | 9 conflict files (`config.cpp`, `engine_bringup.cpp` arena sizing vs our lending, `kv.cpp`, `rad_core.h`, `scheduler.cpp`, tests) | FP-1 trace shows remainder steps |
| 5 | FP-7 | row-tile staging in `mla_attn_sel_sink` + `_paged_sink` (Flash-Next `attn_usel` as pattern) | prefill +3..5 % (deeper context more) | 1-ulp prefill rows, decode/verify untouched | M-L | libr4d only; `afa63933` itself does not apply (add/add on `r4d_attn_usel.h`): re-implement, do not cherry-pick | G-4 owner |
| 6 | FP-8 | kbench check: do Hy4's prefill MoE GEMM shapes take the packed map (K=1536, N=3072) | 0..3 % prefill | yes | S | none | - |
| 7 | FP-6 | DARTree offline sizing for DFly2 (log P and Q top-8 at depth 1, replay RRS with 1..3 siblings) | go / no-go for the L package | yes | S (one short run + script) | none | - |
| 8 | FP-5 | global suffix cache + index take-over | +0..2 % sampled, +3 % greedy retries | yes | M | shares `scheduler.cpp` conflicts with FP-4 | FP-4 (same commit series) |
| 9 | FP-9 | DARTree for DFly2 (implementation) | +0..3 % c1, <= 0 at c >= 4 | yes | L | new tree mask in 3 MLA kernels etc. | FP-6 says go |
| 10 | FP-10 | decode queue beside a prefill chunk (research, beyond Flash-Next) | stall below the chunk-step floor | yes if kernels are unchanged | L, unestimated | engine | FP-1 |
| 11 | FP-11 | `bridge-files` | file rewrites 2x tokens/step | yes | L | huge | harness |

Sequencing: FP-1, FP-6, FP-8 are short and independent (one GPU hour in total). FP-4 and FP-3 are engine cherry-picks
and can run in parallel worktrees. FP-2 waits for FP-1. FP-7 belongs to the kernel owner's G-4.

## 4. Work packages

Common rules: worktree `wt/<wp>` off `claude/peaceful-wright-hhqnhi`, cherry-pick with `-x` from the dartree hash named
below, never rebase or merge dartree, never touch `/root/src/radiance`, GPU only under `flock .../gpu.lock`, 210 W,
dummy-weight tests host-first as in AGENT_RULES.md. Merge via `git cherry-pick -x` of single commits in the order given.

### FP-1: size the arrival stall (S)

What the existing data says (sweep, 2 R9700 TP2, u3, DFly2 coupled, BetterBench workload phase, 16K arrival under
2-4 streams; `work/sweep/bb-*.log`):

| busy-prefill-chunk | worst stall | stall p99 in window | arrival TTFT 16K under load (cold 3.3 s) | bg decode in window, 4 streams (t/s per stream) |
|---|---|---|---|---|
| auto (>= 1024, default) | 812 ms (13x typical gap) | 715 ms | 10.3 s (3.1x) | 24.4 |
| 1024 | 684 ms | 676 ms | 14.8 s (4.4x) | 23.0 |
| 512 | 503 ms | 418 ms | 16.9 s (5.1x) | 26.7 |
| 256 | 444 ms | 348 ms | 24.6 s (7.4x) | 28.0 |

Reading: stall ~ the duration of the chunk step the decoders ride in, and it does not shrink proportionally with the
chunk: 256 rows still cost ~270-350 ms (a 2048-row chunk ~520 ms at 3.9K tok/s). A prefill step has a fixed
cost near 0.2-0.25 s whatever its rows: at 256 rows x 16 experts = 4096 draws over 512 experts every expert of every
layer is touched, so the step reads ~29 GiB of experts a rank (about 55 ms from VRAM) plus the non-resident ~10 %
over the link (about 110 ms per 2048-row chunk, notes/levers-general 1.8) plus ~1500 launches. That is why the
frontier is bad (TTFT x7 for a stall of 0.44 s) and why Flash-Next's remainder step (~86 ms, `decode-headroom` text)
is cheaper than ours would be.

Counter-evidence to keep honest: cold BetterBench TTFT is linear in tokens with intercept ~0 (387 ms at 1534 tokens,
1562 at 5899, slope 0.27 ms/token), i.e. a nearest-resume tail step of a few tokens is *not* visible as a 0.25 s
constant. Either tail steps are cheap at a few rows (an expert read is proportional to the experts touched: 14 rows
touch ~35 % of them) or the benchmark's prompts do not get cut. FP-2's whole value hangs on this.

Plan (one server session, short, under the lock; `prefix-resume nearest`, `prefill-tail-snapshot on` then `off`, the
u3 container, `coupled.yaml` of `work/u3`): (a) a lone prompt of 1 K, 3 K, 6 K tokens plus 1/16/64 tokens past a
16-multiple, step durations from `engine_step_seconds` or the debug step log: the tail step's size and cost;
(b) one decoding client (c1) and four clients while a 2 K prompt arrives, per-token gaps at chunk 64/128/256/512/1024:
fixed step cost; (c) `--debug-placement`/`/stats experts` deltas per step to confirm it is expert reads; (d)
`profile-ops` on one 256-row step: the floor's composition (reads, link, launches). Output: a table "rows -> step ms"
and the tail step's ms. Do not touch the power cap.

### FP-2: in-step tail snapshot for hyv4flash (M)

Engine side is in our tree (ABI 41: `RadBatch::ckpt_cut`, `ckpt_cu`, `RadArchProbe::ckpt_snap_ok`, `rad_ckpt_snap`,
MoeFP8 `split_tok`, `core/engine.h:790`). Missing: the plugin. Needed, mirroring `rad_qsa.h:527-539`,
`rad_block_ple.h:545-560/656-668`:
1. `out->ckpt_snap_ok = 1` in the probe, only when the plugin runs its state in two pieces correctly.
2. Indexer tail ring (`qsa_tail_store`, `kv_index_tail`, 27 KiB a slot): run the ring over the rows before `ckpt_cut`,
   `rad_ckpt_snap(c, kv_index_tail, layer)`, then the rows after.
3. Engram CONV group (`kv_engram_tokens`): same cut and snapshot.
4. The windowed latent group (`kv_latent_sliding`, 17.4 MiB a rank a snapshot, "a checkpoint slot carries its window"):
   **open question for the engine** whether `rad_ckpt_snap` in a step can stage a window group (Flash-Next's PLE windows
   are CONV-like). If not, the first deliverable is the engine change, which is the part with risk.
5. The junction, Engram gather and router are row-wise (no cross-row state); the routed GEMMs already take the split.
6. Test: snapshot hashes byte-identical to the separate tail step (the engine's own comparison), follow-up turn bit
   identical, rollback gate `tools/hy4flash_rollback_gate.py --prefix-cache` TP1/TP2, `mtp_rrs` unaffected.
Gain: removes one step from every nearest-resume prompt and the stall that step is for the decoders. Size from FP-1:
if the tail step is >= 0.1 s the gain per agentic turn is 0.1-0.25 s of TTFT (-5..-13 % on 2-6K deltas); if it is
~30 ms, do not build. Under `prefix-resume exact` (the default) the tail costs no step but the next turn re-prefills up
to a whole chunk; nearest + in-step snapshot gets both.

### FP-4: persistent prefix store (M)

Cherry-pick order (dartree): `a8400909` (store, 14 files, +2316), `9e883a39`, `46c77e35`, `e0976a0a`, `5cb4c2ad`,
`bab7361b`; FP-5 is the same series, take the store first and the suffix files in a second pass. Measured probe: all six
in one `git merge-tree` apply with conflicts in only `core/config.cpp` (our removal stanza for `--prefix-cache-dir`
must be replaced by the option), `core/sched/metrics.h`, `core/sched/scheduler.cpp`, `core/sched/scheduler.h`,
`docs/GUIDE.md`; `engine.cpp`, `mem/kvtier.cpp`, `mem/prefix.cpp`, `rad_core.h`, `server/*`, tests auto-merge.
Hy4-specific checks: (a) a store entry holds paged groups (latent sparse, indexer keys), snapshots hold window + ring +
CONV: write a test with `gpu-mini` TP2 that a restored hit equals the in-memory hit (text and top-5 logprobs), also
across Engram CONV; (b) replicated groups are stored once (`rad_kv_group_replicated` is declared, section 2 #5); (c)
the key includes the binary and every `RADIANCE_*` variable: it is for a deployed build, a rebuild discards it (the
startup line says which input changed); (d) disk: 14.7 KiB a token (`KV 7.33 GiB / 524288`), 100K tokens ~1.4 GiB, the
box has ~112 GB free on `/` (86 % used): set a cap (`prefix-cache-dir-mib`) and the write budget; (e) the directory holds
conversation text unencrypted.
Gain (estimate from the Flash-Next table, scaled to our 2x slower prefill): follow-up turn after a restart 16K ~4 s ->
~0.3 s, 64K ~17 s -> ~0.7 s, 120K ~32 s -> ~1.4 s. Matters when the server restarts or when more conversations are live
than `prefix-cache-mib` holds; no effect on a warm in-memory hit. LRFU + admission keeps single benchmark prompts off the
disk. Lossless.

### FP-3: `--decode-headroom` (S-M)

Cherry-pick `15434c6b` (13 files, +349). Conflicts (merge-tree): `config.cpp`, `engine_bringup.cpp` (activation-arena
sizing, collides with our `shape_probe_ok` / lending), `mem/kv.cpp`, `rad_core.h`, `sched/scheduler.cpp`, `GUIDE.md`, two
tests; resolve by hand, then re-run the arena/lending receipt (the arena grows by 64..256 rows; Flash-Next +12 MiB a
rank). Only useful where a chunk step is planned at the whole budget and the decode rows ride on top: with
`busy-prefill-chunk auto` (>= 1024) at `max-num-batched-tokens 2048` our schedule may already leave the room by cutting
the chunk (that is what `auto` does), in which case the "remainder step" the feature removes does not occur. FP-1's step
log decides: count steps per checkpoint interval during an arrival; if it is 2 (chunk + remainder of one quantum), port
it. A remainder step costs a full expert pass here (section FP-1), so the win per avoided step is larger than on
Flash-Next (86 ms). Lossless in the engine's sense (greedy text identical, rows share other steps).

### FP-5: global suffix cache and suffix-index take-over (M, low)

Same series as FP-4 (`46c77e35`, `9e883a39`, `e0976a0a`; `gsuffix.cpp/.h` new, `scheduler.cpp` hooks). On Hy4 the
verify window is the block (5 drafts + 1), `draft-suffix-max` cannot exceed it (`notes/wp13-8b.md`), so a recurring
response saves at most 5 rows a step. Flash-Next measured neutral for sampled T 1.0 and +3.2 % for T 0 retries; the
suffix-index take-over saves tens of ms once a turn (0.19 us a token: 20 ms at 100K tokens, built under the scheduler
lock; ours builds at the first token) which is small beside a 0.7 s arrival stall. Do it with FP-4 if the scheduler
conflicts are being resolved anyway; not on its own. Worth a second look if agentic use shows many identical tool
outputs (it is what the global cache keys on).

### FP-6: DARTree for DFly2, offline sizing first (S), then the L package only on a go

Why not now: (1) DFly2 has no lattice, so only the serial-head tree applies (grower = batched correction + head launch
per level; the draft side stays about one pass, the head is bandwidth-bound at <= 16 rows); the cost is all on the
verify side. (2) Verify row price 1.0 ms of a 19.4-22 ms step (4.5-5 %); a node pays if its probability of being
accepted exceeds ~0.10-0.12 at c1 (levers-general 1.6) and the bar rises with load. DFly2's first draft is accepted
0.57-0.64 of the time under the coupled law, so a depth-1 runner-up sibling captures part of the other 0.36-0.43:
break-even needs ~0.12. Flash-Next's serial-head trees did not beat the coupled chain at the same row price (memory:
`radiance-dartree-work`); Flint's DFlash2 tree-4 beat chain 3 by 9 % at c1 and 2.6 % at c8 with 0.85 ms rows and a
lattice drafter (docs/FLINT.md). (3) Implementation scope: tree mask in `mla_attn_paged_sink`, `mla_attn_sel_sink`
(and `attn_kv_sink` on the first chunk), per-node indexer selection and `qsa_tail_store` at flat ring rows (dartree
OPS.md:3574 is the model), Engram CONV rollback per path, KV compaction of the accepted path, `dflash_couple_tree`-like
grower on `mtp_couple_propose` states, scheduler: `note_grown`, prefix verify, issue-ahead. Everything under
dartree/flint-dartree, with 137 changed files: re-implementation, not cherry-pick.
Sizing: log per step the first position's coupled law Q_1 (top-8 values, f32 shortlist) and the target law P at the
same row (the verify walk has both; one debug env, ~30 lines in the engine's couple state dump or in `RADIANCE_CONC_LOG`),
run 200 sampled code requests at T 0.9 and one thinking run, then replay recursive rejection sampling offline with
k = 1, 2, 3 siblings (drawn without replacement from Q_1, exactness as in DESIGN-adaptive.md 4) and report the
expected extra accepted tokens per step and the break-even row price. Go only if k = 2 adds >= 0.12 tokens a step at c1
and the net (tokens x 1/step-time) is positive at 1.0 ms a row and 4 clients' bar. Expected outcome (estimate): marginal;
the cheap variant "one sibling at depth 1 only" is the only shape that could pay.

### FP-7: row-tile staging for MLA sel/paged attention (M-L)

Kernel owner's G-4 (`levers-general.md` L4, `kernels/perf.md` 16.4b). Flash-Next gives the recipe and the acceptance:
tiles of 8 launch rows over the union of the rows' page lists, K/V loaded once a tile, exact per-row masks, tiles that
share too little fall back to the single-row kernel; from 256 rows; selftest cases per geometry (`afa63933`,
`libr4d/r4d_attn_usel.h`, 678 lines). For MLA the latent K is shared by all heads of a rank (16 heads at TP2), which
makes the union staging cheaper than for GQA. The sliding layers (30) share a 512-window across consecutive rows, so
`mla_attn_paged_sink` (1.75 ms a call over the four absorbed chunks) benefits too. Target: sel <= 2.8 ms, paged <= 1.0 ms
a call as in G-4. Not bit-identical for prefill rows (softmax key order; 12-13 % of bf16 outputs differ by 1 ulp in
Flash-Next); decode and verify rows must stay on the stable law (16.4a). The commit itself does not apply (add/add on
`r4d_attn_usel.h`, five more conflicts): copy the technique, not the patch.

### FP-8: kbench check of the prefill MoE GEMM shapes (S)

`rad-kbench` the Hy4 shapes (gate_up K=1536 N=3072 u3/IQR, down w4nl64) at 20K..80K routed rows with and without the
packed run map (`RADIANCE_MOE_IQR_MAP_TILES=4|8`, `RADIANCE_MOE_DOWN_MT=2|4`, both in our tree via `8a441abd`); the
profile says routed GEMMs are 38 % of prefill with half of it link reads, so the compute half is ~19 %: a 10 % kernel
gain is ~2 % of prefill. Cheap to rule in or out.

### FP-10 (research, beyond Flash-Next): decode on its own queue beside a prefill chunk

Not a Flash-Next feature. The stall equals the chunk step because decoders ride it; decode steps are launch-bound (1479
tiny dispatches) while a chunk is expert-read and GEMM bound. Running the two on separate HW queues might hide most of a
decode step inside a chunk and cut the stall to ~one decode step. Risks: HBM contention, the engine's single-stream
ordering/haz model, rank barriers at TP2. No estimate; only worth a feasibility look after FP-1 gives the step floor.

## 5. The arrival-stall problem, collected

Facts: worst stall 0.4-0.8 s under 2-4 streams (section FP-1 table); the stall is the chunk step the decoders ride in;
a step's cost has a floor of ~0.2-0.3 s from the expert pass (numbers above), so shrinking the chunk lowers the stall
only slowly and costs TTFT x3..x7.

What Flash-Next offers, in order of usefulness for Hy4:
1. Avoid the prefill altogether: prefix hits (in), persistent store (FP-4), in-step tail snapshot so that a resumed turn
   does not pay the tail step (FP-2). For the user's agentic loop (measured elsewhere: ~90 % prefix hit, prefill 15-20 %
   of waiting) this is the only lever that removes stall steps rather than reshaping them.
2. Reshape: `prefill-interleave auto:N` (in, default `auto:1`): decode-only steps between chunks restore the cadence
   (Flash-Next: decoders 12.3 -> 40.9 tok/s at 2x TTFT); `decode-headroom` (FP-3) lets decode rows ride without stealing
   chunk rows and removes the remainder step; `busy-prefill-chunk` (in) is the sweep in progress, it moves along the
   frontier of the table above and does not beat the floor.
3. Lower the floor: stager (in, +15..25 % prefill, tp2t1), lending (in). Beyond that only formats (friend: fewer or
   smaller experts resident) or FP-10.
Not levers (measured): T2 span-pass, T3 prefetch, T4 split prefill, heat damping.

Recommendation for the running sweep: do not choose `busy-prefill-chunk` from the stall column alone; the table shows
auto (10.3 s TTFT, 0.81 s stall) against 512 (16.9 s, 0.50 s): a 38 % shorter stall costs 65 % more TTFT. FP-1's
"rows -> ms" curve will say whether a 1024 floor (the stager's smallest pass) is the right knee.

## 6. Evidence and caveats

* Option presence: grep of `--draft-tree`, `--draft-adaptive`, `--draft-coupling-temp`, `--draft-suffix-global`,
  `--decode-headroom`, `MOE_VERIFY_BUDGET`, `HCFOLD`, `CKPT_ASYNC`, `SCRATCH_RECLAIM` finds none in our `core/`; present:
  `prefill-span-*`, `prefill-tail-snapshot`, `busy-prefill-chunk`, `prefix-resume`, `draft-depth-load`, `draft-verify`,
  `draft-suffix`, `expert-pin-top`, `expert-profile`, `prefill-interleave`.
* Commit-level: identifier overlap of each dartree commit against our tree (>= 90 % for `c30dbdff`, `3e06911e`, `4b4fbc85`,
  `5a3360a8`, `a11cdf1a`, `1e6adb06`, `a30667e4`, `7b256cc4`, `8a441abd`, `f7338239`, `ae389e1f`; 70-85 % for `d6e6a94c`,
  `64bf299e`, `c2c88f7f` where the misses are Flash-Next-specific hunks). Subject matching is useless (upstream squashed).
* `git merge-tree` probes (read-only, in a scratch repo with object alternates): store series 5 conflict files,
  decode-headroom 9, `afa63933` 7, `bb85d00a` 1 (moot). Dry runs only; no branch touched.
* The dartree branch carries 50 non-merge commits over `origin/main`; local branches checked: `flint-spec/-dartree/-all/
  -perf`, `fix/sync-audit`, `fix/spec-audit`, `bridge-*`, `pr/*` (single-patch upstream candidates), `micro`,
  `split-balance`, `ab/audit-fixed`, worktree-agent-*. Nothing on them beyond the table above is relevant to Hy4
  (`flint-*` carries DFlash2 trees, flint-specific kernels `attn_decode_sink_h128_gqa8`, expert-parallel band `ep_band`).
* Not verified: the cost of a tail step on Hy4 (FP-1), whether the engine can snapshot a windowed group in a step (FP-2),
  whether Hy4's prefill MoE shapes take the packed map (FP-8), the first-sibling capture under DFly2's law (FP-6).
* All Flash-Next percentages are from docs at 210 W unless stated; Hy4's step is more expert-bound than Flash-Next's, so
  prefill-side levers that save a *step* are worth more here than there, levers that save a *row* about the same.
