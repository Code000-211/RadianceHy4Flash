# Audit D: host side (server, text, scheduler, engine step, AQL, sampler)

Read-only audit of `src/core` @ c655f58c (branch claude/peaceful-wright-hhqnhi). Nothing built, nothing run.
Paths are relative to `/root/work/claude-scratch/hy4flash/src/`. "est." = an estimate from code facts plus the
measured figures in COMMON.md; "unverified" = not proven by code I read.

## 0. Result in one page

* The host is not on the decode critical path, as measured (levers-general §1.5). Everything per token on
  the host (sink, detokeniser, SSE, JSON escape, stop scan) is O(1) per token and already tuned (fast paths,
  SIMD escape, segment cache, lock-free sink, wake only while the consumer sleeps). Per-token host cost
  is est. 10-30 us of HTTP-thread CPU plus one futex wake and ~5 syscalls. Not a lever.
* **The largest finding is a scheduling one, provable from the code (D-1).** While decoders ride, every
  2048-token checkpoint interval of a prompt is cut into **two** steps: a chunk of
  `2048 - ceil16(D)` rows and a **remainder step of ceil16(D) = 16-48 rows** (D = the decode rows' tokens).
  Each remainder step pays a whole step floor (an expert pass, live issue, its own interleave run) and
  stalls the decoders once more. This is exactly what FP-3 (`decode-headroom`) removes, so FP-3's
  precondition ("remainder steps exist in our schedule") is now **proven from code**. FP-3 also has a
  **config-only equivalent**: `checkpoint-interval: 1984` with `max-num-batched-tokens: 2048` (D-1).
* **FP-2 refinement (D-6): smaller than assumed.** The tail step is at most 15 rows (quantum 16). Estimated
  cost is a decode pass plus ~15 verify-like rows, ~25-35 ms. That saves ~4 % of the ~0.7 s resumed-turn
  TTFT measured in `work/churn/rs*.out`, not 0.1-0.25 s. Also, the scheduler only takes an in-step snapshot
  under `prefill-span > 1` (`scheduler.cpp:1276-1277`), so the plugin flag alone would change nothing.
* **New stall lever (D-2): layer-sliced prefill chunks.** Run a chunk's 42 layers over m consecutive steps.
  Each step runs the decoders' full pass plus one slice of the chunk's layers. Experts are still read once
  per layer for the chunk's 2048 rows (unlike a small chunk), so prefill throughput stays put. The stall
  drops from ~0.52 s to ~0.52/m + one decode step. L effort, plugin and engine. It beats FP-10:
  FP-10 needs a second collective instance, a second arena and CU partitioning (D-5).
* **Existing knob not in the FP list:** `--mixed-step-target-ms` (off by default). It learns a cap on the
  prefill rows of a mixed step. Like `busy-prefill-chunk`, it moves along the TTFT/stall frontier and does
  not lower the floor. Its decode reserve is applied after the stager floor, so it can push chunks below
  1024 rows (D-4).
* Arrival latency: issue-ahead stops as soon as a request waits (`scheduler.cpp:2962`). A newcomer
  therefore waits 1-2 decode steps (one already issued ahead plus the one in flight) before its first
  chunk step is built. D-3 lets the mixed step itself be issued ahead.
* Smaller host items (D-7..D-20): the suffix-index build at first token is O(prompt) under the scheduler
  lock. The chat template renders the whole history twice per request. Mixed and prefill passes are never
  replayed (live issue; it matters for small mixed steps; measure). The 64-packet first doorbell, the
  condvar barrier with no pinning, and the token stream waiting behind the adaptive-depth walk are also
  listed.

## 1. Thread and wake-up map (code facts)

Threads: one step thread (`Engine::step_loop`, engine.cpp:1314). One rank thread per rank
(`rank_loop`, :1162): 2 at TP2. HTTP workers: `--http-threads`, default one per core and at least 4, up to
128 (`http.cpp:398` ConnectionPool; the log shows 48 on the EPYC 7402P). Plus `rad-xp` / `rad-rows` /
StallWatch helpers. **No thread is pinned** (no `setaffinity` anywhere in core) and none has raised
priority.

Synchronisation primitives:
* `RankBarrier` (runtime/rank_barrier.cpp:6-21): mutex + condvar, n = ranks + 1 = 3. Two rounds per
  release: release, then join.
* Engine host waits: `host_wait` (device/aql.cpp:1769). A KFD `AMDKFD_IOC_WAIT_EVENTS` ioctl on an
  interrupt signal, which blocks with no polling. The code comment prices it at "some tens of us".
* Full ring: `full_nap` nanosleep of 100 us (aql.cpp:1112-1116), only while 4096 slots are full.
* Sink: SPSC ring (server/sink.cpp:67-99). The producer calls `notify_one` only when `asleep_` is set. The
  consumer's `wait_for` is bounded at 25 ms (`RAD_SINK_POLL_MS`, sink.h:30), which is also the liveness
  poll.
* Scheduler wake: `wake_cv_` (scheduler.cpp:374-390). The idle step loop waits `idle*100 us`, growing to a
  20 ms cap (engine.cpp:1408), and is woken at once by `add`.

**Hops from "tokens sampled on the card" to "bytes in the socket" (one stream, steady decode):**
1. The card writes ids to the pinned readback (`collect_issue`, engine.cpp:1759ff, n_rows x 4 bytes). The
   `tok_end_` (early ids) or `step_end_` record raises an interrupt, and the step thread returns from
   the KFD ioctl (**hop 1: interrupt wake**).
2. On the step thread: memcpy of the ids, greedy accept, grammar `accept_token` per rank
   (`collect_step`, :2268), then `Scheduler::commit` under `mu_`, then `sink_push`. While the consumer
   sleeps this is `cv.notify_one` (**hop 2: futex wake**, a syscall on the step thread, inside `mu_`).
3. The HTTP worker returns from `Sink::wait` inside `res.next` and returns empty output. Back in httplib's
   loop (`httplib.cpp:4384`): `wait_writable` = poll(POLLOUT), then `is_peer_alive` = poll(POLLIN, 0)
   (+ `recv(MSG_PEEK)` if readable). Then `next()` again: `peer_closed()` = another poll (+peek)
   (server.cpp:1279); `pump` (sink read, detok push, stop scan, SSE build). Then `sink.write`:
   `write_data` -> `SocketStream::write` = poll(POLLOUT) + `send` (httplib.cpp:7951).
   So **2 wake hops + ~4 polls + 1 send per drain**. One drain carries all tokens of a step for that
   stream (1 + accepted drafts).
4. With DFly2 sampled adaptive, commit runs only after `issue_ahead` (engine.cpp:1629 before :1631).
   `issue_ahead` runs `block_depth_decide` (:1265), which waits for the walk's laws, i.e. for the drafter
   passes. Tokens are therefore committed and streamed one draft pass (~3.9 ms) after they were sampled
   (D-11).

**Hops from "request parsed" to "first step issued":**
* Idle engine: HTTP `add_all` (lock `mu_`, `wake()` notify_all) -> step loop wakes from `wait_idle`
  (**hop 1**). Then `step()` (admission, lookup, plan, `builder_.build` with uploads) -> `prepare_release`
  (arena recall if needed, D-9) -> barrier release: notify_all wakes 2 rank threads (**hop 2**). Each rank
  issues live (a prefill pass is unkeyed or first-seen; ctx.cpp:1424). The first doorbell comes after 64
  packets (D-13).
* Busy engine: no hop is needed, because the request is seen at the next `step()`. But `step_next_locked`
  refuses to issue ahead while `waiting_` is non-empty (scheduler.cpp:2962). If the arrival came after the
  current issue-ahead, the newcomer waits for the step in flight **and** the step already issued ahead:
  1-2 decode steps, 15-60 ms at c1-c8 DFly2.

## 2. The stall on arrival, precisely (part d)

Configuration as served: `max-num-batched-tokens 2048`, `busy-prefill-chunk auto`, `prefill-interleave
auto:1` (share 1.0), `prefix-resume nearest`, `checkpoint-interval 2048`, quantum 16 (log: "chunk geometry:
quantum 16 ... interval 2048"). Hy4 declares `RAD_KV_LINEAR` (indexer tail ring, hyv4flash.cpp:708) and
`RAD_KV_CONV` (Engram, :973). That gives `has_linear = true`, hence `window_ckpt = false` and `span = 1`
(geometry.cpp:151-155): interval checkpoints, the "linear" tail rule, no in-step snapshot.

Timeline with N decoders (DFly2 block drafter, pipelined):
1. **HTTP thread** (server.cpp:500-600, oai.cpp:2300-2370): JSON parse, chat render (twice, D-15),
   encode (segment cache), grammar compile (cached), `submit_all` -> `add_all` under `mu_`.
2. **Next full step** (`Scheduler::step`, scheduler.cpp:2069). Issue-ahead stopped at :2962. The running
   loop (:2296ff) places every decode row first (`want = 1 + n_draft`). Admission (:2511ff) runs
   `admit_one` (:1585): `pc_->restore` (host-tier restore), `lookup` (hash of all prompt blocks,
   prefix.cpp:392), adopt, then `busy_cap` (:1845). With one waiter `areserve = 0`, so `busy_cap` returns
   the whole remaining budget (:1474). **`auto` does not shrink the chunk when the decoders are ahead of
   the newcomer in `running_`.** The chunk is `plan_chunk(.., budget = 2048 - D)`.
3. **Chunk shape** (`plan_chunk`, nearest branch :1222ff). From `n_computed = 0` (or a hit), `next` is the
   next 2048 boundary. `2048 - D` is below it, so no clamp. The quantum alignment (:1305-1318) cuts it to
   `2048 - ceil16(D)` (e.g. 2032). **Next step:** `n_computed = 2032` and `next = 2048`, so
   `n + want > next` and :1250 sets `want = 16`. That is a 16-row (up to 48-row at c8) **remainder step**,
   which carries the 2048 checkpoint. The comment at :1490-1495 confirms that this "step pair" is
   intended. Every later interval repeats the pair.
4. **Each mixed step is one pass over all rows** (decode rows sorted first, :2650-2660; phase MIXED). The
   decoders get their token only at the end of the whole pass: chunk ~0.52 s plus the DFly2 context pass
   over the chunk rows plus the query pass. Their tokens are streamed after `tok_end_` (early ids), before
   the drafter passes finish.
5. **Interleave**: after any step that carried a chunk beside decoders and leaves the prompt unfinished,
   `interleave_left_ = interleave_steps()` (:2770). That is `share * ema_chunk / ema_decode` decode-only
   steps, with the EMA taken over **every** non-decode step's wall time (:3413), remainder and tail steps
   included (D-8). The remainder step arms its own interleave run.
6. **Mixed steps are never issued ahead**: a mid-prompt entry has `!produces_token` (:2979), and a held
   chunk returns at :2965. They are never replayed from a tape (pass key 0 or first sighting:
   ctx.cpp:1378-1427). Every mixed step is issued live by both rank threads (D-12).
7. **Checkpoints at chunk boundaries** (`save_checkpoints`, engine.cpp:2001): async D2H copies on the
   compute stream under `mu_`, no host sync. The size is ring + CONV + window (FP-2 says ~17.4 MiB a rank
   with the window), so ~0.7 ms of copy serialised in-stream per checkpoint, ~0.1 % of a chunk. The async
   variant is CLOSED (`RADIANCE_CKPT_ASYNC`, flashnext-port §0.4).
8. **Last chunk**: `produces_token = n_computed + want >= target` (:2488). **The last prefill chunk samples
   the first token in the same step.** Under nearest, the last chunk ends at `tail = floor16(target)`
   when that lies ≥ 512 past the interval (:1403). The **≤ 15-token tail** then runs as its own step, which
   produces the first token (D-6).
9. **Arena recall** on the first large step after ≥ 2 s of decode only (`arena_tick` floor hold,
   engine.cpp:2520): `prepare_release` -> `arena_step` -> `Mover::reclaim` does
   `rad_stream_sync(mover_)` + `rad_stream_sync(compute_)` (mover.cpp:1735, :1739). It drops the
   experts lent into the arena, so decoders and prefill read them over the link until re-lent (D-9).

What each decoder sees per 2048 prompt tokens: one 0.52 s stall (the chunk), a run of decode-only steps
lasting ~ema_chunk, one short stall (the remainder step, est. 60-120 ms, unverified), and another
decode-only run.

## 3. Findings

### D-1 Remainder steps while decoders ride: kill them by config (checkpoint interval = budget − 64) or FP-3
* area: scheduler (config). Location: `core/sched/scheduler.cpp:1250` (cut to the interval), `:1305-1318`
  (quantum alignment), `:1474` (auto returns the whole budget), comment `:1490-1495`; `geometry.cpp:147-149`
  (interval rounded to the quantum).
* Change, option (a), config only: `checkpoint-interval: 1984` with `max-num-batched-tokens: 2048`. Every
  chunk is then ≤ 1984 rows and ends on an interval boundary, with ≥ 64 rows of headroom for decode rows
  (c8 x (1+5) = 48). The remainder step disappears. A lone prefill takes 1984-row chunks, 3.2 % more steps.
  Option (b): FP-3 `decode-headroom` (cherry-pick 15434c6b): keeps 2048-row chunks and grows the arena by
  the headroom.
* Why it pays: one step less per 2048 prompt tokens under load. Each removed remainder step is a full
  layer stack for 16-48 rows. The stager is not armed below 1024 rows (stager.h), so the pooled ~10 % of
  experts touched are read in place over the link at 66 us a unit. Est. 60-120 ms a step (unverified;
  FP-1's 256-row step is 270-350 ms). It also stops the step from arming its own interleave run of
  ~ema_chunk decode-only steps and from being a second decoder stall per interval. Est. TTFT under load
  −10..20 % for multi-interval prompts, and half the number of decoder stalls. The lone-prefill cost of
  (a) is ~+1.5 % (est.: 3 % more steps, about half of each step's time fixed).
* Lossless: yes. The same math in different chunking; under exact the grid moves with the interval, and
  under nearest only where checkpoints sit.
* Sure: high that the remainder step exists (code path traced line by line); medium on its cost.
  Effort S (a) / S-M (b).
* Verify: `RADIANCE_LOG_STEPS=1` with 2 decoders and a 6K arrival. Count steps per interval: 2 now, 1
  after. Compare the FP-1 sweep table (worst stall, TTFT) between interval 2048 and 1984. Also check
  `sched_test` for interval and chunk expectations.

### D-2 Layer-sliced prefill chunk (decoders ride a slice of the chunk's layers) — new
* area: engine + plugin + scheduler. Location: the mixed-step composition (`scheduler.cpp:2650-2700`,
  `RadBatch::n_seq_decode`), the hyv4flash step function (arch/hyv4flash/hyv4flash.cpp), the pass key
  (`ctx.cpp:1008`).
* Change: plan a chunk as m sub-steps (e.g. m = 3-4). Sub-step j runs the decode rows through all 42
  layers and the chunk rows through layers `[L_j, L_{j+1})`. The chunk's residual and iHC rows and its
  DFly2 taps persist in a `PERSIST` buffer between sub-steps. The KV/indexer/Engram writes of the chunk
  rows happen in their own layers, as now. Per layer, ops take `n_tok_decode` rows outside the slice and
  all rows inside it (row counts per op already vary per op; the attention split by `n_seq_decode`
  exists). The pass key gets `(L_j, L_{j+1})`; mixed passes are unkeyed today anyway.
* Why it pays: decoders get a token every ~(0.52/m + ~0.015) s instead of every 0.52 s. Each layer's
  experts are still read once for the chunk's 2048 rows, so prefill rate is unchanged, unlike
  `busy-prefill-chunk 256` (TTFT x7). The decoders' passes are the same count as interleave already runs.
  The stall falls to 0.13-0.18 s at m = 4 (est.) and TTFT stays near the auto row of the FP-1 table.
* Lossless: yes (each chunk row's per-layer math is unchanged; the rows of a GEMM are the same set inside
  the slice, while decode rows join different GEMMs, as in any mixed step).
* Cost: VRAM for the persistent chunk state, est. 2048 x (iHC rows x 3072) x 2 B per rank. With 4-8
  hyper rows that is 50-100 MiB (unverified: iHC row count). The stager must stage per slice (its "next
  step's first layers" logic, stager.h, already spans passes).
* Sure: medium (design, not yet in code). Effort L. Depends on D-1 (no remainder steps) to keep the
  shapes simple.
* Verify: greedy byte-identity of the prompt's continuation against unsliced (`tools/hy4flash_rollback_gate.py`
  style), `sched_test` shapes, FP-1's stall table.

### D-3 Issue the mixed step ahead (admission and mid-prompt chunks at issue-ahead time)
* area: scheduler/engine. Location: `scheduler.cpp:2959-3042` (`step_next_locked` refuses: `waiting_`
  :2962, held chunk :2965, `!produces_token` :2979); `engine.cpp:1254-1290`.
* Change: let `step_next_locked` build a MIXED plan of device decode rows (`dev = true`, positions from
  the card's acceptance, as now) plus a prefill entry: the newcomer's admitted first chunk, or the next
  chunk of a running prompt, whose position the host already knows. Admission runs there under `mu_`, as
  `step()` would. Keep the remaining refusals (grammar, suffix_hot).
* Why it pays: arrival waits for the in-flight step only, not also for a step already issued ahead. TTFT
  −0.5..1 decode step on average (est. 8-30 ms). The host turnaround between a mixed step and its
  predecessor (interrupt wake + commit + step() + build + barrier + live issue start, est. 0.2-1 ms)
  overlaps the card. Small.
* Lossless: yes (scheduling only). Sure: medium. Effort M: the plan_next phase is hard-coded DECODE
  (:3022), `ends_every_request`, arena level and stager `set_next_prefill` for an issued-ahead step
  (engine.cpp:2482 excludes it).
* Verify: `sched_test` issue-ahead cases; greedy texts identical; `engine_steps_ahead_total`.

### D-4 `--mixed-step-target-ms` exists and is not in the FP list; fix its floor interplay before trying it
* area: scheduler. Location: `scheduler.cpp:1500-1530` (`busy_cap`), `:3453-3470` (observe), config.cpp:80.
* Fact: it learns an aggregate prefill cap per mixed step against a duration target. The decoder reserve
  is applied after the stager floor (:1503), and `fair` can be any multiple of Q (:1525), so the learned
  cap can drive chunks below the stager's 1024-row floor. `set_stage_rows` warns that this case "reads
  every routed layer's pooled experts synchronously".
* Change: clamp the learned cap at `stage_rows_` the way auto does, or skip the target when the cap
  would go below the floor. Then add it to the FP-1 sweep (targets 300/200 ms).
* Gain: a frontier move only (same as `busy-prefill-chunk N`). Lossless. Sure high (code); S.

### D-5 FP-10 refinement: what exists for a decode lane beside a prefill chunk, and what blocks it
* Exists: (1) a second HW queue per rank: `Ctx::lane1_` (`ctx.h:757-799`, `ctx.cpp:955`), built by
  `rad_stream_create`, i.e. `hsa_queue_create`, with HIGH priority available
  (`aql.cpp:2029 hsa_amd_queue_set_priority`). (2) Lane window and joins (`rad_lane`, `rad_lane_join`,
  `rad_lane1_window_begin/end`), used by qwen35_fp8 (`qwen35_fp8.cpp:1047-1051, 1515-1521`) for its
  two-piece pipeline. (3) An SDMA queue for the mover and prefetch (aql.cpp:3260ff). hyv4flash uses none
  of them.
* Blockers in code: (a) "Every collective rides the same lane" (ctx.h:760): the all-reduce's parity slot
  (r4d_ar_seq.h) breaks with two concurrent passes doing junctions, so a decode lane needs its own AR
  instance (scratch + counters) on both ranks. (b) Device-wide counters must be per stream (`split_ctr`,
  ctx.h:765). (c) The arena is planned from one linear order, so a concurrent decode pass needs its own
  arena region or `rad_buf_concurrent` for every buffer. (d) The prefill stager publishes transient VRAM
  addresses for experts inside the prefill pass; a concurrent decode pass reading the same weight tables
  would race them. (e) Sampler buffers, `tok_buf`, `step_end_` and the StepSlots are single.
  (f) Scheduling: one open step plus one issued ahead (`step_open_`, `next_open_`). (g) RDNA queue
  priority orders dispatch, not running waves. Prefill kernels run 1.75-5 ms (`mla_attn_sel_sink`), so
  each of the ~1300 dependent decode launches can wait that long for CUs unless the prefill queue is
  CU-masked (`hsa_amd_queue_cu_set_mask`; not used anywhere in the tree).
* Verdict: FP-10 is L+ with high risk. D-2 gives most of the stall reduction on one queue with none of
  (a)-(g). Keep FP-10 only as research after D-2.

### D-6 FP-2 refinement: the tail step is ≤ 15 rows, and the scheduler gate needs `prefill-span`
* Facts: under nearest, `tail_checkpoint` = `floor16(target)` when it lies ≥ max(Q, I/4) = 512 past the
  interval (`scheduler.cpp:1403`). The last chunk ends there and the remainder (1-15 tokens; none if the
  target is a multiple of 16) is the tail step, which produces the first token. The in-step path needs
  `ckpt_inside_ && prefill_tail_snapshot && geo_.span > 1 && span_now_ && prefill_span > 1 &&
  !window_ckpt` (:1276-1277). With `prefill-span` off (recommended, T2 closed) it never fires, even after
  the plugin sets `ckpt_snap_ok` (engine_bringup.cpp:1309-1313; stage only under nearest).
  `RadBatch::ckpt_cut` is one field (one cut per step; `ckpt_stage_arm` engine.cpp:1971 picks one entry).
  `pass_key` returns 0 for `ckpt_cut > 0` (ctx.cpp:1032).
* Cost of the tail step (est.): a decode pass (13.4 ms) + ≤ 15 rows at ~1.0 ms a verify-like row
  (levers-general §1.4) ≈ 25-30 ms at c1. It is a live-issued mixed/prefill pass (D-12), not issued ahead
  (:2979), so add the host turnaround. Resumed-turn TTFT measured 0.69-0.72 s
  (`work/churn/rs16/rs256.out`, ~220-440 new tokens). **FP-2's gain is ~4 % of TTFT, not 5-13 %.**
* Change if built anyway: plugin per FP-2, plus relax :1276 to allow `inside` without span for a prefill
  whose last chunk carries the tail (one cut per step: only the first such entry).
* Recommendation: let FP-1 measure the tail step first; most likely below the bar. Lossless; sure high on
  the facts, medium on the cost; effort M.

### D-7 Suffix and n-gram index built O(prompt) at the first decode, under the scheduler lock
* area: scheduler. Location: `suffix.h:159-168` (`index_to`), called from `suffix_propose` in `step()`
  (`scheduler.cpp:2361-2363`) and `commit()` (`:3294-3296`). `ngram.h` (incremental, same pattern).
  Both run under `mu_` on the step thread.
* Change: index during prefill. After each committed chunk call `index_to(seq, n_computed)`, while the
  card runs the next chunk and the host is idle. Or index the prompt on the HTTP thread before submission
  and hand the index over with the request. Same tables, same proposals.
* Why: FP-5 prices it at 0.19 us a token: 12 ms at 64K, 20 ms at 100K, once per turn. It currently lands
  in the first decode step or the first-token commit. When that step is not issued ahead it delays the
  card, and it always holds `mu_` (blocks `add`/`cancel`/`publish` for that long).
* Lossless: byte-identical proposals. Sure high (code). Effort S.
* Verify: `sched_test` suffix cases; time with `t_on` phases (plan/commit) at 64K.

### D-8 Interleave EMA mixes every non-decode step
* area: scheduler. Location: `scheduler.cpp:3413` (`step_cost_locked(phase != DECODE, wall)`), `:1542-1547`.
* Change: feed `ema_chunk_s_` only from steps whose prefill rows ≥ the stager floor (or scale it by rows).
  Remainder, tail and small resumed-delta steps pull the EMA down, so fewer decode-only steps follow the
  big chunks and the decoders' cadence after a 0.52 s chunk is worse than `auto:1` intends.
* Gain: steadier decoder cadence under arrivals (no TTFT gain). Lossless. Sure high. S. Becomes moot after
  D-1 except for tail steps.

### D-9 Arena recall at arrival is two stream drains plus dropped lent experts on the TTFT path
* area: engine/memory. Location: `engine.cpp:1234` -> `arena_step` (:2470-2505) -> `Mover::reclaim`
  (`place/mover.cpp:1724-1800`: `rad_stream_sync(mover_)` :1735, `rad_stream_sync(compute_)` :1739, units
  dropped to host :1780ff).
* Fact: the first ≥ 1024-row step after a 2 s decode-only floor recalls the arena loan synchronously on the
  step thread and drops the experts lent there. The chunk and the decoders then read them over the link
  (66 us a unit) until `arena_tick` lends again 2 s later and the slab refills (h2d promotions). This is
  the arrival side of `work/churn`.
* Change (host side): start the recall when a request is queued (at `add`, flagged to the step loop) so
  the drain and donor copies overlap the in-flight decode step. Or keep a minimum arena share unlent while
  any request waits. Effect size belongs to the churn study; listed for the TTFT path.
* Lossless. Sure high on the syncs, low on the size. M.

### D-10 Issue-ahead refusal reasons worth counting
* area: scheduler. Location: `scheduler.cpp:2959-2986`.
* Change: per-reason counters (waiting, held, mid-prompt, grammar, suffix_hot (exists), ends_every,
  KV) in `/stats`. Today only `suffix_held_ahead` is counted. Cheap; tells whether D-3 or the
  `draft-suffix` hold costs steps.

### D-11 Tokens stream only after the adaptive-depth walk (sampled DFly2)
* area: engine. Location: `engine.cpp:1629` (`issue_ahead` -> `block_depth_decide` :1265 waits for the
  drafter's laws) runs before `collect_step`/`commit` (:1631, :1689).
* Change: when `block_depth_decide` will wait (some row samples), run `collect_step` + `commit` first
  (wait on `tok_end_`, stream), then decide and issue ahead. The card still holds the drafter passes
  (3.9-6 ms), which cover the commit (~0.1 ms).
* Gain: every step's tokens reach the client one draft pass earlier (~4-6 ms of latency; rate unchanged,
  levers-general §1.5 closed the throughput side). TTFT −4 ms. Lossless. Sure medium (ordering
  dependencies: the commit must precede the next step's sampler staging, which it does). S-M.

### D-12 Mixed and prefill passes are always issued live; small ones may be host-bound
* area: runtime. Location: `ctx.cpp:1378` (stager-armed passes live), `:1032` (`ckpt_cut`), `:1424`
  (non-decode keys recorded only on second sighting, and the key carries `n_tok`, `max_ctx_len` bucket and
  staging pointers, so prefill shapes rarely repeat). `xs1/d0-*.json`: `issued_unkeyed 34` in one run.
* Fact: a pass issued live walks the issue path per op (band scan, operand resolution, packing; `issue.cpp`)
  for ~1500 ops on each rank thread. For a 520 ms chunk that is hidden. For a ≤ 15-row tail step, a
  16-48-row remainder step or a ~200-row resumed delta, card time is 25-300 ms and the live issue could
  approach it. Unverified: no live per-op figure was found.
* Change: measure first (`RADIANCE_DEBUG_ISSUE=1` prints own/launch us per op, issue.cpp:532). If a small
  mixed step is host-bound, key prefill rows by a bucket (16/64/256) and record mixed passes below the
  stager floor on first sight.
* Lossless (tapes are audited every 1024 plays). Effort S to measure, M to change.

### D-13 First doorbell after 64 packets on a step that starts an idle queue
* area: device. Location: `aql.cpp:1571` (`kPlayRing = 64`), `PlayRun::ring` :1597.
* Change: ring after the first 4-8 packets of a play when the queue was idle (read index == write index),
  then every 64.
* Gain: ~64 x 0.35-0.9 us ≈ 20-50 us earlier card start on steps not issued ahead (mixed, tail, arrival,
  interleave transitions). Small. Lossless; sure high; S. Verify: `RADIANCE_AQL_STAMP` first-dispatch time.

### D-14 Barrier and threads: condvar handoffs, no pinning
* area: engine. Location: `rank_barrier.cpp:6-21`. A DFly2 step does 8 barrier rounds (trunk 2, context 2,
  query 2, issue-ahead 2: engine.cpp:1560-1587, engine_draft.cpp:1038-1041, engine.cpp:1278-1281).
* Change: spin-then-futex barrier (spin ~20-50 us) and pin the step thread and rank threads to one CCX of
  the EPYC 7402P, which also keeps them out of deep C-states.
* Gain: only on steps not issued ahead. Est. 10-100 us per wake avoided. Low. Lossless; S. Verify: the
  `t_on` phase line (`ranks`) and the first-dispatch stamp.

### D-15 Chat template rendered twice per request; host prep time not logged
* area: text/server. Location: `vendor/llama/common/chat.cpp:964-979`
  (`common_chat_template_generation_prompt_impl`: one extra full render besides the prompt's own), called
  from `text/chat.cpp:456` (`common_chat_templates_apply`). Plus `alias_reasoning` copy (`chat.cpp:214`),
  `common_chat_msgs_parse_oaicompat` / `render_message_to_json` copies.
* Change: (1) measurement: time parse / render / encode / grammar / submit per request and put it in the
  existing `req ... done` line (server.cpp record()). (2) If render is ≥ ~10 ms at 64K histories: cache
  the generation-prompt suffix per (template, kwargs, last role, thinking flag), verified on first use by
  doing both renders. That halves the render.
* Gain: unverified; O(history) on the HTTP thread, directly on every turn's TTFT. Lossless if the cached
  suffix is checked. S (1) / M (2).

### D-16 Sink wakes issued inside the scheduler lock
* area: scheduler/server. Location: `scheduler.cpp:3086` (`commit` holds `mu_`), pushes at :3172 / :3232,
  wake at `sink.cpp:97`.
* Change: collect sinks to wake and notify after `mu_` is released. Saves one futex syscall per stream
  per step inside `mu_` (est. 1-3 us each). Tiny; lossless; S.

### D-17 `publish()` rebuilds the full request table every busy step under `mu_`
* area: scheduler. Location: `scheduler.cpp:4013-4027` (idle publishes are throttled to 20 ms, busy ones
  are not), `requests_locked` :1025-1075 (per request, a loop over KV groups, then a sort).
* Change: throttle busy publishes to 10-20 ms too. Est. a few us per step at 8-16 requests. Tiny; S.

### D-18 Penalty history uploaded at the full `max_hist` width
* area: sampler. Location: `sampler.cpp:1241-1247` (`n_seq_ * max_hist * 4` bytes), staging :966-1000.
* Change: upload `max(take)` columns (or per-row ranges). At 8 rows x 6 positions x 2048 x 4 B = 384 KiB
  per rank per step when any request uses penalties/DRY. Tiny (~15 us of h2d); S.

### D-19 Per-token HTTP syscalls (httplib chunked loop)
* area: server. Location: `httplib.cpp:4384` (`wait_writable` + `is_peer_alive` per provider call),
  `:7951` (`write` polls again before `send`), `server.cpp:1279` (`peer_closed()` is one more
  `is_socket_alive`).
* Change: drop our own `peer_closed()` probe (httplib already checks `is_peer_alive` before every
  provider call) and let `SocketStream::write` skip its poll right after `wait_writable` (vendored patch).
  Saves 1-2 poll(+recv) per drain. Est. 1-3 us per token per stream. Tiny; lossless; S.

### D-20 Prefix lookup and publication hash the whole context under `mu_`
* area: scheduler/memory. Location: `prefix.cpp:392` (`chain_all` per lookup, including every retry of an
  `admit_one` that waits), `publish_prefix` (scheduler.cpp:723) at finish.
* Fact: O(tokens/16) chained hashes plus a vector allocation per attempt. Est. 0.1-0.5 ms at 64K
  (unverified). Change: cache the request's chain hashes in `SchedReq` (the prompt is immutable) and
  extend them for the output at publish. Small; lossless; S.

## 4. Format / lossy (the friend's decision)
None from this area.

## 5. Checked: already good or not a lever (do not re-derive)
* Sink: lock-free SPSC, no allocation in `push`, wake only while asleep (sink.cpp:67-99). The 25 ms bound is
  a lost-wake fallback that the `asleep_` protocol repairs on the next push.
* Detokeniser: O(1) per token (`tokenizer.cpp:1833-1895`: incremental closer search, `pending_` small,
  stop prefix ≤ maxstop). Server stop scan restarts maxstop−1 back (server.cpp:790ff);
  `stop_prefix_tail` is budgeted.
* SSE build: cached chunk head (`oai.cpp delta_open`), SIMD `json_quote_append`, `json_stable_prefix` O(1).
  The reply parser feeds only new text. Logprob entries are trimmed after sending.
* Tokeniser: per-segment LRU cache between special tokens (`tokenizer.cpp:270ff, 1511-1526`), so
  re-tokenising the history each turn mostly hits.
* Readback: ids only, n_rows x 4 B to pinned memory, one interrupt wait per step (engine.cpp:1772, 2280).
  Logprobs are summarised on the card (`sample_logprobs`).
* Block tables: delta-restaged (batch.cpp:680-860); bounds bucketed (`bound_bucket`, batch.cpp:48) so
  decode tapes survive context growth. Tape audit every 1024 plays (ctx.h:624), LRU of 48 tapes. One
  measured run recorded 8 of 4128 passes (`work/xs1/d0-B.json`). Still worth one `/stats passes` check
  under c8 DFly2 adaptive (more `n_tok` shapes).
* Host waits are KFD interrupts, not spins (aql.cpp:1769). The full-ring nap (100 us) only happens when
  4096 packets are queued.
* Prefill chunk steps that sample nothing do not wait (`collect_step` returns at `n_rows <= 0`), so lone
  chunk steps are effectively pipelined.
* Grammar: schema->GBNF and compiled-program caches (`sample/grammar.cpp:30ff`). DRY breakers are resolved
  once per request.
* Logging at the default level: nothing per step or per token. `RADIANCE_LOG_STEPS` and the phase line are
  opt-in; INFO lines are per request (`req ... done`) or once per pass shape (`recorded pass`).
* Per-step vectors in the step loop and builder reuse their capacity (`so.*.assign`, `o.seqs.clear()`).
  No per-step heap growth was found on the hot path.
* CLOSED and confirmed by reading: adaptive-depth host wait (levers-general §1.5); sampler chain vs greedy;
  async checkpoint copy (flashnext-port §0.4); span-pass/prefill-span (T2); two-piece pipelined prefill
  (T4); expert prefetch at TP2 (T3).

## 6. Suggested order
D-1(a) config A/B inside FP-1's session (S, biggest, near-free) -> D-7, D-11, D-4 clamp (S) -> D-12 and
D-15(1) measurements -> D-3 (M) -> D-2 design note and prototype (L) -> D-6/FP-2 only if FP-1's tail step is
≥ 0.1 s. D-13/14/16/17/18/19/20 as one host-hygiene batch.
