# Issue #251 — "TP2 silent 60 s hang: stale scalar-cache chain record and body scales": closed, the change not taken

The report (issue #251, krisf) and its fix (PR #252, `42befd90`):

> On TP2 (2 x R9700), under sustained concurrent load the engine goes quiet for ~60 s and the queue watchdog fires
> a fatal dump (exit 70). The dumped queues are idle; the chain `done` counter is short by exactly the block count
> of a kernel that ran but never counted. […] A bare chained launch carries FENCE_SCOPE_NONE, so the device does not
> empty the scalar data cache before the kernel reads its chain record. Ring slots are reused: a slot still holds
> the previous pass's record in the scalar cache, the magic check fails against that stale copy, the wait returns
> early without acquiring, and `done` skips the count […] The same staleness hits `r4d_chain_wait_vec()` body reads
> of scalar scale factors (Sa / Sb / SCL).
>
> Fix: `s_dcache_inv` at the entry of `r4d_chain_wait_impl()`; `r4d_gemm_fp8a8.hip`, `r4d_hc_bf16.hip`,
> `r4d_rmsnorm_quant_i8.hip` use `r4d_chain_wait()` instead of `r4d_chain_wait_vec()`; the audit excludes the entry
> invalidate (`inv > 0`).

**Closed without the change, on the lane's measurement** (`../measurements/issue-251.md`, iks-A, TP2, before
`cfd79b13`, after `f0e43395`): the default branch does not hang under the load the report names, the kernels it
names read no scale through the scalar cache behind their wait, and the change costs 0.6 % of the decode step. The
commit is taken out again in `aa849bcf`; the PR merges what is useful beside it.

## What was checked

1. **The hang before the change.** Four runs of 320 requests (40 waves of eight at ~12K tokens of context) on the
   default branch, two on `nospec.yaml` and two on `coupled03.yaml`, and a decode-heavy one (160 requests of 1,024
   tokens): every request answered, the server serving at the end, no `E` line in the log. Two runs with the change: the same. A read of a record the kernel does not accept is
   not a rare glitch here but a hang of that run (the kernel neither waits nor counts, the counter stays short for
   good), and a run is 9 to 18 million launches over 2,000 to 4,000 laps of a card's argument ring.
2. **The record of a reused slot.** The diagnosis leaves out what `core/device/aql.cpp` already does for exactly
   this read: a chained packet goes out bare only while the ring does not ask for an acquire, and it asks every half
   lap at system scope (`acquire_scope`), which the packet keeps; and the host's write of the slot is flushed
   before the doorbell (`host_writes_out`). Both were in the reporter's base. `tests/device_test.cpp` holds the case
   itself: `a_copy_issued_after_a_pause_finds_the_record_written_for_it` leaves a lap of records no kernel accepts
   and then issues 6,000 chained launches with their packets unacquired, each reporting whether it found its
   record; `a_chained_launch_reads_its_own_arguments_after_the_ring_has_lapped` does three laps. Both pass on the
   default branch on a TP2 pair, three runs of three.
   A stale copy of a slot's last *chained* occupant would also not fail the check the way the report says: it is a
   record that checked out when it was written (same magic, same counter, an older target), so the kernel would
   wait for too little, not skip its count.
3. **The body reads.** `tests/chain_scalar_audit.py --check` on the default branch: 2,310 chained kernels, 798
   skip `s_dcache_inv`, 0 of those have a scalar load behind the acquire. `gemm_i8_kpar_kernel<…, r4dchain>`,
   `hc_read_fused_kernel<…, r4dchain>` and `r4dchain_ar_rmsnorm_quant_i8_sp` are each listed `ok … scalar loads
   behind the acquire: 0`: `gemm_i8_kpar`'s `Sa` is a per-lane (vector) load of a static weight's scale, issued ahead
   of the wait on purpose, and the scalar loads the other two do behind the acquire are kernel arguments. Greedy
   text is the same before and after on both configs, which a stale scale would not leave it.
4. **The cost.** `nospec.yaml`, three interleaved runs a side: c1 13.244 -> 13.328 ms/step (+0.63 %, ranges
   apart), c8 pass wall 23,953 -> 24,159 us (+0.86 %), `gemm_i8_kpar` 16.8 -> 17.1 us a launch (+1.8 %; 125 launches
   a pass). `coupled03.yaml`: inside the run-to-run range at the step, the same +1.7 % on `gemm_i8_kpar` at c1.

## What merges instead

- `core/device/aql.cpp`: krisf's chain line in the stall report, as `chain_line()` — issued blocks, the counter as
  the host reads it, what the slot at the read index needs, the flush outstanding. If a queue ever does stand idle
  on a short counter, the report now shows it, which is what the issue could not.
- `tests/device_test.cpp`: `the_stall_report_gives_a_queues_chain_accounting` (a lap and a half of chained launches,
  unacquired and acquired: one block a launch, the counter never ahead, every launch at its own slot).
- `scripts/backlog/probe.py`: the `stress` phase the runs above used, and `nowarm` as a phase; tested in
  `tests/backlog_stamps_test.py`.

## If it comes back

A hang of this kind leaves `server-exit=70` and, from now on, a `chain: issued N, counter reads M …` line per queue
in `server.log`. With that line, the config and the commit, it can be told apart from the two things that look
like it: a host read of the counter that lags (ended by `chain_flush`), and a run stopped by something else. The
lane saw one of the latter while measuring this: on iks-B (Engram read from disk, `disk=~/ngram`) the `stress` run
stops in wave 20 with `avx_embed_lookup_q … io error -- the engine cannot continue`, three runs of three, with and
without the change, the host at 3 GiB of free memory with two lanes up. That is the disk Engram reader
(`libavx/avx_ngram.cpp`), not a chain wait, and it is open.

To put the scalar half of the acquire back behind every wait for an experiment: build with
`-DR4D_CHAIN_SCINV_ALWAYS` (`libr4d/r4d_chain.h`); the entry invalidate is `42befd90`'s hunk in the same file.
