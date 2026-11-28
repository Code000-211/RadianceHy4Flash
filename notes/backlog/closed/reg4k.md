# reg4k (the pinned host pool on 4 KiB pages): CLOSED — output identical, but the step is not faster and the end to end is flat: pass wall -1.2 % to +0.3 %, c8 +1.4 % under DFly2 (inside the run's own spread, and speculative acceptance, not the pass), +0.1 % under nospec, bring-up +3.2 s

Item as written (issue #310, a follow-up of #307): "A/B the pinned host pool (`host_pool`,
`RAD_MEM_HOST_MAPPED`, `core/device/aql.cpp` `pin_pieces`) on 4 KiB pages (`MADV_NOHUGEPAGE`) instead
of THP, alone. `notes/backlog/measurements/pcie-h2d.md` measured `reg4k` at +20-25 % SDMA H2D
throughput with both TP2 cards copying at once vs THP-backed buffers. Switch:
`RADIANCE_HOST_POOL_PAGES=4k` (default = current behaviour). Merge bar (CARDS.md §2): output identical,
e2e better on DFly2 c8, not worse on c1 / nospec."

This branch built the switch (`ae9401c7`, with its tests in `tests/mem_test.cpp`), verified both arms
through `/proc/<pid>/smaps` of the pool's own mapping (100 % `AnonHugePages` unset, 0 kB under
`RADIANCE_HOST_POOL_PAGES=4k`, four boots an arm), and measured it on the lane `iks-A`, TP2, one binary
serving both arms, six interleaved runs a side a config over two sessions. The full record, with the
raw lines, is `notes/backlog/measurements/reg4k.md`. Against the bar:

1. **Output identical**: yes — the greedy phase's 4 texts a config, one sha256 set each
   (`0075e702dfdf…` DFly2, `0699709c4be2…` nospec) across all 24 runs.
2. **The step measurably faster in ms**: no. Decode pass wall 3861 → 3816 us at c1 under DFly2
   (-1.2 %, the arm's own spread is 35 us), 5039 → 4998 us at c8 (-0.8 %); under nospec, where every
   step is the same work, 12966 → 13000 us at c1 (+0.26 %) and 24072 → 24058 us at c8 (-0.06 %). The
   c1 step is 22.870 → 22.848 ms (DFly2) and 13.144 → 13.166 ms (nospec).
3. **Not worse beyond noise**: c8 under DFly2 +1.4 % (392.6 → 398.1 tok/s), inside the arm's own
   379.0-418.0 spread and driven by speculative acceptance (tok/step 20.28-23.54), not by the pass;
   nospec +0.1 % at c8 and -0.2 % at c1; TTFT the same to the millisecond.

So the 20-25 % of SDMA H2D throughput the `pcie-h2d` bench sees against THP does not reach the served
engine at TP2 — the pool's page size is not what a c1 or a c8 step waits on — and the arm pays 3.2 s of
extra bring-up for the same 8708 MiB pool. `ae9401c7` is reverted in the same branch (`49d98c06`; it stays in the
branch's history) and nothing merges: the note above is the item's deliverable. Whoever wants the
measurement again re-applies that commit; the default stays THP.

The same conclusion was reached twice before (`098e7b16` on the local lane, `2b2e7a4a` … `d1c73b9d` on
iks-A/B with two trees); this run is the third, with one binary serving both arms so no build
difference sits in the numbers.
