Measurement note and closure for X-03 (#130). No engine code is proposed: the card's gain does not exist on the served path, so it closes per CARDS.md §2.

**Result.** Local lane (TP2, u4 disk model, `--decode-checkpoint-interval` 0 against 256, two replicates, 1,024-token replies): turn-2 prefill is **1,108 / 1,098 / 1,096 tokens in both arms** (all 12 arm-runs), the replies are byte-identical, and c1/c8 sit inside the replicate-to-replicate spread. The flag writes its snapshots (`linear_checkpoints_written_total` 4 -> 16-21 on turn 1) and moves the next turn's prefill by not a token.

Two measured reasons, in `notes/backlog/closed/X-03.md`:

- A snapshot that passes the store's own position check is written about **once per 1,000 reply tokens**, not every 256: in a 1,024-token reply, 17 slots were reserved, 16 rows were cut short (the store refuses and returns the slot) and one committed, at 7,712 — 85 tokens into the reply. The documented `n_spec == 0` fallback would raise the rate but force the one-token boundary row the design set out to avoid.
- No position inside a reply is **reachable**: the assistant turn does not re-render token-identically — the reply's 16th token is `13` as generated and `364` when the same text is encoded through the chat template — so the prefix chain departs 13 tokens into the reply (7,640) and the one snapshot can never be matched. It was still in the index at turn 2 (`[1984 3968 5952 7616 7712]`), so retention/LRU is not the blocker and the key and session identity are the same on both sides; the content-addressed chain is.

The store's contract held where it was exercised: the row that committed at exactly its planned position was accepted and indexed. It was never resumed into, so the saved state's content is untested — the identical replies prove the flag is inert, not that the snapshot is right.

**What ran**: two replicates x two arms x (three `longresume24000` conversations + c1 + c8 + four greedy texts) on `127.0.0.1:8041` under `/tmp/hy4-lane.lock`, the house engine stopped for the window and restored after it; plus `ctest -R 'sched_test|mem_test|config_test|arch_test'`. `config_test` needed the new flag in its inventory (fixed here); `mem_test_card` failed only while the house engine held both cards and passes with them free. What did not run is in the note.

The implementation, including the `step_next_locked()` hook without which the handed-over build never reserved a snapshot at all, stays on `krisf/RadianceHy4Flash:x03-impl` (`3f51b541`), unmerged.

Also adds the `longresumeN` phase to `scripts/backlog/probe.py` that produced the numbers.

Fixes #130