Fixes #N

### Item
<!-- The backlog item's ID and title as notes/backlog/optimisation-backlog.md writes them, `K-47: <subject>`. Title the pull request `[measured] K-47: <subject>` when notes/backlog/measurements/<item>.md covers the item's whole merge bar, `[needs testing] K-47: <subject>` when it does not (no GPU). The tag comes first. -->

### What the change does
<!-- Say what changed in sentences a reviewer can check: the kernel, op or step, and what it does differently, and the tests that come with it (a code path without a test in the same tree is a blocking finding).
The merge bar (notes/backlog/CARDS.md §2, lossless items) is three things, all of which must hold:
1. The output is byte-identical, or inside the item's stated bound (ORD / PF).
2. The kernel, op or step the item targets is measurably faster in ms (AQL stamps, rad-kbench or profile-ops), before and after on one machine.
3. End-to-end tok/s or TTFT is not worse beyond noise.
A kernel gain whose end-to-end effect is inside noise still merges. -->

### Tests I ran
<!-- The checks that need no GPU (notes/backlog/CARDS.md §6 step 2), a checklist with the numbers you got, one line for each run. Paste nothing you did not run.
- [x] the build (`./build.sh` or the CMake presets): <result>
- [x] the CPU emulation tests in `tests/r4d_emu/`: <n> passed, <n> failed
- [x] the libref / libavx reference paths: <n> passed, <n> failed
- [x] `tests/chain_scalar_audit.py`: <result>
- [x] the plugin lints (`hyv4flash_arch_test`): <result>
- [x] the scheduler tests (`sched_test`): <result>
- [x] every ctest that runs without a card (`ctest --test-dir build -LE 'gpu|oot'`): <n> passed, <n> failed -->

### Measurement
<!-- Start with one of:
`measured`, then the path of the committed file, `notes/backlog/measurements/<item>.md` (for example `notes/backlog/measurements/K-47.md`);
`needs testing`, then why (no GPU was available);
`not needed`, then the reason.
Write no numbers here: the measurement is only what that committed file holds (under 40 KB), and the review reads no measurement from this body. Per CARDS.md §6 step 3 the file has the before commit (the merge base) and the after commit (the code commit you measured), who measured, the GPUs (model, count, TP), the power cap, the ROCm / driver version, the container and the ngram-placement, and at least three interleaved runs of each side with the raw lines of the rows §2 names, plus the output parity. -->
