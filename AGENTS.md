# AGENTS.md — working on the Hy4-flash backlog from outside (no lane)

You are working on this repository without a GPU lane. Everything you need is in the tree;
read these before working:

- `notes/backlog/CARDS.md` — the working rules. §2 is the baseline and merge bar, §6 is the
  outside path (no lane). It is short; read all of it once.
- `notes/backlog/optimisation-backlog.md` — the items. Each item has an issue here; the issue
  is the unit of work, the PR is only its delivery.

## How an item is taken

1. Take an item through its issue: assign yourself (or comment `/take` with the tooling you
   have).
2. Open a PR whose title or body contains a closing keyword for the item's issue
   (`Fixes #N` / `Closes #N`). Without the keyword, the follow does not bind your PR to the
   issue and nobody reviews it (`#172` was missed exactly that way).
3. Title the PR `[measured] <item>: <subject>` when you measured the item's whole merge bar
   yourself (see below), `[needs testing] <item>: <subject>` when you could not.
4. **Write `/done` as a comment on the item's issue** (not on the PR). Only that starts the
   review, and only you, the PR's author, can. Write it again after each push you want
   reviewed; a PR gets at most three review rounds.

## What a change carries

- The change **and its tests** (CARDS.md §6): a code path without a test in the same tree is
  a blocking finding (this came up on the router-bf16 PR: the branch lived in three sibling
  kernel files, only one had a test).
- Nothing in `third_party/` is edited; the engine, when involved, is touched only as a patch
  (repository rule).
- Commits, PR text and issue comments in English.

## What you can verify (and must, before pushing)

A HIP build uses radiance's own HIP compiler, the one the Dockerfile ships (ROCm's clang over the
VOPD-repeatable libLLVM, `-DRAD_HIP_VOPD_REPEATABLE=ON`): see CARDS.md §1.1 for the exact cmake line.
Stock ROCm clang 24 builds, but with VOPD off and different device code, so its kernel timings are not
the shipped engine's.

No GPU: the CPU build (`./build.sh` or CMake presets), `tests/r4d_emu/`, the libref / libavx
reference paths, `tests/chain_scalar_audit.py`, the plugin lints (`hyv4flash_arch_test`), the
scheduler tests (`sched_test`), every ctest that runs without a card. Run the touched ones;
paste nothing you did not run.

With a GPU, measure the change yourself (CARDS.md §6 step 3): before (merge base) and after
(your code commit) on the same machine, at least three runs each, the stamps / `rad-kbench` /
`profile-ops` rows the item's merge bar names plus output parity. Commit them as
`notes/backlog/measurements/<item>.md` (under 40 KB) with both commit ids, your GPUs, power
cap, ROCm version, container, `ngram-placement` and the raw lines of each run. Any placement,
`ram` included, is fine on your machine. The review sees only the PR's diff and files and the
issue's title and body, not the PR body or comments: a measurement anywhere else does not count.

Without a GPU, open the PR as `[needs testing]`. When the review approves the code, the issue
gets `ready-for-testing`; someone with a free GPU posts a measurement on the issue, you commit
it as the file above and write `/done` again. Never claim numbers you did not run.

## Iterating on one kernel (with a GPU)

Work on one libr4d kernel needs neither a whole-library build nor a server restart. Read
`docs/PLUGIN.md` §14 before using it; the short form:

- **Override:** `cmake -B build -DR4D_OVERRIDE_UNITS=<unit>[;<unit>]` then
  `cmake --build build --target libr4d_override` compiles only those units into
  `build/override_home/kernels/libr4d.so`. Run with
  `RADIANCE_HOME=build/override_home:build/radiance_home`: it shadows libr4d with the same rows,
  flags and selection, and only the hooks those units define are new. Add
  `-DR4D_OVERRIDE_BASE=<path>/libr4d.so` to build over a libr4d built elsewhere (e.g. the merge
  base's), so nothing else compiles.
- **Reload:** rebuild the override in place, then `curl -X POST http://127.0.0.1:<port>/admin/kernels/reload`.
  Every engine switches at a step boundary, in about 0.1 s; the model stays loaded. The reply
  names the rows that took new hooks; a 409 names why nothing changed.
- **Limits:** only hooks reload (`launch`, `scratch` of the same size, `tune_valid`). A change
  to a row's declaration, constraints, layout, shapes, fusion, `init` / `fini`, or a header
  other units include, needs a rebuild of libr4d (name every unit that includes the header, or
  rebuild). Replace the file as a new file (`mv`, `install`, the linker); `cp` over it rewrites
  the file the server has mapped. Each reload keeps the old copy mapped (~100 MB VRAM a card on
  the first one).

An override is for iterating. The measurement that merges is still the before / after of
CARDS.md §6 step 3 on the full build of each commit.

## How the review works here

- `/done` on the issue starts a review round of your PR's current head. The review is posted
  on the PR, with a verdict (`approve` / `changes`) and inline comments.
- Fix findings by pushing to the same branch, then write `/done` again. Never force-push.
- An approve merges the PR at once (code and measurement both held). You never merge, never
  close the PR yourself.

## Feedback loop

If a verdict says `approve`, the merge closes the issue automatically. If it says `changes`,
the blocking findings are mandatory; judgement findings are advice unless the fix is trivial.
State in one line per finding what you changed about it, so the re-review can check it.
