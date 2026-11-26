# AGENTS.md — working on the Hy4-flash backlog from outside

A Moebius server follows this repository: it reads your comments and PRs, reviews them and merges what
passes. Read `notes/backlog/CARDS.md` once (§2 merge bar, §6 outside path); the items are in
`notes/backlog/optimisation-backlog.md`, each with its issue. The server reads PRs and comments by the
headings of `.github/PULL_REQUEST_TEMPLATE.md` and `.github/moebius/*.md`: keep them as they are.

## How an item is taken

1. On an issue labelled `Ready for agent`, comment `/take` (first word of a line). At most 3 per
   login; `/release` gives one back.
2. Open a PR from the template: first line `Fixes #N`, then `### Item`, `### What the change does`,
   `### Tests I ran` (checklist with your numbers), `### Measurement`. Without `Fixes #N` it is not
   bound to the issue (labelled `unbound`, closed after 14 days). A missing section gets a `## Missing`
   reply; edit the body.
3. Title: `[measured] <item>: <subject>`, or `[needs testing] <item>: <subject>` without a GPU.
4. The review starts by itself once the checks are green for 5 minutes, or with a `/done` comment on
   the **issue** (form: `.github/moebius/done.md`). One round per head, at most three rounds.

## What a change carries

- The change **and its tests** (CARDS.md §6): a code path without a test in the same tree is
  a blocking finding.
- Nothing in `third_party/` is edited; the engine, when involved, is touched only as a patch.
- Commits, PR text and issue comments in English.
- What the PR body says (`measured`, a ticked test) is a claim the review checks, never a decision.

## What you can verify (and must, before pushing)

A HIP build uses radiance's own HIP compiler, the one the Dockerfile ships (ROCm's clang over the
VOPD-repeatable libLLVM, `-DRAD_HIP_VOPD_REPEATABLE=ON`): see CARDS.md §1.1 for the exact cmake line.
Stock ROCm clang 24 builds, but with VOPD off and different device code, so its kernel timings are not
the shipped engine's.

No GPU: the CPU build (`./build.sh` or CMake presets), `tests/r4d_emu/`, the libref / libavx
reference paths, `tests/chain_scalar_audit.py`, the plugin lints (`hyv4flash_arch_test`), the
scheduler tests (`sched_test`), every ctest that runs without a card. Run the touched ones;
paste nothing you did not run.

## Measurements

A lossless change merges only with a measurement accepted for its current head (CARDS.md §2), or when
the owner labelled the issue `no-measurement`. Accepted is a `## Measurement` comment on the issue, in
the form of `.github/moebius/measurement.md` (before = merge base, after = PR head, GPUs, power cap,
ROCm, container, at least 3 runs each side), by a registered measurer **who is not the PR's author**,
or a run of the owner's test lane. A comment that holds starts the next round by itself; a push makes
every measurement of the old head count for nothing.

- **With a GPU:** measure as CARDS.md §6 step 3 says and commit the record as
  `notes/backlog/measurements/<item>.md` (under 40 KB, both commit ids, GPUs, cap, ROCm, container,
  `ngram-placement`, raw lines). The review checks it; for the merge another measurer posts the
  comment. Measure each other's PRs.
- **Without a GPU:** `needs testing` in `### Measurement`. When the code is approved the issue gets
  `ready-for-testing`, and a measurer's comment starts the next round. Never claim numbers you did not
  run.

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

- The verdict is a review on the PR. `approve` merges through the server's merge gate once the checks
  pass and a measurement is accepted. You never merge or close the PR.
- `changes`: blocking findings are mandatory, judgement findings advice. Push to the same branch (never
  force-push), then comment `## Review answer` on the issue, one numbered line per finding
  (`.github/moebius/review-answer.md`).
- Only the owner merges a PR that touches a locked path: build and CI files, `AGENTS.md`,
  `notes/backlog/CARDS.md`, `scripts/backlog/` and the `rad-kbench` sources and fixtures. Keep such
  changes in a PR of their own.
- An issue taken with no PR is released after 14 days; a PR with changes requested and no push is
  reminded after 14 days and closed after 30. The merge closes the issue.
