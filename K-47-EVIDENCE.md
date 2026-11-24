# K-47 — `cast` 2D copy grid flattened for narrow rows — GPU evidence

Card: `notes/backlog/optimisation-backlog.md` K-47 (audit B-kernels-prefill K5). Upstream issue #57.
Change: `core/device/aql_blit.hip` only — `aql_copy2d16` + `r4dchain_copy2d16` walk a flat
grid-stride index `i → (r = i / w16, c = i - r*w16)` over `total = w16 * height` pieces; launcher
grid `blocks_for(w16 * height, 256)` (1-D, capped) replaces the 2-D `(blocks_for(w16), min(height,65535))`.
15 insertions / 15 deletions. Byte-identical by construction (same `(r,c)` set, same bytes moved).

## Method
- Standalone probe `k47_copy2d_probe.cpp` (out-of-tree, links `librad_core.a` + vendor archives +
  `libamdhip64`/`libhsa-runtime64`), exercises both the plain and the chained `copy2d16` entry points
  through `rad_memcpy_2d_async` → `rad_aql_blit_copy2d`, byte-compares dst against a host reference,
  and times N launches with a device sync per launch.
- **base** = pristine HEAD `64f9baeb` in a detached worktree (`radiance-k47-base`), built
  `-DCMAKE_BUILD_TYPE=Release -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201` (matches the fix build).
- **new** = branch `k47-cast-grid-flatten` build (`build-k47`, DSH).
- Both run on the **same free R9700 (GPU0, gfx1201, AQL backend)**, back-to-back, 400 iters/shape,
  engine down (zero contention). 7 shapes × {plain, chained}. `narrow128x32768` is the exact indexer
  defect (128-byte rows → w16=8 → 8/256 lanes busy, 32768 one-row blocks); `tall`/`huge` exercise the
  65535 y-cap and grid-stride wrap; `wide*` are full-lane rows (control).

## Chained form (the served prefill/decode path — launches chain)
| shape | base µs | new µs | speedup | base GB/s | new GB/s | BI |
|---|---|---|---|---|---|---|
| narrow128x32768 | 142.02 | 8.40 | 16.91× | 29.5 | 499.5 | OK/OK |
| narrow128x4096pad | 19.42 | 2.02 | 9.61× | 27.0 | 260.1 | OK/OK |
| med256x2048 | 10.65 | 2.02 | 5.27× | 49.2 | 259.9 | OK/OK |
| wide4096x512 | 13.67 | 5.09 | 2.69× | 153.4 | 411.6 | OK/OK |
| wide8192x256 | 14.11 | 5.10 | 2.77× | 148.6 | 411.0 | OK/OK |
| tall128x65535 | 286.74 | 15.40 | 18.62× | 29.3 | 544.9 | OK/OK |
| huge128x100000 | 304.83 | 17.74 | 17.18× | 42.0 | 721.4 | OK/OK |

## Plain (unchained) form
| shape | base µs | new µs | speedup | base GB/s | new GB/s | BI |
|---|---|---|---|---|---|---|
| narrow128x32768 | 39.81 | 5.17 | 7.70× | 105.4 | 811.7 | OK/OK |
| narrow128x4096pad | 6.31 | 2.62 | 2.41× | 83.1 | 200.3 | OK/OK |
| med256x2048 | 4.22 | 2.62 | 1.61× | 124.2 | 199.9 | OK/OK |
| wide4096x512 | 2.88 | 3.51 | 0.82× | 727.6 | 597.0 | OK/OK |
| wide8192x256 | 2.93 | 3.49 | 0.84× | 716.4 | 601.2 | OK/OK |
| tall128x65535 | 79.02 | 9.36 | 8.44× | 106.2 | 895.9 | OK/OK |
| huge128x100000 | 94.45 | 13.57 | 6.96× | 135.5 | 943.1 | OK/OK |

## Verdict
- **Byte-identity: ALL 14 cases OK on both base and new** (merge bar §2.1 ✓).
- **Targeted kernel measurably faster (§2.2 ✓):** the narrow-row defect shapes are **16.9–18.6× faster
  chained** (142→8.4 µs, 287→15.4 µs, 305→17.7 µs) and **7.0–8.4× unchained**. This matches/exceeds the
  card's prediction (0.13–0.15 ms/call → ~0.02 ms; bytes 4–25 MB now run near HBM rate, 500–940 GB/s).
- **Caveat (honest):** full-lane `wide*` rows in the **unchained** form are ~0.6 µs slower (2.88→3.51,
  2.93→3.49 µs; 0.82–0.84×) — the per-piece `udiv`/grid-stride overhead vs direct 2-D indexing when lanes
  are already full. The **chained** wide form is still 2.7× faster, and chained is the served path. Net
  effect on the engine is dominated by the large narrow-row wins; the unchained-wide cost is sub-µs.
- **§2.3 (end-to-end tok/s/TTFT not worse):** requires the canonical TP2 u3 server lane (210 W, cap read
  back, `probe.py` phases). Not run here — this is a kernel micro-benchmark on one free card, not a server
  lane. Per §2 "a kernel gain whose end-to-end effect is inside noise still merges: gains compound"; the
  card estimates +1.0–1.5 % prefill. **PR is `[needs lane]` for the end-to-end confirmation (§6.3–4).**

## Regression suite (DSH, card-less, no GPU)
215 ctest: 197 pass / 14 fail / 4 skip. **All 14 failures reproduced identically on unmodified HEAD**
(restored `aql_blit.hip` from `git show`, rebuilt, re-ran): missing `numpy`, io_uring seccomp, dashboard
no-engine, artifact-hash mismatch, arch/plugin lint, `chain_scalar_audit` BAD entries in
`gemm_i8_kpar`/`hc_read_fused` (not the copy kernels), `r4d_emu` segfault. None touched by this change.
DSH also host-simulated `(r,c)` coverage incl. narrow rows and >2048-block totals: new block count never
larger, wide rows equal under the cap. See `K47-DSH-REPORT.md`.

## Lane note (CARDS.md §1)
Micro-benchmark lane: single free R9700 GPU0 (gfx1201), AQL backend, engine stopped, base/new back-to-back
on the same card. NOT the canonical TP2 u3 210 W server lane — absolute µs here are isolated-blit, not
served-context stamps; the **relative** base→new gain is the merge-bar evidence. A canonical lane run
(`rad-kbench`/`profile-ops` copy2d16 rows + `probe.py` prefill) should confirm §2.3 before merge.
