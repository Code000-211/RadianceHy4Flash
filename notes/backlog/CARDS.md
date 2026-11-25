# Cards from the Hy4-flash backlog: rules

Each card is one item of `optimisation-backlog.md` (or a tight pair of items). Its issue quotes the item in full, so the issue can
be read without this directory. The audits under `audit/` hold the longer derivations.

## 1. Lanes

A lane is a TP2 pair of R9700 at 210 W with the u3 container (`hy4-flash-iqr-v4-u3.rad`, keptn/hy4-flash-iqr-v4).
The lane is the only thing a run needs to know: `scripts/backlog/lane.sh` gives a command the lane's port, model,
tiers and cards, so the same tools and the same `probe.py` phases run on every lane.

| lane | where | cards | port | Engram, host pool |
|---|---|---|---|---|
| `local` | the Moebius host, its two cards | `<lane lock dir>/gpu.lock`, then `gpu2` | 8041 | `ngram-placement: ram` (a pinned copy), host pool 39000 MiB: the served config, `scripts/backlog/*.yaml` as they are |
| `iks-A` | `ssh iks`, a free pair of its four | `gpu2` on iks | 8042 | `ram=/dev/shm/radiance-ngram` (one copy in tmpfs, pinned by this lane), host pool 16000 MiB |
| `iks-B` | `ssh iks`, the other pair | `gpu2` on iks | 8043 | `disk=~/ngram` (a 60 GiB file on the NVMe, read through the row cache), host pool 16000 MiB |

Why the iks lanes are not the local config: iks has 112 GB of RAM and one u3 server pins ~60 GiB of Engram tables,
so two servers at `ram` do not fit, and the driver there has `no_system_mem_limit=N`, so one `/dev/shm` copy pinned
by both engines is refused too (each engine's pin counts against the cap). `iks-A` therefore keeps the single pinned
copy and `iks-B` reads the same table from a file; both take a host pool of 16000 MiB instead of 39000 so that the
two of them fit beside the 60 GiB. §1.2 has what that measured.

- **Server lane:** a full served A/B (stamps, tok/s, TTFT, KL).
- **Kernel lane:** selftest, rad-kbench, emu and the perf modes, on any free pair.
- **One lane per A/B:** a card's before and after run on the same lane and are compared against that lane's baseline,
  `baseline/<lane>.md`. Record the lane in the commit message or note.
- **Lanes differ:** iks-B reads Engram from disk and both iks lanes have a smaller host pool, so do not compare
  numbers across lanes -- not even iks-A with iks-B, which are two cards of the same machine but not two of the same
  table placement.
- **Power cap:** read the 210 W cap back before every run: `lane.sh` does it in `run.sh` and `lane-pair.sh` and
  prints `cap` per card; a card that is not at 210 W means the run's numbers are not the lane's.

### 1.1 Using the lanes (`scripts/backlog/`)

```sh
cd <tree>
scripts/backlog/lane.sh lanes                     # local iks-A iks-B
scripts/backlog/lane.sh --env iks-A               # what a lane exports: port, model, tiers, paths (`~` on iks)
```

- **`lane.sh <lane> <command>`** runs a shell command in the lane (or a program with its arguments). `local` runs it
  here; `iks-A` and `iks-B` tar the tracked and untracked tree to `ssh iks moebius-gpu/run.sh <name> <b64>` and run
  it in `~/moebius-gpu/<name>/src`, where the `build/` directory survives between calls, so the first iks call of a
  tree builds and the later ones are incremental. The tree name is `<repo>-<run>` (`RadianceHy4Flash-run-1654`);
  `RAD_IKS_NAME` overrides it. One iks call runs at a time (the unpack replaces the tree); two lanes together go
  through `lane-pair.sh`.
- **`lane.sh --gpu <lane> <command>`** the same, on the lane's pair: `flock <lock dir>/gpu.lock gpu2` on `local`,
  `gpu2` on iks. Nothing else takes a card: `run.sh` and `lane-pair.sh` start their servers this way. `gpu.lock`
  (default `<lane lock dir>` = `/tmp/radiance-lanes`, `RAD_LANE_LOCK_DIR` overrides) is what a local lane run holds
  for its whole run.
- **The local lane serves `<tree>/build/bin/radiance`**, so that tree is built once. Here the build needs ROCm's
  clang++ as the host compiler: the host's GCC 13 rejects the compound literals in `abi/rad_runtime.h`, and a
  whole-tree build stops on the narrowing errors in `tests/museglimmer_test.cpp`, which the lane does not use.
  ```sh
  cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201
  cmake --build build --target radiance -j 32
  ```
- **`run.sh <tag> <cfg.yaml> <phases> [ENV=VAL ...] [-- radiance args]`** one server in the lane, `probe.py`
  against it, then it is stopped. Everything lands in `runs/<lane>/<tag>/`: the lane's rendered config
  (`<cfg>.yaml` with the lane's port, model, host pool and `ngram-placement`), `server.log`, `run.out`, `<tag>.json`.
  `RADIANCE_AQL_STAMP=1` also leaves `<tag>.stamps.csv` there, the input of `layer.py` and `agg.py`. A lane whose
  `ngram-placement` is `ram=DIR` first clears that directory: the pinned copy belongs to one process and the one a
  killed run leaves behind refuses the next start (§1.2). The arguments after `--` go on the server's command line,
  each one shell-quoted, so `-- --check` reaches radiance intact (radiance takes only `--config FILE [--check]`).
  ```sh
  scripts/backlog/lane.sh local 'scripts/backlog/run.sh c1 nospec.yaml c1,c8'
  scripts/backlog/lane.sh iks-A 'scripts/backlog/run.sh c1 nospec.yaml c1,c8'
  ```
- **`lane-pair.sh <cfg.yaml> <phases> [ENV=VAL ...]`** both iks lanes at once with the same config, both servers up
  together, `probe.py` against each, both stopped: the two pairs of one machine side by side. Run it inside an iks
  tree: `scripts/backlog/lane.sh iks-A 'scripts/backlog/lane-pair.sh nospec.yaml c1,c8'`.
- **`probe.py <port> <out.json> <phases>`** the phases a card quotes:
  `c1`, `c8` (one and eight decoders), the same on a DFly2 config (`coupled03.yaml`, `cpv.yaml`, `cprof.yaml`) for
  the drafter's `c1` / `c8`, `pf6` (6,681 tokens: `22,000` characters of tree source) and `pf20` (24,856:
  `72,000`) — the backlog's 6.7K / 24.9K prompts, because `prompts.py` walks the corpus in the same `shuffle`
  order its own tools did — `stall1` / `stall4` (a 5,698 / 5,881-token arrival on one and on four running
  decoders), and the small prompts `pf2` (2,385: `7,400`), `tailN` (`N` characters, e.g. `tail400`: 159) and
  `c1s` / `c8s`, `idle` (a conversation of `RAD_IDLE_CHARS` characters that finishes and idles for
  `RAD_IDLE_WAIT_S` s, a decode-only request that makes the experts ask for VRAM back, the same prefix again
  (the resume) and a cold prefill of the same size, with `/stats` and `/sessions` read at each step: the
  host tier's `prefix-cache-mib` check, X-01) and `long8` (the item's multi-session case: eight
  conversations of `RAD_LONG8_CHARS` characters at once, then the same idle, demand and return passes), and
  `lctxN` (`N` conversations of `RAD_LCTX_CHARS` characters — the default 190,000 is the 58,903-token
  conversation of §1.3 — filled by a one-token wave and then decoded for `RAD_LCTX_NTOK` tokens: the long-context
  decode stamps at one and four decoders, `lctx1` and `lctx4`, whose two waves land in the JSON as their own
  phases).
  `greedy` is the merge bar's first clause: three prompts and a 5.7K arrival at temperature 0 with
  `ignore_eos` off, each as `gen` / `chars` / `sha256` of the text itself, so two arms' JSON compare byte for
  byte (S-01b).
  Both `idle` and `long8` read the experts' side of the same views, because that is what the freed VRAM is for:
  `expert_units` and `slab_mib` are the residency and the lent VRAM the experts hold, `lend_mib` the most
  the KV pool lets them borrow, and `routed` / `routed_res` the demand's hit rate. It prints each phase as JSON and writes
  `out.json`; phases are comma-separated (`c1,c8,pf6,pf20,stall1,stall4,pf2,tail400`), `nowarm` skips the two warmup
  requests, and an unknown phase is refused instead of leaving a hole in the run. Its requests are sampled at
  `RAD_TEMP` (default 0.9, seeded per request); a host whose `TEMP` is a directory is not a temperature and is
  ignored.
- **`bench.py <port> <tag> <out-dir>`** the served benchmark (decode, prefill, cold/warm, `--reps`, `--temp`), and
  **`prompts.py`** holds the prompts and the code corpus both tools send: they are read from the tree the tools sit
  in (`RAD_SRC` overrides), never from a copy on the measuring host, so a lane's prompt is the same text everywhere.
- **`layer.py <stamps.csv> <probe.json> <phase> <rec> [<li>]`**, **`agg.py <stamps.csv> <probe.json> [--top N] [--match RE]`**,
  **`attr.py <stamps.csv> <probe.json> [--top N]`**: the per-layer and per-op tables from a run's stamps. They are
  plain CPU tools over those two files (run them as `python3 scripts/backlog/agg.py …`; they are 644 in the tree),
  so they run wherever the files are: on the lane, or here over a file copied back from iks with `scp
  iks:moebius-gpu/<name>/src/runs/<lane>/<tag>/<tag>.stamps.csv .`. They read the phases that carry a stamped
  window; a composite phase (`longctx`, `idle`, `long8`) and a gap probe (`stallN`) carry none, so they are named
  and skipped, and the waves the composite ran are what the tables come from. `agg.py --match RE` prints only the
  kernels of that name and their sum per step — the read of one term, e.g. `--match 'qsa_select|idx_score'` for
  K-25's two terms at long context. They need a demangler, and iks has no
  `c++filt`: `demangle.py` takes `c++filt` or `llvm-cxxfilt` from `PATH`, else the `llvm-cxxfilt` next to the
  compiler the tree was built with (`build/CMakeCache.txt`), which is where the ROCm SDK keeps it. The iks trees'
  `runs/` are not kept between calls (the tree is unpacked fresh each call), so copy back what a card needs.
- **`profdiff.py <log> [rank] [min-ms] [top]`** and **`profagg.py <log> [rank]`**: two runs of a `profile-ops: true`
  config (`prof.yaml`, `cprof.yaml`) compared op by op.
- **`steps.py <server.log> [--tail=N]`** the `RADIANCE_LOG_STEPS=1` log as the backlog reads a step: every
  prefill-bearing step (`MIXED n_tok`, `prefill n_tok`) with the decode-only steps that ran before it, so one
  interval beside decoders prints `MIXED 1984` at the served `checkpoint-interval: 1984`, against `MIXED 2033 | 17 |
  MIXED 2033` at the full 2048 batch before S-01 (the remainder step). It is a plain CPU tool over the text, so it runs
  on the lane or here over a copied-back log.
- The configs are the served (local) ones: `nospec.yaml` (no drafter), `coupled03.yaml` (DFly2 coupled, 5 proposals),
  `cpv.yaml` (the same with debug logging), `cprof.yaml` (DFly2 + `profile-ops`), `nsv.yaml` / `nsv1984.yaml`
  (debug logging, and a 1984-token checkpoint interval), `prof.yaml` (`profile-ops`). `lane.sh` rewrites only a
  lane's four keys, so a config's other lines mean the same thing on every lane.

### 1.2 What the lanes measured

Every lane was run on 2026-11-23 with the u3 container and the 210 W cap read back first
(`lane_begin`: 210 W on all four cards of iks and on both cards here), `nospec.yaml`, the same phase list
`c1,c8,pf6,pf20,stall1,stall4,pf2,tail400`, all rc=0 and `server-exit=0`. The prompts are the backlog's own:
`pf6` is 6,680–6,681 tokens, `pf20` 24,856, `pf2` 2,385, `tail400` 159, and the `stall` arrivals 5,698 and
5,881 — the same text on all three lanes (they are read from the tree, and `prompts.py` walks the corpus with
the `shuffle` the backlog's own tools used, so `pf6` lands on 6,681 tokens against the backlog's 6.7K and `pf20`
on 24,856 against its 24.9K, `refs/kernels/levers-general.md`): a lane's number here compares with the backlog's.

| phase | `local` (`run.sh full …`) | `iks-A` (`run.sh full …`) | `iks-B` (`run.sh full2 …`) |
|---|---|---|---|
| `c1` — 1 decoder, 256 tokens | 72.1 tok/s (13.6 ms/step) | 72.6 tok/s (13.5 ms/step) | 67.0 tok/s (14.6 ms/step) |
| `c8` — 8 decoders, 256 tokens each | 309.3 tok/s agg | 313.0 tok/s agg | 274.4 tok/s agg |
| `pf2` — 2,385 tokens | ttft 0.70 s | ttft 0.67 s | ttft 0.93 s |
| `pf6` — 6,681 tokens | ttft 1.93 s | ttft 1.91 s | ttft 2.35 s |
| `pf20` — 24,856 tokens | ttft 7.04 s | ttft 7.09 s | ttft 7.76 s |
| `stall1` — 5,698-token arrival, 1 decoder | arrival ttft 3.05 s | 2.98 s | 4.62 s |
| `stall4` — 5,881-token arrival, 4 decoders | arrival ttft 2.83 s | 2.82 s | 3.97 s |
| `tail400` — 159 tokens | ttft 0.18 s | ttft 0.15 s | ttft 0.17 s |

The first `iks-B` run after a tree unpack (`run.sh full …`) is slower on every phase than the run after
it (`run.sh full2 …`): `c1` 55.3 tok/s against 67.0, `c8` 155.2 against 274.4, `pf6` ttft 11.04 s against
2.35, `pf20` 14.80 against 7.76, `stall1` 6.80 against 4.62, `stall4` 5.03 against 3.97. Its 60 GB
`disk=~/ngram` table is then not in page cache yet and a lookup reads NVMe. Quote `iks-B` from a run whose
table has been read once, or say which of the two it is.

**The drafter on every lane**, `run.sh fly2 coupled03.yaml c1,c8 RADIANCE_AQL_STAMP=1`:

| lane | `c1` | `c8` | full-model pass kind | `layer.py … c1 <rec> 6` |
|---|---|---|---|---|
| `local` | 110.3 tok/s, 2.81 tok/step | 380.1 tok/s agg, 22.02 tok/step | 1,335 records, 91 passes, 17.48 ms median wall | 394.9 µs (29 records, `moe_iqr_kernel` 96.5 µs) |
| `iks-A` | 117.8 tok/s, 2.81 tok/step | 382.6 tok/s agg | 1,335 records, 91 passes, 17.41 ms | 396.0 µs (29 records, `moe_iqr_kernel` 88.4 µs) |
| `iks-B` | 106.2 tok/s, 2.81 tok/step | 323.2 tok/s agg | 1,327 records, 91 passes, 20.47 ms | 395.8 µs (29 records, `moe_iqr_kernel` 87.9 µs) |

Every lane also splits the drafter's short pass (303 records: `local` 92 passes and 4.87 ms median wall,
`iks-A` 4.11 ms, `iks-B` 4.02 ms), so `agg.py` shows two kinds on the stamped `c1` and `layer.py` wants the
full-model one (1,335, or 1,327 on `iks-B`). The file is 42 MB; both tools read it on the lane itself
(`python3 scripts/backlog/agg.py …`, `python3 scripts/backlog/layer.py …` — they are 644 in the tree), and an
iks lane's `runs/` goes away with the next call's unpack, so copy back what a card needs:
`scp iks:moebius-gpu/<name>/src/runs/iks-A/fly2/fly2.stamps.csv .`

**Both iks lanes served at once.** `scripts/backlog/lane.sh iks-A 'scripts/backlog/lane-pair.sh nospec.yaml c1,c8,pf6'`:
`iks-A` answered `/health` on 8042 after 83 s (it fills 60 GiB of tmpfs first), `iks-B` on 8043 after 1 s, both
stayed up and each decoded while the other was idle:

| run | `iks-A` Engram `ram=/dev/shm/radiance-ngram` | `iks-B` Engram `disk=~/ngram` |
|---|---|---|
| `c1` — 1 decoder, 256 tokens | 72.5 tok/s, 13.6 ms/step | 66.8 tok/s, 14.6 ms/step |
| `c8` — 8 decoders, 256 tokens each | 310.8 tok/s agg | 266.1 tok/s agg |
| `pf6` — 6,681-token prompt | ttft 1.92 s | ttft 2.51 s |

Memory with both up is the whole story: `MemTotal` 112 GiB, `MemAvailable` 10.8 GiB after the probes (66.8 GB of
page cache, no swap), so the two servers fit with about 10 GiB to spare and nothing else may run on iks beside
them. `iks-B`'s `disk=~/ngram` placement is what it costs: with its table in page cache it is 8% behind
`iks-A` at `c1` (66.8 against 72.5) and 15% at `c8` (266.1 against 310.8), and 0.6 s behind at `pf6`. The
co-residency itself is within noise (an earlier pair run had `iks-A` at 66.8 tok/s, 8% under its run alone; this
one is level with it), which is why §1 says not to compare numbers across lanes: compare a lane with itself.

What the lanes pinned down beyond the numbers:

- The u3 container on iks (`~/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad`, 138,093,449,672 bytes) hashes to
  `c4a80918139f2bae4d097976fd2794ca3f1f56e9ea7eb60600188c45f4f5e7aa`, the value in its `.sha256` from Hugging
  Face's LFS oid, and the same size as `/srv`'s copy. The download's `dl-u3.log` ends in `DONE`.
- A `ram=DIR` Engram table is one process's pinned copy and is not reused: a copy left in `/dev/shm` by an
  earlier run is 60 GiB of tmpfs that the next start counts against `MemAvailable` and refuses to start beside
  ("50.75 GiB of memory is available … this wants 45.49 GiB"). `lane_ngram_reset` clears it before a lane
  starts, so `iks-A` refills 60 GiB on each start (part of the 85 s) while `iks-B`'s `disk=` table is reused.
- `lane-pair.sh` is the only way two lanes run on iks at once: `lane.sh` sends the tree to one
  `~/moebius-gpu/<name>/src` and an unpack replaces it, so single-lane calls are serialised by
  `flock <lane lock dir>/iks.lock`.
- iks has no `curl`, so `lane_health` asks `/health` with `python3`; and a measuring host whose `TEMP` is a
  directory is not a sampling temperature (`RAD_TEMP` is).

**P-16, the load rule from one sequence, `local`, 2026-11-23.** `coupled03.yaml`, `cpv.yaml` and `cprof.yaml`
now carry `draft-depth-load-from: 1`, the line the served `README.md` config already had, so the load-aware
rule decides a step that drafts for a single sequence too: the server log prints "a step drafting for 1
sequence(s) or more" where it printed "5", and `draft-depth-stop: 0.3` is not read. Three runs each way on
`local`, interleaved before and after (`scripts/backlog/lane.sh local 'scripts/backlog/run.sh p16-b1
coupled03.yaml c1,c4,think'`, and the same with `p16-a1` on a copy of the committed file with that one line
added):

| phase | rule from five sequences (before) | `draft-depth-load-from: 1` (after) |
|---|---|---|
| `c1` — 1 decoder, 256 code tokens | 123.2 tok/s, 2.81 tok/step, 21.9 ms/step (121.7 / 123.2 / 123.9) | **133.1 tok/s, 3.16 tok/step, 22.7 ms/step** (132.9 / 133.1 / 133.1) |
| `c4` — 4 decoders, 256 tokens each | 281.0 tok/s agg, 11.38 tok/step (280.6 / 281.0 / 287.3) | **306.4 tok/s agg, 12.64 tok/step** (288.4 / 306.4 / 311.7) |
| `think` — 1 decoder, 384 tokens, reasoning high | 108.0 tok/s, 2.27 tok/step, 20.3 ms/step (104.8 / 108.1 / 108.0) | **132.7 tok/s, 3.10 tok/step, 22.6 ms/step** (111.2 / 132.7 / 145.0) |

So the item's "code ±0" does not hold on this lane, and its "thinking text +9 % at one client" is where this
lane's slowest `think` run lands, not what it gives on average: code gains 8.0 % at `c1` and 9.0 % at `c4`,
thinking text 22.9 % at the median.
The step costs 3.7 % more at `c1` for 12.5 % more tokens a step (the round the rule now runs at one
sequence, priced by its own tables). `think` is the one noisy number — 111.2 .. 145.0 tok/s after against
104.8 .. 108.1 before — because the rule's online tables and the sampled text move the depth run to run,
while the before runs are tight to the digit (2.27 tok/step three times: the 0.3 threshold prices the same
chain whatever the text). The drafter table above (`c1` 110.3 tok/s, 2.81 tok/step) is the stop rule at one
client — its tok/step is the "before" here to the digit — so read that row as the config before this line.
Greedy text is unchanged, which is the merge bar's first clause: the three `GREEDY` prompts at temperature
0 (256 tokens, `reasoning_effort: none`, `ignore_eos` off) come back 256 / 256 / 103 tokens and
`9f77180a9ab097fb…` / `285d021c7e9a5bdf…` / `d3d2aa2a1091868e…` sha256, both configs, byte for byte — the rule
picks how many candidates a step verifies, not the token the law emits.

### 1.3 X-01: the host KV tier on, in the served configs (local lane)

The served configs carry `prefix-cache-mib: 16384` (`scripts/backlog/*.yaml`, as `README.md:55` does). The
item's claim is "640 units (3 points) per idle 65K session", its verify step "/stats VRAM units after
idle". `idle` with `RAD_IDLE_CHARS=190000` sends one conversation of 58,903 tokens, lets it finish and
idle, sends a decode-only request so the experts ask for VRAM, then the same prefix again and a cold
prefill of the same size. Three servers with the tier on and three with `RAD_PREFIX_CACHE_MIB=0`, local
lane, one build, phase list `c1,c1b,c8,pf6,pf20,stall1,stall4,idle`, all rc=0 and `server-exit=0`:

| `idle`, `/stats` + `/sessions` | tier on, s1 / s2 / s3 | tier off, s1 / s2 / s3 |
|---|---|---|
| expert units in VRAM, before the conversation | 19,194 | 18,749 |
| … with the conversation (58,903 tokens) live | 17,413 / 17,418 / 17,405 | 16,956 / 16,957 / 16,950 |
| … 5 s after it finished (idle) | 17,982 | 16,975 / 16,975 / 16,963 |
| … after a 64-token decode-only request (demanded) | 19,194 / 19,192 / 19,194 | 18,173 |
| expert slab, lent VRAM, MiB (idle → demanded) | 7,178 → 8,809 / 8,808 / 8,815 | 5,644 → 7,362 / 7,363 / 7,369 |
| the loan's cap, MiB (idle → demanded) | 7,260 → 8,997 | 5,763 → 7,480 (cap 8,306) |
| KV in VRAM, MiB (idle → demanded) | 0 → 0.8 | 1,494 → 1,494 |
| the demand's routed-unit hit | 0.9826 / 0.9823 / 0.9826 | 0.9686 / 0.9684 / 0.9672 |
| resume TTFT, 58,896 cached tokens | 0.100 / 0.106 / 0.098 s | 0.031 / 0.031 / 0.030 s (from VRAM) |
| the same prompt cold | 18.62 / 18.63 / 18.68 s | 18.79 / 18.77 / 18.81 s |

The gain, at the same point of the same phase list: with the tier on the experts hold **1,021 more units
and 1,447 MiB more lent VRAM** (19,194 units / 8,809 MiB against 18,173 / 7,362), for one idle 58,903-token
session — the item's 640 units per 65K session, and more, because the conversation's blocks are 819 MiB and
the pooled KV it leaves behind (107,284 tokens with the phases before it) is 1,493 MiB, and the loan counts
what sits above the highest held block (`core/mem/kv.cpp`'s `spare_tokens`). Within a run the tier gives all
of it back (19,194 → 19,194 units, the loan's whole 8,997 MiB); the tier off leaves 576 units and 825 MiB
of the loan held, and its demand misses 1.4 points more (0.9686 against 0.9826). The host restore costs
0.07 s more than a VRAM hit and 18.5 s less than the re-prefill the item is about.

End to end, the same six servers, median of three (ms a step / tok/s / ms of TTFT / s to first token):

| phase | tier on | tier off | `baseline/local.md` (tier off) |
|---|---|---|---|
| `c1` | 13.720 | 13.750 | 13.710 |
| `c1b` | 13.394 | 13.389 | 13.403 |
| `c8` | 309.0 | 306.4 | 309.1 |
| `pf6` | 1,949 | 1,946 | 1,968 |
| `pf20` | 7,131 | 7,147 | 7,173 |
| `stall1` | 2.732 | 3.037 | 3.04 |
| `stall4` | 2.804 | 2.869 | 2.87 |

Nothing is worse: `c1`/`c1b`/`c8`/`pf6`/`pf20` are level with the tier off (inside 1%), and the arrival
stalls are better by 0.30 s (`stall1`, all three reps 2.707–2.732 against 3.037–3.078) and 0.07 s
(`stall4`), in the item's direction: the phases before them leave cached prefixes in VRAM that the tier
takes to host, so the loan the arrival's prefill wants is there. `stall1` at 3.04 s is also what
`baseline/local.md` has, measured at `dd4c2255` with no tier, so the 0.30 s is not build drift.

The item's "Why" is 8 sessions at 16K–60K, not one: `long8` sends eight conversations of
`RAD_LONG8_CHARS=140000` characters at once (37,872–46,720 tokens each, 350K tokens in all — the M longctx
shape, and near the pool's own limit, `refs/notes/longctx.md`), lets them finish, demands, and sends the
same eight back. Two servers each way:

| `long8`, `/stats` + `/sessions` | tier on, s1 / s2 | tier off, s1 / s2 |
|---|---|---|
| expert units in VRAM, before | 19,194 | 19,228 / 19,229 |
| … once the eight finished (idle) | 17,982 / 19,194 | 15,821 / 14,611 |
| … after the decode-only demand | 19,194 / 19,193 | 15,817 / 15,819 |
| expert slab, MiB (idle → demanded) | 7,132 → 8,771 / 8,804 → 8,723 | 3,971 → 3,842 / 2,238 → 3,970 |
| the loan's cap, MiB (demanded) | 8,997 | 4,109 |
| KV in VRAM, MiB (idle → demanded) | 0 → 1.7 | 4,886 → 4,888 |
| host memory held, tier and snapshots, MiB (idle) | 7,045 / 6,732 | 1,985 / 2,090 |
| the demand's routed-unit hit | 0.9797 / 0.9862 | 0.9427 / 0.9325 |
| the eight back: TTFT, first → last | 0.57 – 3.91 s | 0.62 – 2.91 s |
| … from the cache, tokens | 347,456 / 344,240 | 346,176 / 347,456 |

Eight idle sessions hold 4,886 MiB of VRAM KV (351,096 tokens) with the tier off, and the experts lose all
of it: at the same point in the same run they hold 15,818 units and a 3,906 MiB slab against 19,193 and
8,747 with the tier on, **3,375 units and 4,841 MiB** more, and the demand's hit rate is 4–5 points higher
(0.943/0.933 against 0.980/0.986). The eight returns cost the same either way, 0.6–3.9 s against 26.8–117 s
cold — 4.88 GiB over the link is a fraction of the re-prefill — so at 8 long sessions the tier's price is
the 4.9 GiB copied to host memory (7.0 GiB held there with the snapshot store; the tier-off server's
1,985 MiB is the snapshots alone) and its gain is the experts' VRAM, which is what the item's decode
190 → 148 tok/s is made of.

Byte-identical: one more `idle` server with `RADIANCE_DEBUG_TIER_VERIFY=1`, which hashes every entry as it
leaves VRAM and compares on the way back (`TierExec::verify_note` / `verify_check`, and the replicated
group's sum against rank 1's): 58,903 tokens out, 58,896 back from the host tier, **no "came back
different", no "differ between rank" and no failed move in the server log**, and the resume costs 1.141 s
against 0.100 s without the hashing — the check ran. That plus the served output: the tier moves bytes
between memory, it computes nothing, and the resume's 58,896 hit tokens are the same text the cold run
prefills in 18.6 s.

Sizing against the host side: 16,384 MiB holds 1,177,348 tokens (the startup log's own count), i.e. 18
sessions of this size at 0.89 GiB each — E-K1/M-01's idle pinned memory, which is where the item sizes it
from. Host RAM: `MemAvailable` read 109 GiB during the A/B and 44 GiB at the low point during the
`long8` runs (where the model's file is being read again and /stats' occupancy read up to 7.0 GiB), so
its 16 GiB fits beside the 39.5 GiB host pool (19.73 GiB a rank) and the 2.2 GiB of snapshot slots.

`docs/GUIDE.md` already documents the tier — §4.8 gives the size, the 57 KiB a 4-token entry, what stays in
VRAM, and the "KV in VRAM stays high with one session and nothing moves" row in the troubleshooting table —
so it needed no change for this card; the numbers above are the lane's.

The runs, all under `runs/local/` and reproducible with the configs as committed: `x01-on-s1..3` and
`x01-off-s1..3` (`RAD_IDLE_CHARS=190000`, phase list `c1,c1b,c8,pf6,pf20,stall1,stall4,idle`),
`x01-long8-on-s1..2` and `x01-long8-off-s1..2` (phase `long8`, `RAD_LONG8_CHARS=140000`), and
`x01-verify-on` (`idle`, `RADIANCE_DEBUG_TIER_VERIFY=1`); the off ones with `RAD_PREFIX_CACHE_MIB=0` in
front of `scripts/backlog/run.sh`. They ran on this branch's `coupled03.yaml` before P-16's
`draft-depth-load-from: 1` landed (it merged after them), and the two arms of each pair differ in nothing
but the tier, which is what the A/B measures.

## 2. Merge bar (lossless items)

Merge when all three hold:

1. The output is byte-identical, or inside the item's stated bound (ORD / PF; see the field key at the top of the backlog's §3).
2. The kernel, op or step the item targets is measurably faster in ms (AQL stamps, rad-kbench or profile-ops), before and after on one machine (§6 step 3).
3. End-to-end tok/s or TTFT is not worse beyond noise.

A kernel gain whose end-to-end effect is inside noise still merges: gains compound.

Otherwise write `closed/<ID>.md` with the numbers and the reason, and close the card without merging code (like the backlog's §5).
Measurement-only items (K-17, C-01, T-03, M-14, M-15, ...) merge their note, plus the fix when the measurement calls for it.

## 3. Lossy items

- The change sits behind a switch, off by default.
- Report keptn's KL on u3 (mean / p99 KL, top-1, against today's default) plus tok/s / TTFT on one server lane in
  `lossy/<ID>.md`.
- Only the lossy review card proposes defaults, and the repository owner decides.

## 4. Out of scope

- Format or container changes: F-01..F-11, F-13 as a format change, F-18, F-20, F-21.
- The closed items of §5, P-15 (closed in refs/notes/p15.md) and D-08 (a pointer only).
- M-12 is folded into K-19, P-10, P-11 and P-12. D-10's parts are in K-08, K-16, K-18 and the D-10 card (`gemm_nt_f32`).

## 5. Parallel work

Cards run in parallel even when they touch the same file; whoever merges second rebases. `blocked_by` holds only real logical
dependencies.

`[design]` cards write `design/<ID>.md` and may rewrite the bodies of the build cards that wait for them. Run them with the
design agent: Opus 5.5, thinking medium.

## 6. Working from outside (no lane)

A worker without access to the lanes can still do a card, and measures it as far as their own hardware allows:

1. Write the change and its tests.
2. Run what needs no GPU:
   - the build (`./build.sh` or the CMake presets)
   - the CPU emulation tests in `tests/r4d_emu/`
   - the libref / libavx reference paths
   - `tests/chain_scalar_audit.py`
   - the plugin lints (`hyv4flash_arch_test`)
   - the scheduler tests (`sched_test`)
   - every ctest that runs without a card
3. With a GPU, measure the change yourself: before (the PR's merge base) and after (your code commit) on the same
   machine, same settings, at least three runs each, interleaved. Take the rows the item's merge bar (§2) names: the
   stamps, `rad-kbench` or `profile-ops` rows of the targeted kernel, op or step, and the output parity (§2.1). Write
   them to `notes/backlog/measurements/<item>.md` (for example `K-47.md`) and commit that file to the PR, with:
   - the commit ids of before and after (after is the commit of the code you measured; the commit that adds this file
     comes on top of it and changes nothing else),
   - who measured, the GPUs (model, count, TP), power cap, ROCm / driver version,
   - the container and the `ngram-placement` you ran with (any placement is fine on your machine),
   - the raw lines of each run for the rows the bar names (not only a summary). Keep the file under 40 KB: the
     review reads the PR as text and refuses larger files.
   Numbers from your machine are compared with each other only, never with another machine's.
4. Push a branch and open a PR whose title or body holds `Fixes #<issue>`. Title it `[measured] <item>: <subject>`
   when the file covers the item's whole merge bar, `[needs testing] <item>: <subject>` when it does not (no GPU).
5. **Write `/done` as a comment on the item's issue** (not on the PR): only that starts the review, and only the PR's
   author can. After every push you want reviewed, write `/done` again. A PR gets at most three review rounds, so
   push your fixes and the measurement together where you can.
6. The review reads the PR's diff and files and the issue's title and body, nothing else. It checks the code and the
   measurement file: the rows are the ones §2 asks for, before and after come from one machine, the raw lines carry
   the numbers. When both hold it approves and the PR merges at once. When only the measurement is missing, the issue
   gets the label `ready-for-testing` (§6.1).

### 6.1 Testing a `ready-for-testing` issue

No lane on the Moebius server measures anything. Anyone with a free GPU can measure a change that waits on the label
`ready-for-testing`, without writing code:

1. Check out the PR's head and its merge base (`git merge-base origin/<base> <head>`).
2. Measure both as step 3 above says.
3. Post the result as a comment on the issue, starting with a `## Measurement` heading.

For an own card, the board picks it up within the hour: the Reviewer checks it, and the change merges, or the issue is
told what is missing. For a PR from outside, the PR's author commits the posted measurement as
`notes/backlog/measurements/<item>.md` and writes `/done`; the board reminds them on the issue.

Items whose whole value is a measurement (marked "lane required" in their issue) need a lane from the start. Lossy items
keep §3.
