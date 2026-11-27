# radiance · Hy4-flash

This is a fork of [radiance](https://codeberg.org/StillDeadcode/radiance), the C++/HIP inference
server for AMD GPUs, with a plugin for **Tencent Hy4-flash** (`hy_v4_flash`, 220B MoE). It adds the
`hyv4flash` architecture plugin and its kernels:

- MLA attention with sinks: 30 sliding-window layers and 12 sparse layers with a top-k indexer
- Engram n-gram tables, iHC hyper-connections, and the latent MoE (512 experts, top-16, plus a
  shared expert)
- DFly2 block-drafter speculative decoding, with coupled proposals, traversal verify and an
  adaptive verify depth
- tensor parallel at TP2 and TP4, calibration taps and the KL mode for quantising, and vision

It is built and tested on two Radeon AI PRO R9700 cards (gfx1201, 32 GB each), and on four of them
at TP4.

Quantised containers are made by a separate quantisation workflow. The one used below is
[`keptn/hy4-flash-iqr-v4`](https://huggingface.co/keptn/hy4-flash-iqr-v4)
(`hy4-flash-iqr-v4-u3.rad`, 128.6 GiB: u3 trellis experts, w4nl64 GPTQ attention); its README
lists the other variants and their KL figures.

For everything that is not Hy4-specific (other models, Docker, the full option list), see the
upstream README and [`docs/GUIDE.md`](docs/GUIDE.md). The Hy4 details and the TP2 recipe are in
[`docs/ARCHITECTURES.md`](docs/ARCHITECTURES.md) (section "Hy4-flash"), the four-rank one with its
measurements in [`docs/TP4.md`](docs/TP4.md).

## Example config: two R9700 at TP2

```yaml
server:
  host: 127.0.0.1
  port: 8033
  kernels: libr4d,libavx,spechub      # spechub carries the DFly2 coupling / traversal ops
memory:
  gpu-headroom-mib: 128
models:
  hy4-flash:
    model: /srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad
    tp: 2
    placement: expert_tiered          # experts that do not fit stay in pinned host memory
    host-pool-mib: 39000
    ngram-placement: ram=/dev/shm/radiance-ngram   # the Engram table as a file in a tmpfs: a start
                                      # that finds it filled pins it where it lies instead of
                                      # reading the 60 GiB again (see below on what it costs)
    max-model-len: 1000000
    max-num-batched-tokens: 2048
    checkpoint-interval: 1984         # a quantum under the batch: a chunk sharing its step with
                                      # decoding sessions lands on the interval boundary, so an
                                      # interval is one step instead of two (arrival TTFT -16 %
                                      # at one decoder, -12 % at four on the iks-A lane). Chunks
                                      # are a quantum shorter than 2048 rows, so a long prompt's
                                      # answer may differ from a 2048-chunk step at tie level (no
                                      # dtype, kernel or accumulation changes)
    max-num-seqs: 8
    mm-max-patches: 0                 # no vision
    prefix-resume: nearest
    prefix-cache-mib: 16384           # host tier: finished conversations' KV moves to host RAM,
                                      # so the VRAM goes back to the experts
    # prefix-cache-dir: /var/lib/radiance/prefix   # opt-in: the prefix cache survives a restart
                                      # (below); it holds conversation text unencrypted
    busy-prefill-chunk: auto          # 512 halves decode stalls when long prompts arrive under load,
                                      # at about +60 % time to first token for those prompts
    checkpoint-slots: 128             # for many concurrent agentic sessions (default 64)
    num-speculative-tokens: 5         # DFly2 drafts at most 5 tokens
    draft-window: fixed
    draft-coupling: on
    draft-verify: traversal
    draft-depth: adaptive
    draft-depth-load: on
    draft-depth-load-from: 1          # depth rule priced per row, from one sequence (draft-depth-stop
                                      # is then not read); the row price reads each step against its
                                      # neighbours, so a drifting step time is no longer taken for it
    draft-suffix: on                  # suffix steps from the conversation itself, and from earlier
                                      # answers (draft-suffix-global-mib, 256 by default)
```

Run it with:

```sh
radiance --config hy4-tp2.yaml
```

The Engram placement is the key here that decides what a start costs. `ram` reads the 60 GiB table
into a private pinned copy at every start. The four tables are read beside one another and beside the
pass over the container (they used to be read one after the other, 42 s of a 66 s startup), so the
weights phase is the longer of the two rather than their sum. `ram=DIR` keeps that copy as a file,
`DIR/ngram-<id>.tab`, in a tmpfs, so a start that finds it complete checks samples of it against the
container and pins it where it lies instead of reading the table again. Measured on two R9700 at
TP2 with this recipe (u3 container, 210 W, the engine answering `/health`):

| Engram placement | starts, seconds to `/health` |
|---|---|
| `ram` | 31.1, 53.4, 36.0 |
| `ram=DIR`, tmpfs mounted the default way: the fill, then three reuses | 58.9, then 35.8, 36.7, 36.6 |
| `ram=DIR` on a `huge=always` tmpfs: the fill, then three reuses | 32.1, then 22.2, 22.5, 22.5 |

The reused file takes 13.5 s off a 36 s start, and it does that only on huge pages: on a tmpfs
mounted the default way a reuse lands back at the `ram` time and the fill costs 23 s on top, so the
mount is part of the recipe and not a tuning detail. The answer and the rate were unchanged by it
(greedy text identical in every start, 69 tok/s decoding 64 tokens). The last engine to leave
deletes the file, so what a file saves is a start after a crash, or after a stop that killed the
engine before it cleaned up, rather than a planned restart; and a file left behind is 60.24 GiB of
tmpfs that the next start counts against the free memory it checks before it pins anything. So it
pays where the host has the room to leave it in place, and where two engines on separate card sets
hold one table in host memory instead of two. Mount the directory's tmpfs with huge pages, since a
tmpfs takes them only where `shmem_enabled` allows them and that defaults to `never`:

```sh
sudo mkdir -p /dev/shm/radiance-ngram
sudo mount -t tmpfs -o size=70G,huge=always tmpfs /dev/shm/radiance-ngram
```

In a container, mount it on the host and pass it through
(`-v /dev/shm/radiance-ngram:/dev/shm/radiance-ngram`): Docker's own `/dev/shm` is 64 MiB, and
`--tmpfs` refuses `huge=` (`--shm-size` raises the size and takes no page size).

Measured with this config (u3 container, TP2, 210 W power cap, sampled at T 0.9):

| | 1 client | 8 clients |
|---|---|---|
| decode without speculation | about 75 tok/s | about 305 tok/s |
| decode with DFly2 | about 140–150 tok/s | about 380–430 tok/s |
| prefill | 3000–4100 tok/s, depending on prompt length | |

Some of what makes this recipe faster has no key, since it is on by default. The routed experts take
the resident ones first and copy the misses as whole units on a DMA budget ("Weg 2"): the routed block
of a DFly2 eight-client verify pass is 1.0–1.1 ms shorter, and without a drafter its 34 MiB ring costs
0.1–0.4 %. `RADIANCE_W2=0` turns it off; it acts on steps of 16 rows and more. The expert GEMM's
tighter tile grid takes another 0.75 ms off that pass. The global suffix cache is on with
`draft-suffix: on`. Its only measured gain is on answers asked again under load (+2.7 % per request
at eight clients, TP4), and it costs nothing elsewhere. The output is byte-identical with each of them.

Two keys are worth turning on where the host has the room for them:

- `prefix-cache-dir: DIR` writes the prefix cache to `DIR` at a clean stop (SIGTERM), and also every
  `prefix-cache-flush-s` (600 s) while the server is idle. The next start reads it back on first use.
  On this pair a 24,608-token conversation answered its first request after a restart in 1.06 s, where
  a cold prefill took 6.6–7.3 s. The answer and its top-5 logprobs were bit-identical to the cold run.
  The directory is bounded by `prefix-cache-dir-mib` (default: the host tier plus the checkpoint
  store). It holds the conversations as plain text, so give it a directory only the server can read.
- `keep-container-cache: true` keeps the container in the page cache after the load. A restart then
  took 17 s where it took 29–33 s without it, and the first start took 20 s longer (49 s against
  28 s, buffered reads). It is kept only when the whole container (128.6 GiB) plus
  `host-headroom-mib` is still free after the pins. With this recipe's pools that rules out a host of
  170 GB: it says so at startup and drops the pages as usual.

## Example config: four R9700 at TP4

Four ranks cut the experts four ways: each card holds a quarter of them, and each expert's width is
cut in 128-column blocks over the ranks, so a card's share is whole blocks and every block has one
owner. The scheduler keys of the file above (the `draft-*` family, `busy-prefill-chunk`,
`checkpoint-slots`, the Engram placement) are per request and carry over unchanged; the file below
leaves them at their defaults and leaves placement to the planner, which at four ranks fits more of
the experts on the cards than the TP2 recipe can.

```yaml
server:
  host: 127.0.0.1
  port: 8034
  kernels: libr4d,libavx,spechub      # spechub carries the DFly2 coupling / traversal ops
memory:
  gpu-headroom-mib: 128
models:
  hy4-flash:
    model: /srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4.rad
    tp: 4
    num-speculative-tokens: 5         # DFly2 drafts at most 5 tokens
    prefill-span: 2                   # an 8192-token batch is two intervals, run as two steps
    max-num-batched-tokens: 8192
    checkpoint-interval: 4096         # one interval a step, and the step is what gets cut
```

Run it with the prefill pipelined:

```sh
RADIANCE_HY4_PIPE=1 radiance --config hy4-tp4.yaml
```

`RADIANCE_HY4_PIPE=1` runs a step as two row pieces and issues each piece's junction -- its `add`,
its exact all-reduce and the norm the next layer reads -- on the step's second lane, beside the
other piece's expert GEMMs. It needs a step of exactly one checkpoint interval, which is why the
batch is two intervals wide and the span runs them as two steps. A step is cut only when both
pieces have at least 1024 rows (`RADIANCE_HY4_PIPE_MIN` moves that floor, `RADIANCE_HY4_PIPE_ROWS`
the cut, `RADIANCE_HY4_PIPE_LANE` the lane).
`RADIANCE_HY4_PIPE_ATTN=1` (with the pipeline) cuts the attention core into the same two pieces,
so piece 0's attention junction rides the second lane beside piece 1's attention instead of
waiting; the output is the same bytes (`notes/backlog/measurements/P-17.md`).

Two bugs of the pipeline are fixed, so it is safe to use with current code. A staged pipelined step
of 8192 rows could hang the four ranks on a long prompt (issue #2, fixed in #205). The pieces also
read int8 codes that an earlier step had left behind (#282), so a long prompt's answer was off topic
and changed with the requests sent before it. The first measurement here (5577.5 tok/s, +12.0 %) was
taken with that bug. With the fix, on four R9700 (270 W a card, u4 container, the batch, interval
and span above, no drafter), measured on nine cold prompts of about 11K tokens a side:

| | |
|---|---|
| prefill, `checkpoint-interval: 4096` and the 8192-token batch, no pipeline | 4849.1 tok/s |
| the same with `RADIANCE_HY4_PIPE=1` | **5290.0 tok/s** (+9.1 %) |
| decode, 1 client, no drafter | 118.9 tok/s either way |
| long-prompt greedy output with the cut | the same in 3 of 3 runs and on topic; not bit for bit the step run whole (the same opening sentence, the first 14–64 of 64 greedy tokens equal) |

Decode does not go through the cut: with DFly2 the first measurement (210 W) read 147.2 / 551.8
tok/s at 1 / 8 clients, the same greedy tokens with and without the pipeline. The cut costs about
0.1 GiB of activation arena (0.05 GiB at first, 0.04 GiB more with the fix) and 0.32 GiB of the pools.

## Build from source

You need Linux, git, a C++20 compiler, CMake 3.21+, Ninja and ROCm (`hipcc`).

```sh
git clone https://github.com/Code000-211/RadianceHy4Flash.git && cd RadianceHy4Flash
./build.sh                                   # configure, build for the cards present, run the fast tests
```

Set `RADIANCE_HOME=build/radiance_home` when running from the build tree, so the server loads the
kernel plugins built there.

## License

[Apache-2.0](LICENSE), copyright Deadcode and the radiance contributors. Keep [`NOTICE`](NOTICE)
with any copy or derivative. It credits the authors and lists the MIT-licensed code the tree
includes. The model weights are not covered: each keeps the license of the model it was converted
from.
