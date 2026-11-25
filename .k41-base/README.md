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
    busy-prefill-chunk: auto          # 512 halves decode stalls when long prompts arrive under load,
                                      # at about +60 % time to first token for those prompts
    checkpoint-slots: 128             # for many concurrent agentic sessions (default 64)
    num-speculative-tokens: 5         # DFly2 drafts at most 5 tokens
    draft-window: fixed
    draft-coupling: on
    draft-verify: traversal
    draft-depth: adaptive
    draft-depth-load: on
    draft-depth-load-from: 1          # depth rule priced per row, from one sequence
    draft-suffix: on
```

Run it with:

```sh
radiance --config hy4-tp2.yaml
```

The Engram placement is the key here that decides what a start costs. `ram` reads the table into a
private pinned copy at every start: four tables one after the other at 1.4 GB/s, about 42 s of a
66 s startup. `ram=DIR` keeps that copy as a file, `DIR/ngram-<id>.tab`, in a tmpfs, so a start that
finds it complete checks samples of it against the container and pins it where it lies instead of
reading the table again. Measured on two R9700 at TP2 with this recipe (u3 container, 210 W, the
engine answering `/health`):

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
batch is two intervals wide and the span runs them as two steps; pieces under 1024 rows are not
cut (`RADIANCE_HY4_PIPE_ROWS` moves that floor, `RADIANCE_HY4_PIPE_LANE` the lane).

Measured at these settings on four R9700 (32 GB each, 210 W a card), a 10948-token prompt, three
warm runs a config:

| | |
|---|---|
| prefill, `checkpoint-interval: 4096` and the 8192-token batch, no pipeline | 4978.1 tok/s |
| the same with `RADIANCE_HY4_PIPE=1` | **5577.5 tok/s** (+12.0 %, and +13.5 % over four ranks at the 2048-token intervals it ships with) |
| decode, 1 / 8 clients | 147.2 / 551.8 tok/s, the same greedy tokens with and without the cut |
| the cut's cost in memory | 0.05 GiB of activation arena and 0.32 GiB of the pools |
| prefill with the cut, bit for bit | not the step run whole (`448a9473ec` against `e7001a8285`) |

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
