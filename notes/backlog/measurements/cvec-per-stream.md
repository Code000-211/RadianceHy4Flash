# cvec per stream (#179): output parity with the vector off, loaded-but-off and on

Not a backlog item and no speed claim: the change is lossy only when a deployment loads the vector and a request
turns it on (CARDS.md §3: behind `control-vector`, default `disabled`; with `off` a request must ask). What has to
hold is that every other path is the stock model byte for byte. The KL of the vector itself is the author's
(PR body, `data/vectors/hyv4flash-control-vector.json`): mean 0.199 / median 0.082 against stock, 29 refusals of
146 held-out against 135.

- before: `f82f8a1` (the base branch at the time of the run; it contains the PR's merge base `6e76ed7`)
- after: `055c289` (#179's code with the review fixes, merged on `6e76ed7`); the loader's warning fix `0f9b0ee`
  came later and changes only a log line
- measured by the reviewer on the local lane (host "inference"): 2x AMD Radeon AI PRO R9700 (gfx1201), TP2,
  power cap 210 W read back on both cards before each run, HIP 7.16.26385 (`hipconfig --version`, /opt/rocm), Linux 7.0.2-6-pve
- container `/srv/models/hy4-flash-iqr-v4/hy4-flash-iqr-v4-u3.rad`, `scripts/backlog/nospec.yaml`
  (`kernels: libr4d,libavx`, `ngram-placement: ram`); the two vector arms add only `control-vector: off|on`
- probe phases `greedy,c1` (`scripts/backlog/probe.py`): greedy is temperature 0 over three prompts and a 5.7K
  arrival; it is deterministic, so one run an arm decides parity. One run an arm, so the c1 rows are
  context, not a speed measurement.

| arm | binary | vector | greedy sha256 (prefix) / tokens | c1 tok/s |
|---|---|---|---|---|
| base | before | none | d6da4fb33eed/256 0699709c4be2/256 e7a23c5d9a2f/102 466e53a2921d/64 | 72.0 |
| pr-off | after | none | d6da4fb33eed/256 0699709c4be2/256 e7a23c5d9a2f/102 466e53a2921d/64 | 70.9 |
| pr-loaded | after | `control-vector: off` (loaded, no request asks) | d6da4fb33eed/256 0699709c4be2/256 e7a23c5d9a2f/102 466e53a2921d/64 | 70.6 |
| pr-on | after | `control-vector: on` | 62f768e5486d/256 afe91ea1ac5a/256 cc7a3aa6905d/104 96a2eb4d8bf1/64 | 68.6 |

Result: the three arms without a request for the vector give the same four texts byte for byte as the base. With
the vector on the per-stream package loads (`41 directions of 3072 (rank 8, 328 total), layers 1..41`) and steers
every text (similar lengths: 1069/1042/283/250 chars against 1059/862/284/260).

## Raw lines

### base
```
cap 210 W  (/sys/class/drm/card1/device/hwmon/hwmon3/power1_cap)
cap 210 W  (/sys/class/drm/card2/device/hwmon/hwmon4/power1_cap)
run base: <scratch>/base/runs/local/base
cfg <scratch>/base/runs/local/base/nospec.yaml, phases greedy,c1
gpu2: cards 0,1
{"name": "greedy", "texts": [{"gen": 256, "chars": 1059, "sha256": "d6da4fb33eed29de799b81279721e990b1cc5458d5dd780a5e824373a7435ba6"}, {"gen": 256, "chars": 862, "sha256": "0699709c4be2911589a34d9f85274ca1d5686482a04ef3611ea125792b1a5eac"}, {"gen": 102, "chars": 284, "sha256": "e7a23c5d9a2f3a45a8e9976590318c2e0670e5aa4494b268911817d23a0bce60"}, {"gen": 64, "chars": 260, "sha256": "466e53a2921d02165f7b410f283fea9e73b8d9b0e9f1dbf3fcc601b7ec78ae06", "name": "arrival5.7K"}]}
{"name": "c1", "clients": 1, "wall": 3.555, "gen": 256, "prompt": [73], "ttft": [0.109], "agg_tps": 72.007, "sum_req_dec_tps": 74.008, "decode_steps": 255.0, "ms_step": 13.512, "tok_per_step": 1.004, "mono_ns": [22355951714382, 22359506933351], "all_decoding_s": [0.10951805114746094, 3.555082321166992], "stage_layers": 164, "hit": 0.985, "routed_per_step": 696.494, "miss_units_per_step": 10.157, "steps_sliced": 0, "substeps": 0}
server-exit=0
done 2026-11-25T06:41:15Z rc=0
```
### pr-off
```
cap 210 W  (/sys/class/drm/card1/device/hwmon/hwmon3/power1_cap)
cap 210 W  (/sys/class/drm/card2/device/hwmon/hwmon4/power1_cap)
run pr-off: /root/hy4/wt/pr-179/runs/local/pr-off
cfg /root/hy4/wt/pr-179/runs/local/pr-off/nospec.yaml, phases greedy,c1
gpu2: cards 0,1
{"name": "greedy", "texts": [{"gen": 256, "chars": 1059, "sha256": "d6da4fb33eed29de799b81279721e990b1cc5458d5dd780a5e824373a7435ba6"}, {"gen": 256, "chars": 862, "sha256": "0699709c4be2911589a34d9f85274ca1d5686482a04ef3611ea125792b1a5eac"}, {"gen": 102, "chars": 284, "sha256": "e7a23c5d9a2f3a45a8e9976590318c2e0670e5aa4494b268911817d23a0bce60"}, {"gen": 64, "chars": 260, "sha256": "466e53a2921d02165f7b410f283fea9e73b8d9b0e9f1dbf3fcc601b7ec78ae06", "name": "arrival5.7K"}]}
{"name": "c1", "clients": 1, "wall": 3.611, "gen": 256, "prompt": [73], "ttft": [0.111], "agg_tps": 70.9, "sum_req_dec_tps": 72.884, "decode_steps": 255.0, "ms_step": 13.72, "tok_per_step": 1.004, "mono_ns": [22916415295555, 22920026014677], "all_decoding_s": [0.11187267303466797, 3.6105947494506836], "stage_layers": 164, "hit": 0.986, "routed_per_step": 696.494, "miss_units_per_step": 9.871, "steps_sliced": 0, "substeps": 0}
server-exit=0
done 2026-11-25T06:50:35Z rc=0
```
### pr-loaded
```
cap 210 W  (/sys/class/drm/card1/device/hwmon/hwmon3/power1_cap)
cap 210 W  (/sys/class/drm/card2/device/hwmon/hwmon4/power1_cap)
run pr-loaded: /root/hy4/wt/pr-179/runs/local/pr-loaded
cfg /root/hy4/wt/pr-179/runs/local/pr-loaded/cv-off.yaml, phases greedy,c1
gpu2: cards 0,1
{"name": "greedy", "texts": [{"gen": 256, "chars": 1059, "sha256": "d6da4fb33eed29de799b81279721e990b1cc5458d5dd780a5e824373a7435ba6"}, {"gen": 256, "chars": 862, "sha256": "0699709c4be2911589a34d9f85274ca1d5686482a04ef3611ea125792b1a5eac"}, {"gen": 102, "chars": 284, "sha256": "e7a23c5d9a2f3a45a8e9976590318c2e0670e5aa4494b268911817d23a0bce60"}, {"gen": 64, "chars": 260, "sha256": "466e53a2921d02165f7b410f283fea9e73b8d9b0e9f1dbf3fcc601b7ec78ae06", "name": "arrival5.7K"}]}
{"name": "c1", "clients": 1, "wall": 3.628, "gen": 256, "prompt": [73], "ttft": [0.116], "agg_tps": 70.571, "sum_req_dec_tps": 72.625, "decode_steps": 255.0, "ms_step": 13.769, "tok_per_step": 1.004, "mono_ns": [24714364591592, 24717992150403], "all_decoding_s": [0.11626124382019043, 3.6274495124816895], "stage_layers": 164, "hit": 0.984, "routed_per_step": 696.494, "miss_units_per_step": 10.914, "steps_sliced": 0, "substeps": 0}
server-exit=0
done 2026-11-25T07:20:33Z rc=0
```
### pr-on
```
cap 210 W  (/sys/class/drm/card1/device/hwmon/hwmon3/power1_cap)
cap 210 W  (/sys/class/drm/card2/device/hwmon/hwmon4/power1_cap)
run pr-on: /root/hy4/wt/pr-179/runs/local/pr-on
cfg /root/hy4/wt/pr-179/runs/local/pr-on/cv-on.yaml, phases greedy,c1
gpu2: cards 0,1
{"name": "greedy", "texts": [{"gen": 256, "chars": 1069, "sha256": "62f768e5486d052237943d75b8ea1cf06b70d6dfb2226e96d17115563abba9b5"}, {"gen": 256, "chars": 1042, "sha256": "afe91ea1ac5afa1918e5e3c57dfc2b32f222af146a74dd72db06b1fcf56a161a"}, {"gen": 104, "chars": 283, "sha256": "cc7a3aa6905d40d84a9a455d931476d879daeae6f8c36f45d4c8ab5e60e344a0"}, {"gen": 64, "chars": 250, "sha256": "96a2eb4d8bf1f4a2467cfc594e222e088a98130fc109f4f28a95f90c4b68ba1a", "name": "arrival5.7K"}]}
{"name": "c1", "clients": 1, "wall": 3.73, "gen": 256, "prompt": [73], "ttft": [0.111], "agg_tps": 68.642, "sum_req_dec_tps": 70.491, "decode_steps": 255.0, "ms_step": 14.186, "tok_per_step": 1.004, "mono_ns": [25809686514181, 25813416038103], "all_decoding_s": [0.11187553405761719, 3.729336738586426], "stage_layers": 164, "hit": 0.986, "routed_per_step": 696.482, "miss_units_per_step": 9.42, "steps_sliced": 0, "substeps": 0}
server-exit=0
done 2026-11-25T07:38:49Z rc=0
```

pr-loaded / pr-on server.log:
```
I control vectors: /root/hy4/wt/pr-179/build/radiance_home/vectors/hyv4flash-control-vector.gguf: 41 directions of 3072 (rank 8, 328 total), layers 1..41, scale 1.000 -- default ON (--control-vector on; a request can still say false)
I control vectors: /root/hy4/wt/pr-179/build/radiance_home/vectors/hyv4flash-control-vector.gguf: 41 directions of 3072 (rank 8, 328 total), layers 1..41, scale 1.000 -- default off until a request asks for it (the dashboard's switch, or a request's `control_v
```
