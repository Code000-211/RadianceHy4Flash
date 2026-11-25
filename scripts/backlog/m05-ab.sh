#!/bin/bash
# M-05 A/B on one lane: `new` = the served default (min-gain from the plan's residency),
# `old` = RADIANCE_HEAT_MIN_GAIN=1.5 named (the merge base's constant). Three interleaved runs each,
# nospec at max-model-len 8192 (83 % resident, where the rule fires). Run through lane.sh:
#   scripts/backlog/lane.sh iks-A 'scripts/backlog/m05-ab.sh'
set -u
cd "$(dirname "$0")/../.."
if [ ! -x build/bin/radiance ] || [ ! -f build/radiance_home/kernels/libr4d.so ] || [ -z "$(ls build/radiance_home/architectures 2>/dev/null)" ]; then
  [ -f build/CMakeCache.txt ] || "${RAD_CONFIGURE:-cmake}" -S . -B build -G Ninja -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201 >/dev/null
  cmake --build build -j 48 -- -k 0 2>&1 | tail -3   # -k 0: museglimmer_test does not build with clang 18
fi
for i in 1 2 3; do
  scripts/backlog/run.sh m05-new-$i scripts/backlog/nospec8k.yaml c1,c8
  scripts/backlog/run.sh m05-old-$i scripts/backlog/nospec8k.yaml c1,c8 RADIANCE_HEAT_MIN_GAIN=1.5
done
