#!/usr/bin/env python3
"""lane-cfg.py <cfg.yaml> <out.yaml>: a backlog config with the lane's settings put in.

The configs in scripts/backlog/*.yaml are written for the served (local) lane. A lane run rewrites
the keys that differ per lane, from the environment lane.sh exports, and leaves every other
line of the file alone, so a run's exact config sits next to its numbers:

    RAD_PORT               server.port
    RAD_MODEL              models.*.model
    RAD_HOST_POOL_MIB      models.*.host-pool-mib (empty: keep the file's)
    RAD_NGRAM              models.*.ngram-placement (empty: keep the file's)
    RAD_PREFIX_CACHE_MIB   models.*.prefix-cache-mib (empty: keep the file's)
"""
import os, re, sys

KEYS = {"port": "RAD_PORT", "model": "RAD_MODEL",
        "host-pool-mib": "RAD_HOST_POOL_MIB", "ngram-placement": "RAD_NGRAM",
        "prefix-cache-mib": "RAD_PREFIX_CACHE_MIB"}


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: lane-cfg.py <cfg.yaml> <out.yaml>")
    src, out = sys.argv[1], sys.argv[2]
    want = {k: os.environ.get(v, "") for k, v in KEYS.items()}
    seen, lines = set(), []
    for ln in open(src, errors="replace").read().splitlines(True):
        m = re.match(r"(\s*)([a-z0-9-]+):(\s*)(.*)$", ln.rstrip("\n"))
        if m and m.group(2) in want and want[m.group(2)]:
            key = m.group(2)
            lines.append(f"{m.group(1)}{key}: {want[key]}\n")
            seen.add(key)
        else:
            lines.append(ln)
    missing = sorted(k for k, v in want.items() if v and k not in seen)
    if missing:
        sys.exit(f"lane-cfg.py: {src} has no key(s) {', '.join(missing)} to set")
    with open(out, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
