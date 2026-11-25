#!/usr/bin/env python3
"""prompts.py: the prompts and the code corpus every backlog tool sends, read from this tree.

bench.py and probe.py both take their prompts from here, so a run on any lane and in any worktree
sends the same tokens: the corpus is the C++ of the repository this file sits in
(`scripts/backlog/../../`, overridable with RAD_SRC), never a copy on the measuring host.
"""
import os, random, time

SRC = (os.environ.get("RAD_SRC") or
       os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CORPUS_DIRS = ("core", "arch/hyv4flash", "libavx", "libr4d", "core/server", "core/sched")

CODE_PROMPTS = [
    "Write a complete Python module implementing an LRU cache class with get/put, TTL expiry per entry, thread safety via a lock, type hints and docstrings, followed by a pytest test suite covering eviction order, expiry and concurrent access.",
    "Implement in C++17 a lock-free single-producer single-consumer ring buffer template with power-of-two capacity, std::atomic indices with acquire/release ordering, try_push/try_pop, and a small benchmark main(). Explain each memory ordering choice in comments.",
    "Write a Rust command-line tool that recursively scans a directory, groups files by SHA-256 hash to find duplicates, prints the groups sorted by wasted bytes, and supports --min-size and --json flags. Use clap and walkdir; include error handling and unit tests.",
    "Write a TypeScript Express REST API for a todo list with routes for create, list (with pagination and filtering by status), update and delete, input validation with zod, an in-memory repository behind an interface, and Jest tests using supertest.",
    "Here is a slow Python function:\n\n```python\ndef pairs(nums, target):\n    out = []\n    for i in range(len(nums)):\n        for j in range(len(nums)):\n            if i != j and nums[i] + nums[j] == target and (nums[j], nums[i]) not in out:\n                out.append((nums[i], nums[j]))\n    return out\n```\n\nRewrite it to run in O(n), keep the exact output semantics, explain the differences, and add property-based tests with hypothesis comparing it to the original.",
    "Write a Go HTTP reverse proxy with round-robin load balancing over a configurable list of backends, active health checks every 5 seconds, request timeouts, graceful shutdown on SIGTERM and Prometheus metrics for request counts and latencies. Include the full main.go and a README section on usage.",
    "Implement Dijkstra's algorithm and A* search in Python over a weighted grid read from a text file (digits are cell costs, # is a wall), with a CLI that prints the path overlaid on the grid, plus unit tests comparing both algorithms on random grids.",
    "Write a bash script that backs up a PostgreSQL database with pg_dump, compresses it with zstd, uploads it to an S3 bucket with the aws cli, keeps the last 14 daily and 8 weekly backups, logs to syslog and exits non-zero with a clear message on any failure. Then explain every section.",
    "Write a C function library for a growable string builder (create, append, appendf with printf-style formatting, insert, free) with careful overflow checks, then a test program using assert that exercises edge cases such as empty strings and very large appends.",
    "Write a SQL schema for a library system (books, authors, members, loans, reservations) with constraints and indexes, then ten example queries in PostgreSQL covering joins, window functions, CTEs and an upsert, and explain each query's plan considerations.",
    "Write a Python asyncio web crawler that respects robots.txt, limits concurrency per host with semaphores, deduplicates URLs, follows links up to a configurable depth, stores pages in SQLite and exposes a progress report. Include type hints and tests with aiohttp test utilities.",
    "Implement a simple tokenizer, recursive-descent parser and tree-walking interpreter in Python for arithmetic expressions with variables, assignment, if/else and while loops, plus a REPL and tests for operator precedence and error messages.",
]
GREEDY = [CODE_PROMPTS[0], CODE_PROMPTS[4],
          "Fix the bug in this C function and show the corrected code:\n\nint sum(int *a, int n) {\n  int s;\n  for (int i = 0; i <= n; i++) s += a[i];\n  return s;\n}"]


def code_corpus():
    files = []
    for d in CORPUS_DIRS:
        p = os.path.join(SRC, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith((".cpp", ".h", ".hip")):
                files.append(os.path.join(p, f))
    if not files:
        raise SystemExit(f"prompts.py: no .cpp/.h/.hip under {SRC} (set RAD_SRC)")
    return files


CORPUS = code_corpus()


def long_prompt(chars, seed, stamp=False):
    """A prompt of about `chars` characters of repository source. stamp=True makes it unique per call
    (a fresh arrival must not hit the prefix cache; probe.py uses it), off keeps it stable (bench.py)."""
    rnd = random.Random(seed)
    # shuffle, not sample: the same seed then walks the corpus in the order the backlog's own
    # probe.py and bench.py did, so pf6 / pf20 stay the 6.7K / 24.9K prompts their numbers were
    # taken on (notes/backlog/refs/kernels/levers-general.md).
    files = list(CORPUS)
    rnd.shuffle(files)
    parts, n = [], 0
    for f in files:
        t = open(f, errors="replace").read()[: max(0, chars - n)]
        parts.append(f"// ===== {os.path.relpath(f, SRC)} =====\n{t}")
        n += len(t)
        if n >= chars:
            break
    head = f"[review {seed}-{time.time()}]" if stamp else f"[review {seed}]"
    return (head + " Review the following C++ code for bugs, then summarise the design.\n\n"
            + "\n".join(parts))
