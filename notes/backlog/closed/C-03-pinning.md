# C-03 / CCD pinning — dropped, with the numbers

The item (issue #112) asks for two things: a spin-then-park rank barrier, and pinning the step and
rank threads to one CCX. This file records the second as **dropped**, which CARDS.md §2 asks for
when a change ships without part of its item.

## What was dropped

`core/runtime/thread_pin.{h,cpp}` and its two call sites in `core/engine.cpp`: an opt-in
(`RADIANCE_PIN_THREADS`) pinner that placed the step and rank threads on one CCD.

## Why

- **It measured slower.** The pilot arm ran at **−14.9 % tok/s** against the unpinned arm on the
  same box (c1 aggregate, one boot a side — a pilot, not a §6 measurement, but it is the direction
  the item would have to beat).
- **It was never compiled.** No `CMakeLists.txt` entry, so the file was outside the build and
  outside `ctest`.
- **Its documentation described a different implementation.** `thread_pin.h` documented a CCD index
  and a core range the code did not implement; its fallback parse was broken, and its placement rule
  could put two threads on one CPU.
- **No test existed for it**, which AGENTS.md makes blocking on its own.

The barrier half of the item is unaffected and ships in this PR.

## Reopen condition

A pinner is worth re-opening with a §6 measurement (three interleaved runs each side on a lane) that
shows a gain, a placement rule that cannot alias two threads onto one core, its parse and fallback
covered by tests, and a `CMakeLists.txt` entry. Until then this half stays closed.

This file is a record of a **drop**, not a claim that pinning cannot help; the item's barrier term is
measured separately in `notes/backlog/measurements/C-03.md`.