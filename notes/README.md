# Qwen3.8 Windows performance record

This directory is the durable record for the native-Windows Qwen3.8-27B
SGLang work. It contains both compact decision-oriented documents and the full
chronological evidence migrated from the former root `NOTES.md`.

**Last reconciled:** 2026-08-20 11:44 PDT.

## Read only what the task needs

| Need | Read |
|---|---|
| Resume the work or recover after compaction | [`current-state.md`](current-state.md) |
| Choose or review a runtime setting | [`decisions.md`](decisions.md) |
| Reproduce or compare a measurement | [`benchmark-contract.md`](benchmark-contract.md) |
| Understand how the current result was reached | [`timeline.md`](timeline.md) |
| Inspect exact samples, commands, logs, PIDs, failures, and intermediate state | [`experiment-log.md`](experiment-log.md) |

For a normal continuation, `current-state.md` is the complete minimum read.
Add `benchmark-contract.md` before measuring and `decisions.md` before changing
a selected default. Use `experiment-log.md` when exact evidence or recovery
detail matters.

## Source precedence and freshness

1. Fresh source inspection, worktree state, launcher resolution, endpoint
   checks, process ownership, and measurements establish live truth.
2. A later entry in `experiment-log.md` supersedes every earlier notebook
   entry and compact summary.
3. `current-state.md` carries the latest reconciled production conclusion.
4. `decisions.md` records selected and closed choices; `timeline.md` remains an
   orientation map across superseded phases.
5. “Qualified” means a configuration passed the recorded measurement,
   behavior, capacity, and production gates. Runtime liveness always requires
   a fresh check.

Process IDs, GPU contention, installed packages, dependency versions,
worktree state, and reported free VRAM are timestamped observations. Reverify
them before acting.

## Document roles

- `current-state.md` contains the accepted production contract, latest
  measurements, invariants, retained experimental infrastructure, and handoff.
- `decisions.md` contains choices whose reasoning should survive the raw
  experiment stream, including rejected branches and conditions for reopening.
- `benchmark-contract.md` defines comparable workloads, environment capture,
  safety procedure, promotion gates, and interpretation rules.
- `timeline.md` condenses the full sequence and identifies which later result
  supersedes each older “final” checkpoint.
- `experiment-log.md` is the lossless chronological notebook. It preserves the
  former root `NOTES.md` content, including raw samples and incident recovery.

## Maintenance pattern

Keep the two layers synchronized:

- append raw evidence and recovery checkpoints to `experiment-log.md` after
  meaningful changes, launches, measurements, failures, promotions, and
  cleanup;
- update `current-state.md` whenever the qualified winner or immediate handoff
  changes;
- add a row to `decisions.md` when a candidate is selected, rejected, or
  reopened by materially new evidence;
- update `benchmark-contract.md` when workloads, environment requirements,
  safety constraints, or acceptance gates change;
- add one phase to `timeline.md` for a material new direction;
- keep raw logs, transient PIDs, complete sample lists, and incident narration
  in `experiment-log.md`.

The repository root no longer owns a separate notebook. Continue the recovery
ledger in `notes/experiment-log.md` so durable guidance and its evidence remain
together.
