# Qwen3.8 Windows performance working set

This directory is a compact reading layer over [`NOTES.md`](../NOTES.md). It
covers the source through **2026-08-16 10:58 PDT** and leaves the chronological
notebook intact.

## Read only what the task needs

| Need | Read |
|---|---|
| Resume the work or recover after compaction | [`current-state.md`](current-state.md) |
| Choose or review a runtime setting | [`decisions.md`](decisions.md) |
| Reproduce or compare a measurement | [`benchmark-contract.md`](benchmark-contract.md) |
| Understand how the current result was reached | [`timeline.md`](timeline.md) |
| Inspect full evidence, logs, PIDs, and intermediate measurements | [`NOTES.md`](../NOTES.md) |

For a normal continuation, `current-state.md` is the complete minimum read.
Add `benchmark-contract.md` before measuring and `decisions.md` before changing
a selected default.

## Precedence and freshness

1. A later entry in `NOTES.md` supersedes every generated document here.
2. Within this directory, `current-state.md` supersedes older results mentioned
   in `timeline.md` or `decisions.md`.
3. “Qualified” means the configuration passed the measurement and behavior
   gates recorded in the notebook. It does not prove that a server or PID is
   still live.
4. Process IDs, GPU contention, installed packages, and dirty-worktree details
   are snapshots. Verify them before acting.

The notebook contains several checkpoints called “final.” The current
qualified performance baseline comes from the **10:38–10:58 two-step MTP
retest and promotion**. Its `159.973` fixed-work and `117.794` real-sampled
results supersede the 09:47–09:54 target-XQA leader.

The default-only production relaunch was healthy at the source cutoff with all
three intended CUDA-graph modes captured. Process state remains a snapshot.

## Maintenance pattern

Keep this set compact:

- update `current-state.md` whenever the qualified winner or immediate handoff
  changes;
- add one row to `decisions.md` when a candidate is selected or closed;
- update `benchmark-contract.md` only when the workload or acceptance gates
  change;
- add one phase to `timeline.md` for a material new direction;
- retain raw samples, logs, incident detail, and recovery narration in
  `NOTES.md`.
