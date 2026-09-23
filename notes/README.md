# Experiment notes

This directory preserves the Windows CUDA and Apple-silicon MPS/MLX
experiment history, including Qwen3.8 and DiffusionGemma. The full notebook
contains exact samples, commands, failures, and recovery checkpoints.
Recurring lessons live in the repository instructions so each new task can
use them immediately.

## Read only what the task needs

| Need | Read |
|---|---|
| Apply recurring working rules and avoid previously diagnosed failures | [AGENTS.md](../AGENTS.md) |
| Resume work or recover the latest platform handoff | [current-state.md](current-state.md), then later dated entries for that lane |
| Choose or revisit a runtime setting or rejected candidate | [decisions.md](decisions.md) |
| Reproduce or compare a measurement | [benchmark-contract.md](benchmark-contract.md) |
| Understand the sequence of experiments | [timeline.md](timeline.md) |
| Recover exact samples, commands, failures, and incidents | [experiment-log.md](experiment-log.md) |

## Find the platform-specific lessons

| Area | Entry points |
|---|---|
| Shared measurement and correctness | [Measurement discipline](../AGENTS.md#measurement-discipline) and [numerical/state contracts](../AGENTS.md#shared-numerical-and-state-contracts) |
| Windows CUDA | [Launcher contract](../AGENTS.md#qualified-production-contract), [GPU/process safety](../AGENTS.md#native-windows-gpu-and-process-safety), and [kernel boundaries](../AGENTS.md#windows-cuda) |
| macOS / Apple silicon | [Residency, attention, MTP history, and Metal lessons](../AGENTS.md#macos-and-apple-silicon-lessons) |
| Closed experiments | [Reopening rules](../AGENTS.md#closed-branches-and-retained-experiments) and the [detailed decision ledger](decisions.md) |

The complete experiment notebook remains at its existing path. The durable
rules link directly to supporting entries; the notebook retains every
historical sample and incident. Search by date, model, mechanism, or experiment
identifier when more detail is needed.

## Source precedence and freshness

1. Fresh source inspection, worktree state, launcher resolution, endpoint
   checks, process ownership, and measurements establish live truth.
2. A later dated entry for the same platform, checkpoint, and workload can
   supersede an earlier conclusion. Cross-machine appends can appear out of
   date order; document position alone does not establish recency.
3. `current-state.md` carries reconciled handoffs and production conclusions.
   Check the date and scope of the relevant section.
4. `decisions.md` records selected and closed choices; `timeline.md` provides
   an orientation map through the earlier phases.
5. Qualification covers the recorded measurement, behavior, capacity, and
   client gates. Each change of model, sampling, implementation, hardware, or
   workload needs the applicable checks again.

Process IDs, contention, installed packages, dependency versions, worktree
state, and free memory are timestamped observations. Reverify them before
acting. Historical goals and running-server entries remain historical context.

## Keep durable guidance concise

Give each new finding a clear home:

- Put an established, recurring failure-prevention rule in `AGENTS.md`.
  State what to do, when it applies, and link the evidence. Update its existing
  statement when the remedy or scope changes.
- Put a selected configuration, rejected candidate, and reopening conditions
  in `decisions.md`.
- Update `current-state.md` when the qualified winner or immediate handoff
  changes.
- Update `benchmark-contract.md` when workloads, environment requirements,
  measurement definitions, or acceptance gates change.
- Add a phase to `timeline.md` for a material new direction.
- Append evidence to `experiment-log.md` after meaningful changes, launches,
  measurements, failures, promotions, and cleanup, following the writing rules
  at the top of that file. Keep individual samples and incident narration
  there, without process IDs, hashes, or handoff lines.

Preserve original measurements with their actual settings and qualification
limits. Link summaries to those entries so the short instructions and complete
record remain connected.
