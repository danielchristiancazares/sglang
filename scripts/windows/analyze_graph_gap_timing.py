#!/usr/bin/env python3
"""Apply the repeatable 0.75 ms admission gate to CUDA-event graph gaps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires samples")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_gaps(
    records: Sequence[Mapping[str, Any]],
    *,
    warmup_per_transition: int = 3,
    minimum_samples: int = 20,
    threshold_ms: float = 0.75,
    maximum_relative_mad: float = 0.25,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("artifact_type") != "speculative_graph_gap":
            raise ValueError("unexpected graph-gap artifact type")
        category = str(record["gap_before"])
        elapsed = float(record["elapsed_ms"])
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("graph-gap samples must be finite and nonnegative")
        grouped[category].append(elapsed)
    transitions = {}
    for category in sorted(grouped):
        raw = grouped[category]
        samples = raw[warmup_per_transition:]
        if not samples:
            continue
        median = statistics.median(samples)
        absolute_deviations = [abs(sample - median) for sample in samples]
        mad = statistics.median(absolute_deviations)
        relative_mad = mad / median if median > 0.0 else math.inf
        p10 = _percentile(samples, 0.10)
        p90 = _percentile(samples, 0.90)
        relative_p80_span = (
            (p90 - p10) / median if median > 0.0 else math.inf
        )
        repeatable = (
            len(samples) >= minimum_samples
            and relative_mad <= maximum_relative_mad
            and relative_p80_span <= maximum_relative_mad
        )
        recoverable = p10 if repeatable else 0.0
        transitions[category] = {
            "raw_count": len(raw),
            "warmup_discarded": min(warmup_per_transition, len(raw)),
            "sample_count": len(samples),
            "samples_ms": samples,
            "mean_ms": statistics.mean(samples),
            "median_ms": median,
            "p10_ms": p10,
            "p90_ms": p90,
            "min_ms": min(samples),
            "max_ms": max(samples),
            "mad_ms": mad,
            "relative_mad": relative_mad,
            "relative_p80_span": relative_p80_span,
            "repeatable": repeatable,
            "repeatable_recoverable_ms": recoverable,
            "passes_0_75ms_gate": repeatable and recoverable >= threshold_ms,
        }
    best = max(
        (
            (name, values["repeatable_recoverable_ms"])
            for name, values in transitions.items()
        ),
        key=lambda item: item[1],
        default=(None, 0.0),
    )
    return {
        "schema_version": 1,
        "artifact_type": "speculative_graph_gap_summary",
        "admission_gate": {
            "threshold_ms": threshold_ms,
            "minimum_samples": minimum_samples,
            "maximum_relative_mad": maximum_relative_mad,
            "best_transition": best[0],
            "best_repeatable_recoverable_ms": best[1],
            "fund_graph_tail_work": best[1] >= threshold_ms,
        },
        "transitions": transitions,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--threshold-ms", type=float, default=0.75)
    parser.add_argument("--maximum-relative-mad", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = summarize_gaps(
        records,
        warmup_per_transition=args.warmup,
        minimum_samples=args.minimum_samples,
        threshold_ms=args.threshold_ms,
        maximum_relative_mad=args.maximum_relative_mad,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
