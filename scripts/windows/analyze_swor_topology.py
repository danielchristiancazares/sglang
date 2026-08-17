#!/usr/bin/env python3
"""Analyze accepted-node histograms from fixed SWOR topology oracle runs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path


STATS_RE = re.compile(
    r"SWOR_ACCEPT_PATH_STATS .*?verify_ct=(\d+) node_histogram=(\[[0-9, ]*\])"
)


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON: {exc.msg}") from exc
    if not isinstance(parsed, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in parsed
    ):
        raise ValueError(f"{name} must be a JSON array of integers")
    return parsed


def read_log_stats(path: Path, skip: int) -> tuple[int, list[int], int]:
    samples = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := STATS_RE.search(line):
            samples.append((int(match.group(1)), ast.literal_eval(match.group(2))))
    samples = samples[skip:]
    if not samples:
        raise ValueError(f"no SWOR path-stat samples remain in {path}")
    width = max(len(histogram) for _, histogram in samples)
    aggregate = [0] * width
    verify_count = 0
    for sample_verify_count, histogram in samples:
        verify_count += sample_verify_count
        for node, count in enumerate(histogram):
            aggregate[node] += count
    return verify_count, aggregate, len(samples)


def analyze(parents: list[int], verify_count: int, histogram: list[int]) -> dict:
    if not parents or parents[0] != -1:
        raise ValueError("parents must start with root parent -1")
    if verify_count <= 0:
        raise ValueError("verify count must be positive")
    histogram = histogram + [0] * (len(parents) - len(histogram))
    if len(histogram) != len(parents):
        raise ValueError("node histogram is wider than the topology")

    depths = [0]
    children: dict[int, list[int]] = defaultdict(list)
    for node, parent in enumerate(parents[1:], start=1):
        if parent < 0 or parent >= node:
            raise ValueError(f"node {node} has invalid parent {parent}")
        depths.append(depths[parent] + 1)
        children[parent].append(node)

    rows = []
    depth_totals: dict[int, int] = defaultdict(int)
    for parent, child_nodes in children.items():
        reached = verify_count if parent == 0 else histogram[parent]
        accepted_group = sum(histogram[node] for node in child_nodes)
        for rank, node in enumerate(child_nodes):
            accepted = histogram[node]
            depth_totals[depths[node]] += accepted
            rows.append(
                {
                    "node": node,
                    "parent": parent,
                    "depth": depths[node],
                    "sibling_rank": rank,
                    "accepted": accepted,
                    "unconditional": accepted / verify_count,
                    "conditional_on_parent": accepted / reached if reached else 0.0,
                    "parent_reached": reached,
                    "sibling_group_continuation": (
                        accepted_group / reached if reached else 0.0
                    ),
                }
            )

    accepted_drafts = sum(histogram[1:])
    return {
        "verify_count": verify_count,
        "histogram": histogram,
        "accepted_drafts": accepted_drafts,
        "outputs_per_traversal": 1.0 + accepted_drafts / verify_count,
        "depth_accepted": dict(sorted(depth_totals.items())),
        "rows": rows,
    }


def print_report(report: dict) -> None:
    print(f"verify_count={report['verify_count']}")
    print(f"node_histogram={report['histogram']}")
    print(f"accepted_drafts={report['accepted_drafts']}")
    print(f"outputs_per_traversal={report['outputs_per_traversal']:.6f}")
    print("depth_accepted=" + json.dumps(report["depth_accepted"], separators=(",", ":")))
    print()
    print("node parent depth rank accepted parent_reached unconditional conditional group")
    for row in report["rows"]:
        print(
            f"{row['node']:4d} {row['parent']:6d} {row['depth']:5d} "
            f"{row['sibling_rank']:4d} {row['accepted']:8d} "
            f"{row['parent_reached']:14d} {row['unconditional']:13.6f} "
            f"{row['conditional_on_parent']:11.6f} "
            f"{row['sibling_group_continuation']:7.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", required=True, help="JSON parent-node array")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stats-log", type=Path)
    source.add_argument("--histogram", help="JSON accepted-node histogram")
    parser.add_argument("--verify-count", type=int)
    parser.add_argument(
        "--skip-log-samples",
        type=int,
        default=0,
        help="Ignore leading warmup path-stat records",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    parents = parse_int_list(args.parents, "parents")
    if args.stats_log is not None:
        verify_count, histogram, sample_count = read_log_stats(
            args.stats_log, args.skip_log_samples
        )
    else:
        if args.verify_count is None:
            parser.error("--verify-count is required with --histogram")
        verify_count = args.verify_count
        histogram = parse_int_list(args.histogram, "histogram")
        sample_count = 1

    report = analyze(parents, verify_count, histogram)
    report["sample_count"] = sample_count
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"sample_count={sample_count}")
        print_report(report)


if __name__ == "__main__":
    main()
