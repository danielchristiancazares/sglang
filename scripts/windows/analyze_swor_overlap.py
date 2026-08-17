"""Summarize the final SWOR proposal-overlap record in a server log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


RECORD = re.compile(
    r"SWOR_OVERLAP_STATS rid=(?P<rid>\S+) count=(?P<count>\d+) "
    r"scales=(?P<scales>\[[^]]*]) top_ks=(?P<top_ks>\[[^]]*]) "
    r"node_grid=(?P<grid>\[.*])$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--parents", required=True)
    return parser.parse_args()


def tree_depths(parents: list[int]) -> list[int]:
    depths = [0] * len(parents)
    for node, parent in enumerate(parents[1:], start=1):
        if not 0 <= parent < node:
            raise ValueError(f"parents[{node}]={parent} is invalid")
        depths[node] = depths[parent] + 1
    return depths


def main() -> None:
    args = parse_args()
    parents = json.loads(args.parents)
    if not isinstance(parents, list) or not parents or parents[0] != -1:
        raise ValueError("--parents must be a JSON parent array beginning with -1")

    matches = []
    for line in args.log.read_text(errors="replace").splitlines():
        match = RECORD.search(line)
        if match is not None:
            matches.append(match)
    if not matches:
        raise ValueError(f"no SWOR_OVERLAP_STATS record in {args.log}")

    match = matches[-1]
    scales = json.loads(match.group("scales"))
    top_ks = json.loads(match.group("top_ks"))
    grid = json.loads(match.group("grid"))
    if len(grid) != len(parents):
        raise ValueError(
            f"grid has {len(grid)} nodes but topology has {len(parents)}"
        )

    children = [[] for _ in parents]
    for node, parent in enumerate(parents[1:], start=1):
        children[parent].append(node)
    depths = tree_depths(parents)
    baseline_scale = min(range(len(scales)), key=lambda i: abs(scales[i] - 1.0))
    baseline_top_k = max(range(len(top_ks)), key=lambda i: top_ks[i])

    rows = []
    for node, node_children in enumerate(children):
        if not node_children:
            continue
        best = max(
            (
                (metrics[0], scale_index, top_k_index, metrics)
                for scale_index, by_top_k in enumerate(grid[node])
                for top_k_index, metrics in enumerate(by_top_k)
            ),
            key=lambda item: item[0],
        )
        baseline = grid[node][baseline_scale][baseline_top_k]
        rows.append(
            {
                "node": node,
                "depth": depths[node],
                "children": node_children,
                "baseline": {
                    "scale": scales[baseline_scale],
                    "top_k": top_ks[baseline_top_k],
                    "overlap": baseline[0],
                    "q_mass_outside_p": baseline[1],
                    "q_support": baseline[2],
                },
                "best": {
                    "scale": scales[best[1]],
                    "top_k": top_ks[best[2]],
                    "overlap": best[3][0],
                    "q_mass_outside_p": best[3][1],
                    "q_support": best[3][2],
                },
                "overlap_gain": best[3][0] - baseline[0],
            }
        )

    print(
        json.dumps(
            {
                "rid": match.group("rid"),
                "count": int(match.group("count")),
                "internal_nodes": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
