#!/usr/bin/env python3
"""Search fixed SWOR trees using measured ordered-sibling acceptance rates."""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchState:
    parents: tuple[int, ...]
    current_level: tuple[tuple[int, float], ...]
    accepted_yield: float
    depth: int
    level_sizes: tuple[int, ...]

    @property
    def nodes(self) -> int:
        return len(self.parents)

    @property
    def outputs(self) -> float:
        return 1.0 + self.accepted_yield


def parse_rank_probs(values: list[str], width: int) -> list[tuple[float, ...]]:
    rows = []
    for depth, value in enumerate(values, start=1):
        try:
            row = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid depth-{depth} rank JSON: {exc.msg}") from exc
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"depth-{depth} rank row must contain {width} values")
        row = tuple(float(item) for item in row)
        if any(item < 0.0 for item in row) or sum(row) > 1.000001:
            raise ValueError(f"invalid probabilities at depth {depth}: {row}")
        if any(row[index] < row[index + 1] for index in range(width - 1)):
            raise ValueError(f"rank probabilities must be nonincreasing: {row}")
        rows.append(row)
    if not rows:
        raise ValueError("at least one --rank-probs row is required")
    return rows


def rank_probs_for_depth(
    rows: list[tuple[float, ...]], depth: int, tail_decay: float
) -> tuple[float, ...]:
    if depth <= len(rows):
        return rows[depth - 1]
    scale = tail_decay ** (depth - len(rows))
    return tuple(probability * scale for probability in rows[-1])


def add_level(
    state: SearchState,
    child_count: int,
    rank_probs: tuple[float, ...],
    width: int,
) -> SearchState:
    candidate_parents = sorted(
        state.current_level, key=lambda item: (-item[1], item[0])
    )[:width]
    max_children = width * len(candidate_parents)
    if not 1 <= child_count <= max_children:
        raise ValueError("child count is outside the fixed-frontier capacity")

    # Each parent's visible children must be an ordered prefix. The heap chooses
    # the highest-probability next child while exposing rank r+1 only after r.
    heap = []
    for parent_id, parent_probability in candidate_parents:
        probability = parent_probability * rank_probs[0]
        heapq.heappush(heap, (-probability, parent_id, 0, parent_probability))

    chosen = []
    for _ in range(child_count):
        negative_probability, parent_id, rank, parent_probability = heapq.heappop(
            heap
        )
        probability = -negative_probability
        chosen.append((parent_id, rank, probability))
        next_rank = rank + 1
        if next_rank < width:
            next_probability = parent_probability * rank_probs[next_rank]
            heapq.heappush(
                heap,
                (-next_probability, parent_id, next_rank, parent_probability),
            )

    # Node IDs are grouped by depth and then parent/rank, matching the runtime
    # topology validator. Probabilities retain their path identities.
    chosen.sort(key=lambda item: (item[0], item[1]))
    first_node = state.nodes
    parents = state.parents + tuple(parent for parent, _, _ in chosen)
    current_level = tuple(
        (first_node + offset, probability)
        for offset, (_, _, probability) in enumerate(chosen)
    )
    return SearchState(
        parents=parents,
        current_level=current_level,
        accepted_yield=state.accepted_yield
        + sum(probability for _, probability in current_level),
        depth=state.depth + 1,
        level_sizes=state.level_sizes + (len(current_level),),
    )


def beam_key(state: SearchState, remaining_nodes: int, width: int) -> float:
    # Score realized yield plus a deliberately loose future-value estimate.
    # This keeps narrow/deep candidates alive beside broad immediate-yield trees.
    frontier_mass = sum(
        probability
        for _, probability in sorted(
            state.current_level, key=lambda item: -item[1]
        )[:width]
    )
    return state.accepted_yield + min(remaining_nodes, width * width) * frontier_mass


def search(
    rows: list[tuple[float, ...]],
    *,
    width: int,
    max_nodes: int,
    max_depth: int,
    tail_decay: float,
    beam_per_node_count: int,
) -> list[SearchState]:
    initial = SearchState(
        parents=(-1,),
        current_level=((0, 1.0),),
        accepted_yield=0.0,
        depth=0,
        level_sizes=(),
    )
    frontier = [initial]
    finished = [initial]

    for depth in range(1, max_depth + 1):
        rank_probs = rank_probs_for_depth(rows, depth, tail_decay)
        generated = []
        for state in frontier:
            remaining = max_nodes - state.nodes
            parent_count = min(width, len(state.current_level))
            max_children = min(remaining, width * parent_count)
            for child_count in range(1, max_children + 1):
                generated.append(
                    add_level(state, child_count, rank_probs, width)
                )
        if not generated:
            break

        by_nodes: dict[int, list[SearchState]] = {}
        for state in generated:
            by_nodes.setdefault(state.nodes, []).append(state)
        frontier = []
        for node_count, states in by_nodes.items():
            remaining = max_nodes - node_count
            states.sort(
                key=lambda state: beam_key(state, remaining, width), reverse=True
            )
            frontier.extend(states[:beam_per_node_count])
        finished.extend(frontier)

    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank-probs",
        action="append",
        required=True,
        help="JSON ordered sibling probabilities for one successive depth",
    )
    parser.add_argument("--draft-width", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--tail-decay",
        type=float,
        default=0.97,
        help="Multiplicative decay per unmeasured depth after the final row",
    )
    parser.add_argument("--beam-per-node-count", type=int, default=256)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--cycle-fixed-ms",
        type=float,
        default=19.24,
        help="Measured fixed cycle-cost intercept",
    )
    parser.add_argument("--cycle-node-ms", type=float, default=0.27)
    parser.add_argument("--cycle-depth-ms", type=float, default=2.73)
    args = parser.parse_args()

    if args.draft_width < 1 or args.max_nodes < 2 or args.max_depth < 1:
        parser.error("width, node budget, and depth must be positive")
    if not 0.0 < args.tail_decay <= 1.0:
        parser.error("--tail-decay must be in (0, 1]")

    rows = parse_rank_probs(args.rank_probs, args.draft_width)
    states = search(
        rows,
        width=args.draft_width,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        tail_decay=args.tail_decay,
        beam_per_node_count=args.beam_per_node_count,
    )

    candidates = []
    seen = set()
    for state in states:
        if state.nodes < 2 or state.parents in seen:
            continue
        seen.add(state.parents)
        cycle_ms = (
            args.cycle_fixed_ms
            + args.cycle_node_ms * state.nodes
            + args.cycle_depth_ms * state.depth
        )
        predicted_tps = 1000.0 * state.outputs / cycle_ms
        candidates.append((predicted_tps, state.outputs, cycle_ms, state))
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))

    print(
        "model: cycle_ms="
        f"{args.cycle_fixed_ms:.3f}+{args.cycle_node_ms:.3f}*nodes+"
        f"{args.cycle_depth_ms:.3f}*depth; tail_decay={args.tail_decay:.4f}"
    )
    for rank, (predicted_tps, outputs, cycle_ms, state) in enumerate(
        candidates[: args.top], start=1
    ):
        print(
            f"rank={rank} nodes={state.nodes} depth={state.depth} "
            f"levels={list(state.level_sizes)} outputs={outputs:.6f} "
            f"cycle_ms={cycle_ms:.3f} predicted_tps={predicted_tps:.3f} "
            f"parents={json.dumps(state.parents, separators=(',', ':'))}"
        )


if __name__ == "__main__":
    main()
