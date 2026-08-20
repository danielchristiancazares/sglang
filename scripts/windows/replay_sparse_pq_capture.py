#!/usr/bin/env python3
"""Replay deterministic tree policies over one immutable sparse p/q corpus.

Schema version 2 stores branch-exact, post-transform target ``p`` and draft
``q`` on a prefix trie.  Every policy sees the same cycles; only its selected
prefix-closed tree changes.  Full-cycle cost samples are joined by explicit ID
and runtime provenance, so the report never averages per-cycle TPS ratios.

The target-aware policy is intentionally impossible: it may inspect target p
while allocating nodes.  It is a family-rejection ceiling, never production
evidence.  An implementable policy is funded only when its conservative lower
projection reaches the configured funding floor (215 TPS by default) and
strictly clears the best-case TPS of the explicitly measured geometry
frontier.  Counterfactual geometries fail closed unless the proposal lattice
has complete target and draft support.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class CaptureInputError(ValueError):
    """Raised when a captured corpus cannot support auditable replay."""


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CaptureInputError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CaptureInputError(f"{name} must be a JSON array")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureInputError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureInputError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CaptureInputError(f"{name} must be a finite number")
    return result


def _probability(value: Any, name: str) -> float:
    result = _number(value, name)
    if not 0.0 <= result <= 1.0:
        raise CaptureInputError(f"{name} must be within [0, 1]")
    return result


def _name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureInputError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class CapturedState:
    state_id: int
    depth: int
    p_other_mass: float
    q_other_mass: float
    target_support_complete: bool
    draft_support_complete: bool


@dataclass(frozen=True)
class CapturedEdge:
    edge_id: int
    parent_id: int
    child_id: int
    token_id: int
    depth: int
    branch_rank: int
    p: float
    q: float
    current_score: float


@dataclass
class CapturedCycle:
    cycle_id: str
    request_id: str
    root_state_id: int
    max_depth: int
    states: dict[int, CapturedState]
    edges: dict[int, CapturedEdge]
    outgoing: dict[int, list[CapturedEdge]]
    incoming: dict[int, CapturedEdge]
    memberships: dict[str, tuple[int, ...]]

    def support_complete(self) -> bool:
        return all(
            state.target_support_complete and state.draft_support_complete
            for state in self.states.values()
        )


@dataclass(frozen=True)
class WidthCost:
    cost_id: str
    topology_family: str
    logical_width: int
    executed_graph_width: int
    max_depth: int
    samples_ms: tuple[float, ...]
    active_worker: str
    torch_compile_mode: str

    @property
    def point_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def lower_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def upper_ms(self) -> float:
        return max(self.samples_ms)


@dataclass(frozen=True)
class ReplayCorpus:
    vocab_size: int
    runtime: Mapping[str, Any]
    cycles: tuple[CapturedCycle, ...]
    costs: Mapping[str, WidthCost]
    policies: tuple[Mapping[str, Any], ...]
    target_tps: float
    funding_tps: float


def _parse_state(value: Any, index: int) -> CapturedState:
    source = _object(value, f"states[{index}]")
    state_id = _integer(source.get("state_id"), f"states[{index}].state_id")
    depth = _integer(source.get("depth"), f"states[{index}].depth")
    if state_id < 0 or depth < 0:
        raise CaptureInputError("state IDs and depths must be nonnegative")
    p_other = _probability(
        source.get("p_other_mass", 0.0), f"states[{index}].p_other_mass"
    )
    q_other = _probability(
        source.get("q_other_mass", 0.0), f"states[{index}].q_other_mass"
    )
    target_complete = source.get("target_support_complete", False)
    draft_complete = source.get("draft_support_complete", False)
    if not isinstance(target_complete, bool) or not isinstance(draft_complete, bool):
        raise CaptureInputError("support-complete fields must be booleans")
    if target_complete and p_other != 0.0:
        raise CaptureInputError("complete target support cannot have p_other_mass")
    if draft_complete and q_other != 0.0:
        raise CaptureInputError("complete draft support cannot have q_other_mass")
    return CapturedState(
        state_id=state_id,
        depth=depth,
        p_other_mass=p_other,
        q_other_mass=q_other,
        target_support_complete=target_complete,
        draft_support_complete=draft_complete,
    )


def _parse_edge(value: Any, index: int, vocab_size: int) -> CapturedEdge:
    source = _object(value, f"edges[{index}]")
    edge_id = _integer(source.get("edge_id"), f"edges[{index}].edge_id")
    parent_id = _integer(source.get("parent_id"), f"edges[{index}].parent_id")
    child_id = _integer(source.get("child_id"), f"edges[{index}].child_id")
    token_id = _integer(source.get("token_id"), f"edges[{index}].token_id")
    depth = _integer(source.get("depth"), f"edges[{index}].depth")
    branch_rank = _integer(
        source.get("branch_rank"), f"edges[{index}].branch_rank"
    )
    if min(edge_id, parent_id, child_id, depth, branch_rank) < 0:
        raise CaptureInputError("edge IDs, state IDs, depths, and ranks are nonnegative")
    if not 0 <= token_id < vocab_size:
        raise CaptureInputError(
            f"edges[{index}].token_id must be within [0, {vocab_size})"
        )
    q = _probability(source.get("q"), f"edges[{index}].q")
    return CapturedEdge(
        edge_id=edge_id,
        parent_id=parent_id,
        child_id=child_id,
        token_id=token_id,
        depth=depth,
        branch_rank=branch_rank,
        p=_probability(source.get("p"), f"edges[{index}].p"),
        q=q,
        current_score=_probability(
            source.get("current_score", q), f"edges[{index}].current_score"
        ),
    )


def _parse_cycle(value: Any, index: int, vocab_size: int) -> CapturedCycle:
    source = _object(value, f"cycles[{index}]")
    cycle_id = str(source.get("cycle_id", index))
    request_id = str(source.get("request_id", "unknown"))
    root_state_id = _integer(
        source.get("root_state_id"), f"cycles[{index}].root_state_id"
    )
    max_depth = _integer(source.get("max_depth"), f"cycles[{index}].max_depth")
    if max_depth < 1:
        raise CaptureInputError("cycles[].max_depth must be positive")

    states: dict[int, CapturedState] = {}
    for state_index, raw_state in enumerate(_list(source.get("states"), "states")):
        state = _parse_state(raw_state, state_index)
        if state.state_id in states:
            raise CaptureInputError(f"duplicate state_id {state.state_id}")
        states[state.state_id] = state
    if root_state_id not in states or states[root_state_id].depth != 0:
        raise CaptureInputError("root_state_id must identify the unique depth-zero state")
    if sum(state.depth == 0 for state in states.values()) != 1:
        raise CaptureInputError("each cycle must have exactly one depth-zero state")

    edges: dict[int, CapturedEdge] = {}
    outgoing: dict[int, list[CapturedEdge]] = {state_id: [] for state_id in states}
    incoming: dict[int, CapturedEdge] = {}
    per_parent_tokens: set[tuple[int, int]] = set()
    per_parent_ranks: set[tuple[int, int]] = set()
    for edge_index, raw_edge in enumerate(_list(source.get("edges"), "edges")):
        edge = _parse_edge(raw_edge, edge_index, vocab_size)
        if edge.edge_id in edges:
            raise CaptureInputError(f"duplicate edge_id {edge.edge_id}")
        if edge.parent_id not in states or edge.child_id not in states:
            raise CaptureInputError(f"edge {edge.edge_id} references an unknown state")
        parent = states[edge.parent_id]
        child = states[edge.child_id]
        if edge.child_id == root_state_id or edge.child_id in incoming:
            raise CaptureInputError("every non-root state must have one incoming edge")
        if edge.depth != child.depth or child.depth != parent.depth + 1:
            raise CaptureInputError(f"edge {edge.edge_id} has inconsistent depth")
        if edge.depth > max_depth:
            raise CaptureInputError(f"edge {edge.edge_id} exceeds cycle max_depth")
        token_key = (edge.parent_id, edge.token_id)
        rank_key = (edge.parent_id, edge.branch_rank)
        if token_key in per_parent_tokens or rank_key in per_parent_ranks:
            raise CaptureInputError("child tokens and branch ranks must be unique per parent")
        per_parent_tokens.add(token_key)
        per_parent_ranks.add(rank_key)
        edges[edge.edge_id] = edge
        outgoing[edge.parent_id].append(edge)
        incoming[edge.child_id] = edge
    if set(incoming) != set(states) - {root_state_id}:
        raise CaptureInputError("states must form one connected prefix trie")

    tolerance = 2e-6
    for state_id, state_outgoing in outgoing.items():
        state_outgoing.sort(
            key=lambda edge: (edge.branch_rank, edge.token_id, edge.edge_id)
        )
        state = states[state_id]
        p_total = math.fsum(edge.p for edge in state_outgoing) + state.p_other_mass
        q_total = math.fsum(edge.q for edge in state_outgoing) + state.q_other_mass
        if state_outgoing and not math.isclose(p_total, 1.0, abs_tol=tolerance):
            raise CaptureInputError(
                f"state {state_id} target mass sums to {p_total}, expected 1"
            )
        if state_outgoing and not math.isclose(q_total, 1.0, abs_tol=tolerance):
            raise CaptureInputError(
                f"state {state_id} draft mass sums to {q_total}, expected 1"
            )

    memberships: dict[str, tuple[int, ...]] = {}
    raw_memberships = _object(source.get("topology_memberships", {}), "memberships")
    for membership_name, raw_edge_ids in raw_memberships.items():
        name = _name(membership_name, "membership name")
        edge_ids = tuple(
            _integer(edge_id, f"membership {name}")
            for edge_id in _list(raw_edge_ids, f"membership {name}")
        )
        if len(edge_ids) != len(set(edge_ids)) or any(
            edge_id not in edges for edge_id in edge_ids
        ):
            raise CaptureInputError(f"membership {name} has duplicate/unknown edges")
        selected = set(edge_ids)
        for edge_id in edge_ids:
            parent_id = edges[edge_id].parent_id
            parent_edge = incoming.get(parent_id)
            if parent_edge is not None and parent_edge.edge_id not in selected:
                raise CaptureInputError(f"membership {name} is not prefix-closed")
        memberships[name] = edge_ids
    return CapturedCycle(
        cycle_id=cycle_id,
        request_id=request_id,
        root_state_id=root_state_id,
        max_depth=max_depth,
        states=states,
        edges=edges,
        outgoing=outgoing,
        incoming=incoming,
        memberships=memberships,
    )


def _parse_cost(value: Any, index: int, runtime: Mapping[str, Any]) -> WidthCost:
    source = _object(value, f"width_costs[{index}]")
    if source.get("scope") != "full_cycle":
        raise CaptureInputError("width costs used for gates must have scope=full_cycle")
    samples = tuple(
        _number(sample, f"width_costs[{index}].samples_ms")
        for sample in _list(source.get("samples_ms"), "samples_ms")
    )
    if not samples or any(sample <= 0.0 for sample in samples):
        raise CaptureInputError("full-cycle cost samples must be positive")
    worker = _name(source.get("active_worker"), "width cost active_worker")
    compile_mode = _name(
        source.get("torch_compile_mode"), "width cost torch_compile_mode"
    )
    if worker != runtime["active_worker"] or compile_mode != runtime["torch_compile_mode"]:
        raise CaptureInputError("width-cost runtime provenance does not match corpus")
    return WidthCost(
        cost_id=_name(source.get("cost_id"), "width cost ID"),
        topology_family=_name(
            source.get("topology_family"), "width cost topology_family"
        ),
        logical_width=_integer(source.get("logical_width"), "logical_width"),
        executed_graph_width=_integer(
            source.get("executed_graph_width"), "executed_graph_width"
        ),
        max_depth=_integer(source.get("max_depth"), "cost max_depth"),
        samples_ms=samples,
        active_worker=worker,
        torch_compile_mode=compile_mode,
    )


def parse_corpus(document: Any) -> ReplayCorpus:
    source = _object(document, "document")
    if source.get("schema_version") != 2:
        raise CaptureInputError("schema_version must be exactly 2")
    if source.get("artifact_type") != "sparse_pq_capture":
        raise CaptureInputError("artifact_type must be sparse_pq_capture")
    vocab_size = _integer(source.get("vocab_size"), "vocab_size")
    if vocab_size <= 0:
        raise CaptureInputError("vocab_size must be positive")
    runtime = _object(source.get("runtime"), "runtime")
    for field in (
        "requested_worker",
        "active_worker",
        "torch_compile_mode",
        "git_head",
    ):
        _name(runtime.get(field), f"runtime.{field}")
    if not isinstance(runtime.get("torch_compile_enabled"), bool):
        raise CaptureInputError("runtime.torch_compile_enabled must be boolean")
    expected_mode = (
        runtime["torch_compile_mode"]
        if runtime["torch_compile_enabled"]
        else "disabled"
    )
    if runtime["torch_compile_mode"] != expected_mode:
        raise CaptureInputError("disabled torch compile must record mode=disabled")

    sampling = _object(source.get("sampling"), "sampling")
    penalties = _object(sampling.get("penalties"), "sampling.penalties")
    repetition = _number(
        penalties.get("repetition", 1.0), "sampling.penalties.repetition"
    )
    if not 0.0 < repetition <= 2.0:
        raise CaptureInputError("sampling.penalties.repetition must be within (0, 2]")
    order = _list(sampling.get("transform_order"), "sampling.transform_order")
    if not order or any(not isinstance(stage, str) or not stage for stage in order):
        raise CaptureInputError("sampling.transform_order must name every stage")

    cycles = tuple(
        _parse_cycle(raw_cycle, index, vocab_size)
        for index, raw_cycle in enumerate(_list(source.get("cycles"), "cycles"))
    )
    if not cycles:
        raise CaptureInputError("cycles must be nonempty")
    costs: dict[str, WidthCost] = {}
    for index, raw_cost in enumerate(
        _list(source.get("width_costs"), "width_costs")
    ):
        cost = _parse_cost(raw_cost, index, runtime)
        if cost.cost_id in costs:
            raise CaptureInputError(f"duplicate cost_id {cost.cost_id}")
        costs[cost.cost_id] = cost
    if not costs:
        raise CaptureInputError("width_costs must be nonempty")

    policies = tuple(
        _object(policy, f"policies[{index}]")
        for index, policy in enumerate(_list(source.get("policies"), "policies"))
    )
    if not policies:
        raise CaptureInputError("policies must be nonempty")
    policy_names = [_name(policy.get("name"), "policy name") for policy in policies]
    if len(policy_names) != len(set(policy_names)):
        raise CaptureInputError("policy names must be unique")

    gate = _object(source.get("gate", {}), "gate")
    target_tps = _number(gate.get("target_tps", 200.0), "gate.target_tps")
    funding_tps = _number(gate.get("funding_tps", 215.0), "gate.funding_tps")
    if target_tps <= 0.0 or funding_tps <= target_tps:
        raise CaptureInputError("funding_tps must be greater than positive target_tps")
    return ReplayCorpus(
        vocab_size=vocab_size,
        runtime=runtime,
        cycles=cycles,
        costs=costs,
        policies=policies,
        target_tps=target_tps,
        funding_tps=funding_tps,
    )


def _path_probability(
    cycle: CapturedCycle, selected: set[int], field: str
) -> dict[int, float]:
    result: dict[int, float] = {}
    for edge in sorted(cycle.edges.values(), key=lambda item: (item.depth, item.edge_id)):
        if edge.edge_id not in selected:
            continue
        parent_edge = cycle.incoming.get(edge.parent_id)
        parent_probability = (
            1.0
            if parent_edge is None
            else result.get(parent_edge.edge_id, 0.0)
        )
        result[edge.edge_id] = parent_probability * getattr(edge, field)
    return result


def expected_target_only(cycle: CapturedCycle, edge_ids: Sequence[int]) -> float:
    selected = set(edge_ids)
    for edge_id in selected:
        parent_edge = cycle.incoming.get(cycle.edges[edge_id].parent_id)
        if parent_edge is not None and parent_edge.edge_id not in selected:
            raise CaptureInputError("selected replay tree is not prefix-closed")
    return 1.0 + math.fsum(_path_probability(cycle, selected, "p").values())


def _temperature_adjusted(
    cycle: CapturedCycle,
    field: str,
    temperatures: Mapping[int, float],
) -> dict[int, float]:
    adjusted: dict[int, float] = {}
    for parent_id, edges in cycle.outgoing.items():
        if not edges:
            continue
        depth = edges[0].depth
        temperature = float(temperatures.get(depth, 1.0))
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise CaptureInputError("calibration temperatures must be positive")
        state = cycle.states[parent_id]
        if temperature != 1.0 and (
            not state.draft_support_complete or state.q_other_mass != 0.0
        ):
            raise CaptureInputError(
                "q calibration requires complete captured draft support"
            )
        weights = {
            edge.edge_id: getattr(edge, field) ** (1.0 / temperature)
            for edge in edges
            if getattr(edge, field) > 0.0
        }
        total = math.fsum(weights.values())
        if total <= 0.0:
            continue
        adjusted.update({edge_id: value / total for edge_id, value in weights.items()})
    return adjusted


def select_frontier_tree(
    cycle: CapturedCycle,
    *,
    budget: int,
    field: str,
    temperatures: Mapping[int, float] | None = None,
    max_fanout: int | None = None,
    confidence_power: float = 0.0,
) -> tuple[int, ...]:
    if budget < 0:
        raise CaptureInputError("tree budget must be nonnegative")
    temperatures = temperatures or {}
    conditional = _temperature_adjusted(cycle, field, temperatures)
    parent_path_score = {cycle.root_state_id: 1.0}
    selected: list[int] = []
    heap: list[tuple[float, int, int, int, int]] = []

    def push_children(parent_id: int) -> None:
        edges = list(cycle.outgoing[parent_id])
        edges.sort(
            key=lambda edge: (
                -conditional.get(edge.edge_id, 0.0),
                edge.branch_rank,
                edge.token_id,
            )
        )
        if max_fanout is not None:
            edges = edges[:max_fanout]
        confidence = max(
            (conditional.get(edge.edge_id, 0.0) for edge in edges), default=0.0
        )
        for edge in edges:
            score = (
                parent_path_score[parent_id]
                * conditional.get(edge.edge_id, 0.0)
                * (confidence**confidence_power if confidence_power else 1.0)
            )
            heapq.heappush(
                heap,
                (-score, edge.depth, edge.branch_rank, edge.token_id, edge.edge_id),
            )

    push_children(cycle.root_state_id)
    while heap and len(selected) < budget:
        negative_score, _depth, _rank, _token, edge_id = heapq.heappop(heap)
        edge = cycle.edges[edge_id]
        parent_edge = cycle.incoming.get(edge.parent_id)
        if parent_edge is not None and parent_edge.edge_id not in selected:
            continue
        selected.append(edge_id)
        parent_path_score[edge.child_id] = -negative_score
        push_children(edge.child_id)
    return tuple(selected)


def select_confidence_chain(
    cycle: CapturedCycle,
    *,
    short_depth: int,
    long_depth: int,
    threshold: float,
) -> tuple[tuple[int, ...], int]:
    root_edges = sorted(
        cycle.outgoing[cycle.root_state_id],
        key=lambda edge: (-edge.q, edge.branch_rank, edge.token_id),
    )
    if not root_edges:
        return (), short_depth
    confidence = root_edges[0].q
    selected_depth = long_depth if confidence >= threshold else short_depth
    selected: list[int] = []
    state_id = cycle.root_state_id
    for _depth in range(selected_depth):
        edges = sorted(
            cycle.outgoing[state_id],
            key=lambda edge: (-edge.q, edge.branch_rank, edge.token_id),
        )
        if not edges:
            break
        edge = edges[0]
        selected.append(edge.edge_id)
        state_id = edge.child_id
    return tuple(selected), selected_depth


def _normalized(values: Mapping[int, float], name: str) -> dict[int, float]:
    positive = {token: value for token, value in values.items() if value > 0.0}
    total = math.fsum(positive.values())
    if total <= 0.0:
        raise CaptureInputError(f"{name} has no positive mass")
    return {token: value / total for token, value in positive.items()}


def expected_integrated_swor(
    cycle: CapturedCycle, parents: Sequence[int], vocab_size: int
) -> float:
    """Integrate SWOR proposal draws and verifier coins on complete support."""
    topology = tuple(_integer(parent, "SWOR parent") for parent in parents)
    if len(topology) < 2 or topology[0] != -1:
        raise CaptureInputError("SWOR topology must start with root parent -1")
    children: list[list[int]] = [[] for _ in topology]
    for slot, parent in enumerate(topology[1:], start=1):
        if not 0 <= parent < slot:
            raise CaptureInputError("SWOR parents must precede children")
        children[parent].append(slot)
    if not cycle.support_complete():
        raise CaptureInputError("exact integrated SWOR requires complete p/q support")

    def distributions(state_id: int) -> tuple[dict[int, float], dict[int, float]]:
        edges = cycle.outgoing[state_id]
        p = {edge.token_id: edge.p for edge in edges if edge.p > 0.0}
        q = {edge.token_id: edge.q for edge in edges if edge.q > 0.0}
        if not p:
            raise CaptureInputError("SWOR internal state is missing target support")
        return _normalized(p, "target p"), _normalized(q, "draft q") if q else {}

    state_by_token = {
        parent_id: {edge.token_id: edge.child_id for edge in edges}
        for parent_id, edges in cycle.outgoing.items()
    }

    def eval_slot(slot: int, state_id: int) -> float:
        if not children[slot]:
            return 0.0
        target, draft = distributions(state_id)
        return eval_rank(slot, state_id, 0, target, draft, 0)

    def residual(
        target: Mapping[int, float], proposal: Mapping[int, float]
    ) -> dict[int, float]:
        values = {
            token: max(probability - proposal.get(token, 0.0), 0.0)
            for token, probability in target.items()
        }
        return _normalized(values, "SWOR rejection residual")

    def eval_rank(
        slot: int,
        state_id: int,
        rank: int,
        target: Mapping[int, float],
        draft_remaining: Mapping[int, float],
        rejected_count: int,
    ) -> float:
        child_slots = children[slot]
        if rank >= len(child_slots):
            return 0.0
        child_slot = child_slots[rank]
        total = 0.0
        if draft_remaining:
            proposal = _normalized(draft_remaining, "remaining SWOR q")
            next_residual = None
            if rank + 1 < len(child_slots):
                rejected_mass = math.fsum(
                    max(q_value - target.get(token, 0.0), 0.0)
                    for token, q_value in proposal.items()
                )
                if rejected_mass > 0.0:
                    next_residual = residual(target, proposal)
            for token, q_value in proposal.items():
                accepted = min(q_value, target.get(token, 0.0))
                if accepted > 0.0:
                    child_state = state_by_token.get(state_id, {}).get(token)
                    nested = 0.0
                    if child_state is not None:
                        nested = eval_slot(child_slot, child_state)
                    elif children[child_slot]:
                        raise CaptureInputError(
                            "SWOR corpus lacks a reachable accepted branch state"
                        )
                    total += accepted * (1.0 + nested)
                rejected = max(q_value - target.get(token, 0.0), 0.0)
                if rejected > 0.0 and rank + 1 < len(child_slots):
                    assert next_residual is not None
                    next_q = dict(draft_remaining)
                    del next_q[token]
                    total += rejected * eval_rank(
                        slot,
                        state_id,
                        rank + 1,
                        next_residual,
                        next_q,
                        rejected_count + 1,
                    )
            return total

        remaining_vocab = vocab_size - rejected_count
        if remaining_vocab <= 0:
            return 0.0
        uniform = 1.0 / remaining_vocab
        for token, p_value in target.items():
            accepted = min(uniform, p_value)
            if accepted > 0.0:
                child_state = state_by_token.get(state_id, {}).get(token)
                nested = eval_slot(child_slot, child_state) if child_state is not None else 0.0
                total += accepted * (1.0 + nested)
        if rank + 1 < len(child_slots):
            uniform_on_target = {token: uniform for token in target}
            rejected_mass = 1.0 - math.fsum(
                min(uniform, p_value) for p_value in target.values()
            )
            if rejected_mass > 0.0:
                total += rejected_mass * eval_rank(
                    slot,
                    state_id,
                    rank + 1,
                    residual(target, uniform_on_target),
                    {},
                    rejected_count + 1,
                )
        return total

    return 1.0 + eval_slot(0, cycle.root_state_id)


def _cost_summary(cost: WidthCost) -> dict[str, Any]:
    return {
        "cost_id": cost.cost_id,
        "logical_width": cost.logical_width,
        "executed_graph_width": cost.executed_graph_width,
        "max_depth": cost.max_depth,
        "samples_ms": list(cost.samples_ms),
        "point_ms": cost.point_ms,
        "lower_ms": cost.lower_ms,
        "upper_ms": cost.upper_ms,
    }


def _policy_selection(
    corpus: ReplayCorpus,
    cycle: CapturedCycle,
    policy: Mapping[str, Any],
) -> tuple[tuple[int, ...] | None, float, WidthCost, bool]:
    kind = _name(policy.get("kind"), "policy kind")
    oracle_only = bool(policy.get("oracle_only", False))
    cost_id = policy.get("cost_id")
    if kind == "current_deterministic":
        membership = _name(policy.get("membership"), "current membership")
        if membership not in cycle.memberships:
            raise CaptureInputError(f"cycle lacks membership {membership}")
        edges = cycle.memberships[membership]
    elif kind in {
        "aligned_deterministic",
        "irregular_variable_fanout",
        "scalar_depth_calibration",
        "learned_depth_calibration",
        "target_aware_upper_bound",
    }:
        width = _integer(policy.get("logical_width"), "policy logical_width")
        if width < 1:
            raise CaptureInputError("logical_width must be positive")
        temperatures: dict[int, float] = {}
        if kind == "scalar_depth_calibration":
            value = _number(policy.get("temperature"), "policy temperature")
            temperatures = {depth: value for depth in range(1, cycle.max_depth + 1)}
        elif kind == "learned_depth_calibration":
            temperatures = {
                int(depth): _number(value, f"temperature_by_depth.{depth}")
                for depth, value in _object(
                    policy.get("temperature_by_depth"), "temperature_by_depth"
                ).items()
            }
        edges = select_frontier_tree(
            cycle,
            budget=width - 1,
            field="p" if kind == "target_aware_upper_bound" else "q",
            temperatures=temperatures,
            max_fanout=(
                _integer(policy.get("max_fanout"), "max_fanout")
                if policy.get("max_fanout") is not None
                else None
            ),
            confidence_power=(
                _number(policy.get("confidence_power", 1.0), "confidence_power")
                if kind == "irregular_variable_fanout"
                else 0.0
            ),
        )
    elif kind == "confidence_gated_chain":
        edges, selected_depth = select_confidence_chain(
            cycle,
            short_depth=_integer(policy.get("short_depth", 2), "short_depth"),
            long_depth=_integer(policy.get("long_depth", 3), "long_depth"),
            threshold=_probability(policy.get("threshold"), "confidence threshold"),
        )
        cost_map = _object(policy.get("cost_by_depth"), "cost_by_depth")
        cost_id = cost_map.get(str(selected_depth))
    elif kind == "swor":
        parents = _list(policy.get("parents"), "SWOR parents")
        expected = expected_integrated_swor(cycle, parents, corpus.vocab_size)
        if cost_id not in corpus.costs:
            raise CaptureInputError(f"unknown cost_id {cost_id!r}")
        return None, expected, corpus.costs[str(cost_id)], oracle_only
    else:
        raise CaptureInputError(f"unsupported policy kind {kind}")

    if cost_id not in corpus.costs:
        raise CaptureInputError(f"unknown cost_id {cost_id!r}")
    expected = expected_target_only(cycle, edges)
    return tuple(edges), expected, corpus.costs[str(cost_id)], oracle_only


def _project_policy(
    corpus: ReplayCorpus, policy: Mapping[str, Any]
) -> dict[str, Any]:
    kind = _name(policy.get("kind"), "policy kind")
    oracle_only = bool(policy.get("oracle_only", False))
    measured_frontier_baseline = bool(
        policy.get("measured_frontier_baseline", False)
    )
    if measured_frontier_baseline and kind != "current_deterministic":
        raise CaptureInputError(
            "only a captured current_deterministic membership can define the "
            "measured geometry frontier"
        )
    if measured_frontier_baseline and oracle_only:
        raise CaptureInputError("an oracle cannot define the measured frontier")

    cycle_reports = []
    total_expected = 0.0
    total_point_ms = 0.0
    total_lower_ms = 0.0
    total_upper_ms = 0.0
    incomplete = False
    max_emitted = 0.0
    for cycle in corpus.cycles:
        # The current membership is an observation of nodes that were actually
        # proposed and can be replayed from selected-tree capture.  Every other
        # policy is counterfactual: choosing a node absent from the immutable
        # lattice would silently invent p/q mass, ancestry, or branch state.
        if kind != "current_deterministic" and not cycle.support_complete():
            raise CaptureInputError(
                f"policy {kind} requires complete proposal-lattice p/q coverage "
                f"for cycle {cycle.cycle_id}"
            )
        edges, expected, cost, selected_oracle = _policy_selection(
            corpus, cycle, policy
        )
        oracle_only = oracle_only or selected_oracle
        incomplete = incomplete or not cycle.support_complete()
        max_emitted += cost.max_depth + 1.0
        total_expected += expected
        total_point_ms += cost.point_ms
        total_lower_ms += cost.lower_ms
        total_upper_ms += cost.upper_ms
        cycle_reports.append(
            {
                "cycle_id": cycle.cycle_id,
                "request_id": cycle.request_id,
                "selected_edge_ids": list(edges) if edges is not None else None,
                "expected_emitted_tokens": expected,
                "cost_id": cost.cost_id,
            }
        )

    point_tps = 1000.0 * total_expected / total_point_ms
    lower_tps = 1000.0 * total_expected / total_upper_ms
    upper_expected = max_emitted if oracle_only and incomplete else total_expected
    upper_tps = 1000.0 * upper_expected / total_lower_ms
    return {
        "name": _name(policy.get("name"), "policy name"),
        "kind": kind,
        "status": "ready",
        "oracle_only": oracle_only,
        "measured_frontier_baseline": measured_frontier_baseline,
        "geometry_candidate": not oracle_only and not measured_frontier_baseline,
        "support_complete": not incomplete,
        "cycles": cycle_reports,
        "aggregate": {
            "sum_expected_emitted_tokens": total_expected,
            "sum_point_cycle_ms": total_point_ms,
            "sum_lower_cycle_ms": total_lower_ms,
            "sum_upper_cycle_ms": total_upper_ms,
            "projected_tps": {
                "lower": lower_tps,
                "point": point_tps,
                "upper": upper_tps,
            },
        },
    }


def replay_corpus(corpus: ReplayCorpus, corpus_sha256: str | None = None) -> dict[str, Any]:
    policies = []
    for policy in corpus.policies:
        try:
            policies.append(_project_policy(corpus, policy))
        except CaptureInputError as exc:
            policies.append(
                {
                    "name": _name(policy.get("name"), "policy name"),
                    "kind": _name(policy.get("kind"), "policy kind"),
                    "status": "unavailable",
                    "oracle_only": bool(policy.get("oracle_only", False)),
                    "measured_frontier_baseline": bool(
                        policy.get("measured_frontier_baseline", False)
                    ),
                    "geometry_candidate": not bool(policy.get("oracle_only", False))
                    and not bool(policy.get("measured_frontier_baseline", False)),
                    "reason": str(exc),
                }
            )
    ready_policies = [policy for policy in policies if policy["status"] == "ready"]
    frontier_baselines = [
        policy
        for policy in ready_policies
        if policy["measured_frontier_baseline"]
    ]
    frontier_lower = (
        max(
            policy["aggregate"]["projected_tps"]["lower"]
            for policy in frontier_baselines
        )
        if frontier_baselines
        else None
    )
    frontier_point = (
        max(
            policy["aggregate"]["projected_tps"]["point"]
            for policy in frontier_baselines
        )
        if frontier_baselines
        else None
    )
    frontier_upper = (
        max(
            policy["aggregate"]["projected_tps"]["upper"]
            for policy in frontier_baselines
        )
        if frontier_baselines
        else None
    )
    for policy in policies:
        if not policy["geometry_candidate"]:
            policy["measured_frontier_comparison"] = {
                "required": False,
                "available": frontier_upper is not None,
            }
            continue
        candidate_lower = (
            policy["aggregate"]["projected_tps"]["lower"]
            if policy["status"] == "ready"
            else None
        )
        clears = (
            candidate_lower > frontier_upper
            if candidate_lower is not None and frontier_upper is not None
            else False
        )
        policy["measured_frontier_comparison"] = {
            "required": True,
            "available": frontier_upper is not None,
            "candidate_conservative_lower_tps": candidate_lower,
            "measured_frontier_best_case_upper_tps": frontier_upper,
            "headroom_tps": (
                candidate_lower - frontier_upper
                if candidate_lower is not None and frontier_upper is not None
                else None
            ),
            "clears": clears,
        }
    oracle_upper = [
        policy["aggregate"]["projected_tps"]["upper"]
        for policy in ready_policies
        if policy["oracle_only"]
    ]
    implementable_lower = [
        policy["aggregate"]["projected_tps"]["lower"]
        for policy in ready_policies
        if policy["geometry_candidate"]
    ]
    frontier_clearing_lower = [
        policy["aggregate"]["projected_tps"]["lower"]
        for policy in ready_policies
        if policy["geometry_candidate"]
        and policy["support_complete"]
        and policy["measured_frontier_comparison"]["clears"]
    ]
    max_oracle_upper = max(oracle_upper) if oracle_upper else None
    max_implementable_lower = (
        max(implementable_lower) if implementable_lower else None
    )
    max_frontier_clearing_lower = (
        max(frontier_clearing_lower) if frontier_clearing_lower else None
    )
    costs = {cost_id: _cost_summary(cost) for cost_id, cost in corpus.costs.items()}
    hard_ceilings = []
    for cost in corpus.costs.values():
        maximum_emitted = cost.max_depth + 1.0
        hard_ceilings.append(
            {
                "cost_id": cost.cost_id,
                "maximum_emitted_tokens": maximum_emitted,
                "point_tps": 1000.0 * maximum_emitted / cost.point_ms,
                "upper_tps": 1000.0 * maximum_emitted / cost.lower_ms,
                "cost_required_below_ms_for_target": (
                    1000.0 * maximum_emitted / corpus.target_tps
                ),
                "cost_required_at_or_below_ms_for_funding": (
                    1000.0 * maximum_emitted / corpus.funding_tps
                ),
            }
        )
    return {
        "schema_version": 2,
        "artifact_type": "sparse_pq_replay",
        "corpus_sha256": corpus_sha256,
        "runtime": dict(corpus.runtime),
        "costs": costs,
        "hard_ceilings": hard_ceilings,
        "policies": policies,
        "measured_geometry_frontier": {
            "available": frontier_upper is not None,
            "baseline_policy_names": [
                policy["name"] for policy in frontier_baselines
            ],
            "projected_tps": {
                "lower": frontier_lower,
                "point": frontier_point,
                "upper": frontier_upper,
            },
            "comparison_rule": (
                "candidate conservative lower TPS must be strictly greater than "
                "the measured frontier best-case upper TPS"
            ),
        },
        "gate": {
            "target_tps": corpus.target_tps,
            "funding_tps": corpus.funding_tps,
            "max_target_aware_oracle_upper_tps": max_oracle_upper,
            "max_implementable_lower_tps": max_implementable_lower,
            "max_frontier_clearing_implementable_lower_tps": (
                max_frontier_clearing_lower
            ),
            "reject_family": (
                max_oracle_upper <= corpus.target_tps
                if max_oracle_upper is not None
                else None
            ),
            "fund_production_implementation": (
                max_frontier_clearing_lower >= corpus.funding_tps
                if max_frontier_clearing_lower is not None
                else None
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="schema-v2 sparse p/q corpus")
    parser.add_argument("--output", type=Path, help="write deterministic JSON report")
    parser.add_argument("--require-funding-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(raw)
        corpus = parse_corpus(document)
        report = replay_corpus(corpus, hashlib.sha256(raw).hexdigest())
    except (OSError, json.JSONDecodeError, CaptureInputError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")
    if args.require_funding_gate and not report["gate"][
        "fund_production_implementation"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
