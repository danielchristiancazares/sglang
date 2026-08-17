#!/usr/bin/env python3
"""Evaluate an exact sparse p/q SWOR tree with branch-local penalties.

The input is a JSON document containing a fixed parent topology and sparse,
unpenalized target/draft logits for each accepted token suffix.  Omitted token
IDs have exactly zero probability.  The oracle integrates both proposal draws
and verifier coins; it does not condition on one realized proposal tree.

Penalty state is rebuilt independently for every accepted branch from
``initial_token_counts + root_token + accepted_suffix``.  Rejected siblings
never enter another branch's history.

Minimal schema::

    {
      "schema_version": 1,
      "vocab_size": 4,
      "parents": [-1, 0, 0],
      "root_token": 3,
      "initial_token_counts": {"3": 2},
      "penalties": {"presence": 1.5, "frequency": 0.0},
      "target_sampling": {"temperature": 1.0, "top_k": 4, "top_p": 0.95},
      "draft_sampling": {"temperature": 1.0, "top_k": 4, "top_p": 0.95},
      "rows": [
        {
          "suffix": [],
          "target_logits": {"0": 0.0, "1": -1.0},
          "draft_logits": {"0": -1.0, "1": 0.0}
        }
      ],
      "cycle_cost_ms": {"target_graph": 16.0, "draft_graphs": 4.0},
      "gate": {"target_tps": 200.0, "margin_tps": 10.0}
    }

Rows for leaf suffixes may be omitted.  Every reachable internal suffix needs
one row.  The promotion gate is deliberately strict: projected throughput
must be greater than ``target_tps + margin_tps``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


class OracleInputError(ValueError):
    """Raised when an oracle document cannot represent exact semantics."""


@dataclass(frozen=True)
class SamplingTransform:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0


@dataclass(frozen=True)
class AdditivePenalties:
    presence: float = 0.0
    frequency: float = 0.0


@dataclass(frozen=True)
class SparseLogitRow:
    target_logits: Mapping[int, float]
    draft_logits: Mapping[int, float]


@dataclass(frozen=True)
class PromotionGate:
    target_tps: float
    margin_tps: float


@dataclass(frozen=True)
class OracleInput:
    vocab_size: int
    parents: tuple[int, ...]
    root_token: int
    initial_token_counts: Mapping[int, int]
    penalties: AdditivePenalties
    target_sampling: SamplingTransform
    draft_sampling: SamplingTransform
    rows: Mapping[tuple[int, ...], SparseLogitRow]
    cycle_cost_ms: Mapping[str, float]
    gate: PromotionGate


@dataclass
class _Evaluation:
    expected_accepted: float
    accepted_by_node: dict[int, float]

    @classmethod
    def zero(cls) -> _Evaluation:
        return cls(0.0, {})

    def add_scaled(self, other: _Evaluation, scale: float) -> None:
        if scale <= 0.0:
            return
        self.expected_accepted += scale * other.expected_accepted
        for node, probability in other.accepted_by_node.items():
            self.accepted_by_node[node] = (
                self.accepted_by_node.get(node, 0.0) + scale * probability
            )


def _require_object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise OracleInputError(f"{name} must be a JSON object")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleInputError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise OracleInputError(f"{name} must be a finite number")
    return result


def _plain_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OracleInputError(f"{name} must be an integer")
    return value


def _token(value: Any, name: str, vocab_size: int) -> int:
    result = _plain_int(value, name)
    if not 0 <= result < vocab_size:
        raise OracleInputError(f"{name} must be within [0, {vocab_size}), got {result}")
    return result


def _parse_sparse_logits(
    value: Any,
    name: str,
    vocab_size: int,
    *,
    allow_empty: bool,
) -> dict[int, float]:
    source = _require_object(value, name)
    result: dict[int, float] = {}
    for raw_token, raw_logit in source.items():
        try:
            token = int(raw_token)
        except (TypeError, ValueError) as exc:
            raise OracleInputError(
                f"{name} token IDs must be base-10 integers, got {raw_token!r}"
            ) from exc
        if str(token) != str(raw_token):
            raise OracleInputError(
                f"{name} token ID {raw_token!r} is not in canonical integer form"
            )
        _token(token, f"{name}[{raw_token!r}] token", vocab_size)
        result[token] = _finite_float(raw_logit, f"{name}[{raw_token!r}]")
    if not result and not allow_empty:
        raise OracleInputError(f"{name} must contain at least one finite logit")
    return result


def _parse_sampling(value: Any, name: str, vocab_size: int) -> SamplingTransform:
    source = _require_object(value, name)
    unknown = set(source) - {"temperature", "top_k", "top_p"}
    if unknown:
        raise OracleInputError(f"{name} has unknown fields: {sorted(unknown)}")
    temperature = _finite_float(source.get("temperature", 1.0), f"{name}.temperature")
    if temperature <= 0.0:
        raise OracleInputError(f"{name}.temperature must be positive")
    raw_top_k = source.get("top_k")
    top_k = None
    if raw_top_k is not None:
        top_k = _plain_int(raw_top_k, f"{name}.top_k")
        if not 1 <= top_k <= vocab_size:
            raise OracleInputError(f"{name}.top_k must be within [1, {vocab_size}]")
    top_p = _finite_float(source.get("top_p", 1.0), f"{name}.top_p")
    if not 0.0 < top_p <= 1.0:
        raise OracleInputError(f"{name}.top_p must be within (0, 1]")
    return SamplingTransform(temperature=temperature, top_k=top_k, top_p=top_p)


def _parse_parents(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise OracleInputError("parents must be a JSON array with at least two nodes")
    parents = tuple(
        _plain_int(parent, f"parents[{node}]") for node, parent in enumerate(value)
    )
    if parents[0] != -1:
        raise OracleInputError("parents[0] must be the root marker -1")
    for node, parent in enumerate(parents[1:], start=1):
        if not 0 <= parent < node:
            raise OracleInputError(
                f"parents[{node}] must identify an earlier node, got {parent}"
            )
    return parents


def parse_document(document: Any) -> OracleInput:
    """Validate and bind one version-1 oracle document."""
    source = _require_object(document, "document")
    if source.get("schema_version") != 1:
        raise OracleInputError("schema_version must be exactly 1")

    vocab_size = _plain_int(source.get("vocab_size"), "vocab_size")
    if vocab_size <= 0:
        raise OracleInputError("vocab_size must be positive")
    parents = _parse_parents(source.get("parents"))
    root_token = _token(source.get("root_token"), "root_token", vocab_size)

    raw_counts = _require_object(
        source.get("initial_token_counts", {}), "initial_token_counts"
    )
    initial_token_counts: dict[int, int] = {}
    for raw_token, raw_count in raw_counts.items():
        try:
            token = int(raw_token)
        except (TypeError, ValueError) as exc:
            raise OracleInputError(
                f"initial_token_counts token ID {raw_token!r} is invalid"
            ) from exc
        if str(token) != str(raw_token):
            raise OracleInputError(
                "initial_token_counts token IDs must use canonical integer form"
            )
        _token(token, f"initial_token_counts[{raw_token!r}] token", vocab_size)
        count = _plain_int(raw_count, f"initial_token_counts[{raw_token!r}]")
        if count < 0:
            raise OracleInputError("initial token counts cannot be negative")
        if count:
            initial_token_counts[token] = count

    raw_penalties = _require_object(source.get("penalties", {}), "penalties")
    unknown_penalties = set(raw_penalties) - {"presence", "frequency"}
    if unknown_penalties:
        raise OracleInputError(
            f"penalties has unknown fields: {sorted(unknown_penalties)}"
        )
    penalties = AdditivePenalties(
        presence=_finite_float(
            raw_penalties.get("presence", 0.0), "penalties.presence"
        ),
        frequency=_finite_float(
            raw_penalties.get("frequency", 0.0), "penalties.frequency"
        ),
    )

    target_sampling = _parse_sampling(
        source.get("target_sampling", {}), "target_sampling", vocab_size
    )
    draft_sampling = _parse_sampling(
        source.get("draft_sampling", {}), "draft_sampling", vocab_size
    )

    raw_rows = source.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise OracleInputError("rows must be a nonempty JSON array")
    rows: dict[tuple[int, ...], SparseLogitRow] = {}
    for row_index, raw_row in enumerate(raw_rows):
        row = _require_object(raw_row, f"rows[{row_index}]")
        raw_suffix = row.get("suffix")
        if not isinstance(raw_suffix, list):
            raise OracleInputError(f"rows[{row_index}].suffix must be a JSON array")
        suffix = tuple(
            _token(token, f"rows[{row_index}].suffix[{offset}]", vocab_size)
            for offset, token in enumerate(raw_suffix)
        )
        if suffix in rows:
            raise OracleInputError(f"duplicate sparse row for suffix {list(suffix)}")
        rows[suffix] = SparseLogitRow(
            target_logits=_parse_sparse_logits(
                row.get("target_logits"),
                f"rows[{row_index}].target_logits",
                vocab_size,
                allow_empty=False,
            ),
            draft_logits=_parse_sparse_logits(
                row.get("draft_logits"),
                f"rows[{row_index}].draft_logits",
                vocab_size,
                allow_empty=True,
            ),
        )
    if () not in rows:
        raise OracleInputError("rows must include the root suffix []")

    raw_costs = _require_object(source.get("cycle_cost_ms"), "cycle_cost_ms")
    if not raw_costs:
        raise OracleInputError("cycle_cost_ms must contain measured phase costs")
    cycle_cost_ms: dict[str, float] = {}
    for name, raw_cost in raw_costs.items():
        if not isinstance(name, str) or not name.strip():
            raise OracleInputError("cycle_cost_ms phase names must be nonempty strings")
        cost = _finite_float(raw_cost, f"cycle_cost_ms.{name}")
        if cost < 0.0:
            raise OracleInputError("cycle phase costs cannot be negative")
        cycle_cost_ms[name] = cost
    if math.fsum(cycle_cost_ms.values()) <= 0.0:
        raise OracleInputError("total measured cycle cost must be positive")

    raw_gate = _require_object(source.get("gate"), "gate")
    target_tps = _finite_float(raw_gate.get("target_tps"), "gate.target_tps")
    margin_tps = _finite_float(raw_gate.get("margin_tps"), "gate.margin_tps")
    if target_tps <= 0.0:
        raise OracleInputError("gate.target_tps must be positive")
    if margin_tps <= 0.0:
        raise OracleInputError(
            "gate.margin_tps must be positive so the promotion margin is explicit"
        )

    return OracleInput(
        vocab_size=vocab_size,
        parents=parents,
        root_token=root_token,
        initial_token_counts=initial_token_counts,
        penalties=penalties,
        target_sampling=target_sampling,
        draft_sampling=draft_sampling,
        rows=rows,
        cycle_cost_ms=cycle_cost_ms,
        gate=PromotionGate(target_tps=target_tps, margin_tps=margin_tps),
    )


def _normalize(values: Mapping[int, float], name: str) -> dict[int, float]:
    positive = {token: value for token, value in values.items() if value > 0.0}
    total = math.fsum(positive.values())
    if total <= 0.0:
        raise OracleInputError(f"{name} has no positive probability mass")
    return {token: value / total for token, value in positive.items()}


def _distribution_from_logits(
    logits: Mapping[int, float],
    counts: Mapping[int, int],
    penalties: AdditivePenalties,
    sampling: SamplingTransform,
    *,
    allow_empty: bool,
) -> dict[int, float]:
    if not logits:
        if allow_empty:
            return {}
        raise OracleInputError("target sparse logits cannot be empty")

    adjusted: list[tuple[int, float]] = []
    for token, raw_logit in logits.items():
        count = counts.get(token, 0)
        penalty = penalties.frequency * count
        if count:
            penalty += penalties.presence
        adjusted.append((token, (raw_logit - penalty) / sampling.temperature))

    adjusted.sort(key=lambda item: (-item[1], item[0]))
    if sampling.top_k is not None:
        adjusted = adjusted[: sampling.top_k]
    maximum = adjusted[0][1]
    weights = {token: math.exp(logit - maximum) for token, logit in adjusted}
    probabilities = _normalize(weights, "sparse logits")

    if sampling.top_p < 1.0 and len(probabilities) > 1:
        ascending = sorted(probabilities.items(), key=lambda item: (item[1], item[0]))
        threshold = 1.0 - sampling.top_p
        cumulative = 0.0
        cutoff = 0
        while (
            cutoff < len(ascending) - 1
            and cumulative + ascending[cutoff][1] < threshold
        ):
            cumulative += ascending[cutoff][1]
            cutoff += 1
        pivot = ascending[cutoff][1]
        probabilities = _normalize(
            {
                token: probability
                for token, probability in probabilities.items()
                if probability >= pivot
            },
            "top-p sparse logits",
        )
    return probabilities


def project_throughput(
    emitted_tokens_per_cycle: float,
    cycle_cost_ms: Mapping[str, float],
    gate: PromotionGate,
) -> dict[str, Any]:
    """Project TPS and apply the strict target-plus-margin promotion gate."""
    if emitted_tokens_per_cycle <= 0.0 or not math.isfinite(emitted_tokens_per_cycle):
        raise OracleInputError("emitted_tokens_per_cycle must be finite and positive")
    if gate.target_tps <= 0.0 or gate.margin_tps <= 0.0:
        raise OracleInputError("the promotion target and margin must be positive")
    costs = {name: float(cost) for name, cost in cycle_cost_ms.items()}
    if any(not math.isfinite(cost) or cost < 0.0 for cost in costs.values()):
        raise OracleInputError("cycle costs must be finite and nonnegative")
    total_ms = math.fsum(costs.values())
    if total_ms <= 0.0:
        raise OracleInputError("total cycle cost must be positive")

    projected_tps = 1000.0 * emitted_tokens_per_cycle / total_ms
    required_tps = gate.target_tps + gate.margin_tps
    required_emitted = required_tps * total_ms / 1000.0
    return {
        "components_ms": costs,
        "total_ms": total_ms,
        "projected_tps": projected_tps,
        "gate": {
            "target_tps": gate.target_tps,
            "margin_tps": gate.margin_tps,
            "required_tps_exclusive": required_tps,
            "required_emitted_tokens_per_cycle_exclusive": required_emitted,
            "headroom_over_required_tps": projected_tps - required_tps,
            "passes": projected_tps > required_tps,
        },
    }


class SparsePQSworOracle:
    """Exact finite-support SWOR expectation for one fixed tree topology."""

    def __init__(self, oracle_input: OracleInput):
        self.input = oracle_input
        self.children: list[list[int]] = [[] for _ in oracle_input.parents]
        self.depths = [0] * len(oracle_input.parents)
        for node, parent in enumerate(oracle_input.parents[1:], start=1):
            self.children[parent].append(node)
            self.depths[node] = self.depths[parent] + 1
        self._row_cache: dict[
            tuple[int, ...], tuple[dict[int, float], dict[int, float]]
        ] = {}
        self._node_cache: dict[tuple[int, tuple[int, ...]], _Evaluation] = {}

    def _branch_counts(self, suffix: tuple[int, ...]) -> dict[int, int]:
        counts = dict(self.input.initial_token_counts)
        counts[self.input.root_token] = counts.get(self.input.root_token, 0) + 1
        for token in suffix:
            counts[token] = counts.get(token, 0) + 1
        return counts

    def distributions_for_suffix(
        self, suffix: Sequence[int]
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Return branch-local target p and proposal q for an accepted suffix."""
        key = tuple(suffix)
        cached = self._row_cache.get(key)
        if cached is not None:
            return cached
        row = self.input.rows.get(key)
        if row is None:
            raise OracleInputError(
                "missing sparse p/q row for reachable internal suffix " f"{list(key)}"
            )
        counts = self._branch_counts(key)
        target = _distribution_from_logits(
            row.target_logits,
            counts,
            self.input.penalties,
            self.input.target_sampling,
            allow_empty=False,
        )
        draft = _distribution_from_logits(
            row.draft_logits,
            counts,
            self.input.penalties,
            self.input.draft_sampling,
            allow_empty=True,
        )
        cached = (target, draft)
        self._row_cache[key] = cached
        return cached

    @staticmethod
    def _residual_after_rejection(
        target: Mapping[int, float], proposal: Mapping[int, float]
    ) -> dict[int, float]:
        residual = {
            token: max(probability - proposal.get(token, 0.0), 0.0)
            for token, probability in target.items()
        }
        return _normalize(residual, "SWOR rejection residual")

    def _evaluate_node(self, node: int, suffix: tuple[int, ...]) -> _Evaluation:
        if not self.children[node]:
            return _Evaluation.zero()
        key = (node, suffix)
        cached = self._node_cache.get(key)
        if cached is not None:
            return cached
        target, draft = self.distributions_for_suffix(suffix)
        result = self._evaluate_rank(
            node=node,
            suffix=suffix,
            rank=0,
            target=target,
            draft_remaining=draft,
            rejected_count=0,
        )
        self._node_cache[key] = result
        return result

    def _accepted_transition(
        self,
        child: int,
        suffix: tuple[int, ...],
        token: int,
    ) -> _Evaluation:
        nested = self._evaluate_node(child, suffix + (token,))
        result = _Evaluation(1.0 + nested.expected_accepted, {child: 1.0})
        for node, probability in nested.accepted_by_node.items():
            result.accepted_by_node[node] = (
                result.accepted_by_node.get(node, 0.0) + probability
            )
        return result

    def _evaluate_rank(
        self,
        *,
        node: int,
        suffix: tuple[int, ...],
        rank: int,
        target: Mapping[int, float],
        draft_remaining: Mapping[int, float],
        rejected_count: int,
    ) -> _Evaluation:
        child_nodes = self.children[node]
        if rank >= len(child_nodes):
            return _Evaluation.zero()
        child = child_nodes[rank]
        result = _Evaluation.zero()

        if draft_remaining:
            proposal = _normalize(draft_remaining, "remaining SWOR proposal")
            rejected_by_token = {
                token: max(probability - target.get(token, 0.0), 0.0)
                for token, probability in proposal.items()
            }
            rejection_mass = math.fsum(rejected_by_token.values())
            residual = None
            if rejection_mass > 0.0 and rank + 1 < len(child_nodes):
                residual = self._residual_after_rejection(target, proposal)

            for token, proposal_probability in proposal.items():
                accepted = min(proposal_probability, target.get(token, 0.0))
                if accepted > 0.0:
                    result.add_scaled(
                        self._accepted_transition(child, suffix, token), accepted
                    )

                rejected = rejected_by_token[token]
                if rejected > 0.0 and rank + 1 < len(child_nodes):
                    assert residual is not None
                    next_draft = dict(draft_remaining)
                    del next_draft[token]
                    continuation = self._evaluate_rank(
                        node=node,
                        suffix=suffix,
                        rank=rank + 1,
                        target=residual,
                        draft_remaining=next_draft,
                        rejected_count=rejected_count + 1,
                    )
                    result.add_scaled(continuation, rejected)
            return result

        remaining_vocab = self.input.vocab_size - rejected_count
        if remaining_vocab <= 0:
            raise OracleInputError(
                "topology requests more SWOR siblings than "
                f"vocab_size={self.input.vocab_size}"
            )
        proposal_probability = 1.0 / remaining_vocab
        if len(target) > remaining_vocab:
            raise RuntimeError(
                "internal oracle error: sparse target support exceeds the "
                "unrejected vocabulary"
            )
        target_rejection_mass = math.fsum(
            max(proposal_probability - probability, 0.0)
            for probability in target.values()
        )
        other_token_count = remaining_vocab - len(target)
        rejection_mass = (
            target_rejection_mass + other_token_count * proposal_probability
        )
        for token, target_probability in target.items():
            accepted = min(proposal_probability, target_probability)
            if accepted > 0.0:
                result.add_scaled(
                    self._accepted_transition(child, suffix, token), accepted
                )

        if rejection_mass > 0.0 and rank + 1 < len(child_nodes):
            uniform_on_target = {token: proposal_probability for token in target}
            residual = self._residual_after_rejection(target, uniform_on_target)
            continuation = self._evaluate_rank(
                node=node,
                suffix=suffix,
                rank=rank + 1,
                target=residual,
                draft_remaining={},
                rejected_count=rejected_count + 1,
            )
            # All rejected zero-target tokens are exchangeable.  Target-support
            # rejections have the same normalized residual, so one aggregate
            # transition is exact for the full remaining vocabulary.
            result.add_scaled(continuation, rejection_mass)
        return result

    def evaluate(self) -> dict[str, Any]:
        evaluation = self._evaluate_node(0, ())
        accepted_sum = math.fsum(evaluation.accepted_by_node.values())
        if not math.isclose(
            evaluation.expected_accepted,
            accepted_sum,
            rel_tol=1e-11,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "internal oracle error: node probabilities do not sum to "
                "expected drafts"
            )

        accepted_by_node = [0.0] * len(self.input.parents)
        accepted_by_depth: dict[int, float] = {}
        for node, probability in evaluation.accepted_by_node.items():
            accepted_by_node[node] = probability
            depth = self.depths[node]
            accepted_by_depth[depth] = accepted_by_depth.get(depth, 0.0) + probability

        emitted = 1.0 + evaluation.expected_accepted
        projection = project_throughput(
            emitted,
            self.input.cycle_cost_ms,
            self.input.gate,
        )
        return {
            "schema_version": 1,
            "semantics": {
                "proposal": "ordered sparse q sampling without replacement",
                "verification": "exact R=max(R-D,0) rejection residual",
                "expectation": "integrated proposal draws and verifier coins",
                "penalty_scope": (
                    "initial_token_counts + root_token + accepted branch suffix"
                ),
            },
            "topology": {
                "parents": list(self.input.parents),
                "nodes": len(self.input.parents),
                "max_depth": max(self.depths),
            },
            "expected": {
                "accepted_drafts_per_cycle": evaluation.expected_accepted,
                "emitted_tokens_per_cycle": emitted,
                "accepted_probability_by_node": accepted_by_node,
                "accepted_probability_by_depth": {
                    str(depth): accepted_by_depth[depth]
                    for depth in sorted(accepted_by_depth)
                },
            },
            "cycle_projection": projection,
        }


def _print_human(report: Mapping[str, Any]) -> None:
    expected = report["expected"]
    projection = report["cycle_projection"]
    gate = projection["gate"]
    print("semantics=exact_sparse_pq_swor branch_penalties=accepted_path_local")
    print(f"accepted_drafts_per_cycle={expected['accepted_drafts_per_cycle']:.9f}")
    print(f"emitted_tokens_per_cycle={expected['emitted_tokens_per_cycle']:.9f}")
    print(
        "cycle_cost_ms="
        + json.dumps(projection["components_ms"], sort_keys=True, separators=(",", ":"))
    )
    print(f"cycle_total_ms={projection['total_ms']:.9f}")
    print(f"projected_tps={projection['projected_tps']:.9f}")
    print(
        "promotion_gate="
        f">{gate['required_tps_exclusive']:.9f} "
        f"(target={gate['target_tps']:.9f} margin={gate['margin_tps']:.9f}) "
        f"passes={str(gate['passes']).lower()}"
    )
    print(
        "required_emitted_tokens_per_cycle_exclusive="
        f"{gate['required_emitted_tokens_per_cycle_exclusive']:.9f}"
    )
    print(
        "accepted_probability_by_node="
        + json.dumps(expected["accepted_probability_by_node"], separators=(",", ":"))
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="version-1 sparse oracle JSON")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--target-tps",
        type=float,
        help="override gate.target_tps while retaining the recorded document",
    )
    parser.add_argument(
        "--margin-tps",
        type=float,
        help="override gate.margin_tps; must remain positive",
    )
    parser.add_argument(
        "--require-promotion-gate",
        action="store_true",
        help="return exit status 2 unless projected TPS strictly clears the gate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        oracle_input = parse_document(document)
        target_tps = (
            oracle_input.gate.target_tps
            if args.target_tps is None
            else _finite_float(args.target_tps, "--target-tps")
        )
        margin_tps = (
            oracle_input.gate.margin_tps
            if args.margin_tps is None
            else _finite_float(args.margin_tps, "--margin-tps")
        )
        if target_tps <= 0.0 or margin_tps <= 0.0:
            raise OracleInputError(
                "target TPS and margin TPS overrides must be positive"
            )
        oracle_input = replace(
            oracle_input,
            gate=PromotionGate(target_tps=target_tps, margin_tps=margin_tps),
        )
        report = SparsePQSworOracle(oracle_input).evaluate()
    except (OSError, json.JSONDecodeError, OracleInputError) as exc:
        parser.error(str(exc))

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    if args.require_promotion_gate and not report["cycle_projection"]["gate"]["passes"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
