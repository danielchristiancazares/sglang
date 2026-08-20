"""Default-inactive branch-exact sparse p/q diagnostic capture.

The live worker retains raw target/draft logits and the verifier's realized
first-child/next-sibling topology.  ``GenerationBatchResult.copy_to_cpu`` moves
those tensors through its existing pinned asynchronous D2H lifetime.  A bounded
background writer then applies presence, frequency, and sign-aware repetition
state independently on every branch and appends one JSONL cycle record.

Records from a selected tree deliberately mark omitted branches incomplete.
The offline replay may compare policies over captured nodes and may use the
depth hard ceiling, but it must not treat missing proposal-lattice branches as
zero-probability evidence.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import queue
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch


class PQDiagnosticError(ValueError):
    """Raised when an opt-in capture cannot represent exact semantics."""


@dataclass(frozen=True)
class PenaltyConfig:
    presence: float
    frequency: float
    repetition: float


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_k: int
    top_p: float


def branch_counts(
    initial_counts: Mapping[int, int],
    parent_by_node: Sequence[int],
    node_tokens: Sequence[int],
    node: int,
) -> dict[int, int]:
    """Return immutable-prefix counts at the boundary leaving ``node``.

    ``initial_counts`` already contains the root/current token.  Node zero is
    therefore never added a second time; every non-root node contributes its
    incoming token exactly once along its own ancestry.
    """
    counts = dict(initial_counts)
    ancestry = []
    current = node
    while current > 0:
        ancestry.append(current)
        current = parent_by_node[current]
        if current < 0:
            raise PQDiagnosticError("non-root node has no path to root")
    for ancestor in reversed(ancestry):
        token = int(node_tokens[ancestor])
        counts[token] = counts.get(token, 0) + 1
    return counts


def sparse_distribution_from_logits(
    logits: torch.Tensor,
    counts: Mapping[int, int],
    penalties: PenaltyConfig,
    sampling: SamplingConfig,
    *,
    logit_bias: Optional[Mapping[int, float]] = None,
) -> dict[int, float]:
    """Sequential scalar-equivalent distribution for the spec/overlap path.

    Order is additive presence/frequency, sign-aware repetition, logit bias,
    temperature, top-k renormalization, then top-p renormalization.  The input
    is immutable raw model logits; applying this function to probabilities or
    an already penalized tensor is a caller error.
    """
    if logits.ndim != 1 or not logits.numel():
        raise PQDiagnosticError("raw logits must be one nonempty vocabulary row")
    if not 0.0 < sampling.temperature or not 0.0 < sampling.top_p <= 1.0:
        raise PQDiagnosticError("temperature and top-p must be in range")
    if not 1 <= sampling.top_k <= logits.numel():
        raise PQDiagnosticError("top-k must fit the captured vocabulary")
    if not 0.0 < penalties.repetition <= 2.0:
        raise PQDiagnosticError("repetition penalty must be within (0, 2]")

    values = logits.detach().to(dtype=torch.float32, device="cpu").clone()
    for token, count in counts.items():
        if count <= 0 or not 0 <= token < values.numel():
            continue
        adjusted = values[token] - penalties.frequency * count - penalties.presence
        values[token] = (
            adjusted * penalties.repetition
            if adjusted < 0.0
            else adjusted / penalties.repetition
        )
    if logit_bias:
        for token, bias in logit_bias.items():
            if 0 <= token < values.numel():
                values[token] += float(bias)
    values.div_(sampling.temperature)
    probabilities = torch.softmax(values, dim=-1)

    top_values, top_indices = torch.topk(
        probabilities, sampling.top_k, sorted=True
    )
    top_values = top_values / top_values.sum()
    if sampling.top_p < 1.0 and top_values.numel() > 1:
        ascending_values, _ = torch.sort(top_values)
        ascending_cdf = torch.cumsum(ascending_values, dim=0)
        cutoff = int(torch.sum(ascending_cdf < (1.0 - sampling.top_p)).item())
        cutoff = min(cutoff, top_values.numel() - 1)
        pivot = ascending_values[cutoff]
        top_values = torch.where(
            top_values >= pivot, top_values, torch.zeros_like(top_values)
        )
        top_values = top_values / top_values.sum()
    return {
        int(token): float(probability)
        for token, probability in zip(top_indices.tolist(), top_values.tolist())
        if probability > 0.0
    }


def normalize_verifier_topology(
    retrieve_next_token: Sequence[int], retrieve_next_sibling: Sequence[int]
) -> tuple[list[int], list[int], list[int]]:
    """Convert verifier first-child/next-sibling links to parent/depth/rank."""
    if len(retrieve_next_token) != len(retrieve_next_sibling):
        raise PQDiagnosticError("verifier topology arrays must have equal length")
    nodes = len(retrieve_next_token)
    if not nodes:
        raise PQDiagnosticError("verifier topology must contain the root")
    parents = [-2] * nodes
    depths = [-1] * nodes
    ranks = [-1] * nodes
    parents[0] = -1
    depths[0] = 0
    stack = [0]
    while stack:
        parent = stack.pop()
        child = int(retrieve_next_token[parent])
        rank = 0
        siblings = []
        while child != -1:
            if not 0 <= child < nodes:
                raise PQDiagnosticError("verifier child index is out of range")
            if child == 0 or parents[child] != -2:
                raise PQDiagnosticError("verifier topology is cyclic or multiply parented")
            parents[child] = parent
            depths[child] = depths[parent] + 1
            ranks[child] = rank
            siblings.append(child)
            child = int(retrieve_next_sibling[child])
            rank += 1
        stack.extend(reversed(siblings))
    if any(parent == -2 for parent in parents):
        raise PQDiagnosticError("verifier topology contains unreachable nodes")
    return parents, depths, ranks


def _base_counts_for_req(req: Any) -> dict[int, int]:
    tokens = list(req.output_ids)
    if not tokens:
        tokens = [int(req.origin_input_ids[-1])]
    return dict(Counter(int(token) for token in tokens))


def _sampling_config(req: Any, *, top_k: Optional[int] = None) -> SamplingConfig:
    params = req.sampling_params
    return SamplingConfig(
        temperature=float(params.temperature),
        top_k=int(params.top_k if top_k is None else top_k),
        top_p=float(params.top_p),
    )


def _penalty_config(req: Any) -> PenaltyConfig:
    params = req.sampling_params
    return PenaltyConfig(
        presence=float(params.presence_penalty),
        frequency=float(params.frequency_penalty),
        repetition=float(params.repetition_penalty),
    )


def _logit_bias(req: Any) -> dict[int, float]:
    value = req.sampling_params.logit_bias
    return {int(token): float(bias) for token, bias in (value or {}).items()}


@dataclass
class BranchPQCapture:
    output_path: str
    max_cycles: int
    request_id: str
    input_sha256: str
    active_worker: str
    torch_compile_enabled: bool
    torch_compile_mode: str
    speculative_algorithm: str
    draft_sampling_top_k: int
    raw_target_logits: torch.Tensor
    raw_draft_logits: torch.Tensor
    node_tokens: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    initial_counts: dict[int, int]
    penalties: PenaltyConfig
    target_sampling: SamplingConfig
    draft_sampling: SamplingConfig
    logit_bias: dict[int, float]

    def map_device_tensors(self, mapper: Callable[[torch.Tensor], torch.Tensor]) -> None:
        for field in (
            "raw_target_logits",
            "raw_draft_logits",
            "node_tokens",
            "retrieve_next_token",
            "retrieve_next_sibling",
        ):
            value = getattr(self, field)
            setattr(self, field, mapper(value))

    def _cycle_record(self, ordinal: int) -> dict[str, Any]:
        target_logits = self.raw_target_logits
        draft_logits = self.raw_draft_logits
        if target_logits.ndim != 2 or draft_logits.ndim != 2:
            raise PQDiagnosticError("captured target/draft logits must be rank two")
        tokens = [int(token) for token in self.node_tokens.reshape(-1).tolist()]
        next_token = [
            int(value) for value in self.retrieve_next_token.reshape(-1).tolist()
        ]
        next_sibling = [
            int(value) for value in self.retrieve_next_sibling.reshape(-1).tolist()
        ]
        if target_logits.shape[0] != len(tokens):
            raise PQDiagnosticError("target rows must align with verifier nodes")
        parents, depths, ranks = normalize_verifier_topology(next_token, next_sibling)

        states = []
        distributions: dict[int, tuple[dict[int, float], dict[int, float]]] = {}
        for node in range(len(tokens)):
            counts = branch_counts(
                self.initial_counts, parents, tokens, node
            )
            p = sparse_distribution_from_logits(
                target_logits[node],
                counts,
                self.penalties,
                self.target_sampling,
                logit_bias=self.logit_bias,
            )
            q = (
                sparse_distribution_from_logits(
                    draft_logits[node],
                    counts,
                    self.penalties,
                    self.draft_sampling,
                    logit_bias=self.logit_bias,
                )
                if node < draft_logits.shape[0]
                else {}
            )
            distributions[node] = (p, q)
            states.append(
                {
                    "state_id": node,
                    "depth": depths[node],
                    "initial_token_counts": {
                        str(token): count for token, count in sorted(counts.items())
                    },
                    "target_support": {
                        str(token): probability for token, probability in sorted(p.items())
                    },
                    "draft_support": {
                        str(token): probability for token, probability in sorted(q.items())
                    },
                    "target_support_complete": True,
                    "draft_support_complete": bool(q),
                }
            )

        edges = []
        for child in range(1, len(tokens)):
            parent = parents[child]
            p, q = distributions[parent]
            token = tokens[child]
            edges.append(
                {
                    "edge_id": child - 1,
                    "parent_id": parent,
                    "child_id": child,
                    "token_id": token,
                    "depth": depths[child],
                    "branch_rank": ranks[child],
                    "p": p.get(token, 0.0),
                    "q": q.get(token, 0.0),
                    "current_score": q.get(token, 0.0),
                }
            )
        for state in states:
            outgoing = [edge for edge in edges if edge["parent_id"] == state["state_id"]]
            state["p_other_mass"] = max(
                0.0, 1.0 - math.fsum(edge["p"] for edge in outgoing)
            )
            state["q_other_mass"] = max(
                0.0, 1.0 - math.fsum(edge["q"] for edge in outgoing)
            )
            # A selected-tree record contains complete probability supports but
            # lacks child states for omitted tokens.  Counterfactual topology
            # coverage is therefore incomplete unless no probability remains.
            state["target_support_complete"] = state["p_other_mass"] <= 1e-7
            state["draft_support_complete"] = state["q_other_mass"] <= 1e-7

        return {
            "schema_version": 2,
            "artifact_type": "sparse_pq_cycle",
            "capture_scope": "selected_tree",
            "cycle_ordinal": ordinal,
            "request_id": self.request_id,
            "input_sha256": self.input_sha256,
            "runtime": {
                "active_worker": self.active_worker,
                "torch_compile_enabled": self.torch_compile_enabled,
                "torch_compile_mode": self.torch_compile_mode,
                "speculative_algorithm": self.speculative_algorithm,
            },
            "sampling": {
                "penalties": {
                    "presence": self.penalties.presence,
                    "frequency": self.penalties.frequency,
                    "repetition": self.penalties.repetition,
                },
                "penalty_scope": "committed root snapshot plus accepted branch only",
                "root_in_initial_counts": True,
                "transform_order": [
                    "presence_frequency",
                    "repetition",
                    "logit_bias",
                    "temperature",
                    "top_k_renormalize",
                    "top_p_renormalize",
                ],
                "target": self.target_sampling.__dict__,
                "draft": self.draft_sampling.__dict__,
            },
            "cycle": {
                "root_state_id": 0,
                "max_depth": max(depths),
                "states": states,
                "edges": edges,
                "topology_memberships": {
                    "current": [edge["edge_id"] for edge in edges]
                },
            },
        }


class _CaptureWriter:
    def __init__(self, path: str, max_cycles: int):
        self.path = Path(path)
        self.max_cycles = max_cycles
        self.count = 0
        self.queue: queue.Queue[Optional[BranchPQCapture]] = queue.Queue(maxsize=8)
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"pq-capture-{self.path.name}",
            daemon=True,
        )
        self.thread.start()

    def submit(self, capture: BranchPQCapture) -> None:
        if self.error is not None:
            raise RuntimeError("p/q capture writer failed") from self.error
        if self.count >= self.max_cycles:
            return
        try:
            self.queue.put_nowait(capture)
        except queue.Full as exc:
            raise RuntimeError("p/q capture writer queue is full") from exc

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as output:
                while True:
                    capture = self.queue.get()
                    if capture is None:
                        self.queue.task_done()
                        break
                    if self.count < self.max_cycles:
                        record = capture._cycle_record(self.count)
                        output.write(
                            json.dumps(record, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        output.flush()
                        self.count += 1
                    self.queue.task_done()
        except BaseException as exc:  # diagnostic failure is surfaced on submit/flush
            self.error = exc

    def close(self) -> None:
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join()
        if self.error is not None:
            raise RuntimeError("p/q capture writer failed") from self.error


_writers: dict[str, _CaptureWriter] = {}
_writers_lock = threading.Lock()


def submit_capture(capture: BranchPQCapture) -> None:
    with _writers_lock:
        writer = _writers.get(capture.output_path)
        if writer is None:
            writer = _CaptureWriter(capture.output_path, capture.max_cycles)
            _writers[capture.output_path] = writer
        elif writer.max_cycles != capture.max_cycles:
            raise RuntimeError("p/q capture max_cycles changed within one process")
    writer.submit(capture)


def close_capture_writers() -> None:
    with _writers_lock:
        writers = list(_writers.values())
        _writers.clear()
    for writer in writers:
        writer.close()


atexit.register(close_capture_writers)


def make_branch_pq_capture(
    *,
    output_path: str,
    max_cycles: int,
    batch: Any,
    raw_target_logits: torch.Tensor,
    raw_draft_logits: torch.Tensor,
    node_tokens: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_sampling_top_k: int,
    active_worker: str,
    torch_compile_enabled: bool,
    torch_compile_mode: str,
    speculative_algorithm: str,
) -> BranchPQCapture:
    if len(batch.reqs) != 1:
        raise PQDiagnosticError("initial p/q capture supports batch size one")
    if max_cycles <= 0:
        raise PQDiagnosticError("p/q capture max_cycles must be positive")
    req = batch.reqs[0]
    if req.grammar is not None:
        raise PQDiagnosticError("p/q capture does not yet support grammar masks")
    params = req.sampling_params
    if params.min_p != 0.0 or req.custom_logit_processor is not None:
        raise PQDiagnosticError("p/q capture does not support min-p/custom processors")
    prompt_digest = hashlib.sha256(
        json.dumps(list(req.origin_input_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return BranchPQCapture(
        output_path=output_path,
        max_cycles=max_cycles,
        request_id=str(req.rid),
        input_sha256=prompt_digest,
        active_worker=active_worker,
        torch_compile_enabled=torch_compile_enabled,
        torch_compile_mode=torch_compile_mode,
        speculative_algorithm=speculative_algorithm,
        draft_sampling_top_k=draft_sampling_top_k,
        raw_target_logits=raw_target_logits,
        raw_draft_logits=raw_draft_logits,
        node_tokens=node_tokens,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        initial_counts=_base_counts_for_req(req),
        penalties=_penalty_config(req),
        target_sampling=_sampling_config(req),
        draft_sampling=_sampling_config(req, top_k=draft_sampling_top_k),
        logit_bias=_logit_bias(req),
    )
