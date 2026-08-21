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
import base64
import hashlib
import json
import logging
import math
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


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
    if sampling.temperature < 0.0 or not 0.0 < sampling.top_p <= 1.0:
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
    if sampling.temperature == 0.0:
        return {int(torch.argmax(values).item()): 1.0}
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


def sparse_distribution_from_probs(probs: torch.Tensor) -> dict[int, float]:
    """Return the positive finite support of one exact proposal q row."""
    if probs.ndim != 1 or not probs.numel():
        raise PQDiagnosticError("proposal probabilities must be one nonempty row")
    values = probs.detach().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(values).all() or torch.any(values < 0.0):
        raise PQDiagnosticError("proposal probabilities must be finite and nonnegative")
    support = torch.nonzero(values > 0.0, as_tuple=False).flatten()
    if not support.numel():
        raise PQDiagnosticError("proposal probabilities must have positive mass")
    total = float(values[support].sum().item())
    if not math.isfinite(total) or total <= 0.0:
        raise PQDiagnosticError("proposal probabilities have invalid total mass")
    return {
        int(token): float(values[token].item() / total)
        for token in support.tolist()
    }


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _distribution_rank(distribution: Mapping[int, float], token: int) -> Optional[int]:
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    for rank, (candidate, _) in enumerate(ordered):
        if candidate == token:
            return rank
    return None


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
    origin_input_length: int = 0
    committed_output_length: int = 0
    exact_draft_probs: Optional[torch.Tensor] = None
    draft_hidden_states: Optional[torch.Tensor] = None
    target_hidden_states: Optional[torch.Tensor] = None
    accept_lens: Optional[torch.Tensor] = None

    def map_device_tensors(self, mapper: Callable[[torch.Tensor], torch.Tensor]) -> None:
        for field in (
            "raw_target_logits",
            "raw_draft_logits",
            "node_tokens",
            "retrieve_next_token",
            "retrieve_next_sibling",
            "exact_draft_probs",
            "draft_hidden_states",
            "target_hidden_states",
            "accept_lens",
        ):
            value = getattr(self, field)
            if value is not None:
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
        if (
            self.exact_draft_probs is not None
            and self.exact_draft_probs.shape[0] != draft_logits.shape[0]
        ):
            raise PQDiagnosticError("exact q rows must align with draft logits")
        if (
            self.draft_hidden_states is not None
            and self.draft_hidden_states.shape[0] != draft_logits.shape[0]
        ):
            raise PQDiagnosticError("draft hidden rows must align with draft logits")
        if (
            self.target_hidden_states is not None
            and self.target_hidden_states.shape[0] != target_logits.shape[0]
        ):
            raise PQDiagnosticError("target hidden rows must align with target logits")
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
            q = {}
            if node < draft_logits.shape[0] and next_token[node] != -1:
                q = (
                    sparse_distribution_from_probs(self.exact_draft_probs[node])
                    if self.exact_draft_probs is not None
                    else sparse_distribution_from_logits(
                        draft_logits[node],
                        counts,
                        self.penalties,
                        self.draft_sampling,
                        logit_bias=self.logit_bias,
                    )
                )
            distributions[node] = (p, q)
            target_argmax = min(
                token
                for token, probability in p.items()
                if probability == max(p.values())
            )
            state = {
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
                "target_argmax_token": target_argmax,
                "target_argmax_draft_rank": _distribution_rank(q, target_argmax),
                "target_support_complete": True,
                "draft_support_complete": bool(q),
            }
            if (
                self.draft_hidden_states is not None
                and node < self.draft_hidden_states.shape[0]
            ):
                state["draft_hidden"] = _tensor_payload(
                    self.draft_hidden_states[node]
                )
            if (
                self.target_hidden_states is not None
                and node < self.target_hidden_states.shape[0]
            ):
                state["target_hidden"] = _tensor_payload(
                    self.target_hidden_states[node]
                )
            states.append(state)

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
            "origin_input_length": self.origin_input_length,
            # Overlap scheduling can leave req.output_ids one or more device
            # decisions behind. This is provenance only; reconstruct exact
            # committed positions by accumulating realized_accept_length in
            # cycle order for each request.
            "scheduler_output_length_snapshot": self.committed_output_length,
            "realized_accept_length": (
                int(self.accept_lens.reshape(-1)[0].item())
                if self.accept_lens is not None
                else None
            ),
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
                "draft_distribution_source": (
                    "verifier_exact_q"
                    if self.exact_draft_probs is not None
                    else "reconstructed_from_raw_logits"
                ),
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
        self.submitted = 0
        self._submit_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.queue: queue.Queue[Optional[BranchPQCapture]] = queue.Queue(
            maxsize=min(max_cycles, 64)
        )
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"pq-capture-{self.path.name}",
            daemon=True,
        )
        self.thread.start()

    def submit(self, capture: BranchPQCapture) -> None:
        deadline = time.monotonic() + 30.0
        if not self._submit_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise TimeoutError(
                "p/q capture writer admission stayed blocked for 30 seconds"
            )
        try:
            if self.error is not None:
                raise RuntimeError("p/q capture writer failed") from self.error
            if self.submitted >= self.max_cycles:
                return
            # Diagnostic capture is default-off. Bound memory at 64 pending
            # cycles and slow the diagnostic request rather than crashing the
            # scheduler when serialization cannot keep up with GPU decode.
            while True:
                if self.error is not None:
                    raise RuntimeError("p/q capture writer failed") from self.error
                if self._stop_event.is_set():
                    raise RuntimeError("p/q capture writer stopped")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "p/q capture writer stayed backpressured for 30 seconds"
                    )
                try:
                    self.queue.put(capture, timeout=0.1)
                    self.submitted += 1
                    return
                except queue.Full:
                    continue
        finally:
            self._submit_lock.release()

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as output:
                while not self._stop_event.is_set() or not self.queue.empty():
                    try:
                        capture = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    try:
                        if self.count < self.max_cycles:
                            record = capture._cycle_record(self.count)
                            output.write(
                                json.dumps(
                                    record, sort_keys=True, separators=(",", ":")
                                )
                                + "\n"
                            )
                            output.flush()
                            self.count += 1
                    finally:
                        self.queue.task_done()
        except BaseException as exc:
            self.error = exc
        finally:
            self._stop_event.set()
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self.queue.task_done()

    def close(self, timeout: float = 10.0) -> None:
        with self._submit_lock:
            self._stop_event.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            logger.error("p/q capture writer did not stop within 10 seconds: %s", self.path)
        if self.error is not None:
            logger.error(
                "p/q capture writer failed for %s",
                self.path,
                exc_info=(
                    type(self.error),
                    self.error,
                    self.error.__traceback__,
                ),
            )


_writers: dict[str, _CaptureWriter] = {}
_disabled_writer_paths: set[str] = set()
_writers_lock = threading.Lock()


def submit_capture(capture: BranchPQCapture) -> None:
    with _writers_lock:
        if capture.output_path in _disabled_writer_paths:
            return
        writer = _writers.get(capture.output_path)
        if writer is None:
            try:
                writer = _CaptureWriter(capture.output_path, capture.max_cycles)
            except Exception:
                _disabled_writer_paths.add(capture.output_path)
                logger.exception(
                    "Disabling p/q capture writer after construction failure: %s",
                    capture.output_path,
                )
                return
            _writers[capture.output_path] = writer
        elif writer.max_cycles != capture.max_cycles:
            logger.error(
                "Disabling p/q capture because max_cycles changed within one process"
            )
            _writers.pop(capture.output_path, None)
            _disabled_writer_paths.add(capture.output_path)
            writer.close(timeout=0.0)
            return
    try:
        writer.submit(capture)
    except Exception:
        logger.exception("Disabling failed p/q capture writer: %s", capture.output_path)
        with _writers_lock:
            if _writers.get(capture.output_path) is writer:
                _writers.pop(capture.output_path)
            _disabled_writer_paths.add(capture.output_path)
        writer.close(timeout=0.0)


def close_capture_writers() -> None:
    with _writers_lock:
        writers = list(_writers.values())
        _writers.clear()
        _disabled_writer_paths.clear()
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
    exact_draft_probs: Optional[torch.Tensor],
    draft_hidden_states: Optional[torch.Tensor],
    target_hidden_states: Optional[torch.Tensor],
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
        exact_draft_probs=exact_draft_probs,
        draft_hidden_states=draft_hidden_states,
        target_hidden_states=target_hidden_states,
        node_tokens=node_tokens,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        initial_counts=_base_counts_for_req(req),
        origin_input_length=len(req.origin_input_ids),
        committed_output_length=len(req.output_ids),
        penalties=_penalty_config(req),
        target_sampling=_sampling_config(req),
        draft_sampling=_sampling_config(req, top_k=draft_sampling_top_k),
        logit_bias=_logit_bias(req),
    )
