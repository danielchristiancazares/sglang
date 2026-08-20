#!/usr/bin/env python3
"""Attribute target CUDA-graph GEMMs to exact Qwen3.5 model shapes.

PyTorch profiler reports CUDA-graph kernels with stable graph IDs, launch
geometry, stream, and timestamps, but it does not retain GEMM problem M/N/K in
the replay event.  For Qwen3.5 dense checkpoints, the ordered projection list
is fixed by the model configuration.  This analyzer matches that list against
every replay independently and fails the model-role attribution closed if any
kernel-family count differs.

The report keeps summed kernel residency separate from wall exposure.  For
each launch shape and inferred problem shape it records:

* aggregate kernel time, which can exceed graph wall time when streams overlap;
* union wall coverage across all streams;
* serialized residency on the stream containing the graph's terminal kernel;
* exclusive wall exposure, the interval union lost when that shape's kernels
  are removed from the observed schedule.

The latter two are complementary critical-path evidence.  They do not claim a
counterfactual speedup without a remeasurement of the modified graph.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

from analyze_torch_trace import kernel_family


PRIMARY_GEMM_FAMILIES = frozenset(
    {
        "cutlass_nvfp4_gemm",
        "cublas_fp8_gemm",
        "cutlass_bf16_gemm",
        "cublas_bf16_gemv",
    }
)
BF16_GEMM_FAMILIES = frozenset({"cutlass_bf16_gemm", "cublas_bf16_gemv"})


class AttributionError(ValueError):
    """Raised when a trace cannot support exact model-role attribution."""


@dataclass(frozen=True)
class GraphRun:
    graph_id: int
    correlation: int | str
    start_us: float
    end_us: float
    events: tuple[Mapping[str, Any], ...]

    @property
    def span_us(self) -> float:
        return self.end_us - self.start_us

    @property
    def completion_stream(self) -> int | str:
        terminal = max(
            self.events,
            key=lambda event: (
                _event_end_us(event),
                _event_start_us(event),
                str(event.get("name", "")),
            ),
        )
        return terminal.get("args", {}).get("stream", "unknown")


@dataclass(frozen=True)
class GemmRole:
    model_role: str
    layer_id: int | None
    family_group: str
    accepted_families: tuple[str, ...]
    m: int
    n: int
    k: int

    @property
    def problem_key(self) -> tuple[str, int, int, int]:
        return (self.family_group, self.m, self.n, self.k)


def _event_start_us(event: Mapping[str, Any]) -> float:
    return float(event.get("ts", 0.0))


def _event_duration_us(event: Mapping[str, Any]) -> float:
    return float(event.get("dur") or 0.0)


def _event_end_us(event: Mapping[str, Any]) -> float:
    return _event_start_us(event) + _event_duration_us(event)


def _event_intervals(
    events: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    return [
        (_event_start_us(event), _event_end_us(event))
        for event in events
        if _event_duration_us(event) > 0.0
    ]


def interval_union_us(intervals: Sequence[tuple[float, float]]) -> float:
    """Return the exact union length of half-open timestamp intervals."""
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def graph_runs_from_events(events: Sequence[Mapping[str, Any]]) -> list[GraphRun]:
    """Group graph kernels by CUDA graph launch correlation and graph ID."""
    grouped: dict[tuple[int, int | str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for event in events:
        args = event.get("args", {})
        graph_id = args.get("graph id")
        if event.get("cat") != "kernel" or graph_id is None:
            continue
        graph_id = int(graph_id)
        if graph_id <= 0:
            # CUPTI uses graph ID zero for ordinary launches outside a CUDA
            # graph.  Their per-kernel correlations are not graph replays.
            continue
        correlation = args.get("correlation")
        if correlation is None:
            raise AttributionError(
                "graph kernel lacks a CUDA launch correlation; replay boundaries "
                "cannot be established safely"
            )
        grouped[(graph_id, correlation)].append(event)

    runs = []
    for (graph_id, correlation), run_events in grouped.items():
        ordered = tuple(
            sorted(
                run_events,
                key=lambda event: (
                    _event_start_us(event),
                    _event_end_us(event),
                    str(event.get("name", "")),
                ),
            )
        )
        runs.append(
            GraphRun(
                graph_id=graph_id,
                correlation=correlation,
                start_us=min(_event_start_us(event) for event in ordered),
                end_us=max(_event_end_us(event) for event in ordered),
                events=ordered,
            )
        )
    return sorted(runs, key=lambda run: (run.start_us, run.graph_id, str(run.correlation)))


def dominant_graph_id(runs: Sequence[GraphRun]) -> int:
    by_graph: dict[int, list[GraphRun]] = defaultdict(list)
    for run in runs:
        by_graph[run.graph_id].append(run)
    candidates = {graph_id: items for graph_id, items in by_graph.items() if items}
    if not candidates:
        raise AttributionError("trace contains no CUDA graph kernel replays")
    return max(
        candidates,
        key=lambda graph_id: (
            statistics.median(run.span_us for run in candidates[graph_id]),
            len(candidates[graph_id]),
            -graph_id,
        ),
    )


def _sequence_stats_us(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "mean_ms": statistics.fmean(values) / 1000.0,
        "median_ms": statistics.median(values) / 1000.0,
        "min_ms": min(values) / 1000.0,
        "max_ms": max(values) / 1000.0,
    }


def summarize_event_group(
    runs: Sequence[GraphRun],
    selected: Callable[[int, Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """Summarize one kernel group over every replay, preserving overlap."""
    aggregate_per_run = []
    coverage_per_run = []
    completion_stream_per_run = []
    exclusive_per_run = []
    counts = []
    total_aggregate_us = 0.0
    completion_streams: Counter[str] = Counter()

    for run_index, run in enumerate(runs):
        chosen = [event for event in run.events if selected(run_index, event)]
        others = [event for event in run.events if not selected(run_index, event)]
        aggregate_us = math.fsum(_event_duration_us(event) for event in chosen)
        coverage_us = interval_union_us(_event_intervals(chosen))
        all_coverage_us = interval_union_us(_event_intervals(run.events))
        other_coverage_us = interval_union_us(_event_intervals(others))
        exclusive_us = max(0.0, all_coverage_us - other_coverage_us)
        completion_stream = run.completion_stream
        completion_streams[str(completion_stream)] += 1
        completion_us = interval_union_us(
            _event_intervals(
                [
                    event
                    for event in chosen
                    if event.get("args", {}).get("stream", "unknown")
                    == completion_stream
                ]
            )
        )

        counts.append(len(chosen))
        aggregate_per_run.append(aggregate_us)
        coverage_per_run.append(coverage_us)
        completion_stream_per_run.append(completion_us)
        exclusive_per_run.append(exclusive_us)
        total_aggregate_us += aggregate_us

    return {
        "replay_count": len(runs),
        "kernel_count": sum(counts),
        "kernels_per_replay": {
            "mean": statistics.fmean(counts) if counts else 0.0,
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
        },
        "aggregate_kernel_time": {
            "total_trace_ms": total_aggregate_us / 1000.0,
            **_sequence_stats_us(aggregate_per_run),
        },
        "all_stream_wall_coverage": _sequence_stats_us(coverage_per_run),
        "critical_path_exposure": {
            "completion_stream_serialized": _sequence_stats_us(
                completion_stream_per_run
            ),
            "exclusive_observed_wall": _sequence_stats_us(exclusive_per_run),
            "completion_stream_replay_counts": dict(
                sorted(completion_streams.items())
            ),
        },
    }


def _tuple_arg(event: Mapping[str, Any], name: str) -> tuple[int, ...]:
    value = event.get("args", {}).get(name, [])
    if not isinstance(value, list):
        return ()
    return tuple(int(item) for item in value)


def _kernel_variant(name: str) -> str:
    if "BLOCKSCALED" in name:
        return "cutlass_sm120_blockscaled"
    for pattern in (
        r"tilesize[^_]+(?:_stage\d+)?",
        r"cutlass_80_[A-Za-z0-9_]+?align\d+",
    ):
        match = re.search(pattern, name)
        if match:
            return match.group(0)
    return name[:120]


def launch_shape_key(event: Mapping[str, Any]) -> tuple[Hashable, ...]:
    name = str(event.get("name") or "<unnamed>")
    args = event.get("args", {})
    return (
        kernel_family(name),
        hashlib.sha256(name.encode("utf-8")).hexdigest()[:16],
        _tuple_arg(event, "grid"),
        _tuple_arg(event, "block"),
        int(args.get("shared memory") or 0),
    )


def launch_shape_report(runs: Sequence[GraphRun]) -> list[dict[str, Any]]:
    examples: dict[tuple[Hashable, ...], Mapping[str, Any]] = {}
    keys: set[tuple[Hashable, ...]] = set()
    for run in runs:
        for event in run.events:
            family = kernel_family(str(event.get("name") or ""))
            if family not in PRIMARY_GEMM_FAMILIES:
                continue
            key = launch_shape_key(event)
            keys.add(key)
            examples.setdefault(key, event)

    result = []
    for key in keys:
        family, symbol_sha256, grid, block, shared_memory = key
        example = examples[key]
        metrics = summarize_event_group(
            runs, lambda _run_index, event, expected=key: launch_shape_key(event) == expected
        )
        result.append(
            {
                "launch_shape_id": (
                    f"{family}:{symbol_sha256}:grid={list(grid)}:block={list(block)}"
                ),
                "kernel_family": family,
                "kernel_variant": _kernel_variant(str(example.get("name") or "")),
                "kernel_symbol_sha256": symbol_sha256,
                "grid": list(grid),
                "block": list(block),
                "dynamic_shared_memory_bytes": shared_memory,
                **metrics,
            }
        )
    return sorted(
        result,
        key=lambda item: (
            -item["aggregate_kernel_time"]["total_trace_ms"],
            item["launch_shape_id"],
        ),
    )


def kernel_family_report(runs: Sequence[GraphRun]) -> list[dict[str, Any]]:
    families = sorted(
        {
            kernel_family(str(event.get("name") or ""))
            for run in runs
            for event in run.events
        }
    )
    result = []
    for family in families:
        metrics = summarize_event_group(
            runs,
            lambda _run_index, event, wanted=family: kernel_family(
                str(event.get("name") or "")
            )
            == wanted,
        )
        result.append({"kernel_family": family, **metrics})
    return sorted(
        result,
        key=lambda item: -item["aggregate_kernel_time"]["total_trace_ms"],
    )


def graph_summary(runs: Sequence[GraphRun]) -> dict[str, Any]:
    spans = [run.span_us for run in runs]
    aggregate = [
        math.fsum(_event_duration_us(event) for event in run.events) for run in runs
    ]
    return {
        "replay_count": len(runs),
        "span": _sequence_stats_us(spans),
        "aggregate_all_kernel_time": _sequence_stats_us(aggregate),
        "completion_stream_replay_counts": dict(
            sorted(Counter(str(run.completion_stream) for run in runs).items())
        ),
        "kernel_families": kernel_family_report(runs),
    }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttributionError(f"{name} must be a positive integer")
    return value


def qwen35_target_roles(
    config_document: Mapping[str, Any], graph_width: int, tensor_parallel_size: int
) -> list[GemmRole]:
    """Build the ordered primary-GEMM contract for one dense Qwen3.5 replay."""
    if tensor_parallel_size != 1:
        raise AttributionError(
            "exact Qwen3.5 role attribution currently requires tensor parallel size 1"
        )
    if graph_width <= 0:
        raise AttributionError("graph width must be positive")
    text = config_document.get("text_config", config_document)
    if not isinstance(text, dict) or text.get("model_type") != "qwen3_5_text":
        raise AttributionError("model config must describe dense qwen3_5_text")

    hidden = _positive_int(text.get("hidden_size"), "hidden_size")
    intermediate = _positive_int(
        text.get("intermediate_size"), "intermediate_size"
    )
    vocab = _positive_int(text.get("vocab_size"), "vocab_size")
    head_dim = _positive_int(text.get("head_dim"), "head_dim")
    attention_heads = _positive_int(
        text.get("num_attention_heads"), "num_attention_heads"
    )
    kv_heads = _positive_int(text.get("num_key_value_heads"), "num_key_value_heads")
    linear_k_heads = _positive_int(
        text.get("linear_num_key_heads"), "linear_num_key_heads"
    )
    linear_v_heads = _positive_int(
        text.get("linear_num_value_heads"), "linear_num_value_heads"
    )
    linear_k_dim = _positive_int(
        text.get("linear_key_head_dim"), "linear_key_head_dim"
    )
    linear_v_dim = _positive_int(
        text.get("linear_value_head_dim"), "linear_value_head_dim"
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise AttributionError("qwen3_5_text config must provide nonempty layer_types")
    if len(layer_types) != _positive_int(
        text.get("num_hidden_layers"), "num_hidden_layers"
    ):
        raise AttributionError("layer_types length does not match num_hidden_layers")

    quantization = config_document.get("quantization_config", {})
    groups = quantization.get("config_groups", {}) if isinstance(quantization, dict) else {}
    weight_bits = {
        group.get("weights", {}).get("num_bits")
        for group in groups.values()
        if isinstance(group, dict) and isinstance(group.get("weights"), dict)
    }
    if not {4, 8}.issubset(weight_bits):
        raise AttributionError(
            "model config must retain the measured FP8 projection and NVFP4 MLP groups"
        )

    linear_key = linear_k_heads * linear_k_dim
    linear_value = linear_v_heads * linear_v_dim
    full_q = attention_heads * head_dim
    full_kv = kv_heads * head_dim
    roles: list[GemmRole] = []
    for layer_id, layer_type in enumerate(layer_types):
        if layer_type == "linear_attention":
            roles.extend(
                [
                    GemmRole(
                        "linear_attention.in_proj_qkvz",
                        layer_id,
                        "fp8",
                        ("cublas_fp8_gemm",),
                        graph_width,
                        2 * linear_key + 2 * linear_value,
                        hidden,
                    ),
                    GemmRole(
                        "linear_attention.in_proj_ba",
                        layer_id,
                        "bf16",
                        tuple(sorted(BF16_GEMM_FAMILIES)),
                        graph_width,
                        2 * linear_v_heads,
                        hidden,
                    ),
                    GemmRole(
                        "linear_attention.out_proj",
                        layer_id,
                        "fp8",
                        ("cublas_fp8_gemm",),
                        graph_width,
                        hidden,
                        linear_value,
                    ),
                ]
            )
        elif layer_type == "full_attention":
            roles.extend(
                [
                    GemmRole(
                        "full_attention.qkv_proj",
                        layer_id,
                        "fp8",
                        ("cublas_fp8_gemm",),
                        graph_width,
                        full_q + 2 * full_kv,
                        hidden,
                    ),
                    GemmRole(
                        "full_attention.o_proj",
                        layer_id,
                        "fp8",
                        ("cublas_fp8_gemm",),
                        graph_width,
                        hidden,
                        full_q,
                    ),
                ]
            )
        else:
            raise AttributionError(f"unsupported Qwen3.5 layer type {layer_type!r}")
        roles.extend(
            [
                GemmRole(
                    "mlp.gate_up_proj",
                    layer_id,
                    "nvfp4",
                    ("cutlass_nvfp4_gemm",),
                    graph_width,
                    2 * intermediate,
                    hidden,
                ),
                GemmRole(
                    "mlp.down_proj",
                    layer_id,
                    "nvfp4",
                    ("cutlass_nvfp4_gemm",),
                    graph_width,
                    hidden,
                    intermediate,
                ),
            ]
        )
    roles.append(
        GemmRole(
            "lm_head",
            None,
            "nvfp4",
            ("cutlass_nvfp4_gemm",),
            graph_width,
            vocab,
            hidden,
        )
    )
    return roles


def _roles_by_family_group(roles: Sequence[GemmRole]) -> dict[str, list[GemmRole]]:
    result: dict[str, list[GemmRole]] = defaultdict(list)
    for role in roles:
        result[role.family_group].append(role)
    return result


def _actual_family_group(event: Mapping[str, Any]) -> str | None:
    family = kernel_family(str(event.get("name") or ""))
    if family == "cublas_fp8_gemm":
        return "fp8"
    if family == "cutlass_nvfp4_gemm":
        return "nvfp4"
    if family in BF16_GEMM_FAMILIES:
        return "bf16"
    return None


def qwen35_role_attribution(
    runs: Sequence[GraphRun], roles: Sequence[GemmRole]
) -> dict[str, Any]:
    """Match ordered primary kernels in every replay, failing closed on drift."""
    expected = _roles_by_family_group(roles)
    assignments: dict[tuple[int, int], GemmRole] = {}
    observed_counts = []
    for run_index, run in enumerate(runs):
        run_counts: dict[str, int] = {}
        for family_group, expected_roles in expected.items():
            actual = [
                event
                for event in run.events
                if _actual_family_group(event) == family_group
            ]
            run_counts[family_group] = len(actual)
            if len(actual) != len(expected_roles):
                raise AttributionError(
                    f"graph {run.graph_id} correlation {run.correlation} has "
                    f"{len(actual)} {family_group} primary GEMMs; model contract "
                    f"requires {len(expected_roles)}"
                )
            for event, role in zip(actual, expected_roles):
                family = kernel_family(str(event.get("name") or ""))
                if family not in role.accepted_families:
                    raise AttributionError(
                        f"model role {role.model_role} expected {role.accepted_families}, "
                        f"observed {family}"
                    )
                assignments[(run_index, id(event))] = role
        observed_counts.append(run_counts)

    def assigned_role(run_index: int, event: Mapping[str, Any]) -> GemmRole | None:
        return assignments.get((run_index, id(event)))

    role_names = sorted({role.model_role for role in roles})
    role_report = []
    for role_name in role_names:
        matching_roles = [role for role in roles if role.model_role == role_name]
        shape_set = sorted({(role.m, role.n, role.k) for role in matching_roles})
        metrics = summarize_event_group(
            runs,
            lambda run_index, event, wanted=role_name: (
                (role := assigned_role(run_index, event)) is not None
                and role.model_role == wanted
            ),
        )
        launch_ids = sorted(
            {
                launch_shape_key(event)
                for run_index, run in enumerate(runs)
                for event in run.events
                if (role := assigned_role(run_index, event)) is not None
                and role.model_role == role_name
            },
            key=str,
        )
        role_report.append(
            {
                "model_role": role_name,
                "layers_per_replay": len(matching_roles),
                "problem_shapes_mnk": [
                    {"m": m, "n": n, "k": k} for m, n, k in shape_set
                ],
                "launch_shape_ids": [
                    f"{family}:{digest}:grid={list(grid)}:block={list(block)}"
                    for family, digest, grid, block, _shared in launch_ids
                ],
                **metrics,
            }
        )

    problem_keys = sorted({role.problem_key for role in roles}, key=str)
    problem_report = []
    for family_group, m, n, k in problem_keys:
        matching_roles = [role for role in roles if role.problem_key == (family_group, m, n, k)]
        metrics = summarize_event_group(
            runs,
            lambda run_index, event, wanted=(family_group, m, n, k): (
                (role := assigned_role(run_index, event)) is not None
                and role.problem_key == wanted
            ),
        )
        problem_report.append(
            {
                "family_group": family_group,
                "m": m,
                "n": n,
                "k": k,
                "model_roles": sorted({role.model_role for role in matching_roles}),
                "occurrences_per_replay": len(matching_roles),
                **metrics,
            }
        )

    all_primary = summarize_event_group(
        runs,
        lambda run_index, event: assigned_role(run_index, event) is not None,
    )
    return {
        "status": "exact",
        "matching_rule": (
            "ordered Qwen3.5 projection contract matched independently within "
            "each primary GEMM family group on every replay"
        ),
        "expected_primary_gemms_per_replay": dict(
            sorted((group, len(items)) for group, items in expected.items())
        ),
        "observed_primary_gemms_per_replay": observed_counts,
        "all_primary_gemms": all_primary,
        "by_model_role": sorted(
            role_report,
            key=lambda item: -item["aggregate_kernel_time"]["total_trace_ms"],
        ),
        "by_problem_shape": sorted(
            problem_report,
            key=lambda item: -item["aggregate_kernel_time"]["total_trace_ms"],
        ),
    }


def analyze(
    events: Sequence[Mapping[str, Any]],
    config_document: Mapping[str, Any],
    graph_width: int,
    graph_id: int | None = None,
    tensor_parallel_size: int = 1,
) -> dict[str, Any]:
    all_runs = graph_runs_from_events(events)
    selected_graph_id = dominant_graph_id(all_runs) if graph_id is None else graph_id
    runs = [run for run in all_runs if run.graph_id == selected_graph_id]
    if not runs:
        raise AttributionError(f"trace has no replay for graph ID {selected_graph_id}")

    spans = [run.span_us for run in runs]
    starts = [run.start_us for run in runs]
    full_cycle_samples_ms = [
        (current - previous) / 1000.0
        for previous, current in zip(starts, starts[1:])
        if current > previous
    ]
    graph_kernel_aggregate = [
        math.fsum(_event_duration_us(event) for event in run.events) for run in runs
    ]
    roles = qwen35_target_roles(config_document, graph_width, tensor_parallel_size)
    try:
        role_attribution = qwen35_role_attribution(runs, roles)
    except AttributionError as exc:
        role_attribution = {
            "status": "unavailable",
            "reason": str(exc),
            "fail_closed": True,
        }

    return {
        "selected_graph_id": selected_graph_id,
        "graph_width": graph_width,
        "tensor_parallel_size": tensor_parallel_size,
        "replay_count": len(runs),
        "graph_span": _sequence_stats_us(spans),
        "full_cycle_start_to_start": {
            "method": "selected_target_graph_start_to_start",
            "samples_ms": full_cycle_samples_ms,
            "count": len(full_cycle_samples_ms),
            **_sequence_stats_us(
                [sample_ms * 1000.0 for sample_ms in full_cycle_samples_ms]
            ),
        },
        "aggregate_all_kernel_time": _sequence_stats_us(graph_kernel_aggregate),
        "completion_stream_replay_counts": dict(
            sorted(Counter(str(run.completion_stream) for run in runs).items())
        ),
        "all_cuda_graphs": {
            str(current_graph_id): graph_summary(
                [run for run in all_runs if run.graph_id == current_graph_id]
            )
            for current_graph_id in sorted({run.graph_id for run in all_runs})
        },
        "launch_shapes": launch_shape_report(runs),
        "qwen35_model_role_attribution": role_attribution,
        "critical_path_semantics": {
            "completion_stream_serialized": (
                "union residency for the shape on the stream containing the "
                "terminal kernel of each replay"
            ),
            "exclusive_observed_wall": (
                "all-kernel interval union minus the interval union after the "
                "shape is removed; a conservative directly exposed wall-time measure"
            ),
            "remeasurement_required": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--graph-width", type=int, required=True)
    parser.add_argument("--graph-id", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        trace_bytes = args.trace.read_bytes()
        config_bytes = args.model_config.read_bytes()
        with gzip.open(args.trace, "rt", encoding="utf-8") as trace_file:
            events = json.load(trace_file)["traceEvents"]
        config_document = json.loads(config_bytes)
        report = analyze(
            events,
            config_document,
            graph_width=args.graph_width,
            graph_id=args.graph_id,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError, AttributionError) as exc:
        parser.error(str(exc))
    report = {
        "schema_version": 1,
        "artifact_type": "target_graph_gemm_attribution",
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "model_config": str(args.model_config),
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        **report,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
