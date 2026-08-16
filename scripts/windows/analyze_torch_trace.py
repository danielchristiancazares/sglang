"""Summarize GPU activity in a gzipped PyTorch profiler trace."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path


def kernel_family(name: str) -> str:
    """Collapse verbose CUDA symbols into workload-level kernel families."""
    checks = (
        ("BLOCKSCALED", "cutlass_nvfp4_gemm"),
        ("xmmaa_gemm_e4m3", "cublas_fp8_gemm"),
        ("xmma_gemm_e4m3", "cublas_fp8_gemm"),
        ("cutlass_80_wmma_tensorop_bf16", "cutlass_bf16_gemm"),
        ("cutlass_80_tensorop_bf16", "cutlass_bf16_gemm"),
        ("internal5gemvx", "cublas_bf16_gemv"),
        ("cublasLt19splitKreduce", "cublas_splitk_reduce"),
        ("BatchPrefillWithPagedKVCacheKernel", "flashinfer_prefill"),
        ("PersistentVariableLengthMergeStatesKernel", "flashinfer_merge_states"),
        ("kernel_mha", "flashinfer_xqa"),
        ("quantize_with_block_size", "trtllm_nvfp4_quantize"),
        ("flashinfer_fp4_quantize", "flashinfer_nvfp4_quantize"),
        ("gdn_replayssm", "replayssm"),
        ("gated_delta_rule", "gdn"),
        ("causal_conv1d", "causal_conv1d"),
        ("triton_", "triton_generated"),
        ("layer_norm", "normalization"),
        ("rmsnorm", "normalization"),
        ("SoftMax", "softmax"),
        ("elementwise_kernel", "pytorch_elementwise"),
        ("index_elementwise_kernel", "pytorch_index"),
        ("direct_copy", "pytorch_copy"),
        ("CatArrayBatchedCopy", "pytorch_copy"),
        ("gpu_memcpy", "gpu_memcpy"),
    )
    for needle, family in checks:
        if needle in name:
            return family
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Case-insensitive substring to aggregate by exact event name (repeatable).",
    )
    parser.add_argument("--match-top", type=int, default=50)
    args = parser.parse_args()

    with gzip.open(args.trace, "rt", encoding="utf-8") as trace_file:
        events = json.load(trace_file)["traceEvents"]

    categories: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    kernels: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    families: dict[str, list[float | int]] = defaultdict(lambda: [0.0, 0])
    matched: dict[tuple[str, str], list[float | int]] = defaultdict(
        lambda: [0.0, 0]
    )
    graph_kernels: list[dict] = []
    match_terms = tuple(term.casefold() for term in args.match)
    for event in events:
        category = event.get("cat")
        duration_us = float(event.get("dur") or 0.0)
        name = str(event.get("name") or "<unnamed>")
        if category:
            categories[category][0] += duration_us
            categories[category][1] += 1
        if match_terms and any(term in name.casefold() for term in match_terms):
            matched[(str(category or "<none>"), name)][0] += duration_us
            matched[(str(category or "<none>"), name)][1] += 1
        if category == "kernel":
            kernels[name][0] += duration_us
            kernels[name][1] += 1
            families[kernel_family(name)][0] += duration_us
            families[kernel_family(name)][1] += 1
            if event.get("args", {}).get("graph id"):
                graph_kernels.append(event)

    total_kernel_us = sum(values[0] for values in kernels.values())
    top_kernels = sorted(kernels.items(), key=lambda item: item[1][0], reverse=True)[
        : args.top
    ]
    report = {
        "trace": str(args.trace),
        "event_count": len(events),
        "total_kernel_ms": round(total_kernel_us / 1000.0, 3),
        "categories": {
            name: {"duration_ms": round(values[0] / 1000.0, 3), "count": values[1]}
            for name, values in sorted(
                categories.items(), key=lambda item: item[1][0], reverse=True
            )
        },
        "kernel_families": {
            name: {
                "duration_ms": round(values[0] / 1000.0, 3),
                "percent_of_kernel_time": (
                    round(values[0] * 100.0 / total_kernel_us, 3)
                    if total_kernel_us
                    else 0.0
                ),
                "count": values[1],
                "mean_us": round(values[0] / values[1], 3),
            }
            for name, values in sorted(
                families.items(), key=lambda item: item[1][0], reverse=True
            )
        },
        "top_kernels": [
            {
                "name": name,
                "duration_ms": round(values[0] / 1000.0, 3),
                "percent_of_kernel_time": (
                    round(values[0] * 100.0 / total_kernel_us, 3)
                    if total_kernel_us
                    else 0.0
                ),
                "count": values[1],
                "mean_us": round(values[0] / values[1], 3),
            }
            for name, values in top_kernels
        ],
    }

    # Kernels from one graph replay are contiguous with the same graph id on
    # the captured stream. Grouping them exposes target/draft graph wall spans,
    # including gaps between kernels, instead of only summed kernel time.
    graph_replays: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    current_graph_id: int | None = None
    current_start_us = 0.0
    current_end_us = 0.0
    current_kernel_us = 0.0
    current_kernel_count = 0
    for event in sorted(graph_kernels, key=lambda item: float(item.get("ts", 0.0))):
        graph_id = int(event["args"]["graph id"])
        start_us = float(event.get("ts", 0.0))
        end_us = start_us + float(event.get("dur") or 0.0)
        if graph_id != current_graph_id:
            if current_graph_id is not None:
                graph_replays[current_graph_id].append(
                    (
                        current_end_us - current_start_us,
                        current_kernel_us,
                        current_kernel_count,
                    )
                )
            current_graph_id = graph_id
            current_start_us = start_us
            current_end_us = end_us
            current_kernel_us = 0.0
            current_kernel_count = 0
        current_end_us = max(current_end_us, end_us)
        current_kernel_us += float(event.get("dur") or 0.0)
        current_kernel_count += 1
    if current_graph_id is not None:
        graph_replays[current_graph_id].append(
            (
                current_end_us - current_start_us,
                current_kernel_us,
                current_kernel_count,
            )
        )

    report["cuda_graph_replays"] = {
        str(graph_id): {
            "count": len(replays),
            "mean_span_ms": round(
                statistics.mean(item[0] for item in replays) / 1000.0, 3
            ),
            "median_span_ms": round(
                statistics.median(item[0] for item in replays) / 1000.0, 3
            ),
            "min_span_ms": round(min(item[0] for item in replays) / 1000.0, 3),
            "max_span_ms": round(max(item[0] for item in replays) / 1000.0, 3),
            "mean_kernel_ms": round(
                statistics.mean(item[1] for item in replays) / 1000.0, 3
            ),
            "mean_kernel_count": round(
                statistics.mean(item[2] for item in replays), 1
            ),
        }
        for graph_id, replays in sorted(graph_replays.items())
    }
    if match_terms:
        report["matched_events"] = [
            {
                "category": category,
                "name": name,
                "duration_ms": round(values[0] / 1000.0, 3),
                "count": values[1],
                "mean_us": round(values[0] / values[1], 3),
            }
            for (category, name), values in sorted(
                matched.items(), key=lambda item: item[1][0], reverse=True
            )[: args.match_top]
        ]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
