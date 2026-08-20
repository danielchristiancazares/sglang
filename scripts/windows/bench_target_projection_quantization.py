"""Compare production-shaped static FP8 and NVFP4 target projections on SM120."""

from __future__ import annotations

import argparse
import json
from statistics import median

import torch
from flashinfer import fp4_quantize, mm_fp4
from flashinfer.autotuner import autotune

from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear_bmm_flashinfer,
)


SHAPES = (
    # role, M, N, K, occurrences/cycle, observed terminal ms, exclusive ms
    ("linear_attention.in_proj_qkvz", 3, 16384, 5120, 48, 2.851188, 1.675006),
    ("attention.out_proj", 3, 5120, 6144, 64, 1.483579, 1.483579),
    ("full_attention.qkv_proj", 3, 8192, 5120, 16, 0.946352, 0.946352),
)


def elapsed_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def paired_benchmark(fp8_fn, fp4_fn, iterations: int, rounds: int) -> dict:
    for _ in range(16):
        fp8_fn()
        fp4_fn()
    torch.cuda.synchronize()

    fp8_samples = []
    fp4_samples = []
    for round_index in range(rounds):
        if round_index % 2:
            fp4_samples.append(elapsed_ms(fp4_fn, iterations))
            fp8_samples.append(elapsed_ms(fp8_fn, iterations))
        else:
            fp8_samples.append(elapsed_ms(fp8_fn, iterations))
            fp4_samples.append(elapsed_ms(fp4_fn, iterations))
    return {
        "fp8_ms": median(fp8_samples),
        "fp4_ms": median(fp4_samples),
        "fp8_samples_ms": fp8_samples,
        "fp4_samples_ms": fp4_samples,
    }


def capture(fn):
    for _ in range(8):
        output = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()

    def replay():
        graph.replay()
        return output

    return replay, graph, output


def make_operands(m: int, n: int, k: int, occurrences: int):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    fp8_input_scale = x.float().abs().amax().clamp(min=1e-12) / 448.0
    fp4_input_global_scale = 448.0 * 6.0 / x.float().abs().amax().clamp(min=1e-12)

    fp8_operands = []
    fp4_operands = []
    for _ in range(occurrences):
        source_weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        fp8_weight_scale = (
            source_weight.float().abs().amax().clamp(min=1e-12) / 448.0
        )
        fp8_weight = (
            (source_weight.float() / fp8_weight_scale)
            .clamp(min=-448.0, max=448.0)
            .to(torch.float8_e4m3fn)
            .t()
        )
        fp8_operands.append((fp8_weight, fp8_weight_scale))

        fp4_weight_global_scale = (
            448.0 * 6.0 / source_weight.float().abs().amax().clamp(min=1e-12)
        )
        fp4_alpha = 1.0 / (fp4_input_global_scale * fp4_weight_global_scale)
        fp4_weight, fp4_weight_scale = fp4_quantize(
            source_weight, fp4_weight_global_scale
        )
        fp4_operands.append(
            (
                fp4_weight.t(),
                fp4_weight_scale.view(torch.float8_e4m3fn).t(),
                fp4_alpha,
                torch.empty((m, n), device="cuda", dtype=torch.bfloat16),
            )
        )
        del source_weight

    def run_fp8():
        output = None
        for fp8_weight, fp8_weight_scale in fp8_operands:
            output = apply_fp8_linear_bmm_flashinfer(
                x, fp8_weight, fp8_weight_scale, fp8_input_scale
            )
        return output

    def run_fp4():
        output = None
        for fp4_weight, fp4_weight_scale, fp4_alpha, fp4_output in fp4_operands:
            x_fp4, x_scale = fp4_quantize(x, fp4_input_global_scale)
            output = mm_fp4(
                x_fp4,
                fp4_weight,
                x_scale.view(torch.float8_e4m3fn),
                fp4_weight_scale,
                fp4_alpha,
                torch.bfloat16,
                fp4_output,
                backend="cutlass",
            )
        return output

    return run_fp8, run_fp4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    torch.manual_seed(1786950187)
    results = []
    for role, m, n, k, occurrences, observed_ms, exclusive_ms in SHAPES:
        fp8_fn, fp4_fn = make_operands(m, n, k, occurrences)
        with autotune():
            fp4_fn()
        fp8_replay, fp8_graph, fp8_output = capture(fp8_fn)
        fp4_replay, fp4_graph, fp4_output = capture(fp4_fn)
        timings = paired_benchmark(
            fp8_replay, fp4_replay, args.iterations, args.rounds
        )
        serialized_saving = timings["fp8_ms"] - timings["fp4_ms"]
        exposure_ratio = exclusive_ms / observed_ms
        results.append(
            {
                "role": role,
                "shape_mnk": [m, n, k],
                "occurrences_per_cycle": occurrences,
                **timings,
                "serialized_saving_ms": serialized_saving,
                "overlap_adjusted_saving_ms": serialized_saving * exposure_ratio,
            }
        )
        del (
            fp8_fn,
            fp4_fn,
            fp8_replay,
            fp4_replay,
            fp8_graph,
            fp4_graph,
            fp8_output,
            fp4_output,
        )
        torch.cuda.empty_cache()

    total_saving = sum(r["overlap_adjusted_saving_ms"] for r in results)
    baseline_cycle_ms = 19.446
    projected_cycle_ms = baseline_cycle_ms - total_saving
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "execution": "distinct-weight projection-family CUDA graph replay",
                "iterations": args.iterations,
                "rounds": args.rounds,
                "results": results,
                "overlap_adjusted_total_saving_ms": total_saving,
                "projected_m3_cycle_ms": projected_cycle_ms,
                "projected_m3_perfect_ceiling_tps": 3000.0 / projected_cycle_ms,
                "projected_k_plus_1_perfect_ceiling_tps": 4000.0
                / projected_cycle_ms,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
