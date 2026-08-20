<#
.SYNOPSIS
Compiles, correctness-checks, and microbenchmarks native Qwen3.5 CUDA hot paths.
#>

[CmdletBinding()]
param(
    [ValidateRange(10, 100000)]
    [int] $Iterations = 1000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'initialize_cuda_build_env.ps1') -MaxJobs 2

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$SmokeCode = @'
import json
import sys

import torch
import torch.nn.functional as F

from sglang.kernels.ops.layernorm.norm import (
    fused_add_rmsnorm as native_fused_add_rmsnorm,
)
from sglang.srt.layers.activation import silu_and_mul as native_silu_and_mul
from sglang.srt.layers.layernorm import gemma_rmsnorm as native_gemma_rmsnorm
from sglang.srt.layers.layernorm import (
    gemma_fused_add_rmsnorm as native_gemma_fused_add_rmsnorm,
)
from sglang.srt.layers.layernorm import rmsnorm as native_rmsnorm


def elapsed_us(fn, iterations):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


iterations = int(sys.argv[1])
torch.manual_seed(1)
results = []

for rows in (1, 3):
    hidden = 17408
    activation_input = torch.randn(
        (rows, hidden * 2), device="cuda", dtype=torch.bfloat16
    )
    activation_out = torch.empty(
        (rows, hidden), device="cuda", dtype=torch.bfloat16
    )
    activation_ref = torch.empty_like(activation_out)

    def native_activation():
        native_silu_and_mul(activation_input, activation_out)

    def torch_activation():
        activation_ref.copy_(
            F.silu(activation_input[:, :hidden]) * activation_input[:, hidden:]
        )

    native_activation()
    torch_activation()
    torch.cuda.synchronize()
    results.append(
        {
            "op": "silu_mul",
            "rows": rows,
            "native_us": elapsed_us(native_activation, iterations),
            "torch_us": elapsed_us(torch_activation, iterations),
            "max_abs": float((activation_out.float() - activation_ref.float()).abs().max()),
            "exact_fraction": float((activation_out == activation_ref).float().mean()),
        }
    )

for rows in (1, 3):
    hidden = 5120
    norm_input = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    residual_base = torch.randn_like(norm_input)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    norm_out = torch.empty_like(norm_input)
    residual_native = residual_base.clone()
    input_native = norm_input.clone()
    residual_ref = residual_base.clone()
    input_ref = norm_input.clone()
    residual_gemma_native = residual_base.clone()
    input_gemma_native = norm_input.clone()
    residual_gemma_staged = residual_base.clone()
    input_gemma_staged = norm_input.clone()

    def native_norm():
        return native_rmsnorm(norm_input, weight, 1e-6)

    def torch_norm():
        normalized = norm_input.float()
        normalized *= torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        norm_out.copy_((normalized * weight.float()).to(norm_input.dtype))

    native_norm_result = native_norm()
    torch_norm()
    torch_norm_result = norm_out.clone()

    def native_gemma_norm():
        return native_gemma_rmsnorm(norm_input, weight, 1e-6)

    def torch_gemma_norm():
        normalized = norm_input.float()
        normalized *= torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        norm_out.copy_((normalized * (weight.float() + 1.0)).to(norm_input.dtype))

    native_gemma_result = native_gemma_norm()
    torch_gemma_norm()
    torch_gemma_result = norm_out.clone()

    def native_fused():
        input_native.copy_(norm_input)
        residual_native.copy_(residual_base)
        native_fused_add_rmsnorm(input_native, residual_native, weight, 1e-6)

    def torch_fused():
        input_ref.copy_(norm_input)
        residual_ref.copy_(residual_base)
        residual_ref.add_(input_ref)
        normalized = residual_ref.float()
        normalized *= torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        input_ref.copy_((normalized * weight.float()).to(input_ref.dtype))

    def native_gemma_fused():
        input_gemma_native.copy_(norm_input)
        residual_gemma_native.copy_(residual_base)
        native_gemma_fused_add_rmsnorm(
            input_gemma_native,
            residual_gemma_native,
            weight,
            1e-6,
        )

    def staged_gemma_fused():
        input_gemma_staged.copy_(norm_input)
        residual_gemma_staged.copy_(residual_base)
        residual_gemma_staged.add_(input_gemma_staged)
        input_gemma_staged.copy_(
            native_gemma_rmsnorm(residual_gemma_staged, weight, 1e-6)
        )

    native_fused()
    torch_fused()
    native_gemma_fused()
    staged_gemma_fused()
    torch.cuda.synchronize()
    results.append(
        {
            "op": "rmsnorm",
            "rows": rows,
            "native_us": elapsed_us(native_norm, iterations),
            "torch_us": elapsed_us(torch_norm, iterations),
            "max_abs": float(
                (native_norm_result.float() - torch_norm_result.float()).abs().max()
            ),
        }
    )
    results.append(
        {
            "op": "gemma_rmsnorm",
            "rows": rows,
            "native_us": elapsed_us(native_gemma_norm, iterations),
            "torch_us": elapsed_us(torch_gemma_norm, iterations),
            "max_abs": float(
                (native_gemma_result.float() - torch_gemma_result.float()).abs().max()
            ),
        }
    )
    results.append(
        {
            "op": "fused_add_rmsnorm",
            "rows": rows,
            "native_us": elapsed_us(native_fused, iterations),
            "torch_us": elapsed_us(torch_fused, iterations),
            "input_max_abs": float((input_native.float() - input_ref.float()).abs().max()),
            "residual_exact": bool(torch.equal(residual_native, residual_ref)),
        }
    )
    results.append(
        {
            "op": "gemma_fused_add_rmsnorm_direct_out",
            "rows": rows,
            "native_us": elapsed_us(native_gemma_fused, iterations),
            "staged_us": elapsed_us(staged_gemma_fused, iterations),
            "input_exact": bool(
                torch.equal(input_gemma_native, input_gemma_staged)
            ),
            "residual_exact": bool(
                torch.equal(residual_gemma_native, residual_gemma_staged)
            ),
        }
    )

compile_input = torch.randn((1, 10240), device="cuda", dtype=torch.bfloat16)
compile_weight = torch.randn((5120,), device="cuda", dtype=torch.bfloat16)


def native_pipeline(x, weight):
    return native_rmsnorm(native_silu_and_mul(x), weight, 1e-6)


pipeline_reference = native_pipeline(compile_input, compile_weight)
compiled_pipeline = torch.compile(native_pipeline, backend="eager", fullgraph=True)
pipeline_compiled = compiled_pipeline(compile_input, compile_weight)
torch.cuda.synchronize()
results.append(
    {
        "op": "fullgraph_integration",
        "exact": bool(torch.equal(pipeline_reference, pipeline_compiled)),
    }
)

print(json.dumps(results, sort_keys=True))
'@

& $Python -c $SmokeCode $Iterations
exit $LASTEXITCODE
