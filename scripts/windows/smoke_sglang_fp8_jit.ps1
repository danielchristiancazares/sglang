<#
.SYNOPSIS
Compiles and correctness-checks SGLang's native-Windows per-token FP8 JIT kernel.
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'initialize_cuda_build_env.ps1')

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$SmokeCode = @'
import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_quant_fp8,
    static_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import fp8_scaled_mm

torch.manual_seed(1)
x = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)
quantized, scale = sglang_per_token_quant_fp8(x)
reconstructed = quantized.float() * scale
fp8_limit = torch.finfo(torch.float8_e4m3fn).max
static_scale = x.float().abs().max() / fp8_limit
static_quantized, _ = static_quant_fp8(x, static_scale)
static_reference = (
    (x.float() / static_scale)
    .clamp(min=-fp8_limit, max=fp8_limit)
    .to(torch.float8_e4m3fn)
)
weight = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
weight_scale = weight.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
weight_quantized = (
    (weight.float() / weight_scale)
    .clamp(min=-448.0, max=448.0)
    .to(torch.float8_e4m3fn)
    .t()
)
output = fp8_scaled_mm(
    quantized,
    weight_quantized,
    scale,
    weight_scale,
    torch.bfloat16,
)
reference = x @ weight.t()
single_output = fp8_scaled_mm(
    quantized[:1],
    weight_quantized,
    scale[:1],
    weight_scale,
    torch.bfloat16,
)
single_reference = x[:1] @ weight.t()
torch.cuda.synchronize()
relative_mae = (reconstructed - x.float()).abs().mean() / x.float().abs().mean()
gemm_relative_mae = (output - reference).abs().mean() / reference.abs().mean()
single_relative_mae = (
    (single_output - single_reference).abs().mean()
    / single_reference.abs().mean()
)
print(f"output_shape={tuple(quantized.shape)}")
print(f"scale_shape={tuple(scale.shape)}")
print(f"finite={bool(torch.isfinite(reconstructed).all())}")
print(f"relative_mae={float(relative_mae):.6f}")
print(f"static_exact={bool(torch.equal(static_quantized, static_reference))}")
print(f"gemm_output_shape={tuple(output.shape)}")
print(f"gemm_finite={bool(torch.isfinite(output).all())}")
print(f"gemm_relative_mae={float(gemm_relative_mae):.6f}")
print(f"single_output_shape={tuple(single_output.shape)}")
print(f"single_finite={bool(torch.isfinite(single_output).all())}")
print(f"single_relative_mae={float(single_relative_mae):.6f}")
'@

& $Python -c $SmokeCode
exit $LASTEXITCODE
