<#
.SYNOPSIS
Compiles and correctness-checks FlashInfer's native-Windows SM120 NVFP4 path.
#>

[CmdletBinding()]
param(
    [ValidateSet('auto', 'cudnn', 'cutlass', 'cute-dsl')]
    [string] $Backend = 'cutlass'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'initialize_cuda_build_env.ps1')

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$SmokeCode = @'
import torch
import flashinfer
import sys

backend = sys.argv[1]

torch.manual_seed(1)
a = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)
w = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
a_global_scale = (448.0 * 6.0) / a.float().abs().max()
w_global_scale = (448.0 * 6.0) / w.float().abs().max()
a_fp4, a_scale = flashinfer.fp4_quantize(a, a_global_scale)
w_fp4, w_scale = flashinfer.fp4_quantize(w, w_global_scale)
output = flashinfer.mm_fp4(
    a_fp4,
    w_fp4.T,
    a_scale,
    w_scale.T,
    1.0 / (a_global_scale * w_global_scale),
    torch.bfloat16,
    backend=backend,
)
reference = a @ w.T
torch.cuda.synchronize()
relative_mae = (output - reference).abs().mean() / reference.abs().mean()
print(f"flashinfer={flashinfer.__version__}")
print(f"backend={backend}")
print(f"output_shape={tuple(output.shape)}")
print(f"finite={bool(torch.isfinite(output).all())}")
print(f"relative_mae={float(relative_mae):.6f}")
'@

& $Python -c $SmokeCode $Backend
exit $LASTEXITCODE
