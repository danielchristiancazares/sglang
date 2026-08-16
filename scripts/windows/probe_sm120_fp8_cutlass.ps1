<#
.SYNOPSIS
Builds and benchmarks the isolated SM120 CUTLASS channelwise FP8 GEMM.
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'initialize_cuda_build_env.ps1')

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Probe = Join-Path $PSScriptRoot 'probe_sm120_fp8_cutlass.py'
& $Python $Probe
exit $LASTEXITCODE
