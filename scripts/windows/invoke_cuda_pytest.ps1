<#
.SYNOPSIS
Runs native-Windows CUDA tests with the MSVC and CUDA JIT environment loaded.
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PytestArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'initialize_cuda_build_env.ps1') -MaxJobs 2

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
& $Python -m pytest @PytestArgs
exit $LASTEXITCODE
