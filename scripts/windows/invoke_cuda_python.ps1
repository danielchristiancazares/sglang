<#
.SYNOPSIS
Runs a Python script with the native MSVC and CUDA JIT environment loaded.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Script,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ScriptArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $PSScriptRoot 'initialize_cuda_build_env.ps1') -MaxJobs 2

$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$ResolvedScript = (Resolve-Path -LiteralPath $Script).Path
& $Python $ResolvedScript @ScriptArgs
exit $LASTEXITCODE
