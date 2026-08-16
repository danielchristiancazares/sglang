<#
.SYNOPSIS
Initializes the native MSVC and CUDA environment used by Windows JIT kernels.
#>

[CmdletBinding()]
param(
    [string] $CudaHome = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3',
    [ValidateRange(1, 64)]
    [int] $MaxJobs = 12
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$VsWhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $VsWhere -PathType Leaf)) {
    throw "Visual Studio locator not found at $VsWhere"
}

$VsInstall = & $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsInstall) {
    throw 'A Visual Studio installation with the native C++ toolchain was not found'
}

$VsDevShell = Join-Path $VsInstall 'Common7\Tools\Launch-VsDevShell.ps1'
if (-not (Test-Path -LiteralPath $VsDevShell -PathType Leaf)) {
    throw "Visual Studio developer shell not found at $VsDevShell"
}

$ResolvedCudaHome = (Resolve-Path -LiteralPath $CudaHome).Path
& $VsDevShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation

$env:CUDA_HOME = $ResolvedCudaHome
$env:CUDA_PATH = $ResolvedCudaHome
$env:CUDA_ROOT = $ResolvedCudaHome
$env:DISTUTILS_USE_SDK = '1'
$env:MAX_JOBS = [string] $MaxJobs
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:FLASHINFER_CUDA_ARCH_LIST = '12.0'
