<#
.SYNOPSIS
Starts the native NVFP4 Qwen3.8-27B checkpoint through Windows SGLang.

.DESCRIPTION
The measured defaults select the qualified RadixArk checkpoint, an exact 200K
token pool, FlashInfer prefill, TRT-LLM/XQA target and draft decode, its
two-step bundled MTP head, FP4-only FlashInfer autotuning, FlashInfer
sampling, FP8 draft KV, the minimum safe 128 MiB workspace, and partial torch
compilation.
The checkpoint is loaded as a standalone language model, preserving VRAM that
the unused vision encoder would otherwise consume.
#>

[CmdletBinding()]
param(
    [string] $ModelPath = (Join-Path $env:USERPROFILE 'models\Qwen3.8-27B-NVFP4-RadixArk'),
    [string] $ServedModelName = 'qwen3.8-27b',
    [string] $ListenAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int] $Port = 30000,
    [ValidateRange(-1, 2147483647)]
    [int] $RandomSeed = -1,
    [ValidateRange(2048, 262144)]
    [int] $ContextLength = 200000,
    [ValidateRange(2048, 262144)]
    [int] $MaxTotalTokens = 200000,
    [ValidateRange(0.1, 0.98)]
    [double] $MemoryFraction = 0.94,
    [ValidateSet('auto', 'fp8_e4m3', 'bf16', 'bfloat16', 'nvfp4', 'fp4_mx_block16')]
    [string] $KvCacheDtype = 'auto',
    [ValidateSet(1, 8, 16, 32, 64, 128)]
    [int] $PageSize = 64,
    [ValidateRange(1, 1024)]
    [int] $MaxMambaCacheSize = 4,
    [ValidateSet('no_buffer', 'extra_buffer', 'extra_buffer_lazy')]
    [string] $MambaRadixCacheStrategy = 'extra_buffer_lazy',
    [ValidateSet('float32', 'bfloat16', 'float16')]
    [string] $MambaSsmDtype = 'float32',
    [ValidateRange(256, 16384)]
    [int] $ChunkedPrefillSize = 4096,
    [ValidateRange(1, 64)]
    [int] $TritonAttentionNumKvSplits = 16,
    [ValidateRange(1, 16)]
    [int] $CudaGraphMaxBatchSize = 1,
    [ValidateRange(1, 16)]
    [int] $MaxRunningRequests = 1,
    [ValidateRange(1, 256)]
    [int] $SchedulerRecvInterval = 4,
    [ValidateRange(1, 256)]
    [int] $StreamInterval = 4,
    [ValidateSet('triton', 'flashinfer')]
    [string] $AttentionBackend = 'triton',
    [ValidateSet('triton', 'flashinfer', 'trtllm_mha')]
    [string] $PrefillAttentionBackend = 'flashinfer',
    [ValidateSet('triton', 'flashinfer', 'trtllm_mha')]
    [string] $DecodeAttentionBackend = 'trtllm_mha',
    [ValidateSet('triton', 'flashinfer', 'trtllm_mha')]
    [string] $SpeculativeDraftAttentionBackend = 'trtllm_mha',
    [ValidateSet('prefill', 'decode')]
    [string] $SpeculativeAttentionMode = 'decode',
    [ValidateSet('', 'unquant', 'fp8', 'mxfp8', 'nvfp4_online')]
    [string] $SpeculativeDraftModelQuantization = '',
    [ValidateSet('pytorch', 'flashinfer')]
    [string] $SamplingBackend = 'flashinfer',
    [ValidateSet('auto', 'flashinfer_cutlass', 'flashinfer_cutedsl', 'flashinfer_trtllm')]
    [string] $Fp8GemmBackend = 'flashinfer_cutlass',
    [ValidateSet('auto', 'fp8_e4m3', 'bf16', 'bfloat16', 'nvfp4', 'fp4_mx_block16')]
    [string] $SpeculativeDraftKvCacheDtype = 'fp8_e4m3',
    [ValidateSet('triton', 'flashinfer', 'cutedsl')]
    [string] $LinearAttentionDecodeBackend = 'triton',
    [ValidateSet('triton', 'flashinfer', 'cutedsl')]
    [string] $LinearAttentionPrefillBackend = 'triton',
    [ValidateSet('triton', 'flashinfer', 'cutedsl')]
    [string] $LinearAttentionVerifyBackend = 'triton',
    [ValidateSet('auto', 'flashinfer_cutlass', 'flashinfer_cudnn')]
    [string] $Fp4GemmBackend = 'flashinfer_cutlass',
    [ValidateRange(0, 8)]
    [int] $SpeculativeNumSteps = 2,
    [ValidateRange(1, 16)]
    [int] $SpeculativeNumDraftTokens = 3,
    [ValidateRange(1, 16)]
    [int] $SpeculativeEagleTopK = 1,
    [switch] $SpeculativeUseRejectionSampling = $true,
    [switch] $SpeculativeDeviceResidentCycle,
    [ValidateSet(0, 4, 8, 16, 20, 32, 64)]
    [int] $SpeculativeDraftSamplingTopK = 20,
    [switch] $SpeculativeAlignTreeScoring,
    [ValidateSet('target_only', 'swor')]
    [string] $SpeculativeTreeSamplingMode = 'target_only',
    [switch] $SpeculativeSworCollectPathStats,
    [switch] $SpeculativeSworCollectOverlapStats,
    [string] $SpeculativeSworTopology = '',
    [ValidateRange(0.05, 4.0)]
    [double] $SpeculativeTreeDepthDiscount = 1.0,
    [switch] $SpeculativeAdaptive,
    [string] $SpeculativeAdaptiveConfig = '',
    [switch] $EnableFlashInferAutotune = $true,
    [string[]] $FlashInferAutotuneSkipOps = @('fp8_gemm'),
    [ValidateSet('default', 'max-autotune-no-cudagraphs')]
    [string] $TorchCompileMode = 'default',
    [ValidateRange(0, 4096)]
    [int] $FlashInferWorkspaceSizeMB = 128,
    [ValidateRange(0, 16)]
    [int] $SimulateAcceptedLength = 0,
    [switch] $DisableTorchCompile,
    [switch] $DisableIncrementalStreamingOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($MaxTotalTokens -gt $ContextLength) {
    throw "MaxTotalTokens ($MaxTotalTokens) cannot exceed ContextLength ($ContextLength)"
}

$ResolvedModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$SGLang = Join-Path $RepoRoot '.venv\Scripts\sglang.exe'

if (-not (Test-Path -LiteralPath $SGLang -PathType Leaf)) {
    throw "Native SGLang launcher not found at $SGLang"
}

. (Join-Path $PSScriptRoot 'initialize_cuda_build_env.ps1')

$ServeArgs = @(
    'serve'
    '--model-path', $ResolvedModelPath
    '--served-model-name', $ServedModelName
    '--host', $ListenAddress
    '--port', $Port
    '--language-model-only'
    '--reasoning-parser', 'qwen3'
    '--tool-call-parser', 'qwen3_coder'
    '--attention-backend', $AttentionBackend
    '--prefill-attention-backend', $PrefillAttentionBackend
    '--decode-attention-backend', $DecodeAttentionBackend
    '--sampling-backend', $SamplingBackend
    '--fp8-gemm-backend', $Fp8GemmBackend
    '--fp4-gemm-backend', $Fp4GemmBackend
    '--kv-cache-dtype', $KvCacheDtype
    '--page-size', $PageSize
    '--disable-custom-all-reduce'
    '--cuda-graph-backend-decode', 'full'
    '--cuda-graph-max-bs-decode', $CudaGraphMaxBatchSize
    '--cuda-graph-backend-prefill', 'disabled'
    '--mamba-radix-cache-strategy', $MambaRadixCacheStrategy
    '--mamba-ssm-dtype', $MambaSsmDtype
    '--linear-attn-decode-backend', $LinearAttentionDecodeBackend
    '--linear-attn-prefill-backend', $LinearAttentionPrefillBackend
    '--linear-attn-verify-backend', $LinearAttentionVerifyBackend
    '--max-mamba-cache-size', $MaxMambaCacheSize
    '--context-length', $ContextLength
    '--mem-fraction-static', $MemoryFraction
    '--max-total-tokens', $MaxTotalTokens
    '--chunked-prefill-size', $ChunkedPrefillSize
    '--triton-attention-num-kv-splits', $TritonAttentionNumKvSplits
    '--max-running-requests', $MaxRunningRequests
    '--scheduler-recv-interval', $SchedulerRecvInterval
    '--stream-interval', $StreamInterval
)

if ($RandomSeed -ge 0) {
    $ServeArgs += @('--random-seed', $RandomSeed)
}

if (-not $DisableIncrementalStreamingOutput) {
    $ServeArgs += '--incremental-streaming-output'
}

if (-not $EnableFlashInferAutotune) {
    $ServeArgs += '--disable-flashinfer-autotune'
}
elseif ($FlashInferAutotuneSkipOps) {
    $ServeArgs += '--flashinfer-autotune-skip-ops'
    $ServeArgs += $FlashInferAutotuneSkipOps
}

if (-not $DisableTorchCompile) {
    $ServeArgs += @(
        '--enable-torch-compile'
        '--torch-compile-max-bs', $CudaGraphMaxBatchSize
    )
}

if ($SpeculativeNumSteps -gt 0) {
    $ServeArgs += @(
        '--speculative-algorithm', 'NEXTN'
        '--speculative-num-steps', $SpeculativeNumSteps
        '--speculative-eagle-topk', $SpeculativeEagleTopK
        '--speculative-num-draft-tokens', $SpeculativeNumDraftTokens
        '--speculative-draft-attention-backend', $SpeculativeDraftAttentionBackend
        '--speculative-attention-mode', $SpeculativeAttentionMode
        '--speculative-tree-sampling-mode', $SpeculativeTreeSamplingMode
        '--enable-linear-replayssm-spec'
    )
    if ($SpeculativeUseRejectionSampling) {
        $ServeArgs += '--speculative-use-rejection-sampling'
    }
    if ($SpeculativeDeviceResidentCycle) {
        $ServeArgs += '--speculative-device-resident-cycle'
    }
    if ($SpeculativeDraftModelQuantization) {
        $ServeArgs += @(
            '--speculative-draft-model-quantization', $SpeculativeDraftModelQuantization
        )
    }
    if (($SpeculativeUseRejectionSampling -or $SpeculativeAlignTreeScoring -or $SpeculativeTreeSamplingMode -eq 'swor') -and $SpeculativeDraftSamplingTopK -gt 0) {
        $ServeArgs += @(
            '--speculative-draft-sampling-top-k', $SpeculativeDraftSamplingTopK
        )
    }
    if ($SpeculativeTreeDepthDiscount -ne 1.0) {
        $ServeArgs += @(
            '--speculative-tree-depth-discount', $SpeculativeTreeDepthDiscount
        )
    }
    if ($SpeculativeSworCollectPathStats) {
        $ServeArgs += '--speculative-swor-collect-path-stats'
    }
    if ($SpeculativeSworCollectOverlapStats) {
        $ServeArgs += '--speculative-swor-collect-overlap-stats'
    }
    if ($SpeculativeSworTopology) {
        $ServeArgs += @('--speculative-swor-topology', $SpeculativeSworTopology)
    }
    if ($SpeculativeDraftKvCacheDtype) {
        $ServeArgs += @(
            '--speculative-draft-kv-cache-dtype', $SpeculativeDraftKvCacheDtype
        )
    }
    if ($SpeculativeAdaptive) {
        $ServeArgs += '--speculative-adaptive'
        if ($SpeculativeAdaptiveConfig) {
            $ResolvedAdaptiveConfig =
                (Resolve-Path -LiteralPath $SpeculativeAdaptiveConfig).Path
            $ServeArgs += @(
                '--speculative-adaptive-config', $ResolvedAdaptiveConfig
            )
        }
    }
}

$HadTorchCompileMode = Test-Path Env:SGLANG_TORCH_COMPILE_MODE
$PreviousTorchCompileMode = $env:SGLANG_TORCH_COMPILE_MODE
$HadFlashInferWorkspaceSize = Test-Path Env:SGLANG_FLASHINFER_WORKSPACE_SIZE
$PreviousFlashInferWorkspaceSize = $env:SGLANG_FLASHINFER_WORKSPACE_SIZE
$HadSimulateAcceptedLength = Test-Path Env:SGLANG_SIMULATE_ACC_LEN
$PreviousSimulateAcceptedLength = $env:SGLANG_SIMULATE_ACC_LEN

try {
    if (-not $DisableTorchCompile) {
        $env:SGLANG_TORCH_COMPILE_MODE = $TorchCompileMode
    }
    if ($FlashInferWorkspaceSizeMB -gt 0) {
        $env:SGLANG_FLASHINFER_WORKSPACE_SIZE =
            $FlashInferWorkspaceSizeMB * 1MB
    }
    if ($SimulateAcceptedLength -gt 0) {
        $env:SGLANG_SIMULATE_ACC_LEN = $SimulateAcceptedLength
    }
    & $SGLang @ServeArgs
    $ExitCode = $LASTEXITCODE
}
finally {
    if ($HadTorchCompileMode) {
        $env:SGLANG_TORCH_COMPILE_MODE = $PreviousTorchCompileMode
    }
    else {
        Remove-Item Env:SGLANG_TORCH_COMPILE_MODE -ErrorAction SilentlyContinue
    }
    if ($HadFlashInferWorkspaceSize) {
        $env:SGLANG_FLASHINFER_WORKSPACE_SIZE = $PreviousFlashInferWorkspaceSize
    }
    else {
        Remove-Item Env:SGLANG_FLASHINFER_WORKSPACE_SIZE -ErrorAction SilentlyContinue
    }
    if ($HadSimulateAcceptedLength) {
        $env:SGLANG_SIMULATE_ACC_LEN = $PreviousSimulateAcceptedLength
    }
    else {
        Remove-Item Env:SGLANG_SIMULATE_ACC_LEN -ErrorAction SilentlyContinue
    }
}

exit $ExitCode
