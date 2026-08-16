<#
.SYNOPSIS
Starts Qwen3.8-27B GGUF through native-Windows SGLang on an RTX 5090.

.DESCRIPTION
The defaults favor OpenCode2 latency while retaining single-stream throughput,
keep a 128K logical context, and size the active token pool for OpenCode2's 32K
client context. The model path defaults to the local Unsloth Q4_K_XL checkpoint
used by this workstation.
#>

[CmdletBinding()]
param(
    [string] $ModelPath = (Join-Path $env:USERPROFILE 'models\Qwen3.8-27B-UD-Q4_K_XL\Qwen3.8-27B-UD-Q4_K_XL.gguf'),
    [string] $ServedModelName = 'qwen3.8-27b',
    [string] $ListenAddress = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int] $Port = 30000,
    [ValidateRange(2048, 262144)]
    [int] $ContextLength = 131072,
    [ValidateRange(2048, 262144)]
    [int] $MaxTotalTokens = 32768,
    [ValidateRange(0.1, 0.95)]
    [double] $MemoryFraction = 0.88,
    [ValidateRange(1, 1024)]
    [int] $MaxMambaCacheSize = 8,
    [ValidateSet('no_buffer', 'extra_buffer', 'extra_buffer_lazy')]
    [string] $MambaRadixCacheStrategy = 'extra_buffer_lazy',
    [ValidateSet('float32', 'bfloat16', 'float16')]
    [string] $MambaSsmDtype = 'float32',
    [ValidateRange(256, 16384)]
    [int] $ChunkedPrefillSize = 4096,
    [ValidateRange(1, 64)]
    [int] $TritonAttentionNumKvSplits = 16,
    [ValidateRange(1, 16)]
    [int] $CudaGraphMaxBatchSize = 2,
    [ValidateRange(1, 16)]
    [int] $MaxRunningRequests = 2,
    [ValidateRange(1, 16)]
    [int] $ContinuousDecodeSteps = 1,
    [ValidateRange(1, 256)]
    [int] $StreamInterval = 4,
    [switch] $DisableIncrementalStreamingOutput,
    [switch] $EnableLinearReplaySSM,
    [switch] $EnableBundledNextN,
    [ValidateRange(1, 15)]
    [int] $SpeculativeNumSteps = 3,
    [switch] $EnableLinearReplaySSMSpec
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($MaxTotalTokens -gt $ContextLength) {
    throw "MaxTotalTokens ($MaxTotalTokens) cannot exceed ContextLength ($ContextLength)"
}

if ($EnableLinearReplaySSMSpec -and -not $EnableBundledNextN) {
    throw 'EnableLinearReplaySSMSpec requires EnableBundledNextN'
}

$ResolvedModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$SGLang = Join-Path $RepoRoot '.venv\Scripts\sglang.exe'

if (-not (Test-Path -LiteralPath $SGLang -PathType Leaf)) {
    throw "Native SGLang launcher not found at $SGLang"
}

$ServeArgs = @(
    'serve'
    '--model-path', $ResolvedModelPath
    '--served-model-name', $ServedModelName
    '--host', $ListenAddress
    '--port', $Port
    '--reasoning-parser', 'qwen3'
    '--tool-call-parser', 'qwen3_coder'
    '--attention-backend', 'triton'
    '--sampling-backend', 'pytorch'
    '--disable-custom-all-reduce'
    '--cuda-graph-backend-decode', 'full'
    '--cuda-graph-max-bs-decode', $CudaGraphMaxBatchSize
    '--cuda-graph-backend-prefill', 'disabled'
    '--mamba-radix-cache-strategy', $MambaRadixCacheStrategy
    '--mamba-ssm-dtype', $MambaSsmDtype
    '--linear-attn-decode-backend', 'triton'
    '--linear-attn-prefill-backend', 'triton'
    '--max-mamba-cache-size', $MaxMambaCacheSize
    '--context-length', $ContextLength
    '--mem-fraction-static', $MemoryFraction
    '--max-total-tokens', $MaxTotalTokens
    '--chunked-prefill-size', $ChunkedPrefillSize
    '--triton-attention-num-kv-splits', $TritonAttentionNumKvSplits
    '--max-running-requests', $MaxRunningRequests
    '--num-continuous-decode-steps', $ContinuousDecodeSteps
    '--stream-interval', $StreamInterval
)

if (-not $DisableIncrementalStreamingOutput) {
    $ServeArgs += '--incremental-streaming-output'
}

if ($EnableLinearReplaySSM) {
    $ServeArgs += '--enable-linear-replayssm'
}

if ($EnableBundledNextN) {
    $ServeArgs += @(
        '--speculative-algorithm', 'NEXTN'
        '--speculative-draft-model-path', $ResolvedModelPath
        '--speculative-draft-model-quantization', 'gguf'
        '--speculative-num-steps', $SpeculativeNumSteps
        '--speculative-eagle-topk', '1'
        '--speculative-num-draft-tokens', ($SpeculativeNumSteps + 1)
        '--speculative-use-rejection-sampling'
        '--speculative-draft-attention-backend', 'triton'
        '--linear-attn-verify-backend', 'triton'
    )
}

if ($EnableLinearReplaySSMSpec) {
    $ServeArgs += '--enable-linear-replayssm-spec'
}

& $SGLang @ServeArgs
exit $LASTEXITCODE
