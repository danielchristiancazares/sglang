<#
.SYNOPSIS
Runs OpenCode2 with the local Qwen model and a fast non-thinking title agent.

.DESCRIPTION
The title override is process-scoped through OPENCODE_CONFIG_CONTENT. It keeps
an auxiliary title request short and non-thinking, while preserving the main
model's thinking settings and the user's global OpenCode configuration.
Pass OpenCode's --standalone flag when the overlay must be guaranteed; an
already-running background service retains the configuration it started with.

.PARAMETER DisableSnapshots
Disables OpenCode undo/revert snapshots for this invocation. This is the
fastest path in a large dirty worktree, with the explicit tradeoff that the
invocation cannot use snapshot-backed undo or revert.

.PARAMETER MainOutputCap
Sets the process-scoped OpenAI `max_completion_tokens` field for the normal
thinking model and its diagnostic alias. The default matches the user's 8192
token model limit; zero leaves the global main-model request body unchanged.
#>

[CmdletBinding()]
param(
    [switch] $DisableSnapshots,
    [ValidateRange(0, 32768)]
    [int] $MainOutputCap = 8192,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $OpenCodeArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$OpenCode = (Get-Command opencode2.ps1 -ErrorAction Stop).Source
$HadConfigOverlay = Test-Path Env:OPENCODE_CONFIG_CONTENT
$PreviousConfigOverlay = $env:OPENCODE_CONFIG_CONTENT

$TitleAgentOverlayConfig = @{
    providers = @{
        'llama-cpp' = @{
            models = @{
                'qwen3.8-27b-title' = @{
                    modelID = 'qwen3.8-27b'
                    name = 'Qwen3.8 27B local title'
                    body = @{
                        max_completion_tokens = 32
                        chat_template_kwargs = @{
                            enable_thinking = $false
                            preserve_thinking = $false
                        }
                    }
                    capabilities = @{
                        tools = $false
                        input = @('text')
                        output = @('text')
                    }
                    limit = @{
                        context = 200000
                        output = 32
                    }
                }
            }
        }
    }
    agents = @{
        title = @{
            model = 'llama-cpp/qwen3.8-27b-title'
            temperature = 0
        }
    }
}

if ($DisableSnapshots) {
    $TitleAgentOverlayConfig.snapshots = $false
}

if ($MainOutputCap -gt 0) {
    $ThinkingModelConfig = @{
        modelID = 'qwen3.8-27b'
        name = 'Qwen3.8 27B local thinking'
        body = @{
            max_completion_tokens = $MainOutputCap
            temperature = 1.0
            top_p = 0.95
            top_k = 20
            min_p = 0.0
            presence_penalty = 1.5
            repetition_penalty = 1.0
            chat_template_kwargs = @{
                enable_thinking = $true
                preserve_thinking = $true
            }
        }
        compatibility = @{
            reasoningField = 'reasoning_content'
        }
        capabilities = @{
            tools = $true
            input = @('text')
            output = @('text')
        }
        limit = @{
            context = 200000
            output = $MainOutputCap
        }
    }
    $TitleAgentOverlayConfig['providers']['llama-cpp']['models']['qwen3.8-27b'] = $ThinkingModelConfig
    $TitleAgentOverlayConfig['providers']['llama-cpp']['models']['qwen3.8-27b-thinking-capped'] = $ThinkingModelConfig
}

$TitleAgentOverlay = $TitleAgentOverlayConfig | ConvertTo-Json -Depth 12 -Compress

try {
    $env:OPENCODE_CONFIG_CONTENT = $TitleAgentOverlay
    & $OpenCode @OpenCodeArgs
    $ExitCode = $LASTEXITCODE
}
finally {
    if ($HadConfigOverlay) {
        $env:OPENCODE_CONFIG_CONTENT = $PreviousConfigOverlay
    }
    else {
        Remove-Item Env:OPENCODE_CONFIG_CONTENT -ErrorAction SilentlyContinue
    }
}

exit $ExitCode
