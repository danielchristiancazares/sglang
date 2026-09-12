from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bench_opencode_client as benchmark


def test_model_overlay_only_caps_the_selected_model() -> None:
    overlay = benchmark.model_overlay(
        "llama-cpp/qwen3.8-27b", 512, "http://127.0.0.1:30000/v1"
    )

    assert overlay == {
        "provider": {
            "llama-cpp": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "SGLang local",
                "options": {
                    "baseURL": "http://127.0.0.1:30000/v1",
                    "apiKey": "local",
                },
                "models": {
                    "qwen3.8-27b": {
                        "limit": {"context": 200000, "output": 512}
                    }
                }
            }
        }
    }


def test_completion_tokens_includes_reasoning() -> None:
    assert benchmark.completion_tokens({"output": 17, "reasoning": 31}) == 48
    assert benchmark.completion_tokens({"output": 5}) == 5
