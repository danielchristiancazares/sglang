"""Measure the installed OpenCode client against its configured local model."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate real OpenCode per-stream TTFT, decode TPS, and latency."
    )
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--model", default="llama-cpp/qwen3.8-27b")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--agent", default="build")
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--title", default="Qwen interactive throughput gate")
    parser.add_argument(
        "--message",
        default=(
            "Analyze the local checkout without calling tools. Produce a dense, "
            "continuous technical assessment until the output limit."
        ),
    )
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--expected-prompt-tokens", type=int, default=6213)
    parser.add_argument("--min-decode-tps", type=float, default=60.0)
    parser.add_argument("--min-e2e-output-tps", type=float, default=1.323)
    parser.add_argument("--max-ttft", type=float, default=101.229673)
    parser.add_argument("--max-e2e", type=float, default=387.008627)
    parser.add_argument("--pure", action="store_true")
    parser.add_argument(
        "--enforce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def model_overlay(
    model: str, output_tokens: int, base_url: str
) -> dict[str, Any]:
    provider, separator, model_id = model.partition("/")
    if not separator or not provider or not model_id:
        raise ValueError("model must use provider/model form")
    return {
        "provider": {
            provider: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "SGLang local",
                "options": {"baseURL": base_url, "apiKey": "local"},
                "models": {
                    model_id: {
                        "limit": {"context": 200000, "output": output_tokens}
                    }
                }
            }
        }
    }


def read_lines(
    name: str,
    stream: TextIO,
    output: queue.Queue[tuple[str, float, str]],
) -> None:
    for line in stream:
        output.put((name, time.perf_counter(), line.rstrip("\n")))


def completion_tokens(tokens: dict[str, Any]) -> int:
    return int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0)


def main() -> None:
    args = parse_args()
    if args.output_tokens < 2:
        raise ValueError("output token limit must be at least two")

    command = [
        args.opencode,
        "run",
        "--model",
        args.model,
        "--agent",
        args.agent,
        "--format",
        "json",
        "--thinking",
        "--title",
        args.title,
    ]
    if args.pure:
        command.append("--pure")
    command.append(args.message)

    environment = os.environ.copy()
    overlay = model_overlay(args.model, args.output_tokens, args.base_url)
    existing_overlay = environment.get("OPENCODE_CONFIG_CONTENT")
    if existing_overlay:
        raise RuntimeError(
            "OPENCODE_CONFIG_CONTENT is already set; clear it or merge it explicitly"
        )
    environment["OPENCODE_CONFIG_CONTENT"] = json.dumps(
        overlay, separators=(",", ":")
    )

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=args.directory.resolve(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    lines: queue.Queue[tuple[str, float, str]] = queue.Queue()
    readers = [
        threading.Thread(
            target=read_lines, args=("stdout", process.stdout, lines), daemon=True
        ),
        threading.Thread(
            target=read_lines, args=("stderr", process.stderr, lines), daemon=True
        ),
    ]
    for reader in readers:
        reader.start()

    first_token_at: float | None = None
    finished_at: float | None = None
    finish_reason: str | None = None
    tokens: dict[str, Any] = {}
    stderr_tail: list[str] = []
    event_counts: dict[str, int] = {}
    deadline = started + args.timeout

    while process.poll() is None or any(reader.is_alive() for reader in readers):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise TimeoutError(f"OpenCode exceeded {args.timeout:g} seconds")
        try:
            source, arrived_at, line = lines.get(timeout=min(0.25, remaining))
        except queue.Empty:
            continue
        if source == "stderr":
            stderr_tail.append(line)
            stderr_tail = stderr_tail[-20:]
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type") or "unknown")
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        part = event.get("part") or {}
        if event_type in ("reasoning", "text") and part.get("text"):
            if first_token_at is None:
                first_token_at = arrived_at
        if event_type == "step_finish":
            finished_at = arrived_at
            finish_reason = part.get("reason")
            if isinstance(part.get("tokens"), dict):
                tokens = part["tokens"]

    ended = time.perf_counter()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"OpenCode exited {return_code}; stderr tail: {stderr_tail}"
        )
    if first_token_at is None or finished_at is None or not tokens:
        raise RuntimeError(
            f"OpenCode stream lacked timing or usage events: {event_counts}"
        )

    prompt_tokens = int(tokens.get("input") or 0)
    generated_tokens = completion_tokens(tokens)
    ttft = first_token_at - started
    request_e2e = finished_at - started
    decode_elapsed = max(0.0, finished_at - first_token_at)
    decode_tokens = max(0, generated_tokens - 1)
    decode_tps = decode_tokens / decode_elapsed if decode_elapsed else 0.0
    output_tps_e2e = generated_tokens / request_e2e if request_e2e else 0.0
    checks = {
        "prompt_shape": prompt_tokens == args.expected_prompt_tokens,
        "completion_nonempty": generated_tokens >= 2,
        "decode_tps": decode_tps >= args.min_decode_tps,
        "e2e_output_tps": output_tps_e2e > args.min_e2e_output_tps,
        "ttft": ttft < args.max_ttft,
        "e2e_latency": request_e2e < args.max_e2e,
    }
    result = {
        "workload": "opencode_client_interactive",
        "command": command,
        "directory": str(args.directory.resolve()),
        "model": args.model,
        "base_url": args.base_url,
        "agent": args.agent,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": generated_tokens,
        "reasoning_tokens": int(tokens.get("reasoning") or 0),
        "content_tokens": int(tokens.get("output") or 0),
        "ttft_s": round(ttft, 6),
        "decode_tps": round(decode_tps, 3),
        "output_tps_e2e": round(output_tps_e2e, 3),
        "e2e_s": round(request_e2e, 6),
        "client_wall_s": round(ended - started, 6),
        "finish_reason": finish_reason,
        "event_counts": event_counts,
        "thresholds": {
            "expected_prompt_tokens": args.expected_prompt_tokens,
            "min_decode_tps": args.min_decode_tps,
            "min_e2e_output_tps": args.min_e2e_output_tps,
            "max_ttft_s": args.max_ttft,
            "max_e2e_s": args.max_e2e,
        },
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    print(json.dumps(result, sort_keys=True))
    if args.enforce and not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
