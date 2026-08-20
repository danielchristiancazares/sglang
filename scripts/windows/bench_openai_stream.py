"""Measure one local OpenAI-compatible streaming request without remote data."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any


PROMPT_UNIT = (
    "Inspect this local program carefully, preserve its behavior, and identify "
    "the next useful correctness or performance change. "
)
FILLER_UNIT = " x"


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "sglang-local-benchmark/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def chat_template_kwargs(enable_thinking: bool = True) -> dict[str, bool]:
    return {
        "enable_thinking": enable_thinking,
        "preserve_thinking": enable_thinking,
    }


def messages_for(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def token_count(
    base_url: str,
    model: str,
    content: str,
    timeout: float,
    backend: str,
    enable_thinking: bool,
) -> int:
    if backend == "llama":
        template_response = request_json(
            f"{base_url}/apply-template",
            {
                "model": model,
                "messages": messages_for(content),
                "chat_template_kwargs": chat_template_kwargs(enable_thinking),
            },
            timeout,
        )
        prompt = template_response.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError(
                f"Unexpected apply-template response: {template_response}"
            )
        response = request_json(
            f"{base_url}/tokenize",
            {
                "content": prompt,
                "add_special": False,
                "parse_special": True,
                "with_pieces": False,
            },
            timeout,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"Unexpected tokenize response: {response}")
        return len(tokens)

    response = request_json(
        f"{base_url}/v1/tokenize",
        {
            "model": model,
            "messages": messages_for(content),
            "chat_template_kwargs": chat_template_kwargs(enable_thinking),
        },
        timeout,
    )
    count = response.get("count")
    if not isinstance(count, int):
        raise RuntimeError(f"Unexpected tokenize response: {response}")
    return count


def calibrate_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    timeout: float,
    backend: str,
    enable_thinking: bool = True,
) -> tuple[str, int]:
    """Find a deterministic local prompt at or just below the token target."""
    low = 0
    high = target_tokens
    best_repeats = 0
    best_count = token_count(
        base_url, model, "", timeout, backend, enable_thinking
    )
    if best_count > target_tokens:
        raise ValueError(
            f"Target {target_tokens} is below the empty templated prompt "
            f"length {best_count}"
        )

    while low <= high:
        middle = (low + high) // 2
        count = token_count(
            base_url,
            model,
            PROMPT_UNIT * middle,
            timeout,
            backend,
            enable_thinking,
        )
        if count <= target_tokens:
            best_repeats = middle
            best_count = count
            low = middle + 1
        else:
            high = middle - 1

    content = PROMPT_UNIT * best_repeats
    remaining = target_tokens - best_count
    if remaining <= 0:
        return content, best_count

    low = 0
    high = max(16, remaining * 2)
    best_filler = 0
    while low <= high:
        middle = (low + high) // 2
        candidate = content + FILLER_UNIT * middle
        count = token_count(
            base_url, model, candidate, timeout, backend, enable_thinking
        )
        if count <= target_tokens:
            best_filler = middle
            best_count = count
            low = middle + 1
        else:
            high = middle - 1

    return content + FILLER_UNIT * best_filler, best_count


def flush_cache(base_url: str, timeout: float, backend: str, slot_id: int) -> None:
    if backend == "llama":
        request_json(
            f"{base_url}/slots/{slot_id}?action=erase",
            {},
            timeout,
        )
        return

    url = f"{base_url}/flush_cache?timeout={timeout:g}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def stream_request(
    base_url: str,
    model: str,
    content: str,
    output_tokens: int,
    timeout: float,
    seed: int | None,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    enable_thinking: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages_for(content),
        "max_completion_tokens": output_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": chat_template_kwargs(enable_thinking),
    }
    for key, value in (
        ("top_p", top_p),
        ("top_k", top_k),
        ("min_p", min_p),
        ("presence_penalty", presence_penalty),
        ("repetition_penalty", repetition_penalty),
    ):
        if value is not None:
            payload[key] = value
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "sglang-local-benchmark/1",
        },
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    output_fragments: list[str] = []
    reasoning_fragments: list[str] = []
    content_fragments: list[str] = []
    reasoning_chars = 0
    content_chars = 0
    nonempty_delta_count = 0
    reasoning_fragment_count = 0
    content_fragment_count = 0
    first_output_delta_chars: int | None = None
    max_output_delta_chars = 0
    last_output_at: float | None = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                reasoning_fragment = delta.get("reasoning_content")
                content_fragment = delta.get("content")
                delta_chars = 0
                if reasoning_fragment:
                    reasoning_chars += len(reasoning_fragment)
                    reasoning_fragment_count += 1
                    reasoning_fragments.append(reasoning_fragment)
                    output_fragments.append(reasoning_fragment)
                    delta_chars += len(reasoning_fragment)
                if content_fragment:
                    content_chars += len(content_fragment)
                    content_fragment_count += 1
                    content_fragments.append(content_fragment)
                    output_fragments.append(content_fragment)
                    delta_chars += len(content_fragment)
                if delta_chars:
                    now = time.perf_counter()
                    nonempty_delta_count += 1
                    if first_output_delta_chars is None:
                        first_output_delta_chars = delta_chars
                    if first_token_at is None:
                        first_token_at = now
                    last_output_at = now
                    max_output_delta_chars = max(max_output_delta_chars, delta_chars)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {base_url}/v1/chat/completions: {detail}"
        ) from exc

    ended = time.perf_counter()
    if first_token_at is None:
        first_token_at = ended
    if last_output_at is None:
        last_output_at = ended

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    ttft = first_token_at - started
    elapsed = ended - started
    decode_elapsed = max(0.0, ended - first_token_at)
    decode_tokens = max(0, completion_tokens - 1)
    output_text = "".join(output_fragments)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or 0),
        "ttft_s": round(ttft, 6),
        "e2e_s": round(elapsed, 6),
        "observed_prompt_tps": round(prompt_tokens / ttft, 3) if ttft else None,
        "decode_tps": (
            round(decode_tokens / decode_elapsed, 3)
            if decode_tokens and decode_elapsed
            else None
        ),
        "output_tps_e2e": (
            round(completion_tokens / elapsed, 3)
            if completion_tokens and elapsed
            else None
        ),
        "finish_reason": finish_reason,
        "output_chars": len(output_text),
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "nonempty_delta_count": nonempty_delta_count,
        "reasoning_fragment_count": reasoning_fragment_count,
        "content_fragment_count": content_fragment_count,
        "first_output_delta_chars": first_output_delta_chars or 0,
        "max_output_delta_chars": max_output_delta_chars,
        "trailing_after_last_delta_s": round(ended - last_output_at, 6),
        "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
        "reasoning_sha256": hashlib.sha256(
            "".join(reasoning_fragments).encode("utf-8")
        ).hexdigest(),
        "content_sha256": hashlib.sha256(
            "".join(content_fragments).encode("utf-8")
        ).hexdigest(),
    }


def validate_result_counts(
    result: dict[str, Any],
    *,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
    label: str,
) -> None:
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    total_tokens = result["total_tokens"]
    if prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            f"{label} prompt token mismatch: "
            f"expected={expected_prompt_tokens}, actual={prompt_tokens}"
        )
    if completion_tokens != expected_completion_tokens:
        raise RuntimeError(
            f"{label} completion token mismatch: "
            f"expected={expected_completion_tokens}, actual={completion_tokens}"
        )
    if total_tokens != prompt_tokens + completion_tokens:
        raise RuntimeError(
            f"{label} total token mismatch: total={total_tokens}, "
            f"prompt={prompt_tokens}, completion={completion_tokens}"
        )
    if result["finish_reason"] != "length":
        raise RuntimeError(
            f"{label} finish reason mismatch: {result['finish_reason']!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one local OpenAI-compatible streaming request."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument(
        "--backend", choices=("sglang", "llama"), default="sglang"
    )
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--input-tokens", type=int, default=6213)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Optional request seed forwarded to the backend; speculative "
            "proposal sampling may remain nondeterministic"
        ),
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Set enable_thinking and preserve_thinking false in the chat template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_tokens < 1 or args.output_tokens < 1:
        raise ValueError("input and output token counts must be positive")
    if args.warmup_output_tokens < 1:
        raise ValueError("warmup output token count must be positive")
    if args.warmup_runs < 0:
        raise ValueError("warmup run count must be non-negative")

    base_url = args.base_url.rstrip("/")
    enable_thinking = not args.disable_thinking
    content, calibrated_tokens = calibrate_prompt(
        base_url,
        args.model,
        args.input_tokens,
        args.timeout,
        args.backend,
        enable_thinking,
    )

    # Remove the preceding invocation's measured request before warming this
    # exact shape; otherwise a repeated benchmark warms only a cached suffix.
    flush_cache(base_url, args.timeout, args.backend, args.slot_id)
    warmup_runs = 0 if args.skip_warmup else args.warmup_runs
    for warmup_index in range(warmup_runs):
        warmup_result = stream_request(
            base_url,
            args.model,
            content,
            args.warmup_output_tokens,
            args.timeout,
            args.seed,
            args.temperature,
            args.top_p,
            args.top_k,
            args.min_p,
            args.presence_penalty,
            args.repetition_penalty,
            enable_thinking,
        )
        validate_result_counts(
            warmup_result,
            expected_prompt_tokens=calibrated_tokens,
            expected_completion_tokens=args.warmup_output_tokens,
            label=f"warmup {warmup_index + 1}",
        )
        if warmup_index + 1 < warmup_runs:
            flush_cache(base_url, args.timeout, args.backend, args.slot_id)
    flush_cache(base_url, args.timeout, args.backend, args.slot_id)
    result = stream_request(
        base_url,
        args.model,
        content,
        args.output_tokens,
        args.timeout,
        args.seed,
        args.temperature,
        args.top_p,
        args.top_k,
        args.min_p,
        args.presence_penalty,
        args.repetition_penalty,
        enable_thinking,
    )
    validate_result_counts(
        result,
        expected_prompt_tokens=calibrated_tokens,
        expected_completion_tokens=args.output_tokens,
        label="measurement",
    )
    result.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url,
            "model": args.model,
            "backend": args.backend,
            "requested_prompt_tokens": args.input_tokens,
            "calibrated_prompt_tokens": calibrated_tokens,
            "requested_completion_tokens": args.output_tokens,
            "warmup": warmup_runs > 0,
            "warmup_runs": warmup_runs,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "enable_thinking": enable_thinking,
        }
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
