"""Measure speculative acceptance for the local OpenAI benchmark shape."""

from __future__ import annotations

import argparse
import hashlib
import json

from bench_openai_stream import (
    calibrate_prompt,
    chat_template_kwargs,
    flush_cache,
    messages_for,
    request_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--input-tokens", type=int, default=6213)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def generate(
    base_url: str,
    input_ids: list[int],
    output_tokens: int,
    args: argparse.Namespace,
) -> dict:
    return request_json(
        f"{base_url}/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "presence_penalty": args.presence_penalty,
                "ignore_eos": True,
            },
        },
        args.timeout,
    )


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    enable_thinking = not args.disable_thinking
    content, calibrated_tokens = calibrate_prompt(
        base_url,
        args.model,
        args.input_tokens,
        args.timeout,
        "sglang",
        enable_thinking,
    )
    tokenized = request_json(
        f"{base_url}/v1/tokenize",
        {
            "model": args.model,
            "messages": messages_for(content),
            "chat_template_kwargs": chat_template_kwargs(enable_thinking),
        },
        args.timeout,
    )
    input_ids = tokenized["tokens"]
    if len(input_ids) != calibrated_tokens:
        raise RuntimeError(
            f"tokenization changed: calibrated={calibrated_tokens}, actual={len(input_ids)}"
        )

    flush_cache(base_url, args.timeout, "sglang", 0)
    generate(base_url, input_ids, args.warmup_output_tokens, args)
    flush_cache(base_url, args.timeout, "sglang", 0)
    response = generate(base_url, input_ids, args.output_tokens, args)
    meta = response["meta_info"]
    output_text = response.get("text", "")
    print(
        json.dumps(
            {
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "enable_thinking": enable_thinking,
                "e2e_latency": meta.get("e2e_latency"),
                "spec_accept_rate": meta.get("spec_accept_rate"),
                "spec_accept_length": meta.get("spec_accept_length"),
                "spec_num_correct_drafts": meta.get("spec_num_correct_drafts"),
                "spec_num_proposed_drafts": meta.get("spec_num_proposed_drafts"),
                "spec_verify_ct": meta.get("spec_verify_ct"),
                "spec_correct_drafts_histogram": meta.get(
                    "spec_correct_drafts_histogram"
                ),
                "output_sha256": hashlib.sha256(
                    output_text.encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
