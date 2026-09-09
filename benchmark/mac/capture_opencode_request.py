"""Capture the structure of one loopback OpenAI request without its content."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def content_shape(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"kind": "text", "chars": len(content)}
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"kind": type(part).__name__})
                continue
            text = part.get("text")
            parts.append(
                {
                    "kind": part.get("type"),
                    "chars": len(text) if isinstance(text, str) else 0,
                    "keys": sorted(part),
                }
            )
        return {"kind": "parts", "parts": parts}
    return {"kind": type(content).__name__}


def request_shape(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    return {
        "keys": sorted(payload),
        "model": payload.get("model"),
        "stream": payload.get("stream"),
        "sampling": {
            key: payload.get(key)
            for key in (
                "max_completion_tokens",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "frequency_penalty",
            )
            if key in payload
        },
        "messages": [
            {
                "role": message.get("role"),
                "keys": sorted(message),
                "content": content_shape(message.get("content")),
                "tool_calls": len(message.get("tool_calls") or []),
            }
            for message in messages
            if isinstance(message, dict)
        ],
        "tools": [
            {
                "name": (tool.get("function") or {}).get("name"),
                "description_chars": len(
                    (tool.get("function") or {}).get("description") or ""
                ),
                "parameter_chars": len(
                    json.dumps(
                        (tool.get("function") or {}).get("parameters") or {},
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ),
            }
            for tool in tools
            if isinstance(tool, dict)
        ],
    }


class CaptureHandler(BaseHTTPRequestHandler):
    server_version = "OpenCodeCapture/1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("models"):
            self._write_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen3.8-27b",
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                }
            )
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length))
        summary = request_shape(payload)
        tokenizer = getattr(self.server, "tokenizer", None)
        if tokenizer is not None:
            token_ids = tokenizer.apply_chat_template(
                payload.get("messages") or [],
                tools=payload.get("tools") or None,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                preserve_thinking=True,
            )
            if isinstance(token_ids, Mapping):
                token_ids = token_ids["input_ids"]
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            summary["rendered_prompt_tokens"] = len(token_ids)
        print(json.dumps(summary, sort_keys=True), flush=True)
        if payload.get("stream"):
            self._write_stream(payload.get("model") or "qwen3.8-27b")
        else:
            self._write_json(
                {
                    "id": "capture",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload.get("model") or "qwen3.8-27b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "READY"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )

    def _write_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_stream(self, model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        chunks = [
            {
                "id": "capture",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "READY"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "capture",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=39001)
    parser.add_argument("--tokenizer-path")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), CaptureHandler)
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        server.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=False
        )
    print(f"capture server listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
