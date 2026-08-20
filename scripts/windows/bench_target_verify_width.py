"""Profile one live speculative shape for the target-width cost curve.

The server owns the model and CUDA graphs; this client warms it first, records
one local request, and reports the acceptance data beside the generated trace.
Run one server at a time with the desired ``--speculative-num-draft-tokens``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from bench_openai_stream import (
    calibrate_prompt,
    chat_template_kwargs,
    flush_cache,
    messages_for,
    request_json,
)
from bench_spec_acceptance import generate
from analyze_torch_trace import cuda_cycle_samples


def request_profile(url: str, payload: dict, timeout: float) -> str:
    """Profiler endpoints intentionally return plain text, not JSON."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_width_cost_manifest(
    *,
    trace: Path,
    width: int,
    server_info: dict,
) -> dict:
    samples = cuda_cycle_samples(trace)
    if samples is None:
        raise RuntimeError("trace has no repeated target-anchored device cycles")
    configured_width = int(server_info["speculative_num_draft_tokens"])
    if configured_width != width:
        raise RuntimeError(
            f"profile width mismatch: requested={width}, server={configured_width}"
        )
    topk = int(server_info["speculative_eagle_topk"])
    max_depth = int(server_info["speculative_num_steps"])
    raw_topology = server_info.get("speculative_swor_topology")
    if raw_topology:
        parents = json.loads(raw_topology)
    elif topk == 1:
        parents = [-1, *range(width - 1)]
    else:
        parents = None
    shape = {
        "parents": parents,
        "logical_width": width,
        "executed_graph_width": configured_width,
        "max_depth": max_depth,
        "draft_fanout": topk,
        "tree_sampling_mode": server_info["speculative_tree_sampling_mode"],
    }
    shape_json = json.dumps(shape, sort_keys=True, separators=(",", ":"))
    shape_hash = hashlib.sha256(shape_json.encode("utf-8")).hexdigest()
    internal_states = server_info.get("internal_states") or []
    internal = internal_states[0] if internal_states else {}
    active_worker = internal.get("active_speculative_worker")
    compile_mode = internal.get("torch_compile_mode")
    if not active_worker or not compile_mode:
        raise RuntimeError("server_info lacks active worker/torch-compile provenance")
    return {
        "schema_version": 1,
        "artifact_type": "target_width_cost",
        "trace": str(trace),
        "trace_sha256": sha256_file(trace),
        "topology": shape,
        "topology_shape_sha256": shape_hash,
        "cost": {
            "cost_id": f"m{width}-{shape_hash[:12]}",
            "topology_family": "eagle_target_verify",
            "logical_width": width,
            "executed_graph_width": configured_width,
            "max_depth": max_depth,
            "scope": "full_cycle",
            "measurement_method": samples["method"],
            "anchor_graph_id": samples["anchor_graph_id"],
            "samples_ms": samples["samples_ms"],
            "active_worker": active_worker,
            "torch_compile_enabled": internal.get("torch_compile_enabled"),
            "torch_compile_mode": compile_mode,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, default=6213)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-output-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/windows/profiles"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    server_info = get_json(f"{base_url}/server_info", args.timeout)
    content, calibrated_tokens = calibrate_prompt(
        base_url,
        args.model,
        args.input_tokens,
        args.timeout,
        "sglang",
    )
    tokenized = request_json(
        f"{base_url}/v1/tokenize",
        {
            "model": args.model,
            "messages": messages_for(content),
            "chat_template_kwargs": chat_template_kwargs(),
        },
        args.timeout,
    )
    input_ids = tokenized["tokens"]
    if len(input_ids) != calibrated_tokens:
        raise RuntimeError(
            f"tokenization changed: calibrated={calibrated_tokens}, "
            f"actual={len(input_ids)}"
        )

    flush_cache(base_url, args.timeout, "sglang", 0)
    generate(base_url, input_ids, args.warmup_output_tokens, args)
    flush_cache(base_url, args.timeout, "sglang", 0)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    profile_dir = (args.output_dir / f"target_width_m{args.width}-{stamp}").resolve()
    profile_dir.mkdir(parents=True, exist_ok=False)
    request_profile(
        f"{base_url}/start_profile",
        {
            "output_dir": str(profile_dir),
            "activities": ["GPU"],
            "with_stack": False,
            "record_shapes": False,
            "profile_prefix": f"target_width_m{args.width}",
        },
        args.timeout,
    )

    response = None
    try:
        response = generate(base_url, input_ids, args.output_tokens, args)
    finally:
        request_profile(f"{base_url}/stop_profile", {}, args.timeout)

    traces = sorted(profile_dir.glob("*.trace.json.gz"))
    if len(traces) != 1:
        raise RuntimeError(f"expected one trace in {profile_dir}, found {traces}")
    cost_manifest = build_width_cost_manifest(
        trace=traces[0], width=args.width, server_info=server_info
    )
    cost_manifest_path = profile_dir / "width_cost_manifest.json"
    cost_manifest_path.write_text(
        json.dumps(cost_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    meta = response["meta_info"]
    output_text = response.get("text", "")
    print(
        json.dumps(
            {
                "width": args.width,
                "trace": str(traces[0]),
                "width_cost_manifest": str(cost_manifest_path),
                "prompt_tokens": meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "e2e_latency": meta.get("e2e_latency"),
                "spec_accept_length": meta.get("spec_accept_length"),
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
