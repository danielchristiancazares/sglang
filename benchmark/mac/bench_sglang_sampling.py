"""Small concurrent real-sampling benchmark for a local SGLang server."""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import threading
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000/generate")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling cutoff (Qwen3.8's recommended value is 20).",
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    barrier = threading.Barrier(args.concurrency)

    def request(index: int):
        sampling_params = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.output_tokens,
            "ignore_eos": True,
        }
        if args.seed is not None:
            sampling_params["seed"] = args.seed + index
        payload = {
            "text": f"Continue this numbered sequence ({index}): 1, 2, 3,",
            "sampling_params": sampling_params,
        }
        barrier.wait()
        start = time.perf_counter()
        response = requests.post(args.url, json=payload, timeout=600)
        response.raise_for_status()
        body = response.json()
        return time.perf_counter() - start, len(body["output_ids"]), body["text"]

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        results = list(executor.map(request, range(args.concurrency)))
    wall = time.perf_counter() - wall_start
    total_tokens = sum(tokens for _, tokens, _ in results)
    latencies = [latency for latency, _, _ in results]
    print(
        f"requests={args.concurrency} output_tokens={total_tokens} "
        f"temperature={args.temperature:g} top_p={args.top_p:g} "
        f"top_k={args.top_k} "
        f"wall={wall:.3f}s aggregate_tps={total_tokens / wall:.3f} "
        f"median_latency={statistics.median(latencies):.3f}s"
    )
    print(f"sample={results[0][2]!r}")


if __name__ == "__main__":
    main()
