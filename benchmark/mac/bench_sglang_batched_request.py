"""Real-sampling throughput with all sequences submitted in one HTTP request."""

import argparse
import time

import requests


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:30000/generate")
parser.add_argument("--batch-size", type=int, default=24)
parser.add_argument("--output-tokens", type=int, default=32)
args = parser.parse_args()

payload = {
    "text": [
        f"Continue this numbered sequence ({index}): 1, 2, 3,"
        for index in range(args.batch_size)
    ],
    "sampling_params": {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 20,
        "max_new_tokens": args.output_tokens,
        "ignore_eos": True,
    },
}
start = time.perf_counter()
response = requests.post(
    args.url, json=payload, timeout=600
)
response.raise_for_status()
result = response.json()
wall = time.perf_counter() - start
if not isinstance(result, list):
    result = [result]
tokens = sum(len(item["output_ids"]) for item in result)
print(
    f"batch_size={args.batch_size} output_tokens={tokens} wall={wall:.3f}s "
    f"aggregate_tps={tokens / wall:.3f} sample={result[0]['text']!r}"
)
