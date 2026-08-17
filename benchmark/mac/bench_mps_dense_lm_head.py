"""Estimate a chunked dense MPS LM head from GGUF Q6_K rows."""

import argparse
import statistics
import time

import gguf
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("gguf_path")
parser.add_argument("--rows", type=int, default=8192)
parser.add_argument("--batch-size", type=int, default=24)
parser.add_argument("--iterations", type=int, default=7)
parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
args = parser.parse_args()

reader = gguf.GGUFReader(args.gguf_path)
tensor = next(t for t in reader.tensors if t.name == "output.weight")
packed = np.array(tensor.data[: args.rows], copy=True)
dense = torch.from_numpy(gguf.dequantize(packed, tensor.tensor_type))
dtype = torch.float16 if args.dtype == "float16" else torch.float32
weight = dense.to(device="mps", dtype=dtype)
x = torch.randn(args.batch_size, dense.shape[1], device="mps", dtype=dtype)


def run():
    return torch.mm(x, weight.T).to(torch.float32)


run()
torch.mps.synchronize()
timings = []
for _ in range(args.iterations):
    start = time.perf_counter()
    run()
    torch.mps.synchronize()
    timings.append(time.perf_counter() - start)
median = statistics.median(timings)
full_rows = tensor.data.shape[0]
print(
    f"dtype={args.dtype} rows={args.rows} batch={args.batch_size} "
    f"median={median * 1000:.3f}ms "
    f"linear_full_estimate={median * full_rows / args.rows * 1000:.3f}ms "
    f"raw_ms={','.join(f'{value * 1000:.3f}' for value in timings)}"
)
