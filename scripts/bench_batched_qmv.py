"""Honest benchmark for a candidate batched affine-q4 matvec kernel.

Target problem: y = x @ dequant(W).T  for x [N, K] bfloat16, N in 3..16,
W affine 4-bit / group-size 64.  This is the verify step of MTP speculative
decoding (see python/sglang/srt/hardware_backend/mlx/mtp_spec.py); N = depth+1.

Usage:
    python scripts/bench_batched_qmv.py                        # MLX baseline only
    python scripts/bench_batched_qmv.py --candidate mykern:matmul
    python scripts/bench_batched_qmv.py --candidate mykern:matmul --ns 4 --shapes gate

The candidate is any callable ``f(x, w, scales, biases, *, group_size, bits)``
returning [N, M], e.g. a wrapper around ``mx.fast.metal_kernel``.

MEASUREMENT NOTES -- these are not stylistic, they are the difference between
a real result and a fake one (all learned on the original Mac Pro measurement
host):

  * Reps are CHAINED through a scalar tap of the previous output so kernels run
    strictly serially, matching the layer-by-layer decode graph.  Timing
    independent bursts lets ops overlap and hides poor single-op latency; that
    mis-ranked two kernels here before it was caught.
  * min-of-iters, not mean: the machine has occasional multi-ms hiccups.
  * A micro-benchmark win is a HYPOTHESIS.  A saturated decode loop runs at
    different occupancy, and a kernel that won every micro-benchmark here lost
    17% end to end.  Confirm with test/registered/unit/hardware_backend/mlx/
    plus a real decode before believing anything.
"""

import argparse
import importlib
import sys
import time

import mlx.core as mx

# (M, K, instances per decoded token) for Qwen3.8-27B 4-bit, 14.41 GB/token.
SHAPES = {
    "gate": (17408, 5120, 128),     # mlp gate/up  -- 6417 MB/token, the big one
    "down": (5120, 17408, 64),      # mlp down     -- 3209 MB/token
    "qkv": (10240, 5120, 48),       # 1416 MB/token
    "gdn_qkv": (5120, 6144, 64),    # 1133 MB/token
    "gdn_out": (6144, 5120, 48),    # 849 MB/token
    "lm_head": (248320, 5120, 1),   # 715 MB/token
    "gdn_z": (12288, 5120, 16),     # 566 MB/token
    "small": (1024, 5120, 32),      # 94 MB/token
    "tiny": (48, 5120, 96),         # 13 MB/token -- M < one row tile, handle it
}
STREAM_CEILING = 368e9      # historical Mac Pro sequential-read measurement


def serial_time(op, x0, reps=12, iters=3, warmup=3):
    """Time `op` with reps chained so nothing overlaps.  Returns min seconds/op."""
    def burst():
        x = x0
        outs = []
        for _ in range(reps):
            out = op(x)
            outs.append(out)
            # scalar tap: forces op i+1 to wait on op i without changing shapes
            x = x0 + 0 * out[0, 0].astype(x0.dtype)
        mx.eval(outs)

    for _ in range(warmup):
        burst()
    mx.synchronize()
    best = None
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        burst()
        mx.synchronize()
        dt = (time.perf_counter() - t0) / reps
        best = dt if best is None or dt < best else best
    return best


def load_candidate(spec):
    if spec is None:
        return None
    mod_name, _, fn_name = spec.partition(":")
    sys.path.insert(0, ".")
    return getattr(importlib.import_module(mod_name), fn_name or "matmul")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None,
                    help="module:function implementing f(x,w,scales,biases,*,group_size,bits)")
    ap.add_argument("--ns", default="1,2,3,4,5,6,8,16")
    ap.add_argument("--shapes", default="gate,down,lm_head")
    ap.add_argument("--group-size", type=int, default=64)
    a = ap.parse_args()

    candidate = load_candidate(a.candidate)
    ns = [int(v) for v in a.ns.split(",")]
    G = a.group_size

    for name in a.shapes.split(","):
        M, K, per_token = SHAPES[name]
        mx.random.seed(0)
        w, s, b = mx.quantize(
            (mx.random.normal((M, K)) * 0.02).astype(mx.bfloat16),
            group_size=G, bits=4)
        mx.eval(w, s, b)
        nbytes = w.nbytes + s.nbytes + b.nbytes
        roof = nbytes / STREAM_CEILING

        xs = {}
        for n in ns:
            mx.random.seed(1)
            xs[n] = (mx.random.normal((n, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(list(xs.values()))

        def mlx_op(x):
            return mx.quantized_matmul(x, w, s, b, transpose=True,
                                       group_size=G, bits=4)

        t1 = serial_time(mlx_op, xs[1])
        print(f"\n=== {name} [{M},{K}] x{per_token}/token  {nbytes/1e6:.0f} MB  "
              f"roofline {roof*1e6:.0f} us  (N=1 actual {t1*1e6:.0f} us, "
              f"{nbytes/t1/1e9:.0f} GB/s) ===")
        print(f"  {'N':>3} {'MLX us':>9} {'xN=1':>6} | "
              f"{'cand us':>9} {'xN=1':>6} {'GB/s':>6} {'%roof':>6} {'rel err':>8}")

        for n in ns:
            t_mlx = serial_time(mlx_op, xs[n])
            row = f"  {n:>3} {t_mlx*1e6:9.1f} {t_mlx/t1:6.2f} |"
            if candidate is not None:
                ref = mlx_op(xs[n])
                got = candidate(xs[n], w, s, b, group_size=G, bits=4)
                mx.eval(ref, got)
                if got.shape != ref.shape:
                    raise ValueError(f"candidate returned {got.shape}, want {ref.shape}")
                scale = float(mx.max(mx.abs(ref.astype(mx.float32))).item())
                rel = float(mx.max(mx.abs(ref.astype(mx.float32)
                                          - got.astype(mx.float32))).item()) / scale
                t_c = serial_time(
                    lambda x: candidate(x, w, s, b, group_size=G, bits=4), xs[n])
                row += (f" {t_c*1e6:9.1f} {t_c/t1:6.2f} {nbytes/t_c/1e9:6.0f} "
                        f"{roof/t_c*100:5.0f}% {rel:8.1e}")
            print(row, flush=True)

    print("\nGoal: N=4 at <=1.3x the N=1 column, correct to <5e-3 relative.")
    print("Compare ratios only WITHIN one run: these freshly allocated weights")
    print("measure ~270 GB/s at N=1 where the same shape inside the loaded model")
    print("measures ~213 GB/s, so absolute times are not comparable to in-model.")
    print("Then confirm end to end -- micro-benchmarks on this machine lie.")


if __name__ == "__main__":
    main()
