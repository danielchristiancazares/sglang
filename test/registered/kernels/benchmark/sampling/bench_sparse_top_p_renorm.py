import torch
from flashinfer.sampling import top_k_renorm_prob, top_p_renorm_prob

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.sampling.sparse_top_p_renorm import sparse_top_p_renorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)


def flashinfer_impl(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    top_k_probs = top_k_renorm_prob(probs, 20)
    return top_p_renorm_prob(top_k_probs, top_ps)


def jit_impl(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    top_k_probs = top_k_renorm_prob(probs, 20)
    return sparse_top_p_renorm(top_k_probs, top_ps)


FN_MAP = {
    "flashinfer": flashinfer_impl,
    "jit": jit_impl,
}


@marker.parametrize("rows", [1, 3], [3])
@marker.benchmark("impl", ["flashinfer", "jit"], unit="us")
def benchmark(rows: int, impl: str):
    torch.manual_seed(41000 + rows)
    logits = torch.randn(rows, 248320, dtype=torch.float32, device="cuda")
    probs = torch.softmax(logits, dim=-1)
    top_ps = torch.full((rows,), 0.95, dtype=torch.float32, device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(probs, top_ps),
        memory_args=(probs, top_ps),
        graph_clone_args=(0,),
    )


if __name__ == "__main__":
    benchmark.run()
