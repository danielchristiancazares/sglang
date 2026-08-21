import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashinfer_kv_indices_triton,
)
from sglang.kernels.ops.kvcache.flashinfer_page_table import (
    build_flashinfer_page_table,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)


PAGE_SIZE = 64


@marker.parametrize("prefix_tokens", [7680, 192000], [192000])
@marker.benchmark("impl", ["native_page64", "triton_page1"], unit="us")
def benchmark(prefix_tokens: int, impl: str):
    req_to_token = torch.arange(
        PAGE_SIZE,
        prefix_tokens + PAGE_SIZE,
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(0)
    req_pool_indices = torch.zeros(1, dtype=torch.int64, device="cuda")
    if impl == "native_page64":
        page_lens = torch.tensor(
            [prefix_tokens // PAGE_SIZE], dtype=torch.int32, device="cuda"
        )
        page_indptr = torch.tensor(
            [0, prefix_tokens // PAGE_SIZE], dtype=torch.int32, device="cuda"
        )

        def run():
            return build_flashinfer_page_table(
                req_to_token,
                req_pool_indices,
                page_lens,
                page_indptr,
                prefix_tokens // PAGE_SIZE,
                PAGE_SIZE,
                prefix_tokens // PAGE_SIZE + 1,
            )

    else:
        token_lens = torch.tensor(
            [prefix_tokens], dtype=torch.int32, device="cuda"
        )
        token_indptr = torch.tensor(
            [0, prefix_tokens], dtype=torch.int32, device="cuda"
        )

        def run():
            output = torch.empty(
                prefix_tokens + 256, dtype=torch.int32, device="cuda"
            )
            create_flashinfer_kv_indices_triton[(1,)](
                req_to_token,
                req_pool_indices,
                token_lens,
                token_indptr,
                None,
                output,
                req_to_token.shape[1],
            )
            return output

    return marker.do_bench(
        run,
        memory_args=(req_to_token, req_pool_indices),
        graph_clone_args=(),
    )


if __name__ == "__main__":
    benchmark.run()
