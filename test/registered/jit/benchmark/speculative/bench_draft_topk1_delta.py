from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.speculative.draft_topk1_delta import draft_topk1_delta
from sglang.srt.speculative.spec_utils import build_aligned_draft_probs, fast_sample
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def make_sampling_info(logits: torch.Tensor):
    return type(
        "SamplingInfo",
        (),
        {
            "temperatures": torch.ones((logits.shape[0], 1), device=logits.device),
            "top_ps": torch.ones((logits.shape[0],), device=logits.device),
            "acc_additive_penalties": None,
            "logit_bias": None,
            "need_top_p_sampling": False,
        },
    )()


def aligned_topk1(logits: torch.Tensor, sampling_info):
    q = build_aligned_draft_probs(logits, sampling_info, 1)
    topk_p, topk_index = fast_sample(q)
    return q, topk_p, topk_index


@marker.parametrize("rows", [1, 3], [1])
@marker.benchmark("impl", ["native_delta", "aligned_q"])
def benchmark(rows: int, impl: str):
    logits = torch.randn((rows, 248320), dtype=torch.float32, device="cuda")
    sampling_info = make_sampling_info(logits)
    fn = (
        (lambda: draft_topk1_delta(logits))
        if impl == "native_delta"
        else (lambda: aligned_topk1(logits, sampling_info))
    )
    return marker.do_bench(
        fn,
        input_args=(),
        graph_clone_args=(),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
