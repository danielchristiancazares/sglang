import sys

import pytest
import torch
from flashinfer.sampling import top_k_renorm_prob, top_p_renorm_prob

from sglang.kernels.ops.sampling.sparse_top_p_renorm import sparse_top_p_renorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=45,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


def _top_k_probs(seed: int, rows: int, vocab_size: int, top_k: int) -> torch.Tensor:
    torch.manual_seed(seed)
    logits = torch.randn(rows, vocab_size, dtype=torch.float32, device="cuda")
    return top_k_renorm_prob(torch.softmax(logits, dim=-1), top_k)


@pytest.mark.parametrize("seed", [41001, 41002, 41003, 41004])
@pytest.mark.parametrize("top_p", [0.8, 0.95])
def test_sparse_top_p_matches_flashinfer(seed: int, top_p: float) -> None:
    probs = _top_k_probs(seed, rows=3, vocab_size=248320, top_k=20)
    top_ps = torch.full((3,), top_p, dtype=torch.float32, device="cuda")

    expected = top_p_renorm_prob(probs, top_ps)
    actual = sparse_top_p_renorm(probs.clone(), top_ps)

    assert torch.equal(actual, expected)


def test_sparse_top_p_graph_replay() -> None:
    static_probs = _top_k_probs(41100, rows=3, vocab_size=248320, top_k=20)
    top_ps = torch.full((3,), 0.95, dtype=torch.float32, device="cuda")
    sparse_top_p_renorm(static_probs, top_ps)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sparse_top_p_renorm(static_probs, top_ps)

    for seed, top_p in ((41101, 0.95), (41102, 0.8)):
        probs = _top_k_probs(seed, rows=3, vocab_size=248320, top_k=20)
        top_ps.fill_(top_p)
        static_probs.copy_(probs)
        graph.replay()
        expected = top_p_renorm_prob(probs, top_ps)
        assert torch.equal(output, expected)


def test_sparse_top_p_adversarial_boundaries() -> None:
    probs = torch.zeros((3, 248320), dtype=torch.float32, device="cuda")
    supports = [
        torch.full((20,), 0.05, dtype=torch.float32, device="cuda"),
        torch.arange(1, 21, dtype=torch.float32, device="cuda") / 210.0,
        torch.tensor([0.8, 0.15, 0.05], dtype=torch.float32, device="cuda"),
    ]
    for row, support in enumerate(supports):
        probs[row, : support.numel()] = support
    top_ps = torch.full((3,), 0.95, dtype=torch.float32, device="cuda")

    expected = top_p_renorm_prob(probs, top_ps)
    actual = sparse_top_p_renorm(probs.clone(), top_ps)

    assert torch.equal(actual, expected)


def test_sparse_top_p_tie_overflow_matches_flashinfer() -> None:
    probs = torch.zeros((1, 248320), dtype=torch.float32, device="cuda")
    probs[0, :33] = 1.0 / 33.0
    top_ps = torch.tensor([0.95], dtype=torch.float32, device="cuda")

    expected = top_p_renorm_prob(probs, top_ps)
    actual = sparse_top_p_renorm(probs.clone(), top_ps)

    assert torch.equal(actual, expected)


def test_sparse_top_p_mixed_tie_overflow_matches_flashinfer() -> None:
    probs = torch.zeros((1, 248320), dtype=torch.float32, device="cuda")
    probs[0, :3] = torch.tensor([0.5, 0.3, 0.1], device="cuda")
    probs[0, 3:36] = 0.1 / 33.0
    top_ps = torch.tensor([0.85], dtype=torch.float32, device="cuda")

    expected = top_p_renorm_prob(probs, top_ps)
    actual = sparse_top_p_renorm(probs.clone(), top_ps)

    assert torch.equal(actual, expected)


def test_sparse_top_p_full_vocab_tie_matches_flashinfer() -> None:
    probs = torch.full(
        (1, 248320),
        1.0 / 248320,
        dtype=torch.float32,
        device="cuda",
    )
    probs = top_k_renorm_prob(probs, 20)
    assert torch.count_nonzero(probs).item() == probs.numel()
    top_ps = torch.tensor([0.95], dtype=torch.float32, device="cuda")

    expected = top_p_renorm_prob(probs, top_ps)
    actual = sparse_top_p_renorm(probs.clone(), top_ps)

    assert torch.equal(actual, expected)


def test_sparse_top_p_prefix_boundaries_match_flashinfer() -> None:
    support_sizes = (2, 3, 7, 16, 20, 32)
    rows = []
    top_ps = []
    for support_size in support_sizes:
        values = torch.arange(
            support_size,
            0,
            -1,
            dtype=torch.float32,
            device="cuda",
        )
        values /= values.sum()
        for prefix_size in {1, max(1, support_size // 2), support_size - 1}:
            boundary = values[:prefix_size].sum()
            for direction in (-1.0, 0.0, 1.0):
                top_p = (
                    torch.nextafter(
                        boundary,
                        torch.tensor(direction, device="cuda"),
                    )
                    if direction
                    else boundary
                )
                row = torch.zeros(248320, dtype=torch.float32, device="cuda")
                offsets = (
                    torch.arange(support_size, device="cuda") * 1024
                    + len(rows) % 1024
                )
                row[offsets] = values
                rows.append(row)
                top_ps.append(top_p)

    probs = torch.stack(rows)
    top_ps_tensor = torch.stack(top_ps)
    expected = top_p_renorm_prob(probs, top_ps_tensor)
    actual = sparse_top_p_renorm(probs.clone(), top_ps_tensor)

    if not torch.equal(actual, expected):
        mismatch_rows = torch.nonzero(
            torch.any(
                actual.view(torch.int32) != expected.view(torch.int32),
                dim=1,
            )
        ).flatten()
        print(
            {
                "mismatch_rows": mismatch_rows.cpu().tolist(),
                "top_ps": top_ps_tensor[mismatch_rows].cpu().tolist(),
                "actual_nonzero": torch.count_nonzero(
                    actual[mismatch_rows], dim=1
                ).cpu().tolist(),
                "expected_nonzero": torch.count_nonzero(
                    expected[mismatch_rows], dim=1
                ).cpu().tolist(),
            }
        )
    assert torch.equal(actual, expected)


def test_sparse_top_p_rejects_unsupported_static_bound() -> None:
    probs = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    top_ps = torch.tensor([0.95], dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match=r"max_nonzero must be in \[1, 1024\]"):
        sparse_top_p_renorm(probs, top_ps, max_nonzero=1025)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
