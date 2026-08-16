"""Measure the dense Windows target-renorm path against sparse top-k logits."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from flashinfer import top_k_renorm_probs as flashinfer_top_k_renorm_probs
from flashinfer import top_p_renorm_probs as flashinfer_top_p_renorm_probs
from flashinfer import top_k as flashinfer_top_k
from flashinfer import top_k_mask_logits as flashinfer_top_k_mask_logits

from sglang.kernels.ops.sampling.renorm_triton import (
    top_k_renorm_probs_triton,
    top_p_renorm_probs_triton,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)


def _sparse_target(logits: torch.Tensor, top_k: int, top_ps: torch.Tensor):
    values, indices = torch.topk(logits, k=top_k, dim=-1, sorted=True)
    probs = torch.softmax(values.float(), dim=-1)
    ascending = torch.flip(probs, dims=(-1,))
    cdf = torch.cumsum(ascending, dim=-1)
    cutoff = torch.sum(cdf < (1.0 - top_ps[:, None]), dim=-1, keepdim=True)
    cutoff.clamp_(max=top_k - 1)
    pivots = ascending.gather(1, cutoff)
    probs = torch.where(probs >= pivots, probs, torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs, indices


def _flashinfer_sparse_target(
    logits: torch.Tensor, top_k: int, top_ps: torch.Tensor
):
    values, indices = flashinfer_top_k(logits, top_k, sorted=True)
    probs = torch.softmax(values.float(), dim=-1)
    ascending = torch.flip(probs, dims=(-1,))
    cdf = torch.cumsum(ascending, dim=-1)
    cutoff = torch.sum(cdf < (1.0 - top_ps[:, None]), dim=-1, keepdim=True)
    cutoff.clamp_(max=top_k - 1)
    pivots = ascending.gather(1, cutoff)
    probs = torch.where(probs >= pivots, probs, torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs, indices


def _measure(fn, *, warmup: int, repeats: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / iterations / 1000.0)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(20260816)
    logits = torch.randn(
        (args.rows, args.vocab),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    top_ks = torch.full((args.rows,), args.top_k, dtype=torch.int64, device="cuda")
    top_ps = torch.full(
        (args.rows,), args.top_p, dtype=torch.float32, device="cuda"
    )

    def dense():
        probs = F.softmax(logits, dim=-1)
        probs = top_k_renorm_probs_triton(probs, top_ks)
        return top_p_renorm_probs_triton(probs, top_ps)

    def sparse():
        return _sparse_target(logits, args.top_k, top_ps)

    def flashinfer_dense():
        probs = F.softmax(logits, dim=-1)
        probs = flashinfer_top_k_renorm_probs(probs, top_ks)
        return flashinfer_top_p_renorm_probs(probs, top_ps)

    def flashinfer_masked_logits():
        masked = flashinfer_top_k_mask_logits(logits, top_ks)
        probs = F.softmax(masked, dim=-1)
        return flashinfer_top_p_renorm_probs(probs, top_ps)

    def flashinfer_sparse():
        return _flashinfer_sparse_target(logits, args.top_k, top_ps)

    dense_out = dense()
    flashinfer_out = flashinfer_dense()
    flashinfer_masked_out = flashinfer_masked_logits()
    sparse_probs, sparse_indices = sparse()
    fi_sparse_probs, fi_sparse_indices = flashinfer_sparse()
    reconstructed = torch.zeros_like(dense_out).scatter_(
        1, sparse_indices, sparse_probs
    )
    fi_reconstructed = torch.zeros_like(dense_out).scatter_(
        1, fi_sparse_indices, fi_sparse_probs
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(flashinfer_out, dense_out, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(
        flashinfer_masked_out, dense_out, rtol=2e-6, atol=1e-7
    )
    torch.testing.assert_close(reconstructed, dense_out, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(fi_reconstructed, dense_out, rtol=2e-6, atol=1e-7)

    dense_us = _measure(
        dense,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    sparse_us = _measure(
        sparse,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    flashinfer_us = _measure(
        flashinfer_dense,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    flashinfer_masked_us = _measure(
        flashinfer_masked_logits,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    flashinfer_sparse_us = _measure(
        flashinfer_sparse,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    dense_median = statistics.median(dense_us)
    sparse_median = statistics.median(sparse_us)
    flashinfer_median = statistics.median(flashinfer_us)
    flashinfer_masked_median = statistics.median(flashinfer_masked_us)
    flashinfer_sparse_median = statistics.median(flashinfer_sparse_us)

    # The production shape has two draft-q rows and three target-p rows.
    if args.rows >= 3:
        target_probs = flashinfer_out[:3].reshape(1, 3, args.vocab)
        draft_logits = torch.randn(
            (2, args.vocab),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        draft_probs = torch.softmax(draft_logits, dim=-1).reshape(
            1, 2, args.vocab
        )
        candidates = torch.tensor([[0, 1, 2]], dtype=torch.long, device="cuda")
        retrieve_index = torch.arange(3, dtype=torch.long, device="cuda").reshape(
            1, 3
        )
        unused_links = torch.full_like(retrieve_index, -1)
        coins = torch.full((1, 3), 0.5, dtype=torch.float32, device="cuda")
        final_coin = torch.full((1,), 0.5, dtype=torch.float32, device="cuda")
        predicts = torch.zeros((3,), dtype=torch.int32, device="cuda")
        accept_index = torch.full((1, 3), -1, dtype=torch.int32, device="cuda")
        accept_num = torch.empty((1,), dtype=torch.int32, device="cuda")

        def rejection():
            chain_speculative_sampling_triton(
                predicts,
                accept_index,
                accept_num,
                candidates,
                retrieve_index,
                unused_links,
                unused_links,
                coins,
                final_coin,
                target_probs,
                draft_probs,
                1.0,
                1.0,
                True,
            )

        rejection_us = _measure(
            rejection,
            warmup=args.warmup,
            repeats=args.repeats,
            iterations=args.iterations,
        )
        rejection_median = statistics.median(rejection_us)
    else:
        rejection_median = float("nan")

    draft_logits_row = logits[:1]
    draft_top_p = top_ps[:1]

    def full_draft_proposal():
        probs = torch.softmax(draft_logits_row, dim=-1)
        noise = torch.empty_like(probs, dtype=torch.float32).exponential_(1.0)
        sampled = (probs.float() / noise).argmax(dim=-1, keepdim=True)
        return probs, probs.gather(1, sampled), sampled

    def sparse_draft_proposal():
        probs, indices = _flashinfer_sparse_target(
            draft_logits_row, args.top_k, draft_top_p
        )
        noise = torch.empty_like(probs, dtype=torch.float32).exponential_(1.0)
        local = (probs / noise).argmax(dim=-1, keepdim=True)
        sampled = indices.gather(1, local)
        dense = torch.zeros_like(draft_logits_row).scatter_(1, indices, probs)
        return dense, probs.gather(1, local), sampled

    def flashinfer_dense_draft_proposal():
        probs = torch.softmax(draft_logits_row, dim=-1)
        probs = flashinfer_top_k_renorm_probs(probs, args.top_k)
        probs = flashinfer_top_p_renorm_probs(probs, draft_top_p)
        noise = torch.empty_like(probs, dtype=torch.float32).exponential_(1.0)
        sampled = (probs / noise).argmax(dim=-1, keepdim=True)
        return probs, probs.gather(1, sampled), sampled

    full_draft_us = _measure(
        full_draft_proposal,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    sparse_draft_us = _measure(
        sparse_draft_proposal,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    dense_draft_us = _measure(
        flashinfer_dense_draft_proposal,
        warmup=args.warmup,
        repeats=args.repeats,
        iterations=args.iterations,
    )
    full_draft_median = statistics.median(full_draft_us)
    sparse_draft_median = statistics.median(sparse_draft_us)
    dense_draft_median = statistics.median(dense_draft_us)
    print(
        f"shape: rows={args.rows}, vocab={args.vocab}, "
        f"top_k={args.top_k}, top_p={args.top_p}"
    )
    print(f"dense Windows renorm median: {dense_median:.3f} us")
    print(f"FlashInfer dense renorm median: {flashinfer_median:.3f} us")
    print(f"FlashInfer logits-mask median: {flashinfer_masked_median:.3f} us")
    print(f"sparse target median: {sparse_median:.3f} us")
    print(f"FlashInfer sparse target median: {flashinfer_sparse_median:.3f} us")
    print(f"dense Triton rejection median: {rejection_median:.3f} us")
    print(f"full-vocab draft proposal median: {full_draft_median:.3f} us")
    print(f"FlashInfer sparse draft median: {sparse_draft_median:.3f} us")
    print(f"FlashInfer dense aligned draft median: {dense_draft_median:.3f} us")
    print(f"FlashInfer speedup: {dense_median / flashinfer_median:.3f}x")
    print(f"speedup: {dense_median / sparse_median:.3f}x")
    print(f"saved: {dense_median - sparse_median:.3f} us/verify")


if __name__ == "__main__":
    main()
