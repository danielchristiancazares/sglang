import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from sglang.kernels.ops.kvcache.flashinfer_page_table import (
    build_flashinfer_page_table,
)
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterDecode,
    FlashInferIndicesUpdaterPrefill,
    MultiItemScoringParams,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=45,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


PAGE_SIZE = 64


def _write_pages(
    req_to_token: torch.Tensor,
    request_row: int,
    page_ids: list[int],
) -> None:
    offsets = torch.arange(PAGE_SIZE, dtype=torch.int32, device="cuda")
    for logical_page, physical_page in enumerate(page_ids):
        start = logical_page * PAGE_SIZE
        req_to_token[request_row, start : start + PAGE_SIZE] = (
            physical_page * PAGE_SIZE + offsets
        )


class TestFlashInferPageTable(CustomTestCase):
    def test_scattered_physical_pages(self):
        req_pool = ReqToTokenPool(
            size=3,
            max_context_len=512,
            device="cuda",
            enable_memory_saver=False,
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=13 * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=MagicMock(),
            need_sort=False,
        )
        self.assertNotIn(0, allocator.free_pages.tolist())
        allocator.free_pages = torch.tensor(
            [7, 2, 11, 5, 13, 1, 3, 4, 6, 8, 9, 10, 12],
            dtype=torch.int64,
            device="cuda",
        )
        slots = allocator.alloc(5 * PAGE_SIZE).view(5, PAGE_SIZE)
        req_pool.req_to_token[3, : 3 * PAGE_SIZE] = slots[:3].reshape(-1)
        req_pool.req_to_token[1, : 2 * PAGE_SIZE] = slots[3:].reshape(-1)
        req_to_token = req_pool.req_to_token
        req_pool_indices = torch.tensor([3, 1], dtype=torch.int64, device="cuda")
        page_lens = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
        page_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")

        actual = build_flashinfer_page_table(
            req_to_token,
            req_pool_indices,
            page_lens,
            page_indptr,
            total_pages=5,
            page_size=PAGE_SIZE,
            physical_page_count=14,
        )

        torch.testing.assert_close(
            actual[:5],
            torch.tensor([7, 2, 11, 5, 13], dtype=torch.int32, device="cuda"),
            rtol=0,
            atol=0,
        )

    def test_misaligned_graph_replay_fails_loudly(self):
        code = """
import torch
from sglang.kernels.ops.kvcache.flashinfer_page_table import build_flashinfer_page_table
req = torch.zeros((1, 64), dtype=torch.int32, device="cuda")
req[0] = torch.arange(64, 128, dtype=torch.int32, device="cuda")
request = torch.zeros(1, dtype=torch.int64, device="cuda")
lens = torch.ones(1, dtype=torch.int32, device="cuda")
indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    build_flashinfer_page_table(req, request, lens, indptr, 1, 64, 3)
req[0, 0] = 65
graph.replay()
torch.cuda.synchronize()
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [
                    str(Path(__file__).resolve().parents[5] / "python"),
                    env.get("PYTHONPATH"),
                ],
            )
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "device-side assert",
            (result.stdout + result.stderr).lower(),
        )

    def test_mutable_graph_replay(self):
        req_to_token = torch.zeros((2, 256), dtype=torch.int32, device="cuda")
        _write_pages(req_to_token, 1, [3, 8])
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        page_lens = torch.tensor([2], dtype=torch.int32, device="cuda")
        page_indptr = torch.tensor([0, 2], dtype=torch.int32, device="cuda")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = build_flashinfer_page_table(
                req_to_token,
                req_pool_indices,
                page_lens,
                page_indptr,
                total_pages=2,
                page_size=PAGE_SIZE,
                physical_page_count=13,
            )

        _write_pages(req_to_token, 1, [12, 4])
        graph.replay()
        torch.testing.assert_close(
            output[:2],
            torch.tensor([12, 4], dtype=torch.int32, device="cuda"),
            rtol=0,
            atol=0,
        )

    def test_page64_attention_matches_page1(self):
        torch.manual_seed(42043)
        prefix_tokens = 1024
        query_tokens = 64
        num_q_heads = 4
        num_kv_heads = 1
        head_dim = 64
        total_pages = prefix_tokens // PAGE_SIZE + 1
        req_to_token = torch.zeros(
            (2, prefix_tokens), dtype=torch.int32, device="cuda"
        )
        page_ids = list(range(1, total_pages))
        page_ids[2], page_ids[7] = page_ids[7], page_ids[2]
        _write_pages(req_to_token, 1, page_ids)
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")

        page_lens = torch.tensor(
            [prefix_tokens // PAGE_SIZE], dtype=torch.int32, device="cuda"
        )
        page_indptr = torch.tensor(
            [0, prefix_tokens // PAGE_SIZE], dtype=torch.int32, device="cuda"
        )
        page64_indices = build_flashinfer_page_table(
            req_to_token,
            req_pool_indices,
            page_lens,
            page_indptr,
            total_pages=prefix_tokens // PAGE_SIZE,
            page_size=PAGE_SIZE,
            physical_page_count=total_pages,
        )
        token_indices = req_to_token[1, :prefix_tokens].contiguous()

        q = torch.randn(
            query_tokens,
            num_q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_flat = torch.randn(
            total_pages * PAGE_SIZE,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_flat = torch.randn_like(k_flat)
        workspace = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
        qo_indptr = torch.tensor(
            [0, query_tokens], dtype=torch.int32, device="cuda"
        )

        def run(page_size, kv_indices, kv_indptr, last_page_len, kv):
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD", backend="fa2"
            )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=False,
                q_data_type=q.dtype,
                kv_data_type=k_flat.dtype,
            )
            return wrapper.forward_return_lse(q, kv, causal=False)

        page1 = run(
            1,
            token_indices,
            torch.tensor([0, prefix_tokens], dtype=torch.int32, device="cuda"),
            torch.ones(1, dtype=torch.int32, device="cuda"),
            (
                k_flat.view(-1, 1, num_kv_heads, head_dim),
                v_flat.view(-1, 1, num_kv_heads, head_dim),
            ),
        )
        page64 = run(
            PAGE_SIZE,
            page64_indices,
            page_indptr,
            torch.full((1,), PAGE_SIZE, dtype=torch.int32, device="cuda"),
            (
                k_flat.view(-1, PAGE_SIZE, num_kv_heads, head_dim),
                v_flat.view(-1, PAGE_SIZE, num_kv_heads, head_dim),
            ),
        )

        self.assertTrue(torch.equal(page1[0], page64[0]))
        self.assertTrue(torch.equal(page1[1], page64[1]))

    def test_disabled_updater_keeps_page1_metadata(self):
        req_to_token = torch.zeros((2, 256), dtype=torch.int32, device="cuda")
        _write_pages(req_to_token, 1, [6, 10])
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.num_qo_heads = 4
        updater.num_kv_heads = 1
        updater.head_dim = 64
        updater.data_type = torch.bfloat16
        updater.q_data_type = torch.bfloat16
        updater.req_to_token = req_to_token
        updater.kv_last_page_len = torch.ones(
            1, dtype=torch.int32, device="cuda"
        )
        updater._swa_kv_pool = None
        updater.attn_backend = SimpleNamespace(
            page_aligned_prefill=False,
            page_size=PAGE_SIZE,
            dq_paged_kernel_lens=None,
        )
        paged = MagicMock()
        prefix_lens = torch.tensor([128], dtype=torch.int32, device="cuda")
        kv_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
        updater.call_begin_forward(
            MagicMock(),
            paged,
            torch.tensor([1], dtype=torch.int64, device="cuda"),
            prefix_lens,
            128,
            torch.tensor([192], dtype=torch.int32, device="cuda"),
            prefix_lens,
            None,
            kv_indptr,
            torch.zeros(2, dtype=torch.int32, device="cuda"),
            True,
            None,
            multi_item_params=MultiItemScoringParams(),
        )

        args = paged.begin_forward.call_args.args
        self.assertEqual(args[7], 1)
        self.assertEqual(args[3].item(), 1)
        torch.testing.assert_close(
            args[2][:128],
            req_to_token[1, :128],
            rtol=0,
            atol=0,
        )

    def test_prefill_updater_routes_only_aligned_prefix(self):
        req_to_token = torch.zeros((2, 256), dtype=torch.int32, device="cuda")
        _write_pages(req_to_token, 1, [9, 3])
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.num_qo_heads = 4
        updater.num_kv_heads = 1
        updater.head_dim = 64
        updater.data_type = torch.bfloat16
        updater.q_data_type = torch.bfloat16
        updater.req_to_token = req_to_token
        updater.kv_last_page_len = torch.ones(
            1, dtype=torch.int32, device="cuda"
        )
        updater._swa_kv_pool = None
        updater.attn_backend = SimpleNamespace(
            page_aligned_prefill=True,
            page_size=PAGE_SIZE,
            dq_paged_kernel_lens=None,
            prefill_physical_page_count=16,
        )
        paged = MagicMock()
        ragged = MagicMock()
        req_pool_indices = torch.tensor([1], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([192], dtype=torch.int32, device="cuda")
        prefix_lens = torch.tensor([128], dtype=torch.int32, device="cuda")
        kv_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")
        qo_indptr = torch.zeros(2, dtype=torch.int32, device="cuda")

        updater.call_begin_forward(
            ragged,
            paged,
            req_pool_indices,
            prefix_lens,
            128,
            seq_lens,
            prefix_lens,
            None,
            kv_indptr,
            qo_indptr,
            True,
            None,
            multi_item_params=MultiItemScoringParams(),
        )

        args = paged.begin_forward.call_args.args
        self.assertEqual(args[7], PAGE_SIZE)
        torch.testing.assert_close(
            args[1],
            torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            args[2][:2],
            torch.tensor([9, 3], dtype=torch.int32, device="cuda"),
            rtol=0,
            atol=0,
        )
        self.assertEqual(args[3].item(), PAGE_SIZE)

        paged.reset_mock()
        spec_info = SpecInput(SpecInputType.EAGLE_VERIFY)
        spec_info.generate_attn_arg_prefill = MagicMock(
            return_value=(
                req_to_token[1, :128].contiguous(),
                torch.tensor([0, 128], dtype=torch.int32, device="cuda"),
                torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
                None,
            )
        )
        updater.call_begin_forward(
            ragged,
            paged,
            req_pool_indices,
            prefix_lens,
            128,
            seq_lens,
            None,
            None,
            kv_indptr,
            qo_indptr,
            False,
            spec_info,
        )
        self.assertEqual(paged.begin_forward.call_args.args[7], 1)
        self.assertEqual(updater.kv_last_page_len.item(), 1)

        paged.reset_mock()
        unaligned_prefix = torch.tensor([65], dtype=torch.int32, device="cuda")
        updater.call_begin_forward(
            ragged,
            paged,
            req_pool_indices,
            unaligned_prefix,
            65,
            torch.tensor([129], dtype=torch.int32, device="cuda"),
            unaligned_prefix,
            None,
            kv_indptr,
            qo_indptr,
            True,
            None,
            multi_item_params=MultiItemScoringParams(),
        )
        self.assertEqual(paged.begin_forward.call_args.args[7], 1)
        self.assertEqual(updater.kv_last_page_len.item(), 1)

        updater.kv_last_page_len.fill_(PAGE_SIZE)
        decode = FlashInferIndicesUpdaterDecode.__new__(
            FlashInferIndicesUpdaterDecode
        )
        decode.num_qo_heads = 4
        decode.num_kv_heads = 1
        decode.head_dim = 64
        decode.data_type = torch.bfloat16
        decode.q_data_type = torch.bfloat16
        decode.attn_backend = updater.attn_backend
        decode.kv_last_page_len = updater.kv_last_page_len
        decode.req_to_token = req_to_token
        decode._swa_kv_pool = None
        decode_wrapper = MagicMock()
        decode_wrapper.is_cuda_graph_enabled = False
        decode.call_begin_forward(
            decode_wrapper,
            req_pool_indices,
            torch.tensor([128], dtype=torch.int32, device="cuda"),
            128,
            kv_indptr,
            None,
            None,
            seq_lens_cpu=None,
        )
        self.assertEqual(decode_wrapper.begin_forward.call_args.args[6], 1)
        self.assertEqual(decode.kv_last_page_len.item(), 1)


if __name__ == "__main__":
    unittest.main()
