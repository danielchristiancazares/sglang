import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTorchNativeExtend(unittest.TestCase):
    def _run_extend(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        num_query_heads: int,
        causal: bool = True,
        sliding_window_size: int | None = None,
    ) -> torch.Tensor:
        output = torch.empty(
            query.shape[0],
            num_query_heads,
            value_cache.shape[-1],
            dtype=query.dtype,
        )
        TorchNativeAttnBackend._run_sdpa_forward_extend(
            TorchNativeAttnBackend.__new__(TorchNativeAttnBackend),
            query,
            output,
            key_cache,
            value_cache,
            req_to_token,
            req_pool_indices,
            prefix_lens + extend_lens,
            prefix_lens,
            extend_lens,
            scaling=query.shape[-1] ** -0.5,
            enable_gqa=num_query_heads != key_cache.shape[-2],
            causal=causal,
            sliding_window_size=sliding_window_size,
        )
        return output

    @staticmethod
    def _full_reference(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        prefix_len: int,
        causal: bool = True,
        sliding_window_size: int | None = None,
    ) -> torch.Tensor:
        query = query.movedim(0, 1).unsqueeze(0)
        key = key.movedim(0, 1).unsqueeze(0)
        value = value.movedim(0, 1).unsqueeze(0)
        mask = None
        if sliding_window_size is not None:
            seq_len = query.shape[-2]
            query_pos = torch.arange(seq_len).unsqueeze(1)
            key_pos = torch.arange(seq_len).unsqueeze(0)
            mask = (key_pos <= query_pos) & (key_pos >= query_pos - sliding_window_size)
            causal = False
        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            is_causal=causal,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return output.squeeze(0).movedim(0, 1)[prefix_len:]

    def test_gqa_partial_extend_matches_full_sequence_with_shuffled_cache(self):
        torch.manual_seed(31)
        prefix_len, extend_len = 3, 4
        num_query_heads, num_kv_heads, head_dim = 4, 2, 8
        seq_len = prefix_len + extend_len
        full_query = torch.randn(seq_len, num_query_heads, head_dim)
        logical_key = torch.randn(seq_len, num_kv_heads, head_dim)
        logical_value = torch.randn_like(logical_key)
        physical_slots = torch.tensor([6, 1, 8, 0, 7, 3, 5], dtype=torch.int32)
        key_cache = torch.randn(9, num_kv_heads, head_dim)
        value_cache = torch.randn_like(key_cache)
        key_cache[physical_slots.long()] = logical_key
        value_cache[physical_slots.long()] = logical_value

        actual = self._run_extend(
            query=full_query[prefix_len:],
            key_cache=key_cache,
            value_cache=value_cache,
            req_to_token=physical_slots.unsqueeze(0),
            req_pool_indices=torch.tensor([0]),
            prefix_lens=torch.tensor([prefix_len]),
            extend_lens=torch.tensor([extend_len]),
            num_query_heads=num_query_heads,
        )
        expected = self._full_reference(
            full_query,
            logical_key,
            logical_value,
            prefix_len=prefix_len,
        )
        torch.testing.assert_close(actual, expected)

    def test_partial_extend_cannot_read_future_values(self):
        prefix_len, extend_len = 2, 3
        seq_len = prefix_len + extend_len
        query = torch.zeros(extend_len, 1, 1)
        key_cache = torch.zeros(seq_len, 1, 1)
        value_cache = torch.tensor([1.0, 1.0, 2.0, 1_000.0, 100_000.0]).view(
            seq_len, 1, 1
        )
        actual = self._run_extend(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            req_to_token=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
            req_pool_indices=torch.tensor([0]),
            prefix_lens=torch.tensor([prefix_len]),
            extend_lens=torch.tensor([extend_len]),
            num_query_heads=1,
        )
        expected = torch.tensor([4.0 / 3.0, 251.0, 20_200.8]).view(extend_len, 1, 1)
        torch.testing.assert_close(actual, expected)

    def test_sliding_window_partial_extend_matches_full_sequence(self):
        torch.manual_seed(37)
        prefix_len, extend_len = 5, 3
        num_query_heads, num_kv_heads, head_dim = 4, 2, 8
        seq_len = prefix_len + extend_len
        full_query = torch.randn(seq_len, num_query_heads, head_dim)
        key_cache = torch.randn(seq_len, num_kv_heads, head_dim)
        value_cache = torch.randn_like(key_cache)

        actual = self._run_extend(
            query=full_query[prefix_len:],
            key_cache=key_cache,
            value_cache=value_cache,
            req_to_token=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
            req_pool_indices=torch.tensor([0]),
            prefix_lens=torch.tensor([prefix_len]),
            extend_lens=torch.tensor([extend_len]),
            num_query_heads=num_query_heads,
            sliding_window_size=2,
        )
        expected = self._full_reference(
            full_query,
            key_cache,
            value_cache,
            prefix_len=prefix_len,
            sliding_window_size=2,
        )
        torch.testing.assert_close(actual, expected)

    def test_ragged_gqa_batch_matches_each_full_sequence(self):
        torch.manual_seed(41)
        prefix_lens = torch.tensor([2, 4])
        extend_lens = torch.tensor([3, 2])
        num_query_heads, num_kv_heads, head_dim = 4, 2, 8
        full_queries = [
            torch.randn(5, num_query_heads, head_dim),
            torch.randn(6, num_query_heads, head_dim),
        ]
        keys = [
            torch.randn(5, num_kv_heads, head_dim),
            torch.randn(6, num_kv_heads, head_dim),
        ]
        values = [torch.randn_like(key) for key in keys]
        query = torch.cat(
            [
                full_queries[index][prefix_lens[index] :]
                for index in range(len(full_queries))
            ]
        )
        req_to_token = torch.zeros(2, 6, dtype=torch.int32)
        req_to_token[0, :5] = torch.arange(5, dtype=torch.int32)
        req_to_token[1, :6] = torch.arange(5, 11, dtype=torch.int32)

        actual = self._run_extend(
            query=query,
            key_cache=torch.cat(keys),
            value_cache=torch.cat(values),
            req_to_token=req_to_token,
            req_pool_indices=torch.tensor([0, 1]),
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            num_query_heads=num_query_heads,
        )
        expected = torch.cat(
            [
                self._full_reference(
                    full_queries[index],
                    keys[index],
                    values[index],
                    prefix_len=prefix_lens[index],
                )
                for index in range(len(full_queries))
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_noncausal_partial_extend_attends_all_keys(self):
        torch.manual_seed(43)
        prefix_len, extend_len = 3, 2
        num_heads, head_dim = 2, 8
        seq_len = prefix_len + extend_len
        query = torch.randn(extend_len, num_heads, head_dim)
        key_cache = torch.randn(seq_len, num_heads, head_dim)
        value_cache = torch.randn_like(key_cache)
        actual = self._run_extend(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            req_to_token=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
            req_pool_indices=torch.tensor([0]),
            prefix_lens=torch.tensor([prefix_len]),
            extend_lens=torch.tensor([extend_len]),
            num_query_heads=num_heads,
            causal=False,
        )
        expected = (
            scaled_dot_product_attention(
                query.movedim(0, 1).unsqueeze(0),
                key_cache.movedim(0, 1).unsqueeze(0),
                value_cache.movedim(0, 1).unsqueeze(0),
                is_causal=False,
            )
            .squeeze(0)
            .movedim(0, 1)
        )
        torch.testing.assert_close(actual, expected)

    def test_empty_extend_returns_empty_output(self):
        actual = self._run_extend(
            query=torch.empty(0, 2, 8),
            key_cache=torch.randn(3, 2, 8),
            value_cache=torch.randn(3, 2, 8),
            req_to_token=torch.arange(3, dtype=torch.int32).unsqueeze(0),
            req_pool_indices=torch.tensor([0]),
            prefix_lens=torch.tensor([3]),
            extend_lens=torch.tensor([0]),
            num_query_heads=2,
        )
        self.assertEqual(actual.shape, (0, 2, 8))


if __name__ == "__main__":
    unittest.main()
