import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_pools
from sglang.srt.speculative.eagle_worker_common import (
    _finalize_accept_tree_path,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-a")


class _SentinelKVCache:
    def __init__(self, values: torch.Tensor):
        self.values = values

    def move_kv_cache(
        self, target_slots: torch.Tensor, source_slots: torch.Tensor
    ) -> None:
        # Real backends must preserve every source when source and destination
        # overlap. Snapshotting here models that contract explicitly.
        source_values = self.values[source_slots.to(torch.long)].clone()
        self.values[target_slots.to(torch.long)] = source_values


class _SentinelAllocator:
    def __init__(self, cache: _SentinelKVCache):
        self.cache = cache

    def get_kvcache(self) -> _SentinelKVCache:
        return self.cache


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTreeAcceptPathCompaction(CustomTestCase):
    def _build_unified_pool(self, *, use_mla_backend: bool, page_size: int = 1):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=8,
            n_groups=1,
            num_heads=2,
            head_dim=4,
            state_size=4,
            conv_kernel=4,
        )
        params = Mamba2CacheParams(
            shape=shape,
            dtype=Mamba2StateDType(
                conv=torch.float16,
                temporal=torch.float32,
            ),
            layers=[1],
        )
        full_attention_layer_ids = [0, 2] if use_mla_backend else [0]
        return init_unified_mamba_pools(
            device="cuda",
            kv_cache_dtype=torch.float16,
            head_num=4,
            head_dim=16,
            page_size=page_size,
            start_layer=0,
            end_layer=3,
            is_draft_worker=False,
            use_mla_backend=use_mla_backend,
            kv_lora_rank=4 if use_mla_backend else None,
            qk_rope_head_dim=2 if use_mla_backend else None,
            mamba_layer_ids=[1],
            full_attention_layer_ids=full_attention_layer_ids,
            mamba2_cache_params=params,
            model_context_len=64,
            extra_max_context_len=8,
            max_total_num_tokens=32,
            max_mamba_cache_size=4,
            max_num_reqs=2,
            enable_memory_saver=False,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=8,
            disable_overlap_schedule=True,
            need_sort=False,
        )

    @staticmethod
    def _cache_rows(
        buffer: torch.Tensor, physical_slots: torch.Tensor, page_size: int
    ) -> torch.Tensor:
        if buffer.ndim == 4:
            return buffer[
                physical_slots.to(torch.long) // page_size,
                physical_slots.to(torch.long) % page_size,
            ]
        return buffer[physical_slots.to(torch.long)]

    @staticmethod
    def _write_cache_rows(
        buffer: torch.Tensor,
        physical_slots: torch.Tensor,
        page_size: int,
        rows: torch.Tensor,
    ) -> None:
        if buffer.ndim == 4:
            buffer[
                physical_slots.to(torch.long) // page_size,
                physical_slots.to(torch.long) % page_size,
            ] = rows
        else:
            buffer[physical_slots.to(torch.long)] = rows

    def test_hybrid_pool_translates_virtual_slots_before_physical_move(self):
        device = torch.device("cuda")
        virtual_to_physical = torch.arange(128, device=device, dtype=torch.int64)
        virtual_to_physical[20:28] = torch.tensor(
            [70, 62, 91, 55, 83, 47, 99, 38], device=device
        )

        original = torch.stack(
            (
                torch.arange(128, dtype=torch.float32, device=device),
                torch.arange(128, dtype=torch.float32, device=device) + 4000,
            ),
            dim=1,
        )
        physical_cache = _SentinelKVCache(original.clone())
        hybrid_pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        hybrid_pool.full_kv_pool = physical_cache
        hybrid_pool._full_move_translate = lambda slots: virtual_to_physical[
            slots.to(torch.long)
        ]

        target_virtual = torch.tensor([20, 21, 22], device=device)
        source_virtual = torch.tensor([20, 23, 27], device=device)
        target_physical = virtual_to_physical[target_virtual]
        source_physical = virtual_to_physical[source_virtual]
        serial = original[source_physical]

        hybrid_pool.move_kv_cache(target_virtual, source_virtual)
        torch.cuda.synchronize()

        torch.testing.assert_close(physical_cache.values[target_physical], serial)

    def test_factory_wires_physical_move_translation_for_mha_and_mla(self):
        for use_mla_backend in (False, True):
            with self.subTest(use_mla_backend=use_mla_backend):
                bundle = self._build_unified_pool(
                    use_mla_backend=use_mla_backend,
                    page_size=4,
                )
                allocator = bundle.token_to_kv_pool_allocator
                virtual_slots = allocator.alloc(8)
                self.assertIsNotNone(virtual_slots)
                physical_slots = allocator.translate_kv_loc(virtual_slots)
                self.assertFalse(torch.equal(virtual_slots, physical_slots))

                captured = []
                bundle.token_to_kv_pool.full_kv_pool.move_kv_cache = (
                    lambda target, source: captured.append(
                        (target.clone(), source.clone())
                    )
                )
                target_virtual = virtual_slots[:3]
                source_virtual = virtual_slots[
                    torch.tensor([0, 3, 7], device=virtual_slots.device)
                ]
                bundle.token_to_kv_pool.move_kv_cache(
                    target_virtual,
                    source_virtual,
                )
                torch.cuda.synchronize()

                self.assertEqual(len(captured), 1)
                torch.testing.assert_close(
                    captured[0][0], allocator.translate_kv_loc(target_virtual)
                )
                torch.testing.assert_close(
                    captured[0][1], allocator.translate_kv_loc(source_virtual)
                )
                if use_mla_backend:
                    self.assertFalse(
                        torch.equal(
                            captured[0][1],
                            allocator.translate_kv_loc_dense(source_virtual),
                        )
                    )

    def test_non_front_branch_matches_serial_commit_boundary(self):
        device = torch.device("cuda")
        num_nodes = 8
        prefix_len = 10

        # Target verification allocated one distinct physical KV slot per tree
        # node and installed them in tree order at the request's speculative
        # suffix. Every slot carries a visually distinct sentinel payload.
        out_cache_loc = torch.arange(20, 20 + num_nodes, device=device)
        req_to_token = torch.zeros((1, 64), dtype=torch.int64, device=device)
        req_to_token[0, prefix_len : prefix_len + num_nodes] = out_cache_loc
        original_kv = torch.stack(
            (
                torch.arange(64, dtype=torch.float32, device=device),
                torch.arange(64, dtype=torch.float32, device=device) + 1000,
            ),
            dim=1,
        )
        cache = _SentinelKVCache(original_kv.clone())

        # The accepted causal path is deliberately non-front and
        # non-contiguous in tree storage: root 0 -> node 3 -> node 7.
        accept_index = torch.tensor([[0, 3, 7, -1]], dtype=torch.int32, device=device)
        accept_lens = torch.tensor([3], dtype=torch.int32, device=device)
        original_tokens = torch.arange(100, 100 + num_nodes, dtype=torch.int32, device=device)
        original_hidden = torch.stack(
            (
                torch.arange(num_nodes, dtype=torch.float32, device=device) + 2000,
                torch.arange(num_nodes, dtype=torch.float32, device=device) + 3000,
            ),
            dim=1,
        )
        logits_output = SimpleNamespace(hidden_states=original_hidden.clone())
        batch = SimpleNamespace(
            seq_lens=torch.tensor([prefix_len], dtype=torch.int64, device=device),
            req_pool_indices=torch.tensor([0], dtype=torch.int64, device=device),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            out_cache_loc=out_cache_loc,
        )

        serial_nodes = torch.tensor([0, 3, 7], dtype=torch.long, device=device)
        serial_tokens = original_tokens[serial_nodes]
        serial_hidden = original_hidden[serial_nodes]
        serial_kv = original_kv[out_cache_loc[serial_nodes]]

        # This is the failure signature: a downstream front slice taken while
        # state remains in tree order installs sibling nodes 1 and 2.
        self.assertFalse(torch.equal(original_tokens[:3], serial_tokens))
        self.assertFalse(torch.equal(original_hidden[:3], serial_hidden))
        self.assertFalse(torch.equal(original_kv[out_cache_loc[:3]], serial_kv))

        compacted_tokens = _finalize_accept_tree_path(
            batch,
            accept_index,
            accept_lens,
            original_tokens,
            logits_output,
            1,
            token_to_kv_pool_allocator=_SentinelAllocator(cache),
            num_draft_tokens=num_nodes,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(compacted_tokens[:3], serial_tokens)
        torch.testing.assert_close(logits_output.hidden_states[:3], serial_hidden)

        committed_slots = req_to_token[0, prefix_len : prefix_len + 3]
        torch.testing.assert_close(committed_slots, out_cache_loc[:3])
        torch.testing.assert_close(cache.values[committed_slots], serial_kv)

        # Draft extend selects `accept_lens - 1` from the compact front block.
        # Its next token and parent hidden state must come from terminal node 7.
        next_draft_row = int(accept_lens.item()) - 1
        torch.testing.assert_close(
            compacted_tokens[next_draft_row], original_tokens[serial_nodes[-1]]
        )
        torch.testing.assert_close(
            logits_output.hidden_states[next_draft_row],
            original_hidden[serial_nodes[-1]],
        )

    def test_captured_non_front_cycles_match_serial_path(self):
        device = torch.device("cuda")
        bundle = self._build_unified_pool(use_mla_backend=False, page_size=1)
        allocator = bundle.token_to_kv_pool_allocator
        full_pool = bundle.token_to_kv_pool.full_kv_pool
        req_to_token_pool = bundle.req_to_token_pool
        req_index = 1
        num_nodes = 8
        prefix_len = 8

        prefix_slots = allocator.alloc(prefix_len)
        self.assertIsNotNone(prefix_slots)
        req_to_token_pool.req_to_token[req_index, :prefix_len] = prefix_slots

        k_buffer = full_pool.k_buffer[0]
        v_buffer = full_pool.v_buffer[0]
        row_shape = self._cache_rows(
            k_buffer,
            allocator.translate_kv_loc(prefix_slots[:1]),
            allocator.page_size,
        ).shape[1:]
        row_template = torch.arange(
            torch.tensor(row_shape).prod().item(),
            dtype=torch.float32,
            device=device,
        ).view(row_shape)
        prefix_rows = torch.stack(
            [row_template + 100 * i for i in range(prefix_len)]
        ).to(k_buffer.dtype)
        prefix_physical = allocator.translate_kv_loc(prefix_slots)
        self._write_cache_rows(
            k_buffer,
            prefix_physical,
            allocator.page_size,
            prefix_rows,
        )
        self._write_cache_rows(
            v_buffer,
            prefix_physical,
            allocator.page_size,
            prefix_rows + 20_000,
        )
        serial_k_rows = [row.clone() for row in prefix_rows]
        serial_v_rows = [row.clone() + 20_000 for row in prefix_rows]

        seq_lens = torch.tensor([prefix_len], dtype=torch.int64, device=device)
        out_cache_loc = torch.empty(num_nodes, dtype=torch.int64, device=device)
        accept_index = torch.empty((1, 4), dtype=torch.int32, device=device)
        accept_lens = torch.tensor([3], dtype=torch.int32, device=device)
        predict_input = torch.empty(num_nodes, dtype=torch.int32, device=device)
        hidden_input = torch.empty((num_nodes, 3), dtype=torch.float32, device=device)
        logits_output = SimpleNamespace(hidden_states=hidden_input)
        batch = SimpleNamespace(
            seq_lens=seq_lens,
            req_pool_indices=torch.tensor([req_index], dtype=torch.int64, device=device),
            req_to_token_pool=req_to_token_pool,
            out_cache_loc=out_cache_loc,
        )

        paths = (
            (0, 3, 7),
            (0, 2, 6),
            (0, 4, 5),
            (0, 1, 7),
        )
        graph = torch.cuda.CUDAGraph()
        compacted_predict = None
        compacted_hidden = None
        saw_reused_virtual_slot = False
        all_seen_virtual_slots = set(prefix_slots.tolist())

        for cycle, path in enumerate(paths):
            nodes = allocator.alloc(num_nodes)
            self.assertIsNotNone(nodes)
            node_list = nodes.tolist()
            if any(slot in all_seen_virtual_slots for slot in node_list):
                saw_reused_virtual_slot = True
            all_seen_virtual_slots.update(node_list)

            cycle_prefix_len = prefix_len + cycle * 3
            seq_lens.fill_(cycle_prefix_len)
            out_cache_loc.copy_(nodes)
            req_to_token_pool.req_to_token[
                req_index, cycle_prefix_len : cycle_prefix_len + num_nodes
            ] = nodes
            accept_index.copy_(
                torch.tensor([path + (-1,)], dtype=torch.int32, device=device)
            )
            predict_input.copy_(
                torch.arange(
                    cycle * 1000,
                    cycle * 1000 + num_nodes,
                    dtype=torch.int32,
                    device=device,
                )
            )
            hidden_input.copy_(
                torch.stack(
                    (
                        torch.arange(num_nodes, dtype=torch.float32, device=device)
                        + cycle * 1000
                        + 10_000,
                        torch.arange(num_nodes, dtype=torch.float32, device=device)
                        + cycle * 1000
                        + 11_000,
                        torch.arange(num_nodes, dtype=torch.float32, device=device)
                        + cycle * 1000
                        + 12_000,
                    ),
                    dim=1,
                )
            )

            node_physical = allocator.translate_kv_loc(nodes)
            node_k_rows = torch.stack(
                [
                    row_template + cycle * 4000 + node * 100
                    for node in range(num_nodes)
                ]
            ).to(k_buffer.dtype)
            node_v_rows = node_k_rows + 20_000
            self._write_cache_rows(
                k_buffer,
                node_physical,
                allocator.page_size,
                node_k_rows,
            )
            self._write_cache_rows(
                v_buffer,
                node_physical,
                allocator.page_size,
                node_v_rows,
            )

            path_index = torch.tensor(path, dtype=torch.long, device=device)
            expected_tokens = predict_input[path_index].clone()
            expected_hidden = hidden_input[path_index].clone()
            expected_k = node_k_rows[path_index].clone()
            expected_v = node_v_rows[path_index].clone()

            if cycle == 0:
                # Warm every kernel before capture, then restore the sentinel
                # inputs so the captured execution is the first checked cycle.
                _finalize_accept_tree_path(
                    batch,
                    accept_index,
                    accept_lens,
                    predict_input,
                    logits_output,
                    1,
                    token_to_kv_pool_allocator=allocator,
                    num_draft_tokens=num_nodes,
                )
                torch.cuda.synchronize()
                logits_output.hidden_states = hidden_input
                predict_input.copy_(
                    torch.arange(
                        num_nodes,
                        dtype=torch.int32,
                        device=device,
                    )
                )
                hidden_input.copy_(
                    expected_hidden.new_tensor(
                        [
                            [10_000 + i, 11_000 + i, 12_000 + i]
                            for i in range(num_nodes)
                        ]
                    )
                )
                self._write_cache_rows(
                    k_buffer,
                    node_physical,
                    allocator.page_size,
                    node_k_rows,
                )
                self._write_cache_rows(
                    v_buffer,
                    node_physical,
                    allocator.page_size,
                    node_v_rows,
                )
                with torch.cuda.graph(graph):
                    compacted_predict = _finalize_accept_tree_path(
                        batch,
                        accept_index,
                        accept_lens,
                        predict_input,
                        logits_output,
                        1,
                        token_to_kv_pool_allocator=allocator,
                        num_draft_tokens=num_nodes,
                    )
                compacted_hidden = logits_output.hidden_states
                # Replay is the runtime contract. Graph-pool outputs need not
                # retain the values produced while the graph was captured.
                graph.replay()
            else:
                graph.replay()
            torch.cuda.synchronize()

            torch.testing.assert_close(compacted_predict[:3], expected_tokens)
            torch.testing.assert_close(compacted_hidden[:3], expected_hidden)
            torch.testing.assert_close(compacted_predict[2], expected_tokens[-1])
            torch.testing.assert_close(compacted_hidden[2], expected_hidden[-1])

            committed_virtual = req_to_token_pool.req_to_token[
                req_index, cycle_prefix_len : cycle_prefix_len + 3
            ]
            committed_physical = allocator.translate_kv_loc(committed_virtual)
            torch.testing.assert_close(
                self._cache_rows(k_buffer, committed_physical, allocator.page_size),
                expected_k,
            )
            torch.testing.assert_close(
                self._cache_rows(v_buffer, committed_physical, allocator.page_size),
                expected_v,
            )

            serial_k_rows.extend(row.clone() for row in expected_k)
            serial_v_rows.extend(row.clone() for row in expected_v)
            allocator.free(nodes[3:])
            torch.cuda.synchronize()

            whole_prefix_virtual = req_to_token_pool.req_to_token[
                req_index, : cycle_prefix_len + 3
            ]
            whole_prefix_physical = allocator.translate_kv_loc(whole_prefix_virtual)
            torch.testing.assert_close(
                self._cache_rows(
                    k_buffer,
                    whole_prefix_physical,
                    allocator.page_size,
                ),
                torch.stack(serial_k_rows),
            )
            torch.testing.assert_close(
                self._cache_rows(
                    v_buffer,
                    whole_prefix_physical,
                    allocator.page_size,
                ),
                torch.stack(serial_v_rows),
            )

        self.assertTrue(saw_reused_virtual_slot)


if __name__ == "__main__":
    unittest.main()
