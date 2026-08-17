from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.attention.fla.gdn_tree_replay import (
    commit_gdn_tree_replay_all_layers,
    gdn_tree_replay_verify,
)
from sglang.test.test_utils import CustomTestCase


def _reference_tree(
    *,
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    checkpoint,
    state_indices,
    parent,
    scale,
):
    batch_size, num_nodes = parent.shape
    _, num_key_heads, key_dim = q.shape
    num_value_heads = v.shape[1]
    group_size = num_value_heads // num_key_heads
    output = torch.empty_like(v)
    states = [[None] * num_nodes for _ in range(batch_size)]
    for batch in range(batch_size):
        for node in range(num_nodes):
            token = batch * num_nodes + node
            parent_node = int(parent[batch, node]) if node else -1
            for value_head in range(num_value_heads):
                key_head = value_head // group_size
                state = (
                    checkpoint[int(state_indices[batch]), value_head].float().clone()
                    if parent_node < 0
                    else states[batch][parent_node][value_head].clone()
                )
                qv = q[token, key_head].float()
                kv = k[token, key_head].float()
                qv = qv / torch.sqrt(torch.sum(qv * qv) + 1.0e-6)
                kv = kv / torch.sqrt(torch.sum(kv * kv) + 1.0e-6)
                x = a[token, value_head].float() + dt_bias[value_head]
                gate = -torch.exp(A_log[value_head]) * torch.where(
                    x <= 20.0, torch.log1p(torch.exp(x)), x
                )
                beta = torch.sigmoid(b[token, value_head].float())
                state *= torch.exp(gate)
                residual = v[token, value_head].float() - torch.sum(
                    state * kv[None, :], dim=1
                )
                residual *= beta
                state += residual[:, None] * kv[None, :]
                output[token, value_head] = torch.sum(
                    state * (qv * scale)[None, :], dim=1
                ).to(torch.bfloat16)
                if states[batch][node] is None:
                    states[batch][node] = [None] * num_value_heads
                states[batch][node][value_head] = state
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestGdnTreeReplayJit(CustomTestCase):
    def test_tree_verify_matches_materialized_recurrence(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(18341)
        batch_size = 2
        num_nodes = 7
        num_key_heads = 2
        num_value_heads = 4
        dim = 128
        slots = 3
        record_len = 8
        total_tokens = batch_size * num_nodes
        parent = torch.tensor(
            [[-1, 0, 0, 1, 1, 2, 3], [-1, 0, 0, 1, 2, 2, 4]],
            dtype=torch.int32,
            device=device,
        )
        state_indices = torch.tensor([0, 2], dtype=torch.int32, device=device)
        checkpoint = torch.randn(
            (slots, num_value_heads, dim, dim),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ) * 0.02
        checkpoint_before = checkpoint.clone()
        # Qwen produces these as split views of wider projection buffers. Give
        # each source a distinct padded token stride to exercise the zero-copy
        # production contract.
        q = torch.randn(
            (total_tokens, num_key_heads + 1, dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )[:, :num_key_heads]
        k = torch.randn(
            (total_tokens, num_key_heads + 2, dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )[:, :num_key_heads]
        v = (
            torch.randn(
                (total_tokens, num_value_heads + 3, dim),
                generator=generator,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.2
        )[:, :num_value_heads]
        a = torch.randn(
            (total_tokens, num_value_heads + 4),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )[:, :num_value_heads]
        b = torch.randn(
            (total_tokens, num_value_heads + 5),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )[:, :num_value_heads]
        A_log = torch.randn(
            (num_value_heads,), generator=generator, device=device
        ).float() - 2.0
        dt_bias = torch.randn(
            (num_value_heads,), generator=generator, device=device
        ).bfloat16()
        rawv = torch.zeros(
            (slots, num_value_heads, record_len, dim),
            dtype=torch.bfloat16,
            device=device,
        )
        rawk = torch.zeros(
            (slots, num_key_heads, record_len, dim),
            dtype=torch.bfloat16,
            device=device,
        )
        gates = torch.zeros(
            (slots, num_value_heads, record_len),
            dtype=torch.float32,
            device=device,
        )
        betas = torch.zeros_like(gates)
        scale = dim**-0.5

        expected = _reference_tree(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            checkpoint=checkpoint,
            state_indices=state_indices,
            parent=parent,
            scale=scale,
        )
        actual = gdn_tree_replay_verify(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            checkpoint_state=checkpoint,
            state_indices=state_indices,
            parent=parent,
            rawv_cache=rawv,
            rawk_cache=rawk,
            g_cache=gates,
            beta_cache=betas,
            scale=scale,
            max_tree_depth=4,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015625)
        torch.testing.assert_close(checkpoint, checkpoint_before, rtol=0, atol=0)
        for batch, slot in enumerate(state_indices.tolist()):
            token_slice = slice(batch * num_nodes, (batch + 1) * num_nodes)
            torch.testing.assert_close(
                rawv[slot, :, :num_nodes].transpose(0, 1),
                v[token_slice],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                rawk[slot, :, :num_nodes].transpose(0, 1),
                k[token_slice],
                rtol=0,
                atol=0,
            )
            expected_g = -torch.exp(A_log)[None, :] * torch.nn.functional.softplus(
                a[token_slice].float() + dt_bias[None, :]
            )
            torch.testing.assert_close(
                gates[slot, :, :num_nodes].transpose(0, 1),
                expected_g,
                rtol=2e-5,
                atol=2e-6,
            )
            torch.testing.assert_close(
                betas[slot, :, :num_nodes].transpose(0, 1),
                torch.sigmoid(b[token_slice].float()),
                rtol=2e-5,
                atol=2e-6,
            )

    def test_commit_replays_only_accepted_nodes(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(77201)
        layers = 2
        slots = 4
        batch_size = 2
        num_nodes = 7
        max_depth = 3
        num_key_heads = 2
        num_value_heads = 4
        dim = 128
        checkpoint = torch.randn(
            (layers, slots, num_value_heads, dim, dim),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ) * 0.02
        expected = checkpoint.clone()
        rawv = torch.randn(
            (layers, slots, num_value_heads, num_nodes, dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) * 0.2
        rawk = torch.randn(
            (layers, slots, num_key_heads, num_nodes, dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        gates = -torch.rand(
            (layers, slots, num_value_heads, num_nodes),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        betas = torch.rand(
            (layers, slots, num_value_heads, num_nodes),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        state_indices = torch.tensor([0, 2], dtype=torch.int32, device=device)
        accept_index = torch.tensor(
            [[0, 1, 4], [7, 9, 12]], dtype=torch.int32, device=device
        )
        accept_lens = torch.tensor([3, 3], dtype=torch.int32, device=device)
        # Production prefix-cache track slots use the int64 CUDA-graph buffer.
        track_indices = torch.tensor([1, -1], dtype=torch.int64, device=device)
        track_nodes = torch.tensor([1, -1], dtype=torch.int32, device=device)

        for layer in range(layers):
            for batch, slot in enumerate(state_indices.tolist()):
                for value_head in range(num_value_heads):
                    key_head = value_head // (num_value_heads // num_key_heads)
                    state = expected[layer, slot, value_head].clone()
                    for ordinal in range(int(accept_lens[batch])):
                        node = int(accept_index[batch, ordinal]) - batch * num_nodes
                        kv = rawk[layer, slot, key_head, node].float()
                        kv /= torch.sqrt(torch.sum(kv * kv) + 1.0e-6)
                        state *= torch.exp(gates[layer, slot, value_head, node])
                        residual = rawv[layer, slot, value_head, node].float()
                        residual -= torch.sum(state * kv[None, :], dim=1)
                        residual *= betas[layer, slot, value_head, node]
                        state += residual[:, None] * kv[None, :]
                        if node == int(track_nodes[batch]) and int(
                            track_indices[batch]
                        ) >= 0:
                            expected[
                                layer, int(track_indices[batch]), value_head
                            ] = state
                    expected[layer, slot, value_head] = state

        commit_gdn_tree_replay_all_layers(
            checkpoint_state=checkpoint,
            rawv_cache=rawv,
            rawk_cache=rawk,
            g_cache=gates,
            beta_cache=betas,
            state_indices=state_indices,
            accept_index=accept_index,
            accept_lens=accept_lens,
            num_tree_nodes=num_nodes,
            mamba_track_indices=track_indices,
            mamba_track_nodes=track_nodes,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(checkpoint, expected, rtol=2e-5, atol=2e-6)

    def test_cuda_graph_replay_refreshes_tree_and_commits_selected_path(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(92531)
        batch_size = 1
        num_nodes = 7
        num_key_heads = 2
        num_value_heads = 4
        dim = 128
        slots = 3
        total_tokens = batch_size * num_nodes

        parent = torch.tensor(
            [[-1, 0, 0, 1, 1, 2, 3]], dtype=torch.int32, device=device
        )
        state_indices = torch.tensor([0], dtype=torch.int32, device=device)
        checkpoint = torch.randn(
            (1, slots, num_value_heads, dim, dim),
            generator=generator,
            dtype=torch.float32,
            device=device,
        ) * 0.02
        q = torch.randn(
            (total_tokens, num_key_heads, dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        k = torch.randn(
            q.shape, generator=generator, dtype=q.dtype, device=device
        )
        v = torch.randn(
            (total_tokens, num_value_heads, dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) * 0.2
        a = torch.randn(
            (total_tokens, num_value_heads),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        b = torch.randn(
            a.shape, generator=generator, dtype=a.dtype, device=device
        )
        A_log = torch.randn(
            (num_value_heads,), generator=generator, device=device
        ).float() - 2.0
        dt_bias = torch.randn(
            (num_value_heads,), generator=generator, device=device
        ).bfloat16()
        rawv = torch.zeros(
            (1, slots, num_value_heads, num_nodes, dim),
            dtype=torch.bfloat16,
            device=device,
        )
        rawk = torch.zeros(
            (1, slots, num_key_heads, num_nodes, dim),
            dtype=torch.bfloat16,
            device=device,
        )
        gates = torch.zeros(
            (1, slots, num_value_heads, num_nodes),
            dtype=torch.float32,
            device=device,
        )
        betas = torch.zeros_like(gates)
        scale = dim**-0.5

        # Warm the JIT wrapper before capture, then capture fixed tensor pointers.
        gdn_tree_replay_verify(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            checkpoint_state=checkpoint[0],
            state_indices=state_indices,
            parent=parent,
            rawv_cache=rawv[0],
            rawk_cache=rawk[0],
            g_cache=gates[0],
            beta_cache=betas[0],
            scale=scale,
            max_tree_depth=4,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = gdn_tree_replay_verify(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                checkpoint_state=checkpoint[0],
                state_indices=state_indices,
                parent=parent,
                rawv_cache=rawv[0],
                rawk_cache=rawk[0],
                g_cache=gates[0],
                beta_cache=betas[0],
                scale=scale,
                max_tree_depth=4,
            )

        # Refresh every dynamic input, including the slot and topology, then
        # replay the captured kernels. This mirrors server graph metadata replay.
        q.copy_(torch.randn(q.shape, generator=generator, dtype=q.dtype, device=device))
        k.copy_(torch.randn(k.shape, generator=generator, dtype=k.dtype, device=device))
        v.copy_(
            torch.randn(v.shape, generator=generator, dtype=v.dtype, device=device)
            * 0.2
        )
        a.copy_(torch.randn(a.shape, generator=generator, dtype=a.dtype, device=device))
        b.copy_(torch.randn(b.shape, generator=generator, dtype=b.dtype, device=device))
        parent.copy_(
            torch.tensor(
                [[-1, 0, 0, 1, 2, 2, 5]], dtype=torch.int32, device=device
            )
        )
        state_indices.fill_(2)
        checkpoint_before = checkpoint.clone()
        expected_output = _reference_tree(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            checkpoint=checkpoint[0],
            state_indices=state_indices,
            parent=parent,
            scale=scale,
        )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            captured_output, expected_output, rtol=0.02, atol=0.015625
        )
        torch.testing.assert_close(checkpoint, checkpoint_before, rtol=0, atol=0)
        torch.testing.assert_close(
            rawv[0, 2].transpose(0, 1), v, rtol=0, atol=0
        )
        torch.testing.assert_close(
            rawk[0, 2].transpose(0, 1), k, rtol=0, atol=0
        )

        # Commit root -> node 2 -> node 5 without a prefix-track destination.
        expected_state = checkpoint_before.clone()
        for value_head in range(num_value_heads):
            key_head = value_head // (num_value_heads // num_key_heads)
            state = expected_state[0, 2, value_head]
            for node in (0, 2, 5):
                kv = k[node, key_head].float()
                kv /= torch.sqrt(torch.sum(kv * kv) + 1.0e-6)
                gate = -torch.exp(A_log[value_head]) * torch.nn.functional.softplus(
                    a[node, value_head].float() + dt_bias[value_head]
                )
                beta = torch.sigmoid(b[node, value_head].float())
                state *= torch.exp(gate)
                residual = v[node, value_head].float() - torch.sum(
                    state * kv[None, :], dim=1
                )
                state += (beta * residual)[:, None] * kv[None, :]
        commit_gdn_tree_replay_all_layers(
            checkpoint_state=checkpoint,
            rawv_cache=rawv,
            rawk_cache=rawk,
            g_cache=gates,
            beta_cache=betas,
            state_indices=state_indices,
            accept_index=torch.tensor(
                [[0, 2, 5]], dtype=torch.int32, device=device
            ),
            accept_lens=torch.tensor([3], dtype=torch.int32, device=device),
            num_tree_nodes=num_nodes,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(checkpoint, expected_state, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
