from __future__ import annotations

import unittest

import torch

from sglang.srt.model_executor.cuda_graph_composite import CudaGraphChildSequence
from sglang.srt.speculative.spec_utils import fast_sample


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCudaGraphChildSequence(unittest.TestCase):
    def test_child_graphs_share_device_resident_values(self):
        device = torch.device("cuda")
        source = torch.zeros(32, device=device)
        first = torch.empty_like(source)
        second = torch.empty_like(source)
        output = torch.empty_like(source)

        graph_one = torch.cuda.CUDAGraph(keep_graph=True)
        graph_two = torch.cuda.CUDAGraph(keep_graph=True)
        graph_three = torch.cuda.CUDAGraph(keep_graph=True)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph_one):
            torch.mul(source, 2.0, out=first)
        with torch.cuda.graph(graph_two):
            torch.add(first, 1.0, out=second)
        with torch.cuda.graph(graph_three):
            torch.mul(second, 3.0, out=output)

        with CudaGraphChildSequence((graph_one, graph_two, graph_three)) as sequence:
            for value in (4.0, -2.5):
                source.fill_(value)
                sequence.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output, torch.full_like(output, (value * 2.0 + 1.0) * 3.0)
                )

    def test_external_races_refresh_sampling_in_raw_child_graph(self):
        device = torch.device("cuda")
        probs = torch.tensor([[0.5, 0.5]], device=device)
        races = torch.ones_like(probs, dtype=torch.float32)

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            sample_p, sample_index = fast_sample(probs, races=races)

        with CudaGraphChildSequence((graph,)) as sequence:
            for race_values, expected_index in (
                ((0.1, 10.0), 0),
                ((10.0, 0.1), 1),
            ):
                races.copy_(torch.tensor([race_values], device=device))
                sequence.replay()
                torch.cuda.synchronize()
                self.assertEqual(int(sample_index.item()), expected_index)
                torch.testing.assert_close(
                    sample_p,
                    probs[:, expected_index : expected_index + 1],
                )


if __name__ == "__main__":
    unittest.main()
