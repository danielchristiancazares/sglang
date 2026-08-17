from __future__ import annotations

import unittest

import torch

from sglang.srt.model_executor.cuda_graph_composite import CudaGraphChildSequence


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


if __name__ == "__main__":
    unittest.main()
