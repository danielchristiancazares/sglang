"""Small CUDA-graph composition primitives.

The CUDA runtime clones child graphs into the parent graph.  Keeping the
source graph objects alive is still intentional: their static tensors own the
addresses referenced by the cloned kernel and memcpy nodes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

try:
    from cuda.bindings import runtime as cuda_rt
except ImportError:
    cuda_rt = None

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
    checkCudaErrors,
)


def raw_cuda_graph(graph: Any):
    """Return a CUDA runtime graph handle from a supported graph wrapper."""
    raw = getattr(graph, "raw_graph", None)
    if raw is not None:
        return raw
    getter = getattr(graph, "raw_cuda_graph", None)
    if getter is not None:
        return getter()
    raise TypeError(f"Unsupported CUDA graph object: {type(graph).__name__}")


class CudaGraphChildSequence:
    """Launch an ordered sequence of captured graphs as one parent graph.

    Each child is cloned into a fresh parent and depends on the preceding
    child.  Replay therefore costs one ``cudaGraphLaunch`` while preserving
    all device-side data dependencies between the source graphs.
    """

    def __init__(self, children: Iterable[Any]):
        if cuda_rt is None:
            raise RuntimeError(
                "cuda.bindings is required for CUDA child-graph composition"
            )
        self._children = tuple(children)
        if not self._children:
            raise ValueError("A CUDA graph child sequence requires at least one graph")

        self._graph = None
        self._graph_exec = None
        try:
            self._graph = checkCudaErrors(cuda_rt.cudaGraphCreate(0))
            dependency = None
            for child in self._children:
                dependencies = None if dependency is None else [dependency]
                dependency = checkCudaErrors(
                    cuda_rt.cudaGraphAddChildGraphNode(
                        self._graph,
                        dependencies,
                        0 if dependencies is None else len(dependencies),
                        raw_cuda_graph(child),
                    )
                )
            self._graph_exec = checkCudaErrors(
                cuda_rt.cudaGraphInstantiateWithFlags(self._graph, 0)
            )
        except Exception:
            self.close()
            raise

    def replay(self, stream: torch.cuda.Stream | None = None) -> None:
        if self._graph_exec is None:
            raise RuntimeError("CUDA graph child sequence is closed")
        if stream is None:
            stream = torch.cuda.current_stream()
        checkCudaErrors(cuda_rt.cudaGraphLaunch(self._graph_exec, stream.cuda_stream))

    def close(self) -> None:
        graph_exec, self._graph_exec = self._graph_exec, None
        graph, self._graph = self._graph, None
        if cuda_rt is None:
            return
        if graph_exec is not None:
            checkCudaErrors(cuda_rt.cudaGraphExecDestroy(graph_exec))
        if graph is not None:
            checkCudaErrors(cuda_rt.cudaGraphDestroy(graph))

    def __enter__(self) -> "CudaGraphChildSequence":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
