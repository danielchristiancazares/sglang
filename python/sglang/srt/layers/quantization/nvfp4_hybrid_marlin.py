from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.quantization.nvfp4_marlin_relayout import (
    nvfp4_marlin_relayout_,
    preload_nvfp4_marlin_relayout,
)
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_buffer

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_MARLIN_TILE_K = 16
_MARLIN_TILE_N = 64
_MAX_RELAYOUT_WEIGHT_BYTES = 128 << 20
_MAX_MARLIN_TOKENS = 4
_SELECTED_GATE_UP_SHAPE = (34816, 5120)


def use_hybrid_marlin_for_num_tokens(num_tokens: int) -> bool:
    return num_tokens <= _MAX_MARLIN_TOKENS


def prepare_nvfp4_layer_for_hybrid_marlin(
    layer: torch.nn.Module,
    *,
    weight_global_scale: torch.Tensor,
) -> None:
    size_n = layer.output_size_per_partition
    size_k = layer.input_size_per_partition
    weight_bytes = layer.weight.numel() * layer.weight.element_size()
    if (
        (size_n, size_k) != _SELECTED_GATE_UP_SHAPE
        or size_n % _MARLIN_TILE_N != 0
        or size_k % _MARLIN_TILE_K != 0
        or weight_bytes > _MAX_RELAYOUT_WEIGHT_BYTES
        or getattr(layer, "bias", None) is not None
    ):
        layer._nvfp4_hybrid_marlin = False
        return

    param_dtype = getattr(layer, "params_dtype", getattr(layer, "orig_dtype", None))
    if param_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("Hybrid NVFP4 Marlin requires FP16 or BF16 activations.")

    raw_scale = layer.weight_scale.T.contiguous().to(param_dtype)
    marlin_scale = marlin_permute_scales(
        s=raw_scale,
        size_k=size_k,
        size_n=size_n,
        group_size=16,
    )
    marlin_scale = nvfp4_marlin_process_scales(marlin_scale)
    copy_or_rebind_param(layer, "weight_scale_marlin", marlin_scale)

    marlin_global_scale = nvfp4_marlin_process_global_scale(
        weight_global_scale.to(param_dtype)
    )
    copy_or_rebind_param(
        layer,
        "weight_global_scale_marlin",
        marlin_global_scale,
    )
    layer.weight_marlin = layer.weight.view(torch.int32).view(
        size_k // _MARLIN_TILE_K,
        size_n * 2,
    )
    layer.workspace_marlin = marlin_make_workspace(layer.weight.device)
    layer._nvfp4_hybrid_marlin = True


class Nvfp4HybridMarlinManager:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: str,
        is_draft_worker: bool,
        enabled: bool,
    ) -> None:
        prepared_layers = tuple(
            layer
            for layer in model.modules()
            if getattr(layer, "_nvfp4_hybrid_marlin", False)
        )
        active_layers = prepared_layers if enabled else ()
        for layer in prepared_layers:
            layer._nvfp4_hybrid_marlin_active = enabled
        self.layers = active_layers
        self._marlin_layout = False
        self.scratch = None
        if not self.layers:
            return

        preload_nvfp4_marlin_relayout()
        max_bytes = max(layer.weight.numel() for layer in self.layers)
        role = "draft" if is_draft_worker else "target"
        self.scratch = get_buffer(
            f"nvfp4_hybrid_marlin_relayout_{role}_{device}",
            lambda: torch.empty(max_bytes, dtype=torch.uint8, device=device),
        )
        logger.info(
            "Hybrid NVFP4 Marlin enabled for %d layers with %.2f MiB relayout scratch.",
            len(self.layers),
            max_bytes / (1 << 20),
        )

    def prepare_for_forward(self, forward_batch: ForwardBatch) -> None:
        if not self.layers:
            return
        self._switch(use_hybrid_marlin_for_num_tokens(forward_batch.input_ids.numel()))

    def finish_forward(self, forward_batch: ForwardBatch) -> None:
        if (
            not self.layers
            or self._marlin_layout
            or forward_batch.is_prefill_only
            or not forward_batch.contains_last_prefill_chunk
            or forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED)
        ):
            return
        self._switch(True)

    def _switch(self, to_marlin: bool) -> None:
        if self._marlin_layout == to_marlin:
            return
        for layer in self.layers:
            weight = layer.weight.view(torch.uint8).reshape(-1)
            nvfp4_marlin_relayout_(
                weight,
                self.scratch,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                to_marlin=to_marlin,
            )
        self._marlin_layout = to_marlin
