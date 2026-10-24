"""PyTorch MPS device operations for the SRT platform layer."""

from __future__ import annotations

import json
import re
import subprocess
from functools import cached_property
from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


def _recommended_working_set_size() -> int:
    getter = getattr(torch.mps, "recommended_max_memory", None)
    if not callable(getter):
        raise RuntimeError(
            "The SGLang MPS backend requires "
            "torch.mps.recommended_max_memory() from Torch >= 2.13.0"
        )
    total = int(getter())
    if total <= 0:
        raise RuntimeError(
            "torch.mps.recommended_max_memory() returned a non-positive Metal "
            "working-set limit; refusing to size the model and KV pools from "
            "raw system RAM"
        )
    return total


class MpsDeviceMixin(DeviceMixin):
    """Device operations backed by ``torch.mps``."""

    _enum: PlatformEnum = PlatformEnum.MPS
    device_name: str = "mps"
    device_type: str = "mps"

    @cached_property
    def _device_description(self) -> tuple[str, int]:
        name = "Apple Metal GPU"
        total_bytes = 0
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                check=True,
                capture_output=True,
                text=True,
            )
            displays = json.loads(result.stdout).get("SPDisplaysDataType", [])
            candidates = [
                item
                for item in displays
                if item.get("spdisplays_metal") == "spdisplays_supported"
                or item.get("_metal_path")
                or item.get("spdisplays_mtlgpufamilysupport")
            ]
            if candidates:
                device = max(
                    candidates,
                    key=lambda item: self._parse_vram(item.get("spdisplays_vram", "")),
                )
                name = device.get("sppci_model") or device.get("_name") or name
                total_bytes = self._parse_vram(device.get("spdisplays_vram", ""))
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            pass
        if total_bytes <= 0:
            # Integrated GPUs share host memory. This fallback is only used
            # when system_profiler does not disclose a discrete VRAM figure.
            import psutil

            total_bytes = psutil.virtual_memory().total
        return str(name), int(total_bytes)

    @staticmethod
    def _parse_vram(value: str) -> int:
        match = re.search(r"([0-9.]+)\s*(GB|MB)", value, re.IGNORECASE)
        if match is None:
            return 0
        amount = float(match.group(1))
        unit = match.group(2).upper()
        return int(amount * (1024**3 if unit == "GB" else 1024**2))

    def get_device_total_memory(self, device_id: int = 0) -> int:
        # Metal's working-set limit, not total system RAM, bounds model memory.
        return _recommended_working_set_size()

    def get_current_memory_usage(self, device: Optional[torch.device] = None) -> float:
        # Driver allocation includes resident cached MPS buffers.
        return float(torch.mps.driver_allocated_memory())

    def get_device(self, local_rank: int = 0) -> torch.device:
        if local_rank not in (-1, 0):
            raise ValueError(f"MPS exposes one device; got local_rank={local_rank}")
        return torch.device("mps")

    def set_device(self, device: torch.device) -> None:
        if str(device).split(":", 1)[0] != "mps":
            raise ValueError(f"MPS platform cannot select device {device}")
        # MPS has no device-selection API.

    def get_device_name(self, device_id: int = 0) -> str:
        return self._device_description[0]

    def get_device_uuid(self, device_id: int = 0) -> str:
        return self._device_description[0]

    def get_device_capability(self, device_id: int = 0):
        return None

    def empty_cache(self) -> None:
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def synchronize(self) -> None:
        synchronize = getattr(torch.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        total = self.get_device_total_memory(device_id)
        used = int(torch.mps.driver_allocated_memory())
        return max(total - used, 0), total

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"


class MpsSRTPlatform(MpsDeviceMixin, SRTPlatform):
    """Built-in SRT platform for Apple Silicon MPS."""

    def get_default_attention_backend(self) -> str:
        return "torch_native"
