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


class MpsDeviceMixin(DeviceMixin):
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
        return self._device_description[1]

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(torch.mps.driver_allocated_memory())

    def get_device(self, device_id: int = 0) -> "torch.device":
        return torch.device("mps")

    def set_device(self, device: "torch.device") -> None:
        return None

    def get_device_name(self, device_id: int = 0) -> str:
        return self._device_description[0]

    def get_device_uuid(self, device_id: int = 0) -> str:
        return self._device_description[0]

    def get_device_capability(self, device_id: int = 0):
        return None

    def empty_cache(self) -> None:
        torch.mps.empty_cache()

    def synchronize(self) -> None:
        torch.mps.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        total = self.get_device_total_memory(device_id)
        used = int(torch.mps.driver_allocated_memory())
        return max(total - used, 0), total

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"


class MpsSRTPlatform(MpsDeviceMixin, SRTPlatform):
    def get_default_attention_backend(self) -> str:
        return "torch_native"
