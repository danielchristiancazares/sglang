from enum import IntEnum

from sglang.srt.utils import is_cpu


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def default_tree_mask_mode() -> TreeMaskMode:
    # The CPU verify attention kernel (intel_amx) consumes the qlen x qlen
    # QLEN_ONLY tree mask directly; FULL_MASK is for the GPU kernels.
    return TreeMaskMode.QLEN_ONLY if is_cpu() else TreeMaskMode.FULL_MASK
