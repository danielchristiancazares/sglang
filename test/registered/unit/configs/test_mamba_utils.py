from unittest.mock import patch

import torch

from sglang.srt.configs.mamba_utils import mamba2_state_dtype
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_mps_uses_float32_convolution_state_by_default():
    conv_dtype = envs.SGLANG_MAMBA_CONV_DTYPE
    with (
        patch.object(conv_dtype, "get", return_value="bfloat16"),
        patch.object(conv_dtype, "is_set", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert mamba2_state_dtype().conv == torch.float32


def test_mps_preserves_explicit_convolution_state_dtype():
    conv_dtype = envs.SGLANG_MAMBA_CONV_DTYPE
    with (
        patch.object(conv_dtype, "get", return_value="float16"),
        patch.object(conv_dtype, "is_set", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert mamba2_state_dtype().conv == torch.float16
