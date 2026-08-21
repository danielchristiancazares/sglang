import sys
from unittest import mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HIDDEN_SIZE = 17408
INPUT_SCALE = 0.0025692894123494625


def _native_windows_sm120() -> bool:
    return (
        sys.platform == "win32"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 12
    )


pytestmark = pytest.mark.skipif(
    not _native_windows_sm120(),
    reason="Native Windows SM120 NVFP4 specialization only",
)


def _global_scale(value: float = 1.0 / INPUT_SCALE) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float32, device="cuda")


def _reference(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.activation.activation import (
        silu_and_mul_with_activation_rounding,
    )
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    activated = silu_and_mul_with_activation_rounding(input)
    return fp4_quantize(activated, global_scale)


def _actual(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sglang.kernels.ops.quantization.silu_and_mul_nvfp4 import (
        silu_and_mul_nvfp4,
    )

    return silu_and_mul_nvfp4(input, global_scale)


def _assert_bit_exact(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual_q, actual_scale = actual
    expected_q, expected_scale = expected
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(
        actual_scale.view(torch.uint8),
        expected_scale.view(torch.uint8),
    )


def _finite_bf16_values() -> torch.Tensor:
    bits = torch.arange(1 << 16, dtype=torch.int32)
    finite = (bits & 0x7F80) != 0x7F80
    values = bits[finite].to(torch.int16).view(torch.bfloat16)
    assert values.numel() == 65280
    return values


@pytest.mark.parametrize("num_rows", [1, 3, 7000, 7680])
def test_silu_and_mul_nvfp4_is_bit_exact(num_rows: int) -> None:
    torch.manual_seed(2700 + num_rows)
    input = torch.randn(
        num_rows,
        HIDDEN_SIZE * 2,
        dtype=torch.bfloat16,
        device="cuda",
    )
    global_scale = _global_scale()
    _assert_bit_exact(
        _actual(input, global_scale),
        _reference(input, global_scale),
    )


@pytest.mark.parametrize(
    ("num_rows", "hidden_size"),
    [(4080, 16), (1024, 512)],
)
def test_silu_and_mul_nvfp4_all_finite_bf16(
    num_rows: int,
    hidden_size: int,
) -> None:
    values = _finite_bf16_values()
    numel = num_rows * hidden_size
    repeats = (numel + values.numel() - 1) // values.numel()
    gate = values.repeat(repeats)[:numel].reshape(num_rows, hidden_size)
    up = (
        values.roll(7919)
        .repeat(repeats)[:numel]
        .reshape(num_rows, hidden_size)
    )
    input = torch.cat((gate, up), dim=-1).cuda()
    global_scale = _global_scale()
    _assert_bit_exact(
        _actual(input, global_scale),
        _reference(input, global_scale),
    )


@pytest.mark.parametrize("num_rows", [1, 3])
def test_silu_and_mul_nvfp4_graph_replay(num_rows: int) -> None:
    static_input = torch.empty(
        num_rows,
        HIDDEN_SIZE * 2,
        dtype=torch.bfloat16,
        device="cuda",
    )
    static_scale = _global_scale()
    _actual(static_input, static_scale)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = _actual(static_input, static_scale)

    for seed, scale_multiplier in ((81, 0.75), (82, 1.25)):
        torch.manual_seed(seed)
        input_value = torch.randn_like(static_input)
        scale_value = _global_scale((1.0 / INPUT_SCALE) * scale_multiplier)
        static_input.copy_(input_value)
        static_scale.copy_(scale_value)
        graph.replay()
        _assert_bit_exact(
            graph_output,
            _reference(input_value, scale_value),
        )


def test_silu_and_mul_nvfp4_fullgraph() -> None:
    @torch.compile(fullgraph=True)
    def compiled(
        input: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _actual(input, global_scale)

    torch.manual_seed(83)
    input = torch.randn(
        3,
        HIDDEN_SIZE * 2,
        dtype=torch.bfloat16,
        device="cuda",
    )
    global_scale = _global_scale()
    _assert_bit_exact(
        compiled(input, global_scale),
        _reference(input, global_scale),
    )


def test_silu_and_mul_nvfp4_modelopt_tuple_graph() -> None:
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization import fp4_utils
    from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
    from sglang.test.layer_ut_utils import init_single_process_dist, load_linear_weights
    from sglang.test.quant_ref_utils import quantize_nvfp4_shard

    init_single_process_dist()
    quant_config = ModelOptFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        use_per_token_activation=False,
        packed_modules_mapping={},
    )
    with mock.patch.object(
        fp4_utils,
        "FP4_GEMM_RUNNER_BACKEND",
        Fp4GemmRunnerBackend("flashinfer_cutlass"),
    ):
        layer = RowParallelLinear(
            input_size=512,
            output_size=256,
            bias=False,
            params_dtype=torch.bfloat16,
            reduce_results=False,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.down_proj",
            tp_rank=0,
            tp_size=1,
        ).cuda()
        torch.manual_seed(84)
        weight = torch.randn(
            256,
            512,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weight_q, weight_scale, weight_global_scale, _ = quantize_nvfp4_shard(
            weight
        )
        load_linear_weights(
            layer,
            weight=weight_q,
            weight_scale=weight_scale,
            weight_scale_2=(1.0 / weight_global_scale).clone(),
            input_scale=torch.tensor(INPUT_SCALE, device="cuda"),
        )
        layer.quant_method.process_weights_after_loading(layer)
        layer._accepts_prequantized_fp4 = True

        static_input = torch.randn(
            3,
            1024,
            dtype=torch.bfloat16,
            device="cuda",
        )

        def fused_chain(
            input: torch.Tensor,
        ) -> torch.Tensor:
            quantized = _actual(input, layer.input_scale_inv)
            output, _ = layer(quantized)
            return output

        fused_chain(static_input)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fused_chain(static_input)

        for seed in (85, 86):
            torch.manual_seed(seed)
            input_value = torch.randn_like(static_input)
            static_input.copy_(input_value)
            graph.replay()
            activated = _reference(input_value, layer.input_scale_inv)
            expected, _ = layer(activated)
            torch.testing.assert_close(
                graph_output,
                expected,
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
