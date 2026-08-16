import pytest
import torch

from sglang.srt.layers.quantization.fp8_utils import (
    flashinfer_mxfp8_blockscaled_linear,
    flashinfer_mxfp8_quantize,
    mxfp8_group_quantize,
)
from sglang.srt.utils import (
    is_blackwell_supported,
    is_flashinfer_available,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_blackwell_supported()
    or not is_flashinfer_available(),
    reason="native MXFP8 quantization requires Blackwell and FlashInfer",
)


def _reference_mxfp8_quantize(x: torch.Tensor):
    rows, k = x.shape
    groups = x.float().view(rows, k // 32, 32)
    amax = groups.abs().amax(dim=-1)
    raw_scale = (amax / 448.0).contiguous()
    raw_bits = raw_scale.view(torch.int32)
    exponent = ((raw_bits >> 23) & 0xFF) + (
        (raw_bits & 0x7FFFFF) != 0
    ).to(torch.int32)
    quant_multiplier = ((254 - exponent) << 23).view(torch.float32)
    quantized = (
        (groups * quant_multiplier.unsqueeze(-1))
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(rows, k)
    )
    return quantized, exponent.to(torch.uint8)


def _dequantize(q: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    rows, k = q.shape
    scale = torch.exp2(scale_u8.float() - 127.0)
    return (q.float().view(rows, k // 32, 32) * scale.unsqueeze(-1)).view(rows, k)


@pytest.mark.parametrize("shape", [(3, 64), (3, 10240), (129, 128)])
def test_native_mxfp8_group_quantize_matches_reference(shape):
    torch.manual_seed(shape[0] * 100_000 + shape[1])
    x = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 4).contiguous()

    q, scale_u8 = mxfp8_group_quantize(x)
    q_ref, scale_ref = _reference_mxfp8_quantize(x)

    assert q.shape == x.shape
    assert q.dtype == torch.float8_e4m3fn
    assert scale_u8.shape == (shape[0], shape[1] // 32)
    assert scale_u8.dtype == torch.uint8
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(scale_u8, scale_ref)


def test_native_mxfp8_weight_layout_executes_cutlass_dense_gemm():
    from flashinfer import block_scale_interleave

    torch.manual_seed(20260816)
    m, n, k = 3, 256, 512
    x = (torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 4).contiguous()
    weight = (
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 4
    ).contiguous()

    weight_q, weight_scale = mxfp8_group_quantize(weight)
    weight_scale_swizzled = block_scale_interleave(weight_scale).contiguous()
    input_q, input_scale_swizzled = flashinfer_mxfp8_quantize(
        x,
        is_sf_swizzled_layout=True,
        alignment=32,
        backend="cuda",
    )
    _, input_scale = mxfp8_group_quantize(x)

    output = flashinfer_mxfp8_blockscaled_linear(
        input=input_q,
        weight=weight_q,
        weight_scale=weight_scale_swizzled,
        input_scale=input_scale_swizzled,
        output_dtype=torch.bfloat16,
        backend="cutlass",
    )
    output_dynamic = flashinfer_mxfp8_blockscaled_linear(
        input=x,
        weight=weight_q,
        weight_scale=weight_scale_swizzled,
        backend="cutlass",
    )
    reference = (_dequantize(input_q, input_scale) @ _dequantize(weight_q, weight_scale).t()).to(
        torch.bfloat16
    )

    relative_mae = (
        (output.float() - reference.float()).abs().mean()
        / reference.float().abs().mean()
    )
    assert relative_mae.item() < 0.02
    torch.testing.assert_close(output_dynamic, output, rtol=0.0, atol=0.0)
