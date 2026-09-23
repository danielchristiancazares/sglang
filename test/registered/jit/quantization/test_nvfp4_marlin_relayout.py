import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.quantization.gptq_marlin_repack import (
    gptq_marlin_repack,
)
from sglang.kernels.ops.quantization.nvfp4_marlin_relayout import (
    create_nvfp4_hybrid_marlin_state,
    nvfp4_marlin_relayout_,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    nvfp4_marlin_process_scales,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.nvfp4_hybrid_marlin import (
    Nvfp4HybridMarlinManager,
    prepare_nvfp4_layer_for_hybrid_marlin,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import quantize_nvfp4_shard
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestNvfp4MarlinRelayout(CustomTestCase):
    def test_native_nvfp4_wrapper_matches_dequantized_reference(self):
        torch.manual_seed(62064)
        size_m, size_n, size_k = 4, 192, 256
        x = (
            torch.randn(
                (size_m, size_k),
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 10
        )
        weight = (
            torch.randn(
                (size_n, size_k),
                dtype=torch.bfloat16,
                device="cuda",
            )
            / 10
        )
        weight_fp4, weight_scale, quant_scale, weight_ref = quantize_nvfp4_shard(weight)
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight_fp4, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
            weight_global_scale=torch.nn.Parameter(
                (1.0 / quant_scale).to(torch.bfloat16).reshape(1),
                requires_grad=False,
            ),
            output_size_per_partition=size_n,
            input_size_per_partition=size_k,
            params_dtype=torch.bfloat16,
            quant_config=SimpleNamespace(group_size=16),
            bias=None,
        )
        prepare_nvfp4_layer_for_marlin(layer)

        output = apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=size_n,
            size_k=size_k,
        )
        output_ref = torch.matmul(x.float(), weight_ref.T).to(torch.bfloat16)
        torch.cuda.synchronize()

        torch.testing.assert_close(output, output_ref, rtol=0.04, atol=0.04)

    def test_matches_marlin_repack_and_round_trips(self):
        for size_n, size_k in ((64, 128), (256, 256), (512, 640)):
            with self.subTest(size_n=size_n, size_k=size_k):
                cutlass = torch.randint(
                    0,
                    256,
                    (size_n, size_k // 2),
                    dtype=torch.uint8,
                    device="cuda",
                )
                original = cutlass.clone()
                expected = gptq_marlin_repack(
                    b_q_weight=cutlass.view(torch.int32).T.contiguous(),
                    perm=torch.empty(0, dtype=torch.int32, device="cuda"),
                    size_k=size_k,
                    size_n=size_n,
                    num_bits=4,
                )
                storage = cutlass.reshape(-1)
                scratch = torch.empty_like(storage)

                nvfp4_marlin_relayout_(
                    storage,
                    scratch,
                    size_n=size_n,
                    size_k=size_k,
                    to_marlin=True,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    storage.view(torch.int32).view_as(expected),
                    expected,
                    rtol=0,
                    atol=0,
                )

                nvfp4_marlin_relayout_(
                    storage,
                    scratch,
                    size_n=size_n,
                    size_k=size_k,
                    to_marlin=False,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    storage.view_as(original),
                    original,
                    rtol=0,
                    atol=0,
                )

    def _hybrid_fixture(self, *, lazy=True, enabled=True):
        torch.manual_seed(62065)
        model = torch.nn.ModuleList()
        references = []
        for size_n, size_k in ((128, 128), (256, 256)):
            source = (
                torch.randn(size_n, size_k, device="cuda", dtype=torch.bfloat16) / 10
            )
            weight, scales, quant_scale, dequantized = quantize_nvfp4_shard(source)
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
            layer.output_size_per_partition = size_n
            layer.input_size_per_partition = size_k
            layer.params_dtype = torch.bfloat16
            prepare_nvfp4_layer_for_hybrid_marlin(
                layer, weight_global_scale=(1.0 / quant_scale).reshape(1)
            )
            self.assertTrue(layer._nvfp4_hybrid_marlin)
            expected_weight = gptq_marlin_repack(
                b_q_weight=weight.view(torch.int32).T.contiguous(),
                perm=torch.empty(0, dtype=torch.int32, device="cuda"),
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
            expected_scale = nvfp4_marlin_process_scales(
                marlin_permute_scales(
                    scales.T.contiguous().to(torch.bfloat16),
                    size_k=size_k,
                    size_n=size_n,
                    group_size=16,
                )
            )
            swizzled = (
                scales.reshape(size_n // 128, 4, 32, size_k // 64, 4)
                .permute(0, 3, 2, 1, 4)
                .contiguous()
                .reshape_as(scales)
            )
            layer.weight_scale.data.copy_(swizzled)
            references.append(
                (
                    weight.clone(),
                    swizzled.view(torch.uint8).clone(),
                    expected_weight,
                    expected_scale.view(torch.uint8),
                    dequantized,
                )
            )
            model.append(layer)
        with (
            envs.SGLANG_ENABLE_NVFP4_MARLIN_LAZY_RELAYOUT.override(lazy),
            patch(
                "sglang.srt.layers.quantization.nvfp4_hybrid_marlin.sys",
                SimpleNamespace(platform="win32"),
            ),
        ):
            manager = Nvfp4HybridMarlinManager(
                model=model, device="cuda", is_draft_worker=False, enabled=enabled
            )
        return model, manager, references

    def _hybrid_batch(self, num_tokens, **kwargs):
        fields = dict(
            input_ids=torch.empty(num_tokens, device="cuda", dtype=torch.int64),
            forward_mode=ForwardMode.EXTEND,
            is_prefill_only=False,
            contains_last_prefill_chunk=True,
        )
        fields.update(kwargs)
        return SimpleNamespace(**fields)

    def _hybrid_linear(self, layer, x):
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight_marlin,
            weight_scale=layer.weight_scale_marlin,
            weight_global_scale=layer.weight_global_scale_marlin,
            workspace=layer.workspace_marlin,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
        )

    def test_lazy_prefill_preserves_bytes_until_verify(self):
        model, manager, references = self._hybrid_fixture()
        addresses = [
            (layer.weight.data_ptr(), layer.weight_scale.data_ptr()) for layer in model
        ]
        for num_tokens in (9, 4096, 9):
            with self.subTest(num_tokens=num_tokens):
                batch = self._hybrid_batch(num_tokens)
                manager.prepare_for_forward(batch)
                manager.finish_forward(batch)
                self.assertFalse(manager._marlin_layout)
                for layer, ref in zip(model, references):
                    torch.testing.assert_close(layer.weight, ref[0], rtol=0, atol=0)
                    torch.testing.assert_close(
                        layer.weight_scale.view(torch.uint8), ref[1], rtol=0, atol=0
                    )
                for verify_tokens in (8, 1, 8):
                    manager.prepare_for_forward(self._hybrid_batch(verify_tokens))
                    self.assertTrue(manager._marlin_layout)
                    for layer, ref in zip(model, references):
                        torch.testing.assert_close(
                            layer.weight_marlin, ref[2], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            layer.weight_scale_marlin.view(torch.uint8),
                            ref[3],
                            rtol=0,
                            atol=0,
                        )
                self.assertEqual(
                    [
                        (layer.weight.data_ptr(), layer.weight_scale.data_ptr())
                        for layer in model
                    ],
                    addresses,
                )

    def test_lazy_relayout_preserves_captured_gemm_replay(self):
        model, manager, references = self._hybrid_fixture()
        layer = model[0]
        x = (
            torch.randn(
                8, layer.input_size_per_partition, device="cuda", dtype=torch.bfloat16
            )
            / 10
        )
        verify = self._hybrid_batch(8, forward_mode=ForwardMode.TARGET_VERIFY)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            manager.prepare_for_forward(verify)
            for _ in range(3):
                self._hybrid_linear(layer, x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._hybrid_linear(layer, x)
        for num_tokens in (9, 4096, 9):
            batch = self._hybrid_batch(num_tokens)
            manager.prepare_for_forward(batch)
            manager.finish_forward(batch)
            self.assertFalse(manager._marlin_layout)
            x.copy_(torch.randn_like(x) / 10)
            manager.prepare_for_forward(verify)
            graph.replay()
            expected = self._hybrid_linear(layer, x)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                output,
                torch.matmul(x.float(), references[0][4].T).to(x.dtype),
                rtol=0.04,
                atol=0.04,
            )

    def test_hybrid_finish_preserves_default_and_nonfinal_policies(self):
        for lazy, enabled, prefill_only, last, mode in (
            (False, True, False, True, ForwardMode.EXTEND),
            (False, True, False, True, ForwardMode.MIXED),
            (False, True, True, True, ForwardMode.EXTEND),
            (False, True, False, False, ForwardMode.EXTEND),
            (True, True, False, False, ForwardMode.EXTEND),
            (True, True, True, True, ForwardMode.EXTEND),
            (True, False, False, True, ForwardMode.EXTEND),
            (False, True, False, True, ForwardMode.TARGET_VERIFY),
        ):
            with self.subTest(
                lazy=lazy,
                enabled=enabled,
                prefill_only=prefill_only,
                last=last,
                mode=mode,
            ):
                model, manager, references = self._hybrid_fixture(
                    lazy=lazy, enabled=enabled
                )
                batch = self._hybrid_batch(
                    9,
                    forward_mode=mode,
                    is_prefill_only=prefill_only,
                    contains_last_prefill_chunk=last,
                )
                manager.prepare_for_forward(batch)
                manager.finish_forward(batch)
                expected = (
                    enabled
                    and not lazy
                    and not prefill_only
                    and last
                    and mode in (ForwardMode.EXTEND, ForwardMode.MIXED)
                )
                self.assertEqual(manager._marlin_layout, expected)
                if not expected:
                    for layer, ref in zip(model, references):
                        torch.testing.assert_close(layer.weight, ref[0], rtol=0, atol=0)
                        torch.testing.assert_close(
                            layer.weight_scale.view(torch.uint8), ref[1], rtol=0, atol=0
                        )

    def test_native_dispatch_checks_managed_layout_and_cutoff(self):
        _, manager, _ = self._hybrid_fixture()
        state = manager._native_state
        self.assertTrue(state.select_dispatch(8, True))
        self.assertFalse(state.select_dispatch(9, True))
        manager.prepare_for_forward(self._hybrid_batch(9))
        self.assertFalse(state.select_dispatch(9, True))
        self.assertEqual(state.transition_count(), 0)
        for num_tokens in (1, 8):
            with self.subTest(num_tokens=num_tokens):
                with self.assertRaisesRegex(RuntimeError, "prepared weight layout"):
                    state.select_dispatch(num_tokens, True)
        manager.prepare_for_forward(self._hybrid_batch(8))
        self.assertTrue(state.select_dispatch(8, True))
        self.assertEqual(state.transition_count(), 1)
        manager.prepare_for_forward(self._hybrid_batch(1))
        self.assertEqual(state.transition_count(), 1)
        with self.assertRaisesRegex(RuntimeError, "prepared weight layout"):
            state.select_dispatch(9, True)
        with self.assertRaisesRegex(RuntimeError, "prepared weight layout"):
            state.select_dispatch(0, False)
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            state.prepare(manager.scratch, -1)

    def test_native_relayout_obeys_nondefault_stream(self):
        model, manager, references = self._hybrid_fixture()
        layer = model[0]
        x = (
            torch.randn(
                8, layer.input_size_per_partition, device="cuda", dtype=torch.bfloat16
            )
            / 10
        )
        negated_weight = references[0][0] ^ 0x88
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10000000)
            layer.weight.copy_(negated_weight)
            manager.prepare_for_forward(self._hybrid_batch(8))
            output = self._hybrid_linear(layer, x)
        torch.cuda.current_stream().wait_stream(stream)
        expected = torch.matmul(x.float(), -references[0][4].T).to(x.dtype)
        torch.testing.assert_close(output, expected, rtol=0.04, atol=0.04)

    def test_native_relayout_rejects_invalid_scratch_before_mutation(self):
        model, manager, references = self._hybrid_fixture()
        state = manager._native_state
        for scratch in (manager.scratch[:1], model[1].weight.view(-1)):
            with self.subTest(scratch_bytes=scratch.numel()):
                with self.assertRaisesRegex(RuntimeError, "scratch"):
                    state.prepare(scratch, 8)
                self.assertEqual(state.transition_count(), 0)
                for layer, ref in zip(model, references):
                    torch.testing.assert_close(layer.weight, ref[0], rtol=0, atol=0)
                    torch.testing.assert_close(
                        layer.weight_scale.view(torch.uint8), ref[1], rtol=0, atol=0
                    )
        manager.prepare_for_forward(self._hybrid_batch(8))
        self.assertEqual(state.transition_count(), 1)

    def test_native_relayout_rejects_transition_during_capture(self):
        model, manager, references = self._hybrid_fixture()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                manager._native_state.prepare(manager.scratch, 8)
            output = model[0].weight.clone()
        graph.replay()
        torch.testing.assert_close(output, references[0][0], rtol=0, atol=0)
        self.assertEqual(manager._native_state.transition_count(), 0)
        manager.prepare_for_forward(self._hybrid_batch(8))
        self.assertEqual(manager._native_state.transition_count(), 1)

    def test_native_eager_policy_and_registration_validation(self):
        model, manager, _ = self._hybrid_fixture(lazy=False)
        state = create_nvfp4_hybrid_marlin_state(model, lazy_relayout=False)
        self.assertFalse(state.prepare(manager.scratch, 9))
        self.assertFalse(state.finish(manager.scratch, True, True, True))
        self.assertFalse(state.finish(manager.scratch, False, False, True))
        self.assertTrue(state.finish(manager.scratch, False, True, True))
        self.assertEqual(state.transition_count(), 1)
        state.validate_dispatch(True)
        state.prepare(manager.scratch, 9)
        with self.assertRaisesRegex(RuntimeError, "must not alias"):
            create_nvfp4_hybrid_marlin_state([model[0], model[0]], lazy_relayout=True)

    def test_native_weight_updates_fail_before_mutation(self):
        model, manager, references = self._hybrid_fixture()
        updater = WeightUpdater(
            tp_rank=0,
            device="cuda",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: SimpleNamespace(
                nvfp4_hybrid_marlin_manager=manager
            ),
        )
        with get_context().override_server_args(weight_cache_mode="off"):
            with self.assertRaisesRegex(RuntimeError, "Online weight updates"):
                updater.update_weights_from_disk("unused", "auto")
        self.assertEqual(manager._native_state.transition_count(), 0)
        for layer, ref in zip(model, references):
            torch.testing.assert_close(layer.weight, ref[0], rtol=0, atol=0)
            torch.testing.assert_close(
                layer.weight_scale.view(torch.uint8), ref[1], rtol=0, atol=0
            )

    def test_rejects_invalid_public_inputs(self):
        weight = torch.zeros(64 * 64, dtype=torch.uint8, device="cuda")
        scratch = torch.empty_like(weight)
        with self.assertRaisesRegex(ValueError, "weight"):
            nvfp4_marlin_relayout_(
                weight.cpu(),
                scratch,
                size_n=64,
                size_k=128,
                to_marlin=True,
            )
        with self.assertRaisesRegex(ValueError, "scratch"):
            nvfp4_marlin_relayout_(
                weight,
                scratch[:1],
                size_n=64,
                size_k=128,
                to_marlin=True,
            )


if __name__ == "__main__":
    unittest.main()
