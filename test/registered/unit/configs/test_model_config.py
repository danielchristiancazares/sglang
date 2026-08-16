"""Unit tests for hybrid attention model configuration."""

import sys
import unittest
from types import SimpleNamespace

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_embedding_gemma,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.nvfp4_online import NvFp4OnlineConfig
from sglang.srt.model_loader.weight_utils import get_quant_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


class TestDraftModelConfig(CustomTestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows registry contract")
    def test_windows_registers_serialized_modelopt_fp4(self):
        self.assertIs(QUANTIZATION_METHODS["modelopt_fp4"], ModelOptFp4Config)

    @staticmethod
    def _mixed_checkpoint_draft_config(ignored_layers, quantization="nvfp4_online"):
        config = object.__new__(ModelConfig)
        config.quantization = quantization
        config.is_draft_model = True
        config.is_draft_quantization_explicit = True
        config.hf_config = SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "ignore": ignored_layers,
            }
        )
        config._parse_quant_hf_config = lambda: {
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
        }
        config._find_quant_modelslim_config = lambda: None
        return config

    def test_qwen35_mtp_depth_is_synced_to_text_config(self):
        config = object.__new__(ModelConfig)
        config.is_draft_model = True
        config.speculative_algorithm = "EAGLE"
        config.hf_config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        config.hf_text_config = SimpleNamespace()

        config._config_draft_model()

        self.assertEqual(config.hf_config.architectures, ["Qwen3_5ForCausalLMMTP"])
        self.assertEqual(config.hf_config.num_nextn_predict_layers, 1)
        self.assertEqual(config.hf_text_config.num_nextn_predict_layers, 1)

    def test_explicit_online_quantizer_quantizes_excluded_mtp_weights(self):
        for quantization in ("fp8", "mxfp8", "nvfp4_online"):
            with self.subTest(quantization=quantization):
                config = self._mixed_checkpoint_draft_config(
                    ["mtp*", "mtp.layers.0*"], quantization
                )

                config._verify_quantization()

                self.assertEqual(config.quantization, quantization)

    def test_explicit_online_quantizer_does_not_requantize_packed_mtp_weights(self):
        for quantization in ("fp8", "mxfp8", "nvfp4_online"):
            with self.subTest(quantization=quantization):
                config = self._mixed_checkpoint_draft_config([], quantization)

                config._verify_quantization()

                self.assertEqual(config.quantization, "modelopt_mixed")

    def test_explicit_draft_fp8_ignores_shared_target_quant_metadata(self):
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        for quantization, use_mxfp8 in (("fp8", False), ("mxfp8", True)):
            with self.subTest(quantization=quantization):
                model_config = SimpleNamespace(
                    model_path="shared-target-model",
                    quantization=quantization,
                    is_draft_model=True,
                    is_draft_quantization_explicit=True,
                    hf_config=SimpleNamespace(
                        quantization_config={
                            "quant_method": "modelopt_mixed",
                            "quant_algo": "MIXED_PRECISION",
                            "ignore": ["mtp*"],
                        }
                    ),
                )

                config = get_quant_config(
                    model_config, LoadConfig(), packed_modules_mapping
                )

                self.assertIsInstance(config, Fp8Config)
                self.assertFalse(config.is_checkpoint_fp8_serialized)
                self.assertEqual(config.activation_scheme, "dynamic")
                self.assertEqual(config.use_mxfp8, use_mxfp8)
                self.assertEqual(
                    config.packed_modules_mapping, packed_modules_mapping
                )

    def test_explicit_draft_nvfp4_ignores_shared_target_quant_metadata(self):
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        model_config = SimpleNamespace(
            model_path="shared-target-model",
            quantization="nvfp4_online",
            is_draft_model=True,
            is_draft_quantization_explicit=True,
            hf_config=SimpleNamespace(
                quantization_config={
                    "quant_method": "modelopt_mixed",
                    "quant_algo": "MIXED_PRECISION",
                    "ignore": ["mtp*"],
                }
            ),
        )

        config = get_quant_config(model_config, LoadConfig(), packed_modules_mapping)

        self.assertIsInstance(config, NvFp4OnlineConfig)
        self.assertFalse(config.is_checkpoint_fp8_serialized)
        self.assertEqual(config.activation_scheme, "dynamic")
        self.assertEqual(config.exclude_modules, [])
        self.assertEqual(config.packed_modules_mapping, packed_modules_mapping)


if __name__ == "__main__":
    unittest.main()
