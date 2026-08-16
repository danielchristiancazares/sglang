import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.model_loader.gguf_name_maps import build_qwen3_5_name_map
from sglang.srt.utils.hf_transformers import gguf_native


class _FakeField:
    def __init__(self, value):
        self.value = value

    def contents(self):
        return self.value


class TestGGUFMetadataSnapshot(unittest.TestCase):
    def test_reader_is_reused_within_process(self):
        calls = []

        class FakeReader:
            def __init__(self, path):
                calls.append(path)
                self.fields = {
                    "general.architecture": _FakeField("qwen35"),
                    "general.name": _FakeField("fixture"),
                }
                self.tensors = [SimpleNamespace(name="weight", shape=(2, 3))]

        fake_gguf = SimpleNamespace(GGUFReader=FakeReader)
        gguf_native._read_gguf_metadata_snapshot.cache_clear()
        try:
            with patch.dict(sys.modules, {"gguf": fake_gguf}):
                first = gguf_native._read_gguf_metadata_snapshot("model.gguf")
                second = gguf_native._read_gguf_metadata_snapshot("model.gguf")
        finally:
            gguf_native._read_gguf_metadata_snapshot.cache_clear()

        self.assertIs(first, second)
        self.assertEqual(calls, ["model.gguf"])
        self.assertEqual(first[0]["general.architecture"], "qwen35")
        self.assertEqual(first[1], (("weight", (2, 3)),))

    def test_model_max_length_uses_finite_architecture_context(self):
        self.assertEqual(
            gguf_native._gguf_model_max_length(
                {
                    "general.architecture": "qwen35",
                    "qwen35.context_length": 262144,
                }
            ),
            262144,
        )
        self.assertIsNone(
            gguf_native._gguf_model_max_length(
                {"general.architecture": "qwen35"}
            )
        )


class TestQwen35GGUFNameMap(unittest.TestCase):
    def test_bundled_mtp_tail_maps_to_mtp_namespace(self):
        config = SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            mtp_num_hidden_layers=1,
        )

        name_map = build_qwen3_5_name_map(config)

        self.assertEqual(
            name_map["blk.0.ssm_out.weight"],
            "model.layers.0.linear_attn.out_proj.weight",
        )
        self.assertEqual(
            name_map["blk.1.attn_q.weight"],
            "model.layers.1.self_attn.q_proj.weight",
        )
        self.assertEqual(
            name_map["blk.2.attn_q.weight"],
            "mtp.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            name_map["blk.2.nextn.eh_proj.weight"], "mtp.fc.weight"
        )
        self.assertEqual(
            name_map["blk.2.nextn.shared_head_norm.weight"], "mtp.norm.weight"
        )
        self.assertIsNone(
            gguf_native._gguf_model_max_length(
                {
                    "general.architecture": "qwen35",
                    "qwen35.context_length": 1 << 64,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
