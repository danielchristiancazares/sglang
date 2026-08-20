import copy

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from scripts.windows.build_selective_target_nvfp4_checkpoint import (
    convert_runtime_quant_config,
)


def test_convert_runtime_quant_config_keeps_manifests_aligned():
    config = {
        "quantization_config": {
            "config_groups": {
                "fp8": {
                    "weights": {"num_bits": 8},
                    "targets": ["attention", "keep_fp8"],
                },
                "nvfp4": {
                    "weights": {"num_bits": 4},
                    "targets": ["mlp"],
                },
            },
            "quantized_layers": {
                "attention": {"quant_algo": "FP8"},
                "keep_fp8": {"quant_algo": "FP8"},
                "mlp": {"quant_algo": "NVFP4", "group_size": 16},
            },
        }
    }

    converted = convert_runtime_quant_config(copy.deepcopy(config), ["attention"])
    quant = converted["quantization_config"]

    assert quant["quantized_layers"]["attention"] == {
        "quant_algo": "NVFP4",
        "group_size": 16,
    }
    assert quant["quantized_layers"]["keep_fp8"] == {"quant_algo": "FP8"}
    assert quant["config_groups"]["fp8"]["targets"] == ["keep_fp8"]
    assert quant["config_groups"]["nvfp4"]["targets"] == ["attention", "mlp"]
