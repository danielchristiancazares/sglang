# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-architecture GGUF -> HF tensor name maps.

``GGUFModelLoader`` normally derives this map from ``gguf.get_tensor_name_map``,
which only covers architectures upstream gguf-py knows, and from a meta-device
``AutoModelForCausalLM.from_config`` to enumerate the HF parameter names. Neither
works for an architecture that lives outside transformers, so those are supplied
here instead.

A builder returns the complete ``{gguf_tensor_name: hf_param_name}`` map. Any
GGUF tensor left out of the map is skipped by ``gguf_quant_weights_iterator``,
which is how dummy tensors are dropped.
"""

from typing import Callable, Dict

from transformers import PretrainedConfig

# Sandwich naming: ffn_norm is the pre-FFN norm.
_MUSE_GLIMMER_LAYER_TENSORS = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attn_norm",
    "ffn_norm": "post_attention_layernorm",
    "post_ffw_norm": "post_ffn_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "attn_gate": "self_attn.output_gate_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

_MUSE_GLIMMER_GLOBAL_TENSORS = {
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output": "lm_head",
}

# attn_q_norm/attn_k_norm omitted: Muse Glimmer's QK-norm is non-parametric.


def build_muse_glimmer_name_map(config: PretrainedConfig) -> Dict[str, str]:
    name_map = {
        f"{gguf}.weight": f"{hf}.weight"
        for gguf, hf in _MUSE_GLIMMER_GLOBAL_TENSORS.items()
    }
    for layer in range(config.num_hidden_layers):
        for gguf, hf in _MUSE_GLIMMER_LAYER_TENSORS.items():
            name_map[f"blk.{layer}.{gguf}.weight"] = f"model.layers.{layer}.{hf}.weight"
    return name_map


def build_qwen3_5_name_map(config: PretrainedConfig) -> Dict[str, str]:
    """Map llama.cpp's Qwen3.5 names to the canonical HF checkpoint names."""
    # Multimodal Qwen3.5 checkpoints expose the decoder shape through a nested
    # text config even when ``--language-model-only`` selects just that decoder.
    config = getattr(config, "text_config", config)
    name_map = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    common = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
    }
    linear_attention = {
        "ssm_dt.bias": "linear_attn.dt_bias",
        "ssm_a": "linear_attn.A_log",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    }
    full_attention = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
    }

    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        layer_types = config.layers_block_type
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            "Qwen3.5 layer_types must match num_hidden_layers: "
            f"{len(layer_types)} != {config.num_hidden_layers}"
        )
    for layer, layer_type in enumerate(layer_types):
        tensors = dict(common)
        if layer_type == "linear_attention":
            tensors.update(linear_attention)
        elif layer_type == "full_attention":
            tensors.update(full_attention)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type!r}")
        for gguf_name, hf_name in tensors.items():
            name_map[f"blk.{layer}.{gguf_name}"] = f"model.layers.{layer}.{hf_name}"

    # llama.cpp appends Qwen3.5's bundled MTP block after the target layers.
    # Keep the ``mtp.`` prefix in the mapped names: the target loader drops
    # those weights while Qwen3_5ForCausalLMMTP selects and remaps them.
    mtp_layers = int(getattr(config, "mtp_num_hidden_layers", 0) or 0)
    for mtp_layer in range(mtp_layers):
        gguf_layer = config.num_hidden_layers + mtp_layer
        tensors = dict(common)
        tensors.update(full_attention)
        for gguf_name, hf_name in tensors.items():
            name_map[f"blk.{gguf_layer}.{gguf_name}"] = (
                f"mtp.layers.{mtp_layer}.{hf_name}"
            )

    if mtp_layers:
        gguf_layer = config.num_hidden_layers
        name_map.update(
            {
                f"blk.{gguf_layer}.nextn.eh_proj.weight": "mtp.fc.weight",
                f"blk.{gguf_layer}.nextn.enorm.weight": (
                    "mtp.pre_fc_norm_embedding.weight"
                ),
                f"blk.{gguf_layer}.nextn.hnorm.weight": (
                    "mtp.pre_fc_norm_hidden.weight"
                ),
                f"blk.{gguf_layer}.nextn.shared_head_norm.weight": (
                    "mtp.norm.weight"
                ),
            }
        )
    return name_map


# Keyed by HF ``config.model_type`` (loader.py looks it up with that), which is
# not the GGUF ``general.architecture`` that GGUF_NATIVE_CONFIG_BUILDERS uses:
# llama.cpp spells the arch "muse-glimmer" while the HF config says "muse_glimmer".
GGUF_HF_NAME_MAP_BUILDERS: Dict[str, Callable[[PretrainedConfig], Dict[str, str]]] = {
    "muse_glimmer": build_muse_glimmer_name_map,
    "qwen3_5": build_qwen3_5_name_map,
    "qwen3_5_text": build_qwen3_5_name_map,
}
