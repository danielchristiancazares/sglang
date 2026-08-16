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
"""Reading config and tokenizer from a GGUF whose architecture transformers lacks.

``load_gguf_checkpoint`` refuses any architecture outside its own
``GGUF_SUPPORTED_ARCHITECTURES``, and it does so before touching a single field,
so both the config and the tokenizer are unreachable for such a checkpoint --
even though the tokenizer half of that reader is entirely architecture-agnostic
(it dispatches on ``tokenizer.ggml.model``, not on the model architecture).

This module carries SGLang's own path for those checkpoints:

* ``GGUF_NATIVE_CONFIG_BUILDERS`` maps a GGUF ``general.architecture`` to a
  builder returning a fully populated config.
* ``build_gguf_tokenizer`` reuses transformers' own converters, which work fine
  once they are reached directly instead of through the gated loader.

Reaching for these is a last resort: a config.json next to the .gguf still wins,
because the checkpoint author's own config outranks anything reconstructed.
"""

from functools import lru_cache
from typing import Any, Callable, Dict, Optional

from transformers import PretrainedConfig

from sglang.srt.configs.muse_glimmer import MuseGlimmerConfig
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig


@lru_cache(maxsize=4)
def _read_gguf_metadata_snapshot(gguf_path: str):
    """Read immutable GGUF metadata and tensor descriptors once per process."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    meta = {key: field.contents() for key, field in reader.fields.items()}
    tensors = tuple(
        (tensor.name, tuple(int(dim) for dim in tensor.shape))
        for tensor in reader.tensors
    )
    return meta, tensors


def _gguf_model_max_length(meta: Dict[str, Any]) -> Optional[int]:
    """Return the architecture's finite training context when GGUF records it."""
    architecture = meta.get("general.architecture")
    if not isinstance(architecture, str):
        return None
    context_length = meta.get(f"{architecture}.context_length")
    if context_length is None:
        return None
    value = int(context_length)
    return value if 0 < value <= (1 << 63) - 1 else None


def _qwen35_config_from_gguf(gguf_path: str) -> Qwen3_5TextConfig:
    """Reconstruct the text-only Qwen3.5 config emitted by llama.cpp.

    Qwen3.5 GGUFs include the MTP layer in ``block_count``.  SGLang loads the
    target model separately, so that tail must not become a serving layer.
    """
    meta, tensors = _read_gguf_metadata_snapshot(gguf_path)
    tensor_names = {name for name, _ in tensors}

    def get(suffix: str):
        return meta[f"qwen35.{suffix}"]

    def token_id(name: str):
        value = meta.get(f"tokenizer.ggml.{name}_token_id")
        return None if value is None else int(value)

    block_count = int(get("block_count"))
    mtp_layers = int(meta.get("qwen35.nextn_predict_layers", 0))
    num_hidden_layers = block_count - mtp_layers
    if num_hidden_layers <= 0:
        raise ValueError(
            "Invalid Qwen3.5 GGUF layer counts: "
            f"block_count={block_count}, nextn_predict_layers={mtp_layers}"
        )

    full_attention_interval = int(get("full_attention_interval"))
    layer_types = [
        (
            "full_attention"
            if (layer + 1) % full_attention_interval == 0
            else "linear_attention"
        )
        for layer in range(num_hidden_layers)
    ]
    head_dim = int(get("attention.key_length"))
    rope_dim = int(get("rope.dimension_count"))
    rope_sections = [int(value) for value in get("rope.dimension_sections")]
    while rope_sections and rope_sections[-1] == 0:
        rope_sections.pop()
    model_name = str(meta.get("general.name", "")).lower()
    output_gate_type = (
        "swish" if model_name.startswith(("qwen3.6", "qwen3.8")) else None
    )

    config = Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        vocab_size=len(meta["tokenizer.ggml.tokens"]),
        hidden_size=int(get("embedding_length")),
        intermediate_size=int(get("feed_forward_length")),
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=int(get("attention.head_count")),
        num_key_value_heads=int(get("attention.head_count_kv")),
        head_dim=head_dim,
        hidden_act="silu",
        output_gate_type=output_gate_type,
        max_position_embeddings=int(get("context_length")),
        rms_norm_eps=float(get("attention.layer_norm_rms_epsilon")),
        linear_conv_kernel_dim=int(get("ssm.conv_kernel")),
        linear_key_head_dim=int(get("ssm.state_size")),
        linear_value_head_dim=int(get("ssm.state_size")),
        linear_num_key_heads=int(get("ssm.group_count")),
        linear_num_value_heads=int(get("ssm.inner_size")) // int(get("ssm.state_size")),
        layer_types=layer_types,
        full_attention_interval=full_attention_interval,
        attn_output_gate=any(
            name.endswith("attn_q.weight")
            and shape[1] == 2 * int(get("attention.head_count")) * head_dim
            for name, shape in tensors
        ),
        rope_parameters={
            "rope_type": "default",
            "rope_theta": float(get("rope.freq_base")),
            "partial_rotary_factor": rope_dim / head_dim,
            "mrope_interleaved": True,
            "mrope_section": rope_sections,
        },
        partial_rotary_factor=rope_dim / head_dim,
        mtp_num_hidden_layers=mtp_layers,
        mtp_use_dedicated_embeddings=False,
        mamba_ssm_dtype="float32",
        tie_word_embeddings="output.weight" not in tensor_names,
        dtype="bfloat16",
        bos_token_id=token_id("bos"),
        eos_token_id=token_id("eos"),
        pad_token_id=token_id("padding"),
    )
    # Qwen3NextConfig exposes the derived layers through layers_block_type but
    # does not retain the constructor's layer_types argument.
    config.layer_types = layer_types
    return config


GGUF_NATIVE_CONFIG_BUILDERS: Dict[str, Callable[[str], PretrainedConfig]] = {
    "muse-glimmer": MuseGlimmerConfig.from_gguf,
    "qwen35": _qwen35_config_from_gguf,
}


def read_gguf_architecture(gguf_path: str) -> Optional[str]:
    """The ``general.architecture`` string, or None if it cannot be read."""
    try:
        meta, _ = _read_gguf_metadata_snapshot(gguf_path)
        value = meta.get("general.architecture")
        if value is None:
            return None
        return value if isinstance(value, str) else None
    except Exception:
        return None


def has_native_gguf_support(gguf_path: str) -> bool:
    return read_gguf_architecture(gguf_path) in GGUF_NATIVE_CONFIG_BUILDERS


def build_gguf_config(gguf_path: str) -> PretrainedConfig:
    arch = read_gguf_architecture(gguf_path)
    return GGUF_NATIVE_CONFIG_BUILDERS[arch](gguf_path)


_GPT4O_SPLIT_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_PRE_TOKENIZER_REGEX = {
    # LLAMA_VOCAB_PRE_TYPE_LLAMA3
    "llama-bpe": (
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
        r"\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "gpt-4o": _GPT4O_SPLIT_REGEX,
    "llama4": _GPT4O_SPLIT_REGEX,
    "qwen35": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
}

_GGML_TOKEN_TYPE_CONTROL = 3


def build_gguf_generation_config(gguf_path: str):
    """GenerationConfig from GGUF metadata, or None if there is nothing to say.

    llama.cpp records the end-of-generation ids explicitly, and for a
    Harmony-style model the distinction matters: ``eos_token_id`` ends the
    sequence and ``eot_token_id`` ends a turn, so both must stop generation while
    an end-of-*message* id must not -- stopping on that truncates the model
    mid-reasoning, before it answers.
    """
    from transformers import GenerationConfig

    meta, _ = _read_gguf_metadata_snapshot(gguf_path)

    stop_ids = []
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if key in meta:
            value = int(meta[key])
            if value not in stop_ids:
                stop_ids.append(value)
    if not stop_ids:
        return None

    fields: Dict[str, Any] = {
        "eos_token_id": stop_ids if len(stop_ids) > 1 else stop_ids[0]
    }
    if "tokenizer.ggml.bos_token_id" in meta:
        fields["bos_token_id"] = int(meta["tokenizer.ggml.bos_token_id"])
    if "tokenizer.ggml.padding_token_id" in meta:
        fields["pad_token_id"] = int(meta["tokenizer.ggml.padding_token_id"])
    return GenerationConfig(**fields)


def build_gguf_tokenizer(gguf_path: str, **kwargs: Any):
    """Build a fast tokenizer from GGUF metadata alone.

    transformers' own GGUF tokenizer path is unreachable for an architecture its
    checkpoint loader rejects, and its converters key on ``tokenizer.ggml.model``
    (here "gpt2") which loses both the special-token block and the pre-tokenizer
    regex. So the tokenizers spec is assembled directly instead: a byte-level BPE
    over the NORMAL tokens, the CONTROL tokens registered as added specials, and
    the split regex named by ``tokenizer.ggml.pre``.
    """
    import json

    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    meta, _ = _read_gguf_metadata_snapshot(gguf_path)

    tokens = list(meta["tokenizer.ggml.tokens"])
    token_types = [int(t) for t in meta["tokenizer.ggml.token_type"]]
    merges = [tuple(m.split(" ", 1)) for m in meta["tokenizer.ggml.merges"]]

    pre_name = meta.get("tokenizer.ggml.pre")
    if pre_name not in _PRE_TOKENIZER_REGEX:
        raise ValueError(
            f"No pre-tokenizer regex known for tokenizer.ggml.pre={pre_name!r}; "
            f"known: {sorted(_PRE_TOKENIZER_REGEX)}"
        )

    control_ids = [
        i for i, t in enumerate(token_types) if t == _GGML_TOKEN_TYPE_CONTROL
    ]
    # Keep the full id space in the BPE vocabulary. Tokenizers compacts sparse
    # vocabularies, which would silently renumber control tokens above the first
    # hole; registering those same entries as added tokens marks them special
    # while preserving their checkpoint ids.
    vocab = {tok: i for i, tok in enumerate(tokens)}

    def token_of(key):
        idx = meta.get(f"tokenizer.ggml.{key}")
        return None if idx is None else tokens[int(idx)]

    bos = token_of("bos_token_id")

    spec = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": tokens[i],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i in control_ids
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _PRE_TOKENIZER_REGEX[pre_name]},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [list(m) for m in merges],
        },
    }

    if meta.get("tokenizer.ggml.add_bos_token") and bos is not None:
        bos_id = int(meta["tokenizer.ggml.bos_token_id"])
        spec["post_processor"] = {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": bos, "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": bos, "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 0}},
            ],
            "special_tokens": {
                bos: {"id": bos, "ids": [bos_id], "tokens": [bos]},
            },
        }

    backend = Tokenizer.from_str(json.dumps(spec))

    model_max_length = _gguf_model_max_length(meta)
    if model_max_length is not None:
        # PreTrainedTokenizerFast otherwise installs a huge sentinel value.
        # That sentinel exceeds JSON's signed 64-bit range and makes SGLang's
        # OpenAI-compatible /tokenize response fail during serialization.
        kwargs.setdefault("model_max_length", model_max_length)

    named = {
        bos,
        token_of("eos_token_id"),
        token_of("padding_token_id"),
        token_of("unknown_token_id"),
    }
    additional = [tokens[i] for i in control_ids if tokens[i] not in named]

    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=bos,
        eos_token=token_of("eos_token_id"),
        unk_token=token_of("unknown_token_id"),
        pad_token=token_of("padding_token_id"),
        additional_special_tokens=additional,
        chat_template=meta.get("tokenizer.chat_template"),
        **kwargs,
    )
