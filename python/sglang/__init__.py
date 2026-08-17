# SGLang public APIs

# sglang.srt.environ must run before the rest of this file's imports
# (hf_transformers_patches, lang.api, ...), which pull in torch and
# FlashInfer: those claim these cache dirs early, and the first value set is
# the one that sticks. Safe here -- environ has no heavy dependency (no torch).
from sglang.srt.environ import redirect_third_party_caches

redirect_third_party_caches()

# Install stubs early for platforms where certain dependencies are unavailable
# (e.g. macOS/MPS has no triton, and torch.mps lacks Stream / set_device /
# get_device_properties).  This must run before any downstream imports.
import platform as _platform
import sys as _sys

_legacy_intel_mps_torch = False
if _sys.platform == "darwin":
    try:
        import torch as _torch

        if _torch.backends.mps.is_available():
            _torch_version = tuple(
                int(part) for part in _torch.__version__.split("+", 1)[0].split(".")[:2]
            )
            _legacy_intel_mps_torch = (
                _platform.machine() == "x86_64" and _torch_version < (2, 4)
            )
            if _legacy_intel_mps_torch:
                # Transformers 5 registers training-only custom operators at
                # import time. They are unreachable in SGLang inference; a
                # lightweight decorator shim lets its configuration and
                # processor modules import under the final Intel-mac wheel.
                if not hasattr(_torch.library, "custom_op"):
                    _torch.library.custom_op = lambda *args, **kwargs: (
                        lambda function: function
                    )
                if not hasattr(_torch.library, "register_fake"):
                    _torch.library.register_fake = lambda *args, **kwargs: (
                        lambda function: function
                    )
                if not hasattr(_torch.library, "register_autograd"):
                    _torch.library.register_autograd = lambda *args, **kwargs: None
                if not hasattr(_torch.compiler, "is_compiling"):
                    _torch.compiler.is_compiling = lambda: False
                for _unsigned, _signed in (
                    ("uint16", "int16"),
                    ("uint32", "int32"),
                    ("uint64", "int64"),
                ):
                    if not hasattr(_torch, _unsigned):
                        setattr(_torch, _unsigned, getattr(_torch, _signed))
                del _unsigned
                del _signed
            from sglang._triton_stub import install as _install_triton_stub

            _install_triton_stub()
            del _install_triton_stub

            from sglang._mps_stub import install as _install_mps_stub

            _install_mps_stub()
            del _install_mps_stub
        del _torch
    except ImportError:
        pass
del _platform
del _sys

if _legacy_intel_mps_torch:
    # PyTorch 2.2.2 is the final upstream x86_64 macOS wheel. Transformers 5
    # only uses its version floor to gate model imports; SGLang owns the model
    # implementation on this path. Let Transformers initialize its torch
    # surface, then restore metadata immediately so later feature probes still
    # see the real version.
    import importlib.metadata as _metadata

    _metadata_version = _metadata.version

    def _intel_mps_metadata_version(
        distribution_name: str, _real_version=_metadata_version
    ) -> str:
        if distribution_name.lower().replace("_", "-") == "torch":
            return "2.4.0"
        return _real_version(distribution_name)

    _metadata.version = _intel_mps_metadata_version
    try:
        from sglang.srt.utils.hf_transformers_patches import (
            apply_all as _apply_hf_patches,
        )
    finally:
        _metadata.version = _metadata_version
        del _metadata
        del _metadata_version
        del _intel_mps_metadata_version
else:
    from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches

del _legacy_intel_mps_torch

_apply_hf_patches()
del _apply_hf_patches

# Frontend Language APIs
from sglang.global_config import global_config
from sglang.lang.api import (
    Engine,
    Runtime,
    assistant,
    assistant_begin,
    assistant_end,
    flush_cache,
    function,
    gen,
    gen_int,
    gen_string,
    get_server_info,
    image,
    select,
    separate_reasoning,
    set_default_backend,
    system,
    system_begin,
    system_end,
    user,
    user_begin,
    user_end,
    video,
)
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.choices import (
    greedy_token_selection,
    token_length_normalized,
    unconditional_likelihood_normalized,
)

# Lazy import some libraries
from sglang.utils import LazyImport
from sglang.version import __version__

Anthropic = LazyImport("sglang.lang.backend.anthropic", "Anthropic")
Crusoe = LazyImport("sglang.lang.backend.crusoe", "Crusoe")
LiteLLM = LazyImport("sglang.lang.backend.litellm", "LiteLLM")
OpenAI = LazyImport("sglang.lang.backend.openai", "OpenAI")
VertexAI = LazyImport("sglang.lang.backend.vertexai", "VertexAI")

# Runtime Engine APIs
ServerArgs = LazyImport("sglang.srt.server_args", "ServerArgs")
Engine = LazyImport("sglang.srt.entrypoints.engine", "Engine")

__all__ = [
    "Engine",
    "Runtime",
    "assistant",
    "assistant_begin",
    "assistant_end",
    "flush_cache",
    "function",
    "gen",
    "gen_int",
    "gen_string",
    "get_server_info",
    "image",
    "select",
    "separate_reasoning",
    "set_default_backend",
    "system",
    "system_begin",
    "system_end",
    "user",
    "user_begin",
    "user_end",
    "video",
    "RuntimeEndpoint",
    "greedy_token_selection",
    "token_length_normalized",
    "unconditional_likelihood_normalized",
    "ServerArgs",
    "Anthropic",
    "Crusoe",
    "LiteLLM",
    "OpenAI",
    "VertexAI",
    "global_config",
    "__version__",
]
