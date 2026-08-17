"""Ctypes wrapper around the compiled Qwen3.8 MLX C++ engine."""

from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent / "native"
_LIB_PATH = _DIR / "libqwen38_engine.dylib"


def _build_lib() -> Path:
    script = _DIR / "build.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing native engine build script at {script}")
    sources = [
        _DIR / "qwen38_engine.cpp",
        _DIR / "qwen38_c_api.cpp",
        _DIR / "qwen38_engine.h",
        _DIR / "qwen38_c_api.h",
        script,
    ]
    need_build = not _LIB_PATH.is_file()
    if not need_build:
        lib_mtime = _LIB_PATH.stat().st_mtime
        need_build = any(src.stat().st_mtime > lib_mtime for src in sources if src.is_file())
    if need_build:
        subprocess.run(["/bin/zsh", str(script), str(_LIB_PATH)], check=True)
    if not _LIB_PATH.is_file():
        raise FileNotFoundError(f"native engine build did not produce {_LIB_PATH}")
    return _LIB_PATH


class MlxQwen38Config(ctypes.Structure):
    _fields_ = [
        ("hidden_size", ctypes.c_int32),
        ("intermediate_size", ctypes.c_int32),
        ("num_hidden_layers", ctypes.c_int32),
        ("num_attention_heads", ctypes.c_int32),
        ("num_key_value_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("full_attention_interval", ctypes.c_int32),
        ("linear_num_value_heads", ctypes.c_int32),
        ("linear_num_key_heads", ctypes.c_int32),
        ("linear_key_head_dim", ctypes.c_int32),
        ("linear_value_head_dim", ctypes.c_int32),
        ("linear_conv_kernel_dim", ctypes.c_int32),
        ("rms_norm_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("partial_rotary_factor", ctypes.c_float),
        ("quant_group_size", ctypes.c_int32),
        ("quant_bits", ctypes.c_int32),
    ]


def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_build_lib()))
    lib.mlx_qwen38_config_from_json.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(MlxQwen38Config),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_config_from_json.restype = ctypes.c_int
    lib.mlx_qwen38_qlinear.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_qlinear.restype = ctypes.c_int
    lib.mlx_qwen38_gated_delta_step.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_gated_delta_step.restype = ctypes.c_int
    lib.mlx_qwen38_gated_delta_update.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_gated_delta_update.restype = ctypes.c_int
    lib.mlx_qwen38_load.argtypes = [
        ctypes.POINTER(MlxQwen38Config),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_load.restype = ctypes.c_void_p
    lib.mlx_qwen38_reset.argtypes = [ctypes.c_void_p]
    lib.mlx_qwen38_reset.restype = None
    lib.mlx_qwen38_prefill.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_prefill.restype = ctypes.c_int
    lib.mlx_qwen38_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_decode.restype = ctypes.c_int
    lib.mlx_qwen38_load_mtp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.mlx_qwen38_load_mtp.restype = ctypes.c_int
    lib.mlx_qwen38_has_mtp.argtypes = [ctypes.c_void_p]
    lib.mlx_qwen38_has_mtp.restype = ctypes.c_int
    lib.mlx_qwen38_last_spec_width.argtypes = [ctypes.c_void_p]
    lib.mlx_qwen38_last_spec_width.restype = ctypes.c_int
    lib.mlx_qwen38_free.argtypes = [ctypes.c_void_p]
    lib.mlx_qwen38_free.restype = None
    return lib


class NativeQwen38Engine:
    """Shipped C++ Qwen3.8-27B graph (affine q4, Gated DeltaNet + full attn)."""

    def __init__(self, handle: int, lib: ctypes.CDLL):
        self._handle = handle
        self._lib = lib

    @staticmethod
    def config_from_json(text: str) -> MlxQwen38Config:
        lib = _lib()
        cfg = MlxQwen38Config()
        err = ctypes.create_string_buffer(512)
        rc = lib.mlx_qwen38_config_from_json(
            text.encode("utf-8"), ctypes.byref(cfg), err, 512
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return cfg

    @classmethod
    def load(cls, model_dir: str, cfg: MlxQwen38Config | None = None) -> NativeQwen38Engine:
        lib = _lib()
        if cfg is None:
            cfg = cls.config_from_json(Path(model_dir, "config.json").read_text())
        err = ctypes.create_string_buffer(1024)
        handle = lib.mlx_qwen38_load(
            ctypes.byref(cfg), model_dir.encode("utf-8"), err, 1024
        )
        if not handle:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return cls(handle, lib)

    def reset(self) -> None:
        self._lib.mlx_qwen38_reset(self._handle)

    def prefill(
        self, tokens: list[int], *, new_request: bool = True, schedule_decode: bool = True
    ) -> int:
        if new_request:
            self.reset()
        n = len(tokens)
        buf = (ctypes.c_int32 * n)(*tokens)
        out = ctypes.c_int32()
        err = ctypes.create_string_buffer(1024)
        rc = self._lib.mlx_qwen38_prefill(
            self._handle,
            buf,
            n,
            1 if schedule_decode else 0,
            ctypes.byref(out),
            err,
            1024,
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return int(out.value)

    def load_mtp(self, mtp_dir: str) -> None:
        err = ctypes.create_string_buffer(1024)
        rc = self._lib.mlx_qwen38_load_mtp(
            self._handle, mtp_dir.encode("utf-8"), err, 1024
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))

    def has_mtp(self) -> bool:
        return bool(self._lib.mlx_qwen38_has_mtp(self._handle))

    def last_spec_width(self) -> int:
        return int(self._lib.mlx_qwen38_last_spec_width(self._handle))

    def decode(self, token: int) -> int:
        out = ctypes.c_int32()
        err = ctypes.create_string_buffer(1024)
        rc = self._lib.mlx_qwen38_decode(
            self._handle, int(token), ctypes.byref(out), err, 1024
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", "replace"))
        return int(out.value)

    def close(self) -> None:
        if self._handle:
            self._lib.mlx_qwen38_free(self._handle)
            self._handle = 0

    def __del__(self) -> None:
        self.close()


def qlinear(
    x: Any,
    weight: Any,
    scales: Any,
    biases: Any,
    *,
    group_size: int,
    bits: int,
) -> Any:
    """Drive the shipped C++ quantized matmul. Arrays are host float/uint32."""
    import numpy as np

    lib = _lib()
    x_np = np.ascontiguousarray(x, dtype=np.float32)
    w_np = np.ascontiguousarray(weight, dtype=np.uint32)
    s_np = np.ascontiguousarray(scales, dtype=np.float32)
    b_np = np.ascontiguousarray(biases, dtype=np.float32)
    rows, cols = x_np.shape
    out_features = w_np.shape[0]
    y = np.empty((rows, out_features), dtype=np.float32)
    err = ctypes.create_string_buffer(512)
    rc = lib.mlx_qwen38_qlinear(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows,
        cols,
        w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        s_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out_features,
        group_size,
        bits,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        err,
        512,
    )
    if rc != 0:
        raise RuntimeError(err.value.decode("utf-8", "replace"))
    return y


def gated_delta_step(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    state: Any,
) -> tuple[Any, Any]:
    import numpy as np

    lib = _lib()
    q_np = np.ascontiguousarray(q, dtype=np.float32)
    k_np = np.ascontiguousarray(k, dtype=np.float32)
    v_np = np.ascontiguousarray(v, dtype=np.float32)
    g_np = np.ascontiguousarray(g, dtype=np.float32)
    b_np = np.ascontiguousarray(beta, dtype=np.float32)
    st_np = np.ascontiguousarray(state, dtype=np.float32)
    h_k, d_k = q_np.shape
    h_v, d_v = v_np.shape
    y = np.empty((h_v, d_v), dtype=np.float32)
    st_out = np.empty_like(st_np)
    err = ctypes.create_string_buffer(512)
    rc = lib.mlx_qwen38_gated_delta_step(
        q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        st_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        h_k,
        h_v,
        d_k,
        d_v,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        st_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        err,
        512,
    )
    if rc != 0:
        raise RuntimeError(err.value.decode("utf-8", "replace"))
    return y, st_out


def gated_delta_update(
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    state: Any,
) -> tuple[Any, Any]:
    """Drive the shipped Metal Gated-DeltaNet over T steps. B is implicit 1.

    q/k: [T, Hk, Dk], v: [T, Hv, Dv], g/beta: [T, Hv], state: [Hv, Dv, Dk].
    """
    import numpy as np

    lib = _lib()
    q_np = np.ascontiguousarray(q, dtype=np.float32)
    k_np = np.ascontiguousarray(k, dtype=np.float32)
    v_np = np.ascontiguousarray(v, dtype=np.float32)
    g_np = np.ascontiguousarray(g, dtype=np.float32)
    b_np = np.ascontiguousarray(beta, dtype=np.float32)
    st_np = np.ascontiguousarray(state, dtype=np.float32)
    t, h_k, d_k = q_np.shape
    _, h_v, d_v = v_np.shape
    y = np.empty((t, h_v, d_v), dtype=np.float32)
    st_out = np.empty_like(st_np)
    err = ctypes.create_string_buffer(512)
    rc = lib.mlx_qwen38_gated_delta_update(
        q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        st_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        t,
        h_k,
        h_v,
        d_k,
        d_v,
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        st_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        err,
        512,
    )
    if rc != 0:
        raise RuntimeError(err.value.decode("utf-8", "replace"))
    return y, st_out


def config_from_model_dir(model_dir: str) -> MlxQwen38Config:
    return NativeQwen38Engine.config_from_json(
        Path(model_dir, "config.json").read_text()
    )
