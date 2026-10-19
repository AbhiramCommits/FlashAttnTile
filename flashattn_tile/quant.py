"""INT8 / FP8 KV-cache quantisation with per-(page, KV-head) scales.

Scale granularity is one scale per ``(page, kv_head)`` pair, which matches how
paged kernels tile K/V: the decode kernel reads a whole page for one KV head and
can apply a single scalar. Scales are stored as fp32 of shape
``[num_pages, num_kv_heads]``.

Two storage formats are supported:

``float8_e4m3fn``
    Used when the installed PyTorch exposes the dtype *and* the Triton FP8
    kernel path is usable (CUDA GPU). Range is ``[-448, 448]``.
``int8``
    Symmetric per-page-per-head quantisation, range ``[-127, 127]``. This is the
    path exercised on CPU-only hosts, including the committed accuracy numbers.

:func:`default_quant_mode` performs the detection at runtime; the chosen mode is
recorded alongside every measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paging import PagedKVCache, validate_cache

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0
INT8_MAX = 127.0
FP8_MIN_SCALE = 1e-8
INT8_MIN_SCALE = 1e-8


@dataclass
class QuantScales:
    """Per-(page, KV-head) dequantisation scales."""

    k: torch.Tensor
    v: torch.Tensor
    mode: str
    granularity: str = "per_page_per_head"

    def as_dict(self) -> dict:
        return {"k": self.k, "v": self.v, "mode": self.mode, "granularity": self.granularity}


def available_quant_modes() -> list[str]:
    modes = ["int8"]
    if FP8_DTYPE is not None:
        modes.insert(0, "float8_e4m3fn")
    return modes


def default_quant_mode() -> str:
    """Pick the format the installed stack actually supports.

    FP8 is only selected when the dtype exists *and* a CUDA device is present,
    because the quantised kernel that consumes it is Triton/CUDA-only. On this
    build host that evaluates to ``"int8"``.
    """
    if FP8_DTYPE is not None and torch.cuda.is_available():
        return "float8_e4m3fn"
    return "int8"


def _quant_dtype(mode: str) -> torch.dtype:
    if mode == "auto":
        mode = default_quant_mode()
    if mode == "int8":
        return torch.int8
    if mode in ("float8_e4m3fn", "fp8"):
        if FP8_DTYPE is None:
            raise RuntimeError("this PyTorch build has no torch.float8_e4m3fn")
        return FP8_DTYPE
    raise ValueError(f"unknown quantisation mode {mode!r}; expected 'int8' or 'float8_e4m3fn'")


def _max_representable(mode: str) -> float:
    return INT8_MAX if mode == "int8" else FP8_MAX


def quantize_kv_pages(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mode: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Quantise paged K/V tensors.

    ``k_cache``/``v_cache`` are ``[num_pages, page_size, num_kv_heads, head_dim]``
    floating point tensors. Returns ``(qk, qv, scale_k, scale_v, info)`` where
    the scales have shape ``[num_pages, num_kv_heads]`` and ``info`` records the
    format actually used.
    """
    if k_cache.shape != v_cache.shape:
        raise ValueError(f"k/v shape mismatch: {tuple(k_cache.shape)} vs {tuple(v_cache.shape)}")
    if k_cache.dim() != 4:
        raise ValueError("cache must be [num_pages, page_size, num_kv_heads, head_dim]")
    if not k_cache.is_floating_point():
        raise ValueError(f"quantize_kv_pages expects a float cache, got {k_cache.dtype}")
    if mode == "auto":
        mode = default_quant_mode()
    dtype = _quant_dtype(mode)
    max_val = _max_representable(mode)
    min_scale = FP8_MIN_SCALE if mode != "int8" else INT8_MIN_SCALE

    _, page_size, kv_heads, head_dim = k_cache.shape
    k_f32 = k_cache.detach().to(torch.float32)
    v_f32 = v_cache.detach().to(torch.float32)

    # Reduce over the token and head-dim axes per (page, kv_head). Keep the
    # tensor in NHD layout so quantised values are never reshuffled.
    k_amax = k_f32.abs().amax(dim=(1, 3))
    v_amax = v_f32.abs().amax(dim=(1, 3))
    k_scale = (k_amax / max_val).clamp_min(min_scale)
    v_scale = (v_amax / max_val).clamp_min(min_scale)

    k_scaled = k_f32 / k_scale[:, None, :, None]
    v_scaled = v_f32 / v_scale[:, None, :, None]

    if mode == "int8":
        qk = torch.round(k_scaled).clamp(-max_val, max_val).to(dtype)
        qv = torch.round(v_scaled).clamp(-max_val, max_val).to(dtype)
    else:
        qk = k_scaled.clamp(-max_val, max_val).to(dtype)
        qv = v_scaled.clamp(-max_val, max_val).to(dtype)

    info = {
        "mode": mode,
        "dtype": str(qk.dtype),
        "max_representable": max_val,
        "granularity": "per_page_per_head",
        "page_size": int(page_size),
        "num_kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "scale_dtype": str(k_scale.dtype),
        "scale_min": float(k_scale.min().item()),
        "scale_max": float(k_scale.max().item()),
        "rounding": "round-to-nearest" if mode == "int8" else "cast-to-fp8",
    }
    return qk, qv, k_scale, v_scale, info


def dequantize_kv_pages(
    qk: torch.Tensor,
    qv: torch.Tensor,
    scale_k: torch.Tensor,
    scale_v: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantise pages back to a floating point cache."""
    if qk.shape != qv.shape:
        raise ValueError("quantised k/v shapes must match")
    if scale_k.shape != (qk.shape[0], qk.shape[2]):
        raise ValueError(
            f"scale_k shape {tuple(scale_k.shape)} must be [{qk.shape[0]}, {qk.shape[2]}]"
        )
    k = qk.to(torch.float32) * scale_k.to(torch.float32).unsqueeze(1).unsqueeze(-1)
    v = qv.to(torch.float32) * scale_v.to(torch.float32).unsqueeze(1).unsqueeze(-1)
    return k.to(out_dtype), v.to(out_dtype)


def quantize_cache(cache: PagedKVCache, mode: str = "auto") -> tuple[PagedKVCache, QuantScales]:
    """Return a new :class:`PagedKVCache` holding quantised K/V plus its scales."""
    validate_cache(cache)
    qk, qv, sk, sv, info = quantize_kv_pages(cache.k_cache, cache.v_cache, mode=mode)
    quantized = PagedKVCache(
        k_cache=qk,
        v_cache=qv,
        page_table=cache.page_table,
        seq_lens=cache.seq_lens,
        page_size=cache.page_size,
    )
    return quantized, QuantScales(k=sk, v=sv, mode=info["mode"])


def dequantize_cache(
    cache: PagedKVCache, scales: QuantScales | dict, out_dtype: torch.dtype = torch.float16
) -> PagedKVCache:
    """Return a float copy of a quantised cache (reference/debug path)."""
    sk = scales["k"] if isinstance(scales, dict) else scales.k
    sv = scales["v"] if isinstance(scales, dict) else scales.v
    k, v = dequantize_kv_pages(cache.k_cache, cache.v_cache, sk, sv, out_dtype=out_dtype)
    return PagedKVCache(
        k_cache=k,
        v_cache=v,
        page_table=cache.page_table,
        seq_lens=cache.seq_lens,
        page_size=cache.page_size,
    )


def kv_bytes_per_token(
    num_kv_heads: int,
    head_dim: int,
    storage: str = "fp16",
    page_size: int = 16,
) -> float:
    """Analytic bytes of K+V per cached token, including scale overhead.

    ``storage`` is one of ``fp16`` (2 B), ``bf16`` (2 B), ``int8`` (1 B) or
    ``fp8`` (1 B). Scales are fp32, two per (page, KV head).
    """
    elem = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "fp8": 1}[storage]
    kv = 2 * num_kv_heads * head_dim * elem
    scales = 2 * num_kv_heads * 4 / page_size
    return float(kv + scales)
