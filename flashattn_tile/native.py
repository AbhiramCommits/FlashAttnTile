"""Optional CUDA C++ extension for the paged-decode baseline.

The extension is built lazily with :func:`torch.utils.cpp_extension.load`. On a
host without CUDA or without ``nvcc``, :func:`is_available` returns ``False`` and
the reason is exposed through :func:`unavailable_reason`; importing this module
never raises.
"""

from __future__ import annotations

import os
import shutil
import threading

import torch

_LOCK = threading.Lock()
_EXT = None
_ERROR: str | None = None
_TRIED = False


def _build() -> None:
    global _EXT, _ERROR, _TRIED
    with _LOCK:
        if _TRIED:
            return
        _TRIED = True
        if not torch.cuda.is_available():
            _ERROR = "no CUDA device detected"
            return
        if shutil.which("nvcc") is None and not os.environ.get("CUDA_HOME"):
            _ERROR = "nvcc not found and CUDA_HOME is unset"
            return
        try:
            from torch.utils.cpp_extension import load

            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _EXT = load(
                name="flashattn_native",
                sources=[
                    os.path.join(here, "csrc", "paged_decode.cu"),
                    os.path.join(here, "csrc", "bindings.cpp"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - build-host dependent
            _ERROR = f"{type(exc).__name__}: {exc}"
            _EXT = None


def is_available() -> bool:
    """Try (once) to build the extension; return whether it is usable."""
    _build()
    return _EXT is not None


def unavailable_reason() -> str:
    """Human-readable reason why the native path is unavailable ('' if built)."""
    _build()
    if _EXT is not None:
        return ""
    return _ERROR or "CUDA C++ extension not built"


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Decode through the native CUDA C++ kernel (requires a built extension)."""
    if not is_available():
        raise RuntimeError(f"native CUDA path unavailable: {unavailable_reason()}")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    return _EXT.paged_decode(  # type: ignore[union-attr]
        q, k_cache, v_cache, page_table, seq_lens, int(page_size), float(softmax_scale)
    )
