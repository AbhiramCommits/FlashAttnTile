"""Test-suite bootstrap: make the in-tree package importable without install."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashattn_tile.env import triton_available  # noqa: E402

HAS_CUDA = __import__("torch").cuda.is_available()
HAS_TRITON = triton_available()

requires_gpu = pytest.mark.skipif(
    not (HAS_CUDA and HAS_TRITON),
    reason=f"requires CUDA + Triton (cuda={HAS_CUDA}, triton={HAS_TRITON})",
)


@pytest.fixture(autouse=True)
def _reset_kernel_fallback_warnings():
    """Kernel modules warn once per process; tests assert each warning fresh."""
    from flashattn_tile.kernels import decode, prefill

    prefill._WARNED.clear()
    decode._WARNED.clear()
    yield
