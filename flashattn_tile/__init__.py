"""FlashAttnTile: fused paged-attention kernels for LLM prefill and decode.

The public API is intentionally small:

* :func:`flashattn_tile.reference.ref_attention_varlen` -- fp32 ground truth
* :func:`flashattn_tile.reference.ref_attention_paged` -- paged ground truth
* :class:`flashattn_tile.paging.PagedKVCache` -- page-table contract
* :func:`flashattn_tile.kernels.prefill.prefill_attention` -- varlen prefill
* :func:`flashattn_tile.kernels.decode.paged_decode_attention` -- paged decode

Everything is importable without CUDA or Triton. When either is missing the
kernel wrappers warn once and execute the numerically equivalent PyTorch
reference path.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .paging import PagedKVCache, allocate, append_tokens, gather_contiguous
from .reference import ref_attention_paged, ref_attention_varlen

__all__ = [
    "__version__",
    "PagedKVCache",
    "allocate",
    "append_tokens",
    "gather_contiguous",
    "ref_attention_varlen",
    "ref_attention_paged",
]
