"""PyTorch custom ops for FlashAttnTile, with fake kernels for ``torch.compile``.

Two ops are registered in the ``flashattn_tile`` namespace:

``flashattn_tile::paged_decode``
    ``(q, k_cache, v_cache, page_table, seq_lens, page_size, softmax_scale,
    split_k) -> [batch, heads, head_dim]``
``flashattn_tile::prefill_varlen``
    ``(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal,
    softmax_scale) -> [total_q, heads, head_dim]``

Both register ``register_fake`` implementations that only depend on tensor
metadata, so a ``torch.compile(fullgraph=True)`` graph can contain them without
graph breaks. The real implementations route to the Triton wrappers, which fall
back to the fp32 reference on CPU.
"""

from __future__ import annotations

import torch

from .kernels.decode import paged_decode_attention
from .kernels.prefill import prefill_attention
from .paging import PagedKVCache

__all__ = [
    "paged_decode",
    "prefill_varlen",
    "ops_available",
    "PAGED_DECODE_OP",
    "PREFILL_OP",
]


@torch.library.custom_op("flashattn_tile::paged_decode", mutates_args=())
def paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    softmax_scale: float,
    split_k: int,
) -> torch.Tensor:
    """Custom-op implementation of paged decode attention."""
    cache = PagedKVCache(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        page_size=int(page_size),
    )
    return paged_decode_attention(q, cache, softmax_scale=softmax_scale,
                                  config={"split_k": int(split_k)})


@paged_decode.register_fake
def _paged_decode_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    softmax_scale: float,
    split_k: int,
) -> torch.Tensor:
    torch._check(q.dim() == 3, lambda: f"q must be 3-D, got {q.dim()}")
    torch._check(q.shape[0] == seq_lens.shape[0], lambda: "q batch must match seq_lens")
    torch._check(q.shape[2] == k_cache.shape[3], lambda: "q head_dim must match cache")
    torch._check(q.shape[1] % k_cache.shape[2] == 0, lambda: "heads must be divisible by KV heads")
    return q.new_empty(q.shape)


@torch.library.custom_op("flashattn_tile::prefill_varlen", mutates_args=())
def prefill_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """Custom-op implementation of packed varlen prefill attention."""
    return prefill_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        causal=bool(causal),
        softmax_scale=softmax_scale,
    )


@prefill_varlen.register_fake
def _prefill_varlen_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    torch._check(q.dim() == 3, lambda: f"q must be 3-D, got {q.dim()}")
    torch._check(k.shape == v.shape, lambda: "k and v shapes must match")
    torch._check(q.shape[2] == k.shape[2], lambda: "q and k head_dim must match")
    torch._check(q.shape[1] % k.shape[1] == 0, lambda: "heads must be divisible by KV heads")
    return q.new_empty(q.shape)


PAGED_DECODE_OP = "flashattn_tile::paged_decode"
PREFILL_OP = "flashattn_tile::prefill_varlen"


def ops_available() -> bool:
    """True when both custom ops registered in this process."""
    return (
        hasattr(torch.ops.flashattn_tile, "paged_decode")
        and hasattr(torch.ops.flashattn_tile, "prefill_varlen")
    )
