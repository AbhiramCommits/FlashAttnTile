"""A minimal, vLLM-shaped attention backend for FlashAttnTile.

This is **not** a vLLM plugin: it mirrors the interface shape so the contract is
concrete and testable. The pieces match vLLM's attention backend abstraction:

``get_name()``
    Stable backend identifier, e.g. ``"flashattn_tile"``.
``get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_dim)``
    vLLM convention: ``(2, num_blocks, block_size, num_kv_heads, head_dim)``,
    where the leading ``2`` selects K or V. :class:`PagedKVCache` stores each
    tensor separately, so its per-tensor shape is the trailing four dims.
``forward(query, kv_cache, attn_metadata)``
    Routes to prefill when ``attn_metadata.is_prefill`` is true, otherwise to
    paged decode. ``kv_cache`` is a :class:`PagedKVCache`; ``attn_metadata``
    carries ``page_table``, ``seq_lens``, ``cu_seqlens_q``/``cu_seqlens_k`` and
    the two ``max_seqlen_*`` grid bounds.

Shapes
------
decode query: ``[batch, num_heads, head_dim]`` (a trailing ``[batch, 1, ...]``
is squeezed). prefill query: packed ``[total_q, num_heads, head_dim]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kernels.decode import paged_decode_attention
from .kernels.prefill import prefill_attention
from .paging import PagedKVCache, allocate

KV_CACHE_NDIM = 4


@dataclass
class AttentionMetadata:
    """Per-step metadata passed to :meth:`FlashAttnTileBackend.forward`."""

    page_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    is_prefill: bool = False
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    # Packed K/V for the prefill call. Prefill attention needs the K/V of the
    # tokens being prefilled; the paged cache is written by the surrounding
    # serving engine. Decode reads K/V from ``kv_cache`` instead.
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    softmax_scale: float | None = None


def get_kv_cache_shape(
    num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int
) -> tuple[int, int, int, int, int]:
    """vLLM-style cache shape: ``(2, blocks, block_size, kv_heads, head_dim)``."""
    return (2, int(num_blocks), int(block_size), int(num_kv_heads), int(head_dim))


class FlashAttnTileBackend:
    """Paged attention backend routing prefill and decode to FlashAttnTile."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.dtype = dtype
        self.device = device

    def get_name(self) -> str:
        return "flashattn_tile"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int
    ) -> tuple[int, int, int, int, int]:
        return get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_dim)

    @staticmethod
    def get_kv_cache_shape_per_tensor(
        num_blocks: int, block_size: int, num_kv_heads: int, head_dim: int
    ) -> tuple[int, int, int, int]:
        return (int(num_blocks), int(block_size), int(num_kv_heads), int(head_dim))

    def allocate_kv_cache(
        self,
        batch: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
    ) -> PagedKVCache:
        """Allocate a cache with the backend's head geometry."""
        return allocate(
            batch=batch,
            max_seq_len=max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            dtype=dtype,
            device=device,
        )

    def _resolve_cache(self, kv_cache: PagedKVCache, attn_metadata: AttentionMetadata) -> PagedKVCache:
        page_table = attn_metadata.page_table
        seq_lens = attn_metadata.seq_lens
        if page_table is None and seq_lens is None:
            return kv_cache
        return PagedKVCache(
            k_cache=kv_cache.k_cache,
            v_cache=kv_cache.v_cache,
            page_table=kv_cache.page_table if page_table is None else page_table,
            seq_lens=kv_cache.seq_lens if seq_lens is None else seq_lens,
            page_size=kv_cache.page_size,
        )

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: PagedKVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Route ``query`` through prefill or paged decode."""
        if not isinstance(kv_cache, PagedKVCache):
            raise TypeError("kv_cache must be a PagedKVCache")
        cache = self._resolve_cache(kv_cache, attn_metadata)
        if not attn_metadata.is_prefill and query.dim() == 4:
            query = query.squeeze(1)
        if query.dim() != 3:
            raise ValueError(f"query must be 3-D, got {tuple(query.shape)}")
        if query.shape[1] != self.num_heads:
            raise ValueError(f"query has {query.shape[1]} heads, backend expects {self.num_heads}")

        if attn_metadata.is_prefill:
            if attn_metadata.cu_seqlens_q is None or attn_metadata.cu_seqlens_k is None:
                raise ValueError("prefill routing requires cu_seqlens_q and cu_seqlens_k")
            max_q = attn_metadata.max_seqlen_q
            max_k = attn_metadata.max_seqlen_k
            if max_q is None:
                max_q = int((attn_metadata.cu_seqlens_q[1:] - attn_metadata.cu_seqlens_q[:-1]).max())
            if max_k is None:
                max_k = int((attn_metadata.cu_seqlens_k[1:] - attn_metadata.cu_seqlens_k[:-1]).max())
            if attn_metadata.k is None or attn_metadata.v is None:
                raise ValueError(
                    "prefill routing requires attn_metadata.k and attn_metadata.v "
                    "(packed [total_k, num_kv_heads, head_dim])"
                )
            return prefill_attention(
                query,
                attn_metadata.k,
                attn_metadata.v,
                attn_metadata.cu_seqlens_q,
                attn_metadata.cu_seqlens_k,
                int(max_q),
                int(max_k),
                causal=True,
                softmax_scale=attn_metadata.softmax_scale,
            )

        return paged_decode_attention(query, cache, softmax_scale=attn_metadata.softmax_scale)
