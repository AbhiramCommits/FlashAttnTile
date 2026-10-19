"""Paged KV-cache contract (vLLM / FlashInfer style NHD layout).

Layout
------
``k_cache`` / ``v_cache``
    ``[num_pages, page_size, num_kv_heads, head_dim]``. Page ``p`` holds
    ``page_size`` consecutive tokens for every KV head. This is the "NHD"
    layout used by vLLM's default paged-attention backend.
``page_table``
    ``int32 [batch, max_pages_per_seq]``. ``page_table[b, i]`` is the physical
    page id holding logical block ``i`` of sequence ``b``. Entries past the
    sequence length are ignored.
``seq_lens``
    ``int32 [batch]``, number of valid tokens per sequence. A token at logical
    position ``t`` lives in physical page ``page_table[b, t // page_size]`` at
    slot ``t % page_size``.

Allocation here is a simple contiguous mapping; the kernels only rely on the
page-table indirection, so a serving engine is free to permute the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PagedKVCache:
    """Paged K/V cache plus its page table.

    Attributes
    ----------
    k_cache, v_cache:
        ``[num_pages, page_size, num_kv_heads, head_dim]``, same dtype/device.
        A quantized cache stores ``int8`` or ``float8_e4m3fn`` here instead.
    page_table:
        ``int32 [batch, max_pages_per_seq]`` physical page ids.
    seq_lens:
        ``int32 [batch]`` valid token counts.
    page_size:
        Tokens per page. Must divide the cache's second dimension.
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    page_size: int

    @property
    def batch_size(self) -> int:
        return int(self.seq_lens.shape[0])

    @property
    def num_pages(self) -> int:
        return int(self.k_cache.shape[0])

    @property
    def num_kv_heads(self) -> int:
        return int(self.k_cache.shape[2])

    @property
    def head_dim(self) -> int:
        return int(self.k_cache.shape[3])

    @property
    def max_pages_per_seq(self) -> int:
        return int(self.page_table.shape[1])

    @property
    def capacity(self) -> int:
        """Maximum tokens per sequence addressable by the page table."""
        return self.max_pages_per_seq * self.page_size


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_cache(cache: PagedKVCache, *, check_pages: bool = True) -> None:
    """Validate shapes, dtypes, page bounds and sequence lengths.

    Raises ``AssertionError`` with a message that names the offending field.
    ``check_pages`` walks every *active* page table entry, which is cheap for
    the batch sizes used here and catches out-of-bounds tables loudly.
    """
    _assert(isinstance(cache, PagedKVCache), "cache must be a PagedKVCache")
    k, v = cache.k_cache, cache.v_cache
    _assert(k.dim() == 4, f"k_cache must be 4-D [pages, page_size, kv_heads, dim], got {tuple(k.shape)}")
    _assert(k.shape == v.shape, f"k_cache {tuple(k.shape)} and v_cache {tuple(v.shape)} must match")
    _assert(k.dtype == v.dtype, f"k_cache dtype {k.dtype} != v_cache dtype {v.dtype}")
    _assert(k.device == v.device, "k_cache and v_cache must live on the same device")
    _assert(cache.page_size > 0, "page_size must be positive")
    _assert(
        k.shape[1] == cache.page_size,
        f"k_cache page dim {k.shape[1]} != page_size {cache.page_size}",
    )
    _assert(cache.page_table.dtype == torch.int32, f"page_table must be int32, got {cache.page_table.dtype}")
    _assert(cache.seq_lens.dtype == torch.int32, f"seq_lens must be int32, got {cache.seq_lens.dtype}")
    _assert(cache.page_table.dim() == 2, "page_table must be [batch, max_pages_per_seq]")
    _assert(cache.seq_lens.dim() == 1, "seq_lens must be [batch]")
    _assert(
        cache.page_table.shape[0] == cache.seq_lens.shape[0],
        "page_table and seq_lens batch dimensions disagree",
    )
    _assert(
        cache.page_table.device == cache.seq_lens.device,
        "page_table and seq_lens must live on the same device",
    )
    _assert(
        int(cache.seq_lens.min().item()) >= 0,
        "seq_lens must be non-negative",
    )

    capacity = cache.max_pages_per_seq * cache.page_size
    max_len = int(cache.seq_lens.max().item()) if cache.seq_lens.numel() else 0
    _assert(max_len <= capacity, f"seq_len {max_len} exceeds page-table capacity {capacity}")

    if check_pages and max_len > 0:
        pages_needed = math.ceil(max_len / cache.page_size)
        used = cache.page_table[:, :pages_needed].long()
        _assert(int(used.min().item()) >= 0, "page_table contains a negative page id")
        _assert(
            int(used.max().item()) < cache.num_pages,
            f"page_table references page {int(used.max().item())} but cache has {cache.num_pages} pages",
        )


def allocate(
    batch: int,
    max_seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
) -> PagedKVCache:
    """Allocate a zeroed paged KV cache with a contiguous page mapping."""
    _assert(batch > 0, "batch must be positive")
    _assert(max_seq_len > 0, "max_seq_len must be positive")
    _assert(num_kv_heads > 0 and head_dim > 0, "num_kv_heads and head_dim must be positive")
    _assert(page_size > 0, "page_size must be positive")
    _assert(dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8,
                      getattr(torch, "float8_e4m3fn", torch.float16)),
            f"unsupported cache dtype {dtype}")

    pages_per_seq = math.ceil(max_seq_len / page_size)
    num_pages = batch * pages_per_seq
    k_cache = torch.zeros(
        (num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        batch, pages_per_seq
    )
    seq_lens = torch.zeros(batch, dtype=torch.int32, device=device)
    cache = PagedKVCache(
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        seq_lens=seq_lens,
        page_size=page_size,
    )
    validate_cache(cache)
    return cache


def append_tokens(cache: PagedKVCache, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
    """Append exactly one token per sequence to ``cache``.

    ``new_k``/``new_v`` are ``[batch, num_kv_heads, head_dim]``. Writes go
    through ``page_table`` at logical position ``seq_lens[b]``; ``seq_lens`` is
    incremented in place afterwards.
    """
    validate_cache(cache, check_pages=False)
    _assert(
        new_k.shape == (cache.batch_size, cache.num_kv_heads, cache.head_dim),
        f"new_k must be [{cache.batch_size}, {cache.num_kv_heads}, {cache.head_dim}], "
        f"got {tuple(new_k.shape)}",
    )
    _assert(new_v.shape == new_k.shape, f"new_v shape {tuple(new_v.shape)} != new_k shape {tuple(new_k.shape)}")
    _assert(new_k.dtype == cache.k_cache.dtype, f"new_k dtype {new_k.dtype} != cache dtype {cache.k_cache.dtype}")
    _assert(new_v.dtype == cache.k_cache.dtype, f"new_v dtype {new_v.dtype} != cache dtype {cache.k_cache.dtype}")
    _assert(new_k.device == cache.k_cache.device, "new_k must be on the cache device")
    _assert(new_v.device == cache.k_cache.device, "new_v must be on the cache device")

    positions = cache.seq_lens.long()
    _assert(
        int(positions.max().item()) < cache.capacity,
        f"append would exceed page-table capacity {cache.capacity}",
    )
    max_pages = int(cache.page_table.shape[1])
    page_idx = torch.div(positions, cache.page_size, rounding_mode="floor")
    _assert(int(page_idx.max().item()) < max_pages, "append would index past the page table")
    slots = positions % cache.page_size

    batch_idx = torch.arange(cache.batch_size, device=cache.k_cache.device)
    phys = cache.page_table[batch_idx, page_idx.to(cache.page_table.device)].long()
    _assert(
        int(phys.min().item()) >= 0 and int(phys.max().item()) < cache.num_pages,
        f"page_table maps to a page outside [0, {cache.num_pages})",
    )
    slots = slots.to(cache.k_cache.device)
    cache.k_cache[phys, slots] = new_k
    cache.v_cache[phys, slots] = new_v
    cache.seq_lens += 1


def gather_paged(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a padded contiguous view ``[batch, max_len, kv_heads, dim]``.

    Rows past each sequence's length are zero. This is the reference path used
    to compare against SDPA, never the kernel path.
    """
    _assert(k_cache.shape == v_cache.shape, "k_cache and v_cache shapes must match")
    _assert(page_table.dtype == torch.int32, "page_table must be int32")
    _assert(seq_lens.dtype == torch.int32, "seq_lens must be int32")
    _assert(page_table.dim() == 2 and seq_lens.dim() == 1, "expected page_table [B, P] and seq_lens [B]")
    _assert(page_table.shape[0] == seq_lens.shape[0], "page_table / seq_lens batch mismatch")
    _assert(k_cache.shape[1] == page_size, "k_cache page dim != page_size")

    batch = int(seq_lens.shape[0])
    max_len = int(seq_lens.max().item()) if batch else 0
    out_k = k_cache.new_zeros((batch, max_len, k_cache.shape[2], k_cache.shape[3]))
    out_v = v_cache.new_zeros((batch, max_len, v_cache.shape[2], v_cache.shape[3]))
    if max_len == 0:
        return out_k, out_v

    capacity = int(page_table.shape[1]) * page_size
    _assert(max_len <= capacity, f"seq_len {max_len} exceeds page-table capacity {capacity}")

    for b in range(batch):
        length = int(seq_lens[b].item())
        if length == 0:
            continue
        pages_needed = math.ceil(length / page_size)
        ids = page_table[b, :pages_needed].long()
        _assert(
            int(ids.min().item()) >= 0 and int(ids.max().item()) < int(k_cache.shape[0]),
            f"sequence {b} page id outside [0, {int(k_cache.shape[0])})",
        )
        k_seq = k_cache.index_select(0, ids).reshape(-1, k_cache.shape[2], k_cache.shape[3])[:length]
        v_seq = v_cache.index_select(0, ids).reshape(-1, v_cache.shape[2], v_cache.shape[3])[:length]
        out_k[b, :length] = k_seq
        out_v[b, :length] = v_seq
    return out_k, out_v


def gather_contiguous(cache: PagedKVCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ``cache`` into padded contiguous K/V tensors (reference path)."""
    validate_cache(cache)
    return gather_paged(cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens, cache.page_size)
