"""Page-table contract tests: append/gather round-trips and loud validation."""

from __future__ import annotations

import math

import pytest
import torch

from flashattn_tile.paging import (
    PagedKVCache,
    allocate,
    append_tokens,
    gather_contiguous,
    gather_paged,
    validate_cache,
)
from flashattn_tile.reference import ref_attention_paged

DTYPE = torch.float32


def _append_steps(cache: PagedKVCache, ks: torch.Tensor, vs: torch.Tensor) -> None:
    """Append ``ks.shape[0]`` lockstep decode steps of shape ``[B, kv_heads, dim]``."""
    for t in range(ks.shape[0]):
        append_tokens(cache, ks[t], vs[t])


@pytest.mark.parametrize(
    "batch, lens, page_size, kv_heads, head_dim",
    [
        (3, [1, 5, 17], 4, 2, 8),          # ragged, non-multiple lengths
        (1, [16], 16, 1, 4),               # exactly one full page
        (4, [31, 32, 33, 1], 16, 2, 16),   # spans page boundaries, incl. len 1
        (2, [7, 8], 8, 4, 8),              # exact page boundary
    ],
)
def test_append_then_gather_roundtrip(batch, lens, page_size, kv_heads, head_dim):
    torch.manual_seed(0)
    cache = allocate(
        batch=batch,
        max_seq_len=max(lens),
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        dtype=DTYPE,
        device="cpu",
    )
    assert cache.num_pages == batch * math.ceil(max(lens) / page_size)

    # Decode appends one token per sequence per step; sequences are truncated to
    # ragged lengths afterwards to exercise gather's tail handling.
    steps = max(lens)
    ks = torch.randn(steps, batch, kv_heads, head_dim)
    vs = torch.randn(steps, batch, kv_heads, head_dim)
    _append_steps(cache, ks, vs)
    cache.seq_lens[:] = torch.tensor(lens, dtype=torch.int32)
    assert cache.seq_lens.tolist() == lens

    expected_k = torch.zeros(batch, steps, kv_heads, head_dim)
    expected_v = torch.zeros_like(expected_k)
    for b, length in enumerate(lens):
        expected_k[b, :length] = ks[:length, b]
        expected_v[b, :length] = vs[:length, b]

    gathered_k, gathered_v = gather_contiguous(cache)
    assert gathered_k.shape == (batch, steps, kv_heads, head_dim)
    torch.testing.assert_close(gathered_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(gathered_v, expected_v, rtol=0, atol=0)


def test_page_table_indirection_with_permuted_pages():
    """A permuted (but valid) page table must still address the same tokens."""
    torch.manual_seed(1)
    batch, length, page_size, kv_heads, head_dim = 2, 13, 4, 2, 8
    cache = allocate(batch, length, kv_heads, head_dim, page_size, dtype=DTYPE)
    num_pages = cache.num_pages
    perm = torch.randperm(num_pages, dtype=torch.int64)
    cache.page_table = perm[cache.page_table.long()].to(torch.int32)
    assert sorted(cache.page_table.flatten().tolist()) == list(range(num_pages))
    validate_cache(cache)

    ks = torch.randn(length, batch, kv_heads, head_dim)
    vs = torch.randn(length, batch, kv_heads, head_dim)
    _append_steps(cache, ks, vs)
    gathered_k, gathered_v = gather_contiguous(cache)
    for b in range(batch):
        torch.testing.assert_close(gathered_k[b], ks[:, b], rtol=0, atol=0)
        torch.testing.assert_close(gathered_v[b], vs[:, b], rtol=0, atol=0)


def test_append_respects_page_capacity():
    cache = allocate(1, max_seq_len=4, num_kv_heads=1, head_dim=4, page_size=2, dtype=DTYPE)
    assert cache.capacity == 4
    for _ in range(4):
        append_tokens(cache, torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    assert cache.seq_lens.tolist() == [4]
    with pytest.raises(AssertionError, match="capacity"):
        append_tokens(cache, torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))


def test_out_of_bounds_page_table_is_rejected():
    cache = allocate(2, 8, 1, 4, page_size=4, dtype=DTYPE)
    cache.seq_lens[:] = 8
    cache.page_table[0, 0] = cache.num_pages  # one past the end
    with pytest.raises(AssertionError, match="references page"):
        validate_cache(cache)
    with pytest.raises(AssertionError, match="page id outside"):
        gather_paged(cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens, cache.page_size)

    cache.page_table[0, 0] = -1
    with pytest.raises(AssertionError, match="negative page id"):
        validate_cache(cache)


def test_gqa_head_mapping_in_reference():
    """Query heads must map to KV head ``h // (H // H_kv)``."""
    torch.manual_seed(2)
    batch, length, page_size = 1, 9, 4
    num_heads, num_kv_heads, head_dim = 4, 2, 8
    cache = allocate(batch, length, num_kv_heads, head_dim, page_size, dtype=DTYPE)
    k = torch.zeros(length, num_kv_heads, head_dim)
    v = torch.zeros(length, num_kv_heads, head_dim)
    v[:, 0, :] = 1.0  # KV head 0 -> value 1
    v[:, 1, :] = 2.0  # KV head 1 -> value 2
    _append_steps(
        cache,
        k.unsqueeze(1).expand(-1, batch, -1, -1).contiguous(),
        v.unsqueeze(1).expand(-1, batch, -1, -1).contiguous(),
    )

    q = torch.zeros(batch, num_heads, head_dim)
    out = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0])
    for h in range(num_heads):
        torch.testing.assert_close(out[0, h], torch.full((head_dim,), expected[h]), atol=1e-6, rtol=0)


def test_append_validates_dtype_and_shape():
    cache = allocate(2, 4, 2, 4, page_size=2, dtype=torch.float16)
    bad_dtype = torch.zeros(2, 2, 4, dtype=torch.float32)
    with pytest.raises(AssertionError, match="dtype"):
        append_tokens(cache, bad_dtype, bad_dtype)
    bad_shape = torch.zeros(2, 3, 4, dtype=torch.float16)
    with pytest.raises(AssertionError, match="new_k must be"):
        append_tokens(cache, bad_shape, bad_shape)


@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.int16])
def test_page_table_dtype_is_strict(bad_dtype):
    cache = allocate(1, 4, 1, 4, page_size=2, dtype=DTYPE)
    cache.page_table = cache.page_table.to(bad_dtype)
    with pytest.raises(AssertionError, match="page_table must be int32"):
        validate_cache(cache)


def test_gather_paged_handles_zero_length_sequences():
    cache = allocate(3, 8, 2, 4, page_size=4, dtype=DTYPE)
    for t in range(5):
        append_tokens(
            cache,
            torch.full((3, 2, 4), float(t)),
            torch.full((3, 2, 4), float(t) + 100.0),
        )
    cache.seq_lens[:] = torch.tensor([3, 0, 2], dtype=torch.int32)

    gathered_k, gathered_v = gather_contiguous(cache)
    assert gathered_k.shape == (3, 3, 2, 4)
    assert torch.all(gathered_k[0, :3] == torch.tensor([0.0, 1.0, 2.0]).view(3, 1, 1))
    assert torch.all(gathered_k[1] == 0)
    assert torch.all(gathered_k[2, :2] == torch.tensor([0.0, 1.0]).view(2, 1, 1))
    assert torch.all(gathered_v[2, :2] == torch.tensor([100.0, 101.0]).view(2, 1, 1))
