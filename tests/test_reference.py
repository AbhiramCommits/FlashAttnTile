"""Sanity tests for the fp32 reference attention (CPU, always runnable)."""

from __future__ import annotations

import math

import pytest
import torch

from flashattn_tile.paging import allocate, append_tokens, gather_contiguous
from flashattn_tile.reference import (
    ref_attention_dense,
    ref_attention_paged,
    ref_attention_varlen,
)


def _manual_attention(q, k, v, causal, scale):
    """Independent per-sequence loop used to validate the reference itself."""
    if q.dim() == 2:
        q = q.unsqueeze(0)
    q_len, heads, dim = q.shape
    kv_len, kv_heads, _ = k.shape
    group = heads // kv_heads
    kb = k.repeat_interleave(group, dim=1).float()
    vb = v.repeat_interleave(group, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), kb) * scale
    if causal:
        offset = kv_len - q_len
        allowed = torch.arange(kv_len)[None, :] <= (torch.arange(q_len)[:, None] + offset)
        scores = scores.masked_fill(~allowed[None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, vb)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (8, 2), (8, 1)])
def test_varlen_matches_manual(causal, num_heads, num_kv_heads):
    torch.manual_seed(0)
    lens = [1, 9, 17]
    head_dim = 16
    q = torch.randn(sum(lens), num_heads, head_dim)
    k = torch.randn(sum(lens), num_kv_heads, head_dim)
    v = torch.randn(sum(lens), num_kv_heads, head_dim)
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)

    out = ref_attention_varlen(q, k, v, cu, cu, causal=causal)
    assert out.shape == q.shape
    assert out.dtype == q.dtype

    offset = 0
    for length in lens:
        expected = _manual_attention(
            q[offset : offset + length], k[offset : offset + length], v[offset : offset + length],
            causal, head_dim ** -0.5,
        )
        torch.testing.assert_close(out[offset : offset + length], expected, atol=1e-6, rtol=1e-6)
        offset += length


def test_decode_single_query_sees_all_keys():
    """q_len=1 must attend to the full KV cache under causal masking."""
    torch.manual_seed(1)
    num_heads, num_kv_heads, head_dim, kv_len = 4, 2, 8, 23
    q = torch.randn(1, num_heads, head_dim)
    k = torch.randn(kv_len, num_kv_heads, head_dim)
    v = torch.randn(kv_len, num_kv_heads, head_dim)
    out = ref_attention_varlen(q, k, v, [0, 1], [0, kv_len], causal=True)
    expected = _manual_attention(q, k, v, True, head_dim ** -0.5)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_dtype_and_cast_error(dtype):
    torch.manual_seed(2)
    lens = [5, 11]
    head_dim = 32
    q = torch.randn(sum(lens), 4, head_dim)
    k = torch.randn(sum(lens), 2, head_dim)
    v = torch.randn(sum(lens), 2, head_dim)
    cu = [0, lens[0], sum(lens)]

    fp32_in = ref_attention_varlen(q, k, v, cu, cu)
    lowp = ref_attention_varlen(q.to(dtype), k.to(dtype), v.to(dtype), cu, cu)
    assert lowp.dtype == dtype
    err = (lowp.float() - fp32_in).abs().max().item()
    # Round-trip through a 16-bit storage format only; keep a generous bound.
    assert err < (0.05 if dtype == torch.bfloat16 else 0.01)


def test_paged_reference_matches_contiguous_gather():
    torch.manual_seed(3)
    batch, length, page_size = 3, 19, 8
    num_heads, num_kv_heads, head_dim = 8, 2, 16
    cache = allocate(batch, length, num_kv_heads, head_dim, page_size, dtype=torch.float32)
    for _ in range(length):
        append_tokens(
            cache,
            torch.randn(batch, num_kv_heads, head_dim),
            torch.randn(batch, num_kv_heads, head_dim),
        )
    q = torch.randn(batch, num_heads, head_dim)
    out = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)

    gathered_k, gathered_v = gather_contiguous(cache)
    for b in range(batch):
        expected = ref_attention_varlen(
            q[b : b + 1], gathered_k[b, :length], gathered_v[b, :length], [0, 1], [0, length]
        )
        torch.testing.assert_close(out[b : b + 1], expected, atol=1e-6, rtol=1e-6)


def test_dense_wrapper_matches_varlen():
    torch.manual_seed(4)
    batch, heads, kv_heads, q_len, kv_len, dim = 2, 4, 2, 6, 6, 16
    q = torch.randn(batch, heads, q_len, dim)
    k = torch.randn(batch, kv_heads, kv_len, dim)
    v = torch.randn(batch, kv_heads, kv_len, dim)
    out = ref_attention_dense(q, k, v, causal=True)

    for b in range(batch):
        expected = ref_attention_varlen(
            q[b].transpose(0, 1), k[b].transpose(0, 1), v[b].transpose(0, 1), [0, q_len], [0, kv_len]
        ).transpose(0, 1)
        torch.testing.assert_close(out[b], expected, atol=1e-6, rtol=1e-6)


def test_causal_requires_kv_at_least_as_long_as_q():
    q = torch.randn(4, 1, 8)
    k = torch.randn(2, 1, 8)
    with pytest.raises(ValueError, match="kv_len >= q_len"):
        ref_attention_varlen(q, k, k, [0, 4], [0, 2], causal=True)


def test_invalid_cu_seqlens_are_rejected():
    q = torch.randn(4, 1, 8)
    with pytest.raises(ValueError, match="must start at 0"):
        ref_attention_varlen(q, q, q, [1, 4], [0, 4])
    with pytest.raises(ValueError, match="non-decreasing"):
        ref_attention_varlen(q, q, q, [0, 4, 2], [0, 2, 4])
    with pytest.raises(ValueError, match="ends at"):
        ref_attention_varlen(q, q, q, [0, 3], [0, 4])


def test_mismatched_head_counts_are_rejected():
    q = torch.randn(3, 5, 8)
    k = torch.randn(3, 2, 8)
    with pytest.raises(ValueError, match="divisible"):
        ref_attention_varlen(q, k, k, [0, 3], [0, 3])


def test_softmax_scale_is_honoured():
    torch.manual_seed(5)
    q = torch.randn(6, 2, 8)
    k = torch.randn(6, 2, 8)
    v = torch.randn(6, 2, 8)
    scaled = ref_attention_varlen(q, k, v, [0, 6], [0, 6], softmax_scale=0.5)
    manual = _manual_attention(q, k, v, True, 0.5)
    torch.testing.assert_close(scaled, manual, atol=1e-6, rtol=1e-6)
    assert not math.isnan(scaled.sum().item())
