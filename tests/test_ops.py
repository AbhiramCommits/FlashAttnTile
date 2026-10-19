"""Custom-op registration, fake kernels, torch.compile, and backend routing."""

from __future__ import annotations

import warnings

import torch

from flashattn_tile.backend import AttentionMetadata, FlashAttnTileBackend
from flashattn_tile.ops import PAGED_DECODE_OP, PREFILL_OP, ops_available
from flashattn_tile.paging import allocate, append_tokens
from flashattn_tile.reference import ref_attention_paged, ref_attention_varlen


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def _make_cache(batch=2, length=12, page_size=16, kv_heads=2, head_dim=8):
    cache = allocate(batch, length, kv_heads, head_dim, page_size, dtype=torch.float32)
    for _ in range(length):
        append_tokens(cache, torch.randn(batch, kv_heads, head_dim),
                      torch.randn(batch, kv_heads, head_dim))
    return cache


def test_custom_ops_registered():
    assert ops_available()
    assert hasattr(torch.ops.flashattn_tile, "paged_decode")
    assert hasattr(torch.ops.flashattn_tile, "prefill_varlen")
    schemas = str(torch.ops.flashattn_tile.paged_decode._schemas)
    assert PAGED_DECODE_OP in schemas
    assert PREFILL_OP in str(torch.ops.flashattn_tile.prefill_varlen._schemas)


def test_fake_paged_decode_shapes():
    with torch._subclasses.fake_tensor.FakeTensorMode():
        q = torch.empty(3, 4, 8, device="meta")
        k = torch.empty(12, 16, 2, 8, device="meta")
        pt = torch.empty(3, 3, dtype=torch.int32, device="meta")
        sl = torch.full((3,), 5, dtype=torch.int32, device="meta")
        out = torch.ops.flashattn_tile.paged_decode(q, k, k, pt, sl, 16, 0.35, 1)
    assert out.shape == (3, 4, 8)
    assert out.dtype == torch.float32
    assert out.device.type == "meta"


def test_fake_prefill_varlen_shapes():
    with torch._subclasses.fake_tensor.FakeTensorMode():
        q = torch.empty(11, 4, 8, device="meta")
        k = torch.empty(11, 2, 8, device="meta")
        cu = torch.full((3,), 0, dtype=torch.int32, device="meta")
        out = torch.ops.flashattn_tile.prefill_varlen(q, k, k, cu, cu, 7, 7, True, 0.35)
    assert out.shape == (11, 4, 8)
    assert out.device.type == "meta"


def test_compile_fullgraph_has_zero_graph_breaks():
    cache = _make_cache()
    q = torch.randn(2, 4, 8)

    def fn(x):
        return torch.ops.flashattn_tile.paged_decode(
            x, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens, 16, 0.35, 1
        )

    explanation = torch._dynamo.explain(fn)(q)
    assert explanation.graph_break_count == 0, explanation.break_reasons

    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    out = compiled(q)
    ref = _quiet(ref_attention_paged, q, cache.k_cache, cache.v_cache, cache.page_table,
                 cache.seq_lens, softmax_scale=0.35)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_compile_prefill_fullgraph():
    lens = [3, 6]
    total = sum(lens)
    q = torch.randn(total, 4, 8)
    k = torch.randn(total, 2, 8)
    v = torch.randn(total, 2, 8)
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32)

    def fn(x):
        return torch.ops.flashattn_tile.prefill_varlen(x, k, v, cu, cu, 6, 6, True, 0.35)

    explanation = torch._dynamo.explain(fn)(q)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    out = compiled(q)
    ref = _quiet(ref_attention_varlen, q, k, v, cu, cu, True, 0.35)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_backend_kv_cache_shape_matches_allocate():
    backend = FlashAttnTileBackend(num_heads=8, num_kv_heads=2, head_dim=16, page_size=16)
    cache = backend.allocate_kv_cache(batch=2, max_seq_len=160, dtype=torch.float32, device="cpu")
    shape = backend.get_kv_cache_shape(num_blocks=cache.num_pages, block_size=16,
                                       num_kv_heads=2, head_dim=16)
    assert shape == (2, cache.num_pages, 16, 2, 16)
    assert shape[1:] == cache.k_cache.shape
    assert backend.get_kv_cache_shape_per_tensor(
        cache.num_pages, 16, 2, 16
    ) == cache.k_cache.shape
    assert backend.get_name() == "flashattn_tile"


def test_backend_decode_routing_matches_reference():
    backend = FlashAttnTileBackend(num_heads=4, num_kv_heads=2, head_dim=8, page_size=16)
    cache = _make_cache()
    q = torch.randn(2, 4, 8)
    meta = AttentionMetadata(page_table=cache.page_table, seq_lens=cache.seq_lens,
                             is_prefill=False, softmax_scale=0.35)
    out = _quiet(backend.forward, q, cache, meta)
    ref = _quiet(ref_attention_paged, q, cache.k_cache, cache.v_cache, cache.page_table,
                 cache.seq_lens, softmax_scale=0.35)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

    # [batch, 1, heads, dim] must be accepted too.
    out_4d = _quiet(backend.forward, q.unsqueeze(1), cache, meta)
    torch.testing.assert_close(out_4d, ref, rtol=0, atol=0)


def test_backend_prefill_routing_matches_reference():
    backend = FlashAttnTileBackend(num_heads=4, num_kv_heads=2, head_dim=8, page_size=16)
    cache = allocate(2, 16, 2, 8, 16, dtype=torch.float32)
    lens = [3, 7]
    total = sum(lens)
    q = torch.randn(total, 4, 8)
    k = torch.randn(total, 2, 8)
    v = torch.randn(total, 2, 8)
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32)
    meta = AttentionMetadata(cu_seqlens_q=cu, cu_seqlens_k=cu, is_prefill=True,
                             max_seqlen_q=max(lens), max_seqlen_k=max(lens), k=k, v=v,
                             softmax_scale=0.35)
    out = _quiet(backend.forward, q, cache, meta)
    ref = _quiet(ref_attention_varlen, q, k, v, cu, cu, True, 0.35)
    assert out.shape == (total, 4, 8)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_backend_validation():
    backend = FlashAttnTileBackend(num_heads=4, num_kv_heads=2, head_dim=8, page_size=16)
    cache = _make_cache()
    meta = AttentionMetadata(is_prefill=True)
    try:
        backend.forward(torch.randn(2, 4, 8), cache, meta)
    except ValueError as exc:
        assert "cu_seqlens" in str(exc) or "attn_metadata.k" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing cu_seqlens")
    try:
        FlashAttnTileBackend(num_heads=5, num_kv_heads=2, head_dim=8, page_size=16)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-divisible heads")
