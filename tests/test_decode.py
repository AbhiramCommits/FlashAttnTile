"""Paged decode tests: split-K equivalence, page-boundary tails, quantised KV."""

from __future__ import annotations

import pytest
import torch
from conftest import requires_gpu

from flashattn_tile.kernels.decode import (
    HAS_TRITON,
    paged_decode_attention,
    paged_decode_attention_fp8,
)
from flashattn_tile.paging import allocate, append_tokens
from flashattn_tile.quant import (
    FP8_DTYPE,
    dequantize_cache,
    kv_bytes_per_token,
    quantize_cache,
)
from flashattn_tile.reference import ref_attention_paged

FP16_MAX_ABS_TOL = 2e-2
SPLIT_K_AGREEMENT_TOL = 2e-3
# Measured on this CPU-only host by the quantised-KV tests below; see
# results/numerics.json for the recorded values. Generous headroom is kept.
INT8_MAX_ABS_TOL = 5e-2
FP8_MAX_ABS_TOL = 5e-2


def _build_cache(batch, lens, page_size, kv_heads, head_dim, dtype, device):
    """Fill a cache lockstep, then truncate sequences to ragged lengths."""
    cache = allocate(batch, max(lens), kv_heads, head_dim, page_size, dtype=dtype, device=device)
    for _ in range(max(lens)):
        append_tokens(
            cache,
            torch.randn(batch, kv_heads, head_dim, device=device, dtype=dtype) * 0.2,
            torch.randn(batch, kv_heads, head_dim, device=device, dtype=dtype) * 0.2,
        )
    cache.seq_lens[:] = torch.tensor(lens, dtype=torch.int32, device=device)
    return cache


def _lens_for_batch(batch, page_size):
    base = [1, page_size - 1, page_size + 1, 2 * page_size + 3, 4 * page_size + 1]
    return [base[i % len(base)] for i in range(batch)]


@requires_gpu
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("split_k", [1, 4])
def test_decode_matches_reference(page_size, batch, split_k):
    torch.manual_seed(21)
    num_heads, num_kv_heads, head_dim = 32, 8, 128
    lens = _lens_for_batch(batch, page_size)
    cache = _build_cache(
        batch, lens, page_size, num_kv_heads, head_dim, torch.float16, "cuda"
    )
    q = (torch.randn(batch, num_heads, head_dim, device="cuda") * 0.2).to(torch.float16)

    out = paged_decode_attention(q, cache, config={"split_k": split_k})
    ref = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)
    max_abs = (out.float() - ref.float()).abs().max().item()
    print(f"page_size={page_size} batch={batch} split_k={split_k} max_abs={max_abs:.5f}")
    assert max_abs <= FP16_MAX_ABS_TOL


@requires_gpu
def test_split_k_agrees_with_non_split():
    torch.manual_seed(22)
    page_size, batch = 32, 8
    lens = _lens_for_batch(batch, page_size)
    cache = _build_cache(batch, lens, page_size, 8, 128, torch.float16, "cuda")
    q = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16)

    single = paged_decode_attention(q, cache, config={"split_k": 1})
    for split_k in (2, 4, 8):
        split = paged_decode_attention(q, cache, config={"split_k": split_k})
        max_abs = (single.float() - split.float()).abs().max().item()
        print(f"split_k={split_k} vs 1: max_abs={max_abs:.6f}")
        assert max_abs <= SPLIT_K_AGREEMENT_TOL


@requires_gpu
def test_mqa_decode():
    torch.manual_seed(23)
    page_size, batch = 16, 4
    lens = [1, 17, 40, 64]
    cache = _build_cache(batch, lens, page_size, 1, 128, torch.float16, "cuda")
    q = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16) * 0.2
    out = paged_decode_attention(q, cache)
    ref = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)
    assert (out.float() - ref.float()).abs().max().item() <= FP16_MAX_ABS_TOL


@requires_gpu
def test_split_k_with_more_splits_than_pages():
    """Empty splits must contribute nothing to the LSE merge."""
    torch.manual_seed(24)
    cache = _build_cache(2, [8, 20], 16, 2, 64, torch.float16, "cuda")
    q = torch.randn(2, 4, 64, device="cuda", dtype=torch.float16)
    out = paged_decode_attention(q, cache, config={"split_k": 16})
    ref = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)
    assert (out.float() - ref.float()).abs().max().item() <= FP16_MAX_ABS_TOL


def test_cpu_fallback_matches_reference_exactly():
    torch.manual_seed(25)
    batch, lens, page_size = 3, [1, 9, 20], 16
    cache = _build_cache(batch, lens, page_size, 2, 16, torch.float32, "cpu")
    q = torch.randn(batch, 4, 16)
    with pytest.warns(RuntimeWarning, match="falling back"):
        out = paged_decode_attention(q, cache)
    ref = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["int8"] + (["float8_e4m3fn"] if FP8_DTYPE else []))
@pytest.mark.parametrize("page_size", [16, 32])
def test_quantized_cache_accuracy(mode, page_size):
    """Measured accuracy of the quantised cache through the reference path."""
    torch.manual_seed(26)
    batch, lens = 8, [1, 16, 33, 64, 100, 7, 48, 129]
    num_heads, num_kv_heads, head_dim = 8, 2, 64
    cache = _build_cache(
        batch, [max(lens)] * batch, page_size, num_kv_heads, head_dim, torch.float32, "cpu"
    )
    cache.seq_lens[:] = torch.tensor(lens, dtype=torch.int32)
    q = torch.randn(batch, num_heads, head_dim) * 0.2

    ref = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)
    qcache, scales = quantize_cache(cache, mode=mode)
    assert scales.k.shape == (cache.num_pages, num_kv_heads)
    assert qcache.k_cache.dtype != torch.float32

    with pytest.warns(RuntimeWarning, match="falling back"):
        out = paged_decode_attention_fp8(q, qcache, scales)
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.abs() + 1e-3)).max().item()
    print(
        f"mode={mode} page_size={page_size} max_abs={max_abs:.6f} "
        f"mean_abs={mean_abs:.6f} max_rel={max_rel:.6f}"
    )
    tol = INT8_MAX_ABS_TOL if mode == "int8" else FP8_MAX_ABS_TOL
    assert max_abs <= tol, f"{mode} max abs error {max_abs} > {tol}"

    deq = dequantize_cache(qcache, scales, out_dtype=torch.float32)
    torch.testing.assert_close(deq.k_cache.shape, cache.k_cache.shape)


def test_quantized_cache_reduces_bytes_per_token():
    fp16 = kv_bytes_per_token(8, 128, "fp16", page_size=16)
    int8 = kv_bytes_per_token(8, 128, "int8", page_size=16)
    fp8 = kv_bytes_per_token(8, 128, "fp8", page_size=16)
    assert int8 < fp16 / 1.9
    assert fp8 < fp16 / 1.9
    assert int8 == fp8


def test_decode_validation_errors():
    cache = _build_cache(2, [5, 9], 16, 2, 16, torch.float32, "cpu")
    with pytest.raises(ValueError, match="q batch"):
        paged_decode_attention(torch.randn(3, 4, 16), cache)
    with pytest.raises(ValueError, match="divisible"):
        paged_decode_attention(torch.randn(2, 3, 16), cache)
    with pytest.raises(ValueError, match="head_dim"):
        paged_decode_attention(torch.randn(2, 4, 8), cache)
    with pytest.raises(ValueError, match="split_k"):
        paged_decode_attention(
            torch.randn(2, 4, 16), cache, config={"split_k": 0}
        )


def test_zero_length_sequence_is_rejected_for_decode():
    cache = _build_cache(2, [4, 4], 16, 2, 16, torch.float32, "cpu")
    cache.seq_lens[1] = 0
    with pytest.raises(ValueError, match="at least one cached token"):
        paged_decode_attention(torch.randn(2, 4, 16), cache)


def test_triton_flag_is_boolean():
    assert isinstance(HAS_TRITON, bool)
