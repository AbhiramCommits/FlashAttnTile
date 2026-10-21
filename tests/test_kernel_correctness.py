"""Cross-checks: every kernel vs the fp32 reference and vs SDPA.

These run only on a CUDA+Triton host (the same ``requires_gpu`` gate as the
other GPU tests). Each kernel is compared against the fp32-accumulating PyTorch
reference and, where a dense formulation exists, against
``torch.nn.functional.scaled_dot_product_attention`` across fp16/bf16, head dims
64/128, GQA/MQA ratios, causal on/off, odd sequence lengths, page sizes, and
mixed varlen batches.

Tolerances are dtype-appropriate (``torch.testing.assert_close``, i.e.
``|diff| <= atol + rtol * |ref|``): the kernels form QK^T/PV in the input dtype
while the reference accumulates in fp32, so fp16/bf16 roundoff is the expected
error floor. They are deliberately not loosened to hide a wrong kernel.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F
from conftest import requires_gpu

from flashattn_tile.kernels.decode import paged_decode_attention, paged_decode_attention_fp8
from flashattn_tile.kernels.prefill import prefill_attention
from flashattn_tile.paging import allocate, append_tokens, gather_contiguous
from flashattn_tile.quant import dequantize_cache, quantize_cache
from flashattn_tile.reference import ref_attention_paged, ref_attention_varlen

DEV = "cuda"

GQA_CASES = [(8, 8), (8, 2), (8, 1)]  # GQA=1, GQA=4, MQA
ODD_LENS = [1, 17, 255, 1000]


def _tol(dtype):
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1.6e-2, 1.6e-2


def _make_varlen(lens, num_heads, num_kv_heads, head_dim, dtype, seed=0):
    torch.manual_seed(seed)
    total = sum(lens)
    q = (torch.randn(total, num_heads, head_dim, device=DEV) * 0.2).to(dtype)
    k = (torch.randn(total, num_kv_heads, head_dim, device=DEV) * 0.2).to(dtype)
    v = (torch.randn(total, num_kv_heads, head_dim, device=DEV) * 0.2).to(dtype)
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()), dtype=torch.int32, device=DEV
    )
    return q, k, v, cu


def _make_cache(batch, lens, page_size, kv_heads, head_dim, dtype, seed=0):
    torch.manual_seed(seed)
    cache = allocate(batch, max(lens), kv_heads, head_dim, page_size, dtype=dtype, device=DEV)
    for _ in range(max(lens)):
        append_tokens(
            cache,
            torch.randn(batch, kv_heads, head_dim, device=DEV, dtype=dtype) * 0.2,
            torch.randn(batch, kv_heads, head_dim, device=DEV, dtype=dtype) * 0.2,
        )
    cache.seq_lens[:] = torch.tensor(lens, dtype=torch.int32, device=DEV)
    return cache


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA_CASES)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_varlen_matches_reference(head_dim, num_heads, num_kv_heads, causal, dtype):
    q, k, v, cu = _make_varlen(ODD_LENS, num_heads, num_kv_heads, head_dim, dtype, seed=7)
    out = prefill_attention(
        q, k, v, cu, cu, max_seqlen_q=max(ODD_LENS), max_seqlen_k=max(ODD_LENS), causal=causal
    )
    ref = ref_attention_varlen(q, k, v, cu, cu, causal=causal)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@requires_gpu
@pytest.mark.parametrize("seqlen", ODD_LENS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_single_odd_length(seqlen, dtype):
    q, k, v, cu = _make_varlen([seqlen], 8, 2, 128, dtype, seed=8)
    for causal in (True, False):
        out = prefill_attention(q, k, v, cu, cu, seqlen, seqlen, causal=causal)
        ref = ref_attention_varlen(q, k, v, cu, cu, causal=causal)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@requires_gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA_CASES)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_matches_sdpa(head_dim, num_heads, num_kv_heads, causal, dtype):
    torch.manual_seed(9)
    batch, seqlen = 2, 255
    q = (torch.randn(batch, num_heads, seqlen, head_dim, device=DEV) * 0.2).to(dtype)
    k = (torch.randn(batch, num_kv_heads, seqlen, head_dim, device=DEV) * 0.2).to(dtype)
    v = (torch.randn(batch, num_kv_heads, seqlen, head_dim, device=DEV) * 0.2).to(dtype)

    qt = q.transpose(1, 2).reshape(batch * seqlen, num_heads, head_dim)
    kt = k.transpose(1, 2).reshape(batch * seqlen, num_kv_heads, head_dim)
    vt = v.transpose(1, 2).reshape(batch * seqlen, num_kv_heads, head_dim)
    cu = torch.tensor([b * seqlen for b in range(batch + 1)], dtype=torch.int32, device=DEV)
    out = prefill_attention(qt, kt, vt, cu, cu, seqlen, seqlen, causal=causal)
    out = out.reshape(batch, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()

    group = num_heads // num_kv_heads
    sdpa = F.scaled_dot_product_attention(
        q, k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1), is_causal=causal
    )
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, sdpa, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA_CASES)
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_matches_reference(head_dim, num_heads, num_kv_heads, page_size, dtype):
    batch = len(ODD_LENS)
    cache = _make_cache(batch, ODD_LENS, page_size, num_kv_heads, head_dim, dtype, seed=10)
    q = (torch.randn(batch, num_heads, head_dim, device=DEV) * 0.2).to(dtype)
    out = paged_decode_attention(q, cache)
    ref = ref_attention_paged(q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens)
    # Decode accumulates elementwise in fp32, so it matches the reference far
    # more tightly than the prefill tensor-core path.
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@requires_gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads,num_kv_heads", GQA_CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_matches_sdpa(head_dim, num_heads, num_kv_heads, dtype):
    lens = [17, 255]
    batch = len(lens)
    page_size = 16
    cache = _make_cache(batch, lens, page_size, num_kv_heads, head_dim, dtype, seed=11)
    q = (torch.randn(batch, num_heads, head_dim, device=DEV) * 0.2).to(dtype)
    out = paged_decode_attention(q, cache)

    gk, gv = gather_contiguous(cache)
    group = num_heads // num_kv_heads
    kexp = gk.repeat_interleave(group, dim=2).transpose(1, 2)
    vexp = gv.repeat_interleave(group, dim=2).transpose(1, 2)
    qq = q.unsqueeze(2)
    maxlen = max(lens)
    mask = torch.arange(maxlen, device=DEV)[None, None, None, :] < torch.tensor(
        lens, device=DEV
    )[:, None, None, None]
    sdpa = F.scaled_dot_product_attention(qq, kexp, vexp, attn_mask=mask).squeeze(2)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, sdpa, rtol=rtol, atol=atol)


@requires_gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_split_k_matches_single(head_dim, dtype):
    lens = [1, 17, 255, 1000]
    cache = _make_cache(4, lens, 16, 2, head_dim, dtype, seed=12)
    q = (torch.randn(4, 8, head_dim, device=DEV) * 0.2).to(dtype)
    single = paged_decode_attention(q, cache, config={"split_k": 1})
    for split_k in (2, 4, 8):
        split = paged_decode_attention(q, cache, config={"split_k": split_k})
        torch.testing.assert_close(single, split, rtol=1e-3, atol=1e-3)


@requires_gpu
@pytest.mark.parametrize("mode", ["int8", "float8_e4m3fn"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantized_decode_kernel_matches_dequantized_reference(mode, dtype):
    lens = [1, 17, 255, 1000]
    cache = _make_cache(4, lens, 16, 2, 64, dtype, seed=13)
    q = (torch.randn(4, 8, 64, device=DEV) * 0.2).to(dtype)
    qcache, scales = quantize_cache(cache, mode=mode)
    out = paged_decode_attention_fp8(q, qcache, scales)
    deq = dequantize_cache(qcache, scales, out_dtype=dtype)
    ref = ref_attention_paged(q, deq.k_cache, deq.v_cache, deq.page_table, deq.seq_lens)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Triton path is actually taken (not the reference fallback)
# ---------------------------------------------------------------------------
@requires_gpu
def test_triton_path_selected_no_fallback():
    q, k, v, cu = _make_varlen([1, 17], 8, 2, 64, torch.float16, seed=14)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        prefill_attention(q, k, v, cu, cu, 17, 17, causal=True)

    cache = _make_cache(2, [1, 17], 16, 2, 64, torch.float16, seed=15)
    dq = (torch.randn(2, 8, 64, device=DEV) * 0.2).to(torch.float16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        paged_decode_attention(dq, cache)
