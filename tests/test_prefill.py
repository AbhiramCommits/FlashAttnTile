"""Correctness tests for the varlen prefill kernel.

GPU tolerances below are provisional: this repository was built on a CPU-only
host, so they have not been validated against a real CUDA run. The measured
numbers that *were* collected on this machine live in ``results/numerics.json``
and cover the CPU reference path only. The CPU fallback test is exact.
"""

from __future__ import annotations

import pytest
import torch
from conftest import requires_gpu

from flashattn_tile.kernels.prefill import HAS_TRITON, prefill_attention
from flashattn_tile.reference import ref_attention_varlen

# Provisional GPU tolerances (fp16 / bf16 inputs, fp32 accumulation).
FP16_MAX_ABS_TOL = 2e-2
FP16_MAX_REL_TOL = 2e-2
BF16_MAX_ABS_TOL = 4e-2
BF16_MAX_REL_TOL = 4e-2
MEAN_REL_TOL = 5e-3

RAGGED_LENS = [1, 7, 33, 64, 129]


def _make_varlen(lens, num_heads, num_kv_heads, head_dim, dtype, device):
    torch.manual_seed(11)
    total = sum(lens)
    q = (torch.randn(total, num_heads, head_dim, device=device) * 0.2).to(dtype)
    k = (torch.randn(total, num_kv_heads, head_dim, device=device) * 0.2).to(dtype)
    v = (torch.randn(total, num_kv_heads, head_dim, device=device) * 0.2).to(dtype)
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()),
                      dtype=torch.int32, device=device)
    return q, k, v, cu


def _errors(out, ref):
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    max_rel = (diff / (ref.float().abs() + 1e-3)).max().item()
    mean_rel = (diff / (ref.float().abs() + 1e-3)).mean().item()
    return max_abs, mean_abs, max_rel, mean_rel


def _tolerances(dtype):
    if dtype == torch.float16:
        return FP16_MAX_ABS_TOL, FP16_MAX_REL_TOL
    return BF16_MAX_ABS_TOL, BF16_MAX_REL_TOL


@requires_gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 8), (8, 2), (8, 1)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_kernel_matches_reference(head_dim, num_heads, num_kv_heads, causal, dtype):
    q, k, v, cu = _make_varlen(
        RAGGED_LENS, num_heads, num_kv_heads, head_dim, dtype, "cuda"
    )
    out = prefill_attention(
        q, k, v, cu, cu, max_seqlen_q=max(RAGGED_LENS), max_seqlen_k=max(RAGGED_LENS),
        causal=causal,
    )
    ref = ref_attention_varlen(q, k, v, cu, cu, causal=causal)

    assert out.shape == q.shape and out.dtype == dtype
    max_abs, mean_abs, max_rel, mean_rel = _errors(out, ref)
    atol, rtol = _tolerances(dtype)
    print(
        f"dtype={dtype} D={head_dim} H={num_heads}/{num_kv_heads} causal={causal} "
        f"max_abs={max_abs:.5f} mean_abs={mean_abs:.6f} max_rel={max_rel:.5f} "
        f"mean_rel={mean_rel:.6f}"
    )
    assert max_abs <= atol, f"max abs error {max_abs} > {atol}"
    assert max_rel <= rtol, f"max rel error {max_rel} > {rtol}"
    assert mean_rel <= MEAN_REL_TOL, f"mean rel error {mean_rel} > {MEAN_REL_TOL}"


@requires_gpu
def test_prefill_explicit_config_and_odd_block_sizes():
    """BLOCK_M larger than the sequence must still mask correctly (len 1)."""
    lens = [1, 5, 70]
    q, k, v, cu = _make_varlen(lens, 4, 2, 64, torch.float16, "cuda")
    out = prefill_attention(
        q, k, v, cu, cu, max_seqlen_q=max(lens), max_seqlen_k=max(lens),
        config={"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
    )
    ref = ref_attention_varlen(q, k, v, cu, cu, causal=True)
    torch.testing.assert_close(out, ref, atol=FP16_MAX_ABS_TOL, rtol=FP16_MAX_REL_TOL)


def test_cpu_fallback_matches_reference_exactly():
    lens = [1, 9, 17]
    q, k, v, cu = _make_varlen(lens, 4, 2, 32, torch.float32, "cpu")
    with pytest.warns(RuntimeWarning, match="falling back"):
        out = prefill_attention(q, k, v, cu, cu, max_seqlen_q=max(lens), max_seqlen_k=max(lens))
    ref = ref_attention_varlen(q, k, v, cu, cu)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_config_validation_rejects_bad_blocks():
    lens = [3]
    q, k, v, cu = _make_varlen(lens, 2, 2, 16, torch.float32, "cpu")
    with pytest.raises(ValueError, match="power of two"):
        prefill_attention(q, k, v, cu, cu, 3, 3, config={"BLOCK_M": 48})
    with pytest.raises(ValueError, match="unknown prefill config keys"):
        prefill_attention(q, k, v, cu, cu, 3, 3, config={"BLOCK_K": 32})


def test_invalid_batch_offsets_rejected():
    q = torch.randn(4, 2, 16)
    bad = torch.tensor([0, 5], dtype=torch.int32)
    with pytest.raises(ValueError, match="must equal q.shape"):
        prefill_attention(q, q, q, bad, bad, 4, 4)


def test_triton_flag_is_boolean():
    assert isinstance(HAS_TRITON, bool)
