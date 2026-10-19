"""Fused FlashAttention-style varlen prefill kernel (Triton).

The kernel streams over K/V blocks, maintaining a running max ``m_i``, running
normaliser ``l_i`` and an fp32 output accumulator, so the ``[Lq, Lk]`` score
matrix is never materialised. ``tl.dot`` drives both QK^T and PV on tensor
cores; GQA is handled by mapping query head ``h`` to KV head
``h // (num_heads // num_kv_heads)``.

The module is importable without Triton. On a CPU-only host
:func:`prefill_attention` warns once and dispatches to the fp32 reference.
"""

from __future__ import annotations

import warnings

import torch

from ..reference import ref_attention_varlen

try:  # pragma: no cover - depends on host
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fwd_varlen_kernel(
        Q,
        K,
        V,
        Out,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_qt,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_ot,
        stride_oh,
        stride_od,
        G: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        q_start = tl.load(cu_seqlens_q + off_b).to(tl.int32)
        q_end = tl.load(cu_seqlens_q + off_b + 1).to(tl.int32)
        k_start = tl.load(cu_seqlens_k + off_b).to(tl.int32)
        k_end = tl.load(cu_seqlens_k + off_b + 1).to(tl.int32)
        q_len = q_end - q_start
        kv_len = k_end - k_start

        # Program covers [pid_m * BLOCK_M, (pid_m + 1) * BLOCK_M) queries.
        if pid_m * BLOCK_M >= q_len:
            return

        kv_head = off_h // G
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q_ptrs = (
            Q
            + (q_start + offs_m)[:, None] * stride_qt
            + off_h * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_m[:, None] < q_len, other=0.0)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        # Bottom-right causal alignment: query i sees keys j <= i + (kv_len - q_len).
        if IS_CAUSAL:
            offset = kv_len - q_len
            hi = tl.minimum(kv_len, (pid_m + 1) * BLOCK_M + offset)
        else:
            hi = kv_len

        for start_n in range(0, hi, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_mask = offs_n < kv_len

            k_ptrs = (
                K
                + (k_start + offs_n)[:, None] * stride_kt
                + kv_head * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * softmax_scale
            qk = tl.where(n_mask[None, :], qk, -float("inf"))
            if IS_CAUSAL:
                qk = tl.where(offs_n[None, :] <= (offs_m + offset)[:, None], qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = (
                V
                + (k_start + offs_n)[:, None] * stride_vt
                + kv_head * stride_vh
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            acc = tl.dot(p.to(q.dtype), v, acc=acc)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_ptrs = (
            Out
            + (q_start + offs_m)[:, None] * stride_ot
            + off_h * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptrs, acc.to(q.dtype), mask=offs_m[:, None] < q_len)


else:  # pragma: no cover - CPU-only hosts
    _fwd_varlen_kernel = None


DEFAULT_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "num_warps": 4,
    "num_stages": 3,
}

_WARNED: set[str] = set()


def _warn_fallback(reason: str) -> None:
    if reason in _WARNED:
        return
    _WARNED.add(reason)
    warnings.warn(
        f"prefill_attention: {reason}; falling back to the PyTorch reference path. "
        "Kernel throughput requires a CUDA GPU with Triton installed.",
        RuntimeWarning,
        stacklevel=3,
    )


def _resolve_config(config: dict | None, signature: str) -> dict:
    """Merge an explicit config with the persisted autotuned pick, if any."""
    resolved = dict(DEFAULT_CONFIG)
    if config is None:
        try:
            from ..autotune import best_config as _best_config

            persisted = _best_config("prefill", signature)
            if persisted:
                resolved.update({k: v for k, v in persisted.items() if k in resolved})
        except Exception:
            pass
    else:
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown prefill config keys: {sorted(unknown)}")
        resolved.update(config)
    for key in ("BLOCK_M", "BLOCK_N"):
        if resolved[key] <= 0 or resolved[key] & (resolved[key] - 1):
            raise ValueError(f"{key} must be a positive power of two, got {resolved[key]}")
    if resolved["num_warps"] not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"num_warps must be a power of two <= 32, got {resolved['num_warps']}")
    if resolved["num_stages"] < 1:
        raise ValueError(f"num_stages must be >= 1, got {resolved['num_stages']}")
    return resolved


def _validate_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, causal):
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, v must be 3-D [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"k shape {tuple(k.shape)} != v shape {tuple(v.shape)}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q head_dim {q.shape[-1]} != k head_dim {k.shape[-1]}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"dtype mismatch: q {q.dtype}, k {k.dtype}, v {v.dtype}")
    if not q.is_floating_point():
        raise ValueError(f"expected floating point inputs, got {q.dtype}")
    if q.device != k.device or k.device != v.device:
        raise ValueError("q, k, v must live on the same device")
    if cu_seqlens_q.dim() != 1 or cu_seqlens_k.dim() != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be 1-D")
    if cu_seqlens_q.shape[0] != cu_seqlens_k.shape[0]:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same batch dimension")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens must contain at least batch + 1 entries")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens_q and cu_seqlens_k must be int32")
    if cu_seqlens_q.device != q.device or cu_seqlens_k.device != q.device:
        raise ValueError("cu_seqlens must live on the same device as q")
    if int(cu_seqlens_q[0].item()) != 0 or int(cu_seqlens_k[0].item()) != 0:
        raise ValueError("cu_seqlens must start at 0")
    if int(cu_seqlens_q[-1].item()) != q.shape[0]:
        raise ValueError("cu_seqlens_q[-1] must equal q.shape[0]")
    if int(cu_seqlens_k[-1].item()) != k.shape[0]:
        raise ValueError("cu_seqlens_k[-1] must equal k.shape[0]")
    if not bool(torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1]).item()):
        raise ValueError("cu_seqlens_q must be non-decreasing")
    if not bool(torch.all(cu_seqlens_k[1:] >= cu_seqlens_k[:-1]).item()):
        raise ValueError("cu_seqlens_k must be non-decreasing")
    if causal and bool(torch.any((cu_seqlens_k[1:] - cu_seqlens_k[:-1])
                                 < (cu_seqlens_q[1:] - cu_seqlens_q[:-1])).item()):
        raise ValueError("causal bottom-right alignment requires kv_len >= q_len per sequence")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"num_heads ({q.shape[1]}) must be divisible by num_kv_heads ({k.shape[1]})"
        )


def prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = True,
    softmax_scale: float | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Fused varlen prefill attention.

    Parameters
    ----------
    q: ``[total_q, num_heads, head_dim]`` fp16/bf16 on CUDA for the kernel path.
    k, v: ``[total_k, num_kv_heads, head_dim]`` packed varlen.
    cu_seqlens_q/k: int32 cumulative offsets on the same device, length ``B+1``.
    max_seqlen_q/k: maximum per-sequence lengths (kernel grid bound).
    causal: bottom-right-aligned causal masking.
    softmax_scale: defaults to ``head_dim ** -0.5``.
    config: optional ``BLOCK_M``, ``BLOCK_N``, ``num_warps``, ``num_stages``.

    Returns
    -------
    ``[total_q, num_heads, head_dim]`` in ``q.dtype``. Falls back to the fp32
    reference with a warning when CUDA or Triton is unavailable.
    """
    _validate_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    batch = cu_seqlens_q.numel() - 1
    head_dim = q.shape[-1]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if max_seqlen_q <= 0:
        raise ValueError(f"max_seqlen_q must be positive, got {max_seqlen_q}")
    if max_seqlen_k <= 0:
        raise ValueError(f"max_seqlen_k must be positive, got {max_seqlen_k}")

    signature = f"B{batch}_H{num_heads}_Hkv{num_kv_heads}_D{head_dim}_Lq{max_seqlen_q}_Lk{max_seqlen_k}_c{int(causal)}"
    cfg = _resolve_config(config, signature)

    if not (HAS_TRITON and q.is_cuda):
        _warn_fallback(
            "CUDA + Triton not available" if not HAS_TRITON else "input tensor is not on CUDA"
        )
        return ref_attention_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal, softmax_scale=scale
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        _warn_fallback(f"dtype {q.dtype} is not fp16/bf16")
        return ref_attention_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal, softmax_scale=scale
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    grid = (triton.cdiv(max_seqlen_q, cfg["BLOCK_M"]), num_heads, batch)
    _fwd_varlen_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads // num_kv_heads,
        causal,
        cfg["BLOCK_M"],
        cfg["BLOCK_N"],
        head_dim,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out
