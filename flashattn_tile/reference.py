"""Pure-PyTorch fp32-accumulating attention, used as ground truth.

The reference never runs a fused kernel: it materialises the score matrix in
fp32, softmaxes in fp32, and only casts the result back to the input dtype. That
makes it slow and memory hungry but numerically honest, which is exactly what a
correctness oracle needs.

Layouts
-------
varlen
    ``q [total_q, H, D]``, ``k/v [total_k, H_kv, D]`` with ``cu_seqlens_*``
    of length ``batch + 1``. GQA/MQA is supported when ``H % H_kv == 0``; query
    head ``h`` reads KV head ``h // (H // H_kv)``.
paged
    ``q [batch, H, D]`` (one decode query per sequence) plus a page table and
    per-sequence lengths. Pages are gathered into a contiguous varlen batch
    before calling :func:`ref_attention_varlen`.

Causal masking uses bottom-right alignment: query ``i`` (0-based within its
sequence) can see key ``j`` when ``j <= i + (kv_len - q_len)``. For prefill
``q_len == kv_len`` this is the usual triangle; for decode (``q_len == 1``) the
single query sees every key.
"""

from __future__ import annotations

import torch

from .paging import gather_paged

__all__ = ["ref_attention_varlen", "ref_attention_paged", "ref_attention_dense"]


def _as_int_list(cu_seqlens, name: str) -> list[int]:
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.dim() != 1:
            raise ValueError(f"{name} must be 1-D, got shape {tuple(cu_seqlens.shape)}")
        values = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    else:
        values = [int(x) for x in cu_seqlens]
    if len(values) < 2:
        raise ValueError(f"{name} must have at least 2 entries (batch + 1)")
    if values[0] != 0:
        raise ValueError(f"{name} must start at 0, got {values[0]}")
    if any(b < a for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be non-decreasing, got {values}")
    return values


def ref_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference attention on packed varlen tensors.

    Parameters
    ----------
    q:
        ``[total_q, num_heads, head_dim]``.
    k, v:
        ``[total_k, num_kv_heads, head_dim]``. ``num_heads % num_kv_heads == 0``.
    cu_seqlens_q, cu_seqlens_k:
        Cumulative sequence offsets, length ``batch + 1``.
    causal:
        Apply bottom-right-aligned causal masking.
    softmax_scale:
        Defaults to ``head_dim ** -0.5``.

    Returns
    -------
    Tensor with the same shape/dtype/device as ``q``.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, v must be 3-D [tokens, heads, head_dim]")
    if q.shape[-1] != k.shape[-1] or k.shape != v.shape:
        raise ValueError(f"shape mismatch: q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"dtype mismatch: q {q.dtype}, k {k.dtype}, v {v.dtype}")
    if not q.is_floating_point():
        raise ValueError(f"expected floating point inputs, got {q.dtype}")
    if q.device != k.device or k.device != v.device:
        raise ValueError("q, k, v must live on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("the last dimension of q, k, v must be contiguous")

    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    group = num_heads // num_kv_heads

    cu_q = _as_int_list(cu_seqlens_q, "cu_seqlens_q")
    cu_k = _as_int_list(cu_seqlens_k, "cu_seqlens_k")
    if len(cu_q) != len(cu_k):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")
    if cu_q[-1] != q.shape[0]:
        raise ValueError(f"cu_seqlens_q ends at {cu_q[-1]} but q has {q.shape[0]} tokens")
    if cu_k[-1] != k.shape[0]:
        raise ValueError(f"cu_seqlens_k ends at {cu_k[-1]} but k has {k.shape[0]} tokens")

    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    out = torch.empty_like(q)

    for b in range(len(cu_q) - 1):
        q_start, q_end = cu_q[b], cu_q[b + 1]
        k_start, k_end = cu_k[b], cu_k[b + 1]
        q_len, kv_len = q_end - q_start, k_end - k_start
        if q_len == 0:
            continue
        if kv_len == 0:
            raise ValueError(f"sequence {b} has {q_len} queries but no keys")
        if causal and kv_len < q_len:
            raise ValueError(
                f"causal bottom-right alignment requires kv_len >= q_len (sequence {b}: "
                f"{kv_len} < {q_len})"
            )

        qb = q[q_start:q_end].to(torch.float32)
        kb = k[k_start:k_end].to(torch.float32)
        vb = v[k_start:k_end].to(torch.float32)
        if group != 1:
            kb = kb.repeat_interleave(group, dim=1)
            vb = vb.repeat_interleave(group, dim=1)

        # [H, q_len, kv_len] fp32 scores.
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        if causal:
            offset = kv_len - q_len
            qi = torch.arange(q_len, device=q.device).unsqueeze(1) + offset
            kj = torch.arange(kv_len, device=q.device).unsqueeze(0)
            allowed = kj <= qi
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ob = torch.einsum("hqk,khd->qhd", probs, vb)  # [q_len, H, head_dim]
        out[q_start:q_end] = ob.to(q.dtype)

    return out


def ref_attention_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference attention for one decode query per sequence in a paged cache.

    ``q`` is ``[batch, num_heads, head_dim]``. ``k_cache``/``v_cache`` are
    ``[num_pages, page_size, num_kv_heads, head_dim]`` float tensors (dequantize
    an FP8/int8 cache first). Returns ``[batch, num_heads, head_dim]``.
    """
    if q.dim() != 3:
        raise ValueError(f"q must be [batch, heads, head_dim], got {tuple(q.shape)}")
    if q.shape[0] != seq_lens.shape[0]:
        raise ValueError(f"q batch {q.shape[0]} != seq_lens batch {seq_lens.shape[0]}")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError(f"q head_dim {q.shape[-1]} != cache head_dim {k_cache.shape[-1]}")
    if q.shape[1] % k_cache.shape[2] != 0:
        raise ValueError(f"num_heads {q.shape[1]} not divisible by num_kv_heads {k_cache.shape[2]}")
    if not (k_cache.is_floating_point() or k_cache.dtype == torch.float32):
        raise ValueError(
            f"reference attention needs a dequantized float cache, got {k_cache.dtype}; "
            "call flashattn_tile.quant.dequantize_cache first"
        )
    if q.dtype != k_cache.dtype:
        raise ValueError(f"q dtype {q.dtype} != cache dtype {k_cache.dtype}")

    batch = int(q.shape[0])
    page_size = int(k_cache.shape[1])
    gathered_k, gathered_v = gather_paged(k_cache, v_cache, page_table, seq_lens, page_size)

    lengths = [int(seq_lens[b].item()) for b in range(batch)]
    if batch == 0 or sum(lengths) == 0:
        raise ValueError("ref_attention_paged requires at least one cached token")
    k_packed = torch.cat([gathered_k[b, : lengths[b]] for b in range(batch)], dim=0)
    v_packed = torch.cat([gathered_v[b, : lengths[b]] for b in range(batch)], dim=0)

    cu_k = [0]
    for b in range(batch):
        cu_k.append(cu_k[-1] + int(seq_lens[b].item()))
    cu_q = list(range(batch + 1))

    out = ref_attention_varlen(
        q, k_packed, v_packed, cu_q, cu_k, causal=causal, softmax_scale=softmax_scale
    )
    return out


def ref_attention_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference attention on dense ``[batch, heads, seq, head_dim]`` tensors.

    Convenience wrapper around :func:`ref_attention_varlen` that transposes to
    token-major layout. ``k``/``v`` may have fewer heads than ``q`` (GQA).
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4-D [batch, heads, seq, head_dim]")
    batch, q_heads, q_len, _ = q.shape
    kv_len = k.shape[2]
    q_t = q.transpose(1, 2).reshape(batch * q_len, q_heads, q.shape[-1])
    k_t = k.transpose(1, 2).reshape(batch * kv_len, k.shape[1], k.shape[-1])
    v_t = v.transpose(1, 2).reshape(batch * kv_len, v.shape[1], v.shape[-1])
    cu_q = [b * q_len for b in range(batch + 1)]
    cu_k = [b * kv_len for b in range(batch + 1)]
    out = ref_attention_varlen(q_t, k_t, v_t, cu_q, cu_k, causal=causal, softmax_scale=softmax_scale)
    return out.reshape(batch, q_len, q_heads, q.shape[-1]).transpose(1, 2).contiguous()
