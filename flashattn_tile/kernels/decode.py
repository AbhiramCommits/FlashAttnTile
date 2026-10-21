"""Paged decode attention kernels (single-query decode + split-K + quantised KV).

Decode reads a single query per sequence and therefore has arithmetic intensity
close to ``2 * num_kv_heads * head_dim / (2 bytes per K/V element)`` -- it is
HBM-bandwidth bound. The kernel walks the page table one page at a time, applies
online softmax in fp32, and never materialises the score vector.

Long contexts are optionally split across ``split_k`` programs. Each program
writes partial ``(acc, m, l)`` triples; :func:`_decode_reduce_kernel` merges them
with

    m* = max_s m_s
    l* = sum_s l_s * exp(m_s - m*)
    o  = sum_s acc_s * exp(m_s - m*) / l*

which is the standard log-sum-exp merge used by split-K FlashAttention.

An INT8/FP8 cache variant dequantises each page inside the kernel using
per-(page, KV-head) scales (see :mod:`flashattn_tile.quant`) and accumulates in
fp32. Quantised decode does not use split-K here.
"""

from __future__ import annotations

import warnings

import torch

from ..paging import PagedKVCache, validate_cache
from ..quant import dequantize_cache
from ..reference import ref_attention_paged

try:  # pragma: no cover - host dependent
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _decode_paged_kernel(
        Q,
        KCache,
        VCache,
        PageTable,
        SeqLens,
        Out,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kp,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vp,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_pb,
        stride_pp,
        stride_ob,
        stride_oh,
        stride_od,
        PAGE_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        kv_head = h // GROUP
        seq_len = tl.load(SeqLens + b)

        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d * stride_qd)
        q = q.to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], tl.float32)

        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        for page_i in range(0, num_pages):
            phys = tl.load(PageTable + b * stride_pb + page_i * stride_pp).to(tl.int64)
            offs_n = tl.arange(0, PAGE_SIZE)
            logical = page_i * PAGE_SIZE + offs_n
            mask = logical < seq_len

            k_ptrs = (
                KCache
                + phys * stride_kp
                + offs_n[:, None] * stride_kt
                + kv_head * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1) * softmax_scale
            scores = tl.where(mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            v_ptrs = (
                VCache
                + phys * stride_vp
                + offs_n[:, None] * stride_vt
                + kv_head * stride_vh
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(
            Out + b * stride_ob + h * stride_oh + offs_d * stride_od,
            out.to(Out.dtype.element_ty),
        )

    @triton.jit
    def _decode_paged_split_kernel(
        Q,
        KCache,
        VCache,
        PageTable,
        SeqLens,
        Acc,
        M,
        L,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kp,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vp,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_pb,
        stride_pp,
        stride_ab,
        stride_ah,
        stride_as,
        stride_ad,
        stride_mb,
        stride_mh,
        stride_ms,
        PAGE_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        split = tl.program_id(2)
        kv_head = h // GROUP
        seq_len = tl.load(SeqLens + b)

        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d * stride_qd).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], tl.float32)

        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pages_per_split = (num_pages + SPLIT_K - 1) // SPLIT_K
        start = split * pages_per_split
        end = tl.minimum(num_pages, start + pages_per_split)

        for page_i in range(start, end):
            phys = tl.load(PageTable + b * stride_pb + page_i * stride_pp).to(tl.int64)
            offs_n = tl.arange(0, PAGE_SIZE)
            logical = page_i * PAGE_SIZE + offs_n
            mask = logical < seq_len

            k_ptrs = (
                KCache
                + phys * stride_kp
                + offs_n[:, None] * stride_kt
                + kv_head * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            scores = tl.sum(q[None, :] * k, axis=1) * softmax_scale
            scores = tl.where(mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            v_ptrs = (
                VCache
                + phys * stride_vp
                + offs_n[:, None] * stride_vt
                + kv_head * stride_vh
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        tl.store(Acc + b * stride_ab + h * stride_ah + split * stride_as
                 + offs_d * stride_ad, acc)
        tl.store(M + b * stride_mb + h * stride_mh + split * stride_ms, m_i)
        tl.store(L + b * stride_mb + h * stride_mh + split * stride_ms, l_i)

    @triton.jit
    def _decode_reduce_kernel(
        Acc,
        M,
        L,
        Out,
        stride_ab,
        stride_ah,
        stride_as,
        stride_ad,
        stride_mb,
        stride_mh,
        stride_ms,
        stride_ob,
        stride_oh,
        stride_od,
        NUM_SPLITS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        offs_s = tl.arange(0, NUM_SPLITS)
        offs_d = tl.arange(0, HEAD_DIM)

        m = tl.load(M + b * stride_mb + h * stride_mh + offs_s * stride_ms)
        l_part = tl.load(L + b * stride_mb + h * stride_mh + offs_s * stride_ms)
        global_m = tl.max(m)
        weights = tl.exp(m - global_m)
        global_l = tl.sum(l_part * weights)

        acc = tl.load(
            Acc
            + b * stride_ab
            + h * stride_ah
            + offs_s[:, None] * stride_as
            + offs_d[None, :] * stride_ad
        )
        out = tl.sum(acc * weights[:, None], axis=0) / global_l
        tl.store(
            Out + b * stride_ob + h * stride_oh + offs_d * stride_od,
            out.to(Out.dtype.element_ty),
        )

    @triton.jit
    def _decode_paged_quant_kernel(
        Q,
        KCache,
        VCache,
        ScaleK,
        ScaleV,
        PageTable,
        SeqLens,
        Out,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kp,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_vp,
        stride_vt,
        stride_vh,
        stride_vd,
        stride_skp,
        stride_skh,
        stride_svp,
        stride_svh,
        stride_pb,
        stride_pp,
        stride_ob,
        stride_oh,
        stride_od,
        PAGE_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        kv_head = h // GROUP
        seq_len = tl.load(SeqLens + b)

        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d * stride_qd).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], tl.float32)

        num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        for page_i in range(0, num_pages):
            phys = tl.load(PageTable + b * stride_pb + page_i * stride_pp).to(tl.int64)
            k_scale = tl.load(ScaleK + phys * stride_skp + kv_head * stride_skh).to(tl.float32)
            v_scale = tl.load(ScaleV + phys * stride_svp + kv_head * stride_svh).to(tl.float32)

            offs_n = tl.arange(0, PAGE_SIZE)
            logical = page_i * PAGE_SIZE + offs_n
            mask = logical < seq_len

            k_ptrs = (
                KCache
                + phys * stride_kp
                + offs_n[:, None] * stride_kt
                + kv_head * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32) * k_scale
            scores = tl.sum(q[None, :] * k, axis=1) * softmax_scale
            scores = tl.where(mask, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            l_i = l_i * alpha + tl.sum(p)
            acc = acc * alpha

            v_ptrs = (
                VCache
                + phys * stride_vp
                + offs_n[:, None] * stride_vt
                + kv_head * stride_vh
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32) * v_scale
            acc += tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(
            Out + b * stride_ob + h * stride_oh + offs_d * stride_od,
            out.to(Out.dtype.element_ty),
        )


else:  # pragma: no cover - CPU-only hosts
    _decode_paged_kernel = None
    _decode_paged_split_kernel = None
    _decode_reduce_kernel = None
    _decode_paged_quant_kernel = None


DEFAULT_CONFIG = {"split_k": 1, "num_warps": 4, "num_stages": 2}
_WARNED: set[str] = set()


def _warn_fallback(reason: str) -> None:
    if reason in _WARNED:
        return
    _WARNED.add(reason)
    warnings.warn(
        f"paged_decode_attention: {reason}; falling back to the PyTorch reference path. "
        "Kernel throughput requires a CUDA GPU with Triton installed.",
        RuntimeWarning,
        stacklevel=3,
    )


def _resolve_config(config: dict | None, signature: str) -> dict:
    resolved = dict(DEFAULT_CONFIG)
    if config is None:
        try:
            from ..autotune import best_config as _best_config

            persisted = _best_config("decode", signature)
            if persisted:
                for key in resolved:
                    if key in persisted:
                        resolved[key] = persisted[key]
        except Exception:
            pass
    else:
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown decode config keys: {sorted(unknown)}")
        resolved.update(config)
    if not isinstance(resolved["split_k"], int) or resolved["split_k"] < 1:
        raise ValueError(f"split_k must be a positive integer, got {resolved['split_k']}")
    if resolved["split_k"] > 64:
        raise ValueError(f"split_k must be <= 64, got {resolved['split_k']}")
    if resolved["num_warps"] not in (1, 2, 4, 8, 16, 32):
        raise ValueError(f"num_warps must be a power of two <= 32, got {resolved['num_warps']}")
    if resolved["num_stages"] < 1:
        raise ValueError(f"num_stages must be >= 1, got {resolved['num_stages']}")
    return resolved


def _validate_decode_inputs(q: torch.Tensor, cache: PagedKVCache) -> None:
    if not isinstance(cache, PagedKVCache):
        raise TypeError("cache must be a PagedKVCache")
    validate_cache(cache)
    if q.dim() != 3:
        raise ValueError(f"q must be [batch, heads, head_dim], got {tuple(q.shape)}")
    if q.shape[0] != cache.batch_size:
        raise ValueError(f"q batch {q.shape[0]} != cache batch {cache.batch_size}")
    if q.shape[2] != cache.head_dim:
        raise ValueError(f"q head_dim {q.shape[2]} != cache head_dim {cache.head_dim}")
    if q.shape[1] % cache.num_kv_heads != 0:
        raise ValueError(
            f"num_heads {q.shape[1]} must be divisible by num_kv_heads {cache.num_kv_heads}"
        )
    if q.device != cache.k_cache.device:
        raise ValueError("q and cache must live on the same device")
    if not q.is_floating_point():
        raise ValueError(f"q must be floating point, got {q.dtype}")
    if int(cache.seq_lens.min().item()) <= 0:
        raise ValueError("decode requires every sequence to hold at least one cached token")


def _signature(q, cache, split_k):
    max_len = int(cache.seq_lens.max().item())
    return (
        f"B{cache.batch_size}_H{q.shape[1]}_Hkv{cache.num_kv_heads}_D{cache.head_dim}"
        f"_P{cache.page_size}_L{max_len}_sk{split_k}"
    )


def paged_decode_attention(
    q: torch.Tensor,
    cache: PagedKVCache,
    softmax_scale: float | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Paged decode attention for one query per sequence.

    Parameters
    ----------
    q: ``[batch, num_heads, head_dim]`` fp16/bf16 (fp32 works on the CPU path).
    cache: :class:`PagedKVCache` with ``[pages, page_size, kv_heads, dim]`` K/V.
    softmax_scale: defaults to ``head_dim ** -0.5``.
    config: ``split_k`` (1 disables split-K), ``num_warps``, ``num_stages``.

    Returns
    -------
    ``[batch, num_heads, head_dim]`` in ``q.dtype``.
    """
    _validate_decode_inputs(q, cache)
    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    cfg = _resolve_config(config, _signature(q, cache, (config or {}).get("split_k", 1)))

    if not (HAS_TRITON and q.is_cuda):
        _warn_fallback(
            "CUDA + Triton not available" if not HAS_TRITON else "input tensor is not on CUDA"
        )
        return ref_attention_paged(
            q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens,
            causal=True, softmax_scale=scale,
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        _warn_fallback(f"dtype {q.dtype} is not fp16/bf16")
        return ref_attention_paged(
            q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens,
            causal=True, softmax_scale=scale,
        )
    if cache.k_cache.dtype != q.dtype:
        raise ValueError(f"q dtype {q.dtype} must match cache dtype {cache.k_cache.dtype}")

    out = torch.empty_like(q)
    split_k = cfg["split_k"]
    grid = (cache.batch_size, q.shape[1], split_k)
    q = q.contiguous()
    k_cache = cache.k_cache
    v_cache = cache.v_cache
    page_table = cache.page_table
    seq_lens = cache.seq_lens
    group = q.shape[1] // cache.num_kv_heads

    if split_k == 1:
        _decode_paged_kernel[grid](
            q, k_cache, v_cache, page_table, seq_lens, out, scale,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            page_table.stride(0), page_table.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            cache.page_size, cache.head_dim, group,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
        )
        return out

    acc = torch.empty(
        (cache.batch_size, q.shape[1], split_k, cache.head_dim),
        dtype=torch.float32, device=q.device,
    )
    m_part = torch.empty(
        (cache.batch_size, q.shape[1], split_k), dtype=torch.float32, device=q.device
    )
    l_part = torch.empty_like(m_part)
    _decode_paged_split_kernel[grid](
        q, k_cache, v_cache, page_table, seq_lens, acc, m_part, l_part, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        page_table.stride(0), page_table.stride(1),
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        cache.page_size, cache.head_dim, group, split_k,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    _decode_reduce_kernel[(cache.batch_size, q.shape[1])](
        acc, m_part, l_part, out,
        acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        m_part.stride(0), m_part.stride(1), m_part.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        split_k, cache.head_dim,
        num_warps=1, num_stages=1,
    )
    return out


def paged_decode_attention_fp8(
    q: torch.Tensor,
    cache: PagedKVCache,
    scales,
    softmax_scale: float | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Decode against an INT8 or FP8 KV cache with per-page-per-head scales.

    ``cache.k_cache``/``v_cache`` must be ``torch.int8`` or
    ``torch.float8_e4m3fn``; ``scales`` is a
    :class:`flashattn_tile.quant.QuantScales` (or dict with ``"k"``/``"v"``).
    On a CPU-only host the cache is dequantised and the fp32 reference runs.
    """
    _validate_decode_inputs(q, cache)
    k_scale = scales["k"] if isinstance(scales, dict) else scales.k
    v_scale = scales["v"] if isinstance(scales, dict) else scales.v
    quant_dtype = cache.k_cache.dtype
    if quant_dtype not in (torch.int8, getattr(torch, "float8_e4m3fn", torch.int8)):
        raise ValueError(f"expected int8/float8 cache, got {quant_dtype}")
    expected = (cache.num_pages, cache.num_kv_heads)
    if tuple(k_scale.shape) != expected or tuple(v_scale.shape) != expected:
        raise ValueError(f"scales must have shape {expected}, got {tuple(k_scale.shape)}")

    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    cfg = _resolve_config(config, _signature(q, cache, 1))
    if cfg["split_k"] != 1:
        raise ValueError("the quantised decode path does not support split_k")

    if not (HAS_TRITON and q.is_cuda):
        _warn_fallback(
            "CUDA + Triton not available" if not HAS_TRITON else "input tensor is not on CUDA"
        )
        deq = dequantize_cache(cache, scales, out_dtype=q.dtype)
        return ref_attention_paged(
            q, deq.k_cache, deq.v_cache, deq.page_table, deq.seq_lens,
            causal=True, softmax_scale=scale,
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"quantised decode requires fp16/bf16 queries, got {q.dtype}")

    q = q.contiguous()
    out = torch.empty_like(q)
    group = q.shape[1] // cache.num_kv_heads
    k_cache = cache.k_cache
    v_cache = cache.v_cache
    page_table = cache.page_table
    _decode_paged_quant_kernel[(cache.batch_size, q.shape[1])](
        q, k_cache, v_cache, k_scale, v_scale, page_table, cache.seq_lens, out, scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        k_scale.stride(0), k_scale.stride(1),
        v_scale.stride(0), v_scale.stride(1),
        page_table.stride(0), page_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        cache.page_size, cache.head_dim, group,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out
