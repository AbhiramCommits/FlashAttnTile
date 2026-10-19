"""Numerics table: storage dtype, quantised KV, split-K, and optional paths.

Every record is written to ``results/numerics.json``. On a CPU-only host the
reference path is exercised for real (fp16/bf16 storage and INT8/FP8 cache
quantisation); Triton/CUDA rows are recorded as skipped rather than fabricated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from flashattn_tile.env import device_info, triton_available
from flashattn_tile.kernels.decode import paged_decode_attention
from flashattn_tile.quant import (
    FP8_DTYPE,
    dequantize_cache,
    kv_bytes_per_token,
    quantize_cache,
)
from flashattn_tile.reference import ref_attention_dense, ref_attention_paged

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "numerics.json"

# Provisional bounds for the CPU reference path. The fp16/bf16 rows measure
# storage rounding; the quantised rows measure INT8/FP8 cache error. Values are
# tightened/loosened only from actual runs recorded in results/numerics.json.
FP16_MAX_ABS_TOL = 2e-2
BF16_MAX_ABS_TOL = 5e-2
INT8_MAX_ABS_TOL = 5e-2
FP8_MAX_ABS_TOL = 5e-2
SPLIT_K_MAX_ABS_TOL = 1e-6

BATCH = 4
LENS = [1, 16, 33, 64]
PAGE_SIZE = 16
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64


def _errors(out, ref):
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs() + 1e-3
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
        "mean_rel": float((diff / denom).mean().item()),
    }


def _record(path, dtype, shape, errors, tolerance, note, status="measured"):
    passed = None if status == "skipped" else bool(errors["max_abs"] <= tolerance)
    return {
        "path": path,
        "dtype": dtype,
        "shape": shape,
        "status": status,
        "max_abs": None if errors is None else errors["max_abs"],
        "mean_abs": None if errors is None else errors["mean_abs"],
        "max_rel": None if errors is None else errors["max_rel"],
        "mean_rel": None if errors is None else errors["mean_rel"],
        "tolerance": tolerance,
        "passed": passed,
        "note": note,
    }


def _build_cache():
    torch.manual_seed(101)
    from flashattn_tile.paging import allocate, append_tokens

    cache = allocate(BATCH, max(LENS), NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, dtype=torch.float32)
    for _ in range(max(LENS)):
        append_tokens(cache, torch.randn(BATCH, NUM_KV_HEADS, HEAD_DIM) * 0.2,
                      torch.randn(BATCH, NUM_KV_HEADS, HEAD_DIM) * 0.2)
    cache.seq_lens[:] = torch.tensor(LENS, dtype=torch.int32)
    q = torch.randn(BATCH, NUM_HEADS, HEAD_DIM) * 0.2
    return cache, q


def _dense_inputs():
    torch.manual_seed(202)
    batch, seq = 2, 64
    q = torch.randn(batch, NUM_HEADS, seq, HEAD_DIM) * 0.2
    k = torch.randn(batch, NUM_KV_HEADS, seq, HEAD_DIM) * 0.2
    v = torch.randn(batch, NUM_KV_HEADS, seq, HEAD_DIM) * 0.2
    return q, k, v


def _run_checks():
    cache, q = _build_cache()
    shape = f"B{BATCH}_H{NUM_HEADS}_Hkv{NUM_KV_HEADS}_D{HEAD_DIM}_P{PAGE_SIZE}_lens{max(LENS)}"
    ref = ref_attention_paged(
        q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens
    )
    records = [
        _record("paged_reference_fp32", "float32", shape,
                _errors(ref, ref), 0.0,
                "fp32 reference vs itself (sanity)"),
    ]

    # 16-bit storage of the cache and query.
    for dtype, name, tol in (
        (torch.float16, "fp16", FP16_MAX_ABS_TOL),
        (torch.bfloat16, "bf16", BF16_MAX_ABS_TOL),
    ):
        low_cache_k = cache.k_cache.to(dtype)
        low_cache_v = cache.v_cache.to(dtype)
        low_q = q.to(dtype)
        out = ref_attention_paged(
            low_q, low_cache_k, low_cache_v, cache.page_table, cache.seq_lens
        )
        records.append(
            _record(f"paged_storage_{name}", str(dtype).replace("torch.", ""), shape,
                    _errors(out, ref), tol,
                    "fp32 reference with 16-bit cache/query storage; output cast back to 16-bit")
        )

    qd, kd, vd = _dense_inputs()
    dense_shape = f"B2_H{NUM_HEADS}_Hkv{NUM_KV_HEADS}_L64_D{HEAD_DIM}"
    ref_dense = ref_attention_dense(qd, kd, vd, causal=True)
    for dtype, name, tol in (
        (torch.float16, "fp16", FP16_MAX_ABS_TOL),
        (torch.bfloat16, "bf16", BF16_MAX_ABS_TOL),
    ):
        out = ref_attention_dense(
            qd.to(dtype), kd.to(dtype), vd.to(dtype), causal=True
        )
        records.append(
            _record(f"prefill_storage_{name}", str(dtype).replace("torch.", ""), dense_shape,
                    _errors(out, ref_dense), tol,
                    "fp32 reference prefill with 16-bit inputs; output cast back to 16-bit")
        )

    # Quantised KV caches (per-page, per-KV-head scales).
    for mode, name, tol in (
        ("int8", "int8", INT8_MAX_ABS_TOL),
        ("float8_e4m3fn", "fp8_e4m3fn", FP8_MAX_ABS_TOL),
    ):
        if mode == "float8_e4m3fn" and FP8_DTYPE is None:
            records.append(
                _record(f"kv_{name}", name, shape, None, tol, "fp8 dtype unavailable", "skipped")
            )
            continue
        qcache, scales = quantize_cache(cache, mode=mode)
        deq = dequantize_cache(qcache, scales, out_dtype=torch.float32)
        out = ref_attention_paged(
            q, deq.k_cache, deq.v_cache, deq.page_table, deq.seq_lens
        )
        records.append(
            _record(f"kv_{name}", name, shape, _errors(out, ref), tol,
                    "KV cache quantised per (page, KV-head), dequantised to fp32, "
                    "then fp32 reference attention")
        )

    # Split-K on the CPU falls back to the same reference implementation.
    split = paged_decode_attention(q, cache, config={"split_k": 4})
    records.append(
        _record("split_k_4_cpu_fallback", "float32", shape,
                _errors(split, ref), SPLIT_K_MAX_ABS_TOL,
                "CPU fallback ignores split_k (identical reference path); the GPU "
                "split-K LSE merge is validated by tests/test_decode.py")
    )

    # Paths that need hardware/tooling not present here.
    cuda = torch.cuda.is_available()
    triton = triton_available()
    reason = []
    if not cuda:
        reason.append("no CUDA device")
    if not triton:
        reason.append("Triton not importable")
    missing = "; ".join(reason) if reason else ""
    for path, note in (
        ("triton_decode", "Triton paged decode kernel"),
        ("triton_prefill", "Triton varlen prefill kernel"),
    ):
        records.append(
            _record(path, "float16/bf16", shape, None, None,
                    f"not run here ({missing}); validated by the GPU test suite"
                    if missing else "GPU kernel numerics are covered by tests/test_prefill.py "
                                    "and tests/test_decode.py",
                    "skipped")
        )
    records.append(
        _record("native_cuda_decode", "float16/bf16", shape, None, None,
                f"CUDA C++ baseline not built ({missing or 'build not attempted'})",
                "skipped")
    )
    return records, ref


def test_numerics_table():
    records, _ = _run_checks()
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": device_info(),
        "precision": "fp32 accumulation throughout the reference; 16-bit rows measure "
                     "input/output storage rounding",
        "tolerance_basis": "provisional CPU-path bounds; GPU kernel tolerances are checked "
                           "in tests/test_prefill.py and tests/test_decode.py",
        "kv_bytes_per_token": {
            "config": {"num_kv_heads": 8, "head_dim": 128, "page_size": 16},
            "fp16": kv_bytes_per_token(8, 128, "fp16", 16),
            "bf16": kv_bytes_per_token(8, 128, "bf16", 16),
            "int8": kv_bytes_per_token(8, 128, "int8", 16),
            "fp8": kv_bytes_per_token(8, 128, "fp8", 16),
            "fp16_to_int8_factor": kv_bytes_per_token(8, 128, "fp16", 16)
            / kv_bytes_per_token(8, 128, "int8", 16),
            "note": "analytic: K+V elements plus two fp32 scales per (page, KV head); "
                    "scale overhead included",
        },
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    failures = [r for r in records if r["status"] == "measured" and not r["passed"]]
    assert not failures, f"numerics failures: {[(r['path'], r['max_abs']) for r in failures]}"
    print(f"wrote {OUT.relative_to(ROOT)} with {len(records)} records")
