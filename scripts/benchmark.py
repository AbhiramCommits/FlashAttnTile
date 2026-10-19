#!/usr/bin/env python
"""Head-to-head benchmark harness for FlashAttnTile.

Modes
-----
``--mode decode``
    Paged decode: batch x context-length sweep at ``page_size=16``.
``--mode prefill``
    Varlen prefill: sequence-length sweep (batch 1).

On a CUDA host with Triton this times our kernels (after autotuning) against
PyTorch SDPA, FlashAttention-2 and FlashInfer when importable, plus the optional
CUDA C++ baseline. On a CPU-only host the script exits successfully with
``status: "no_gpu"`` and records a much smaller *CPU reference* sweep (labelled
as such) so the repository still ships measured artifacts.

Artifacts
---------
``results/raw/bench_<mode>.csv``   full per-run rows (gitignored)
``results/bench_<mode>.csv``       committed copy of the same rows
``results/benchmarks.json``        merged payload with status + speedups
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashattn_tile.autotune import (  # noqa: E402
    arithmetic_intensity,
    autotune,
    best_config,
    build_config_space,
    bytes_attention,
    flops_attention,
    initialize_store,
    summarize_times,
    time_callable,
)
from flashattn_tile.env import probe, triton_available  # noqa: E402
from flashattn_tile.kernels.decode import paged_decode_attention  # noqa: E402
from flashattn_tile.kernels.prefill import prefill_attention  # noqa: E402
from flashattn_tile.paging import allocate, gather_paged  # noqa: E402
from flashattn_tile.reference import ref_attention_dense, ref_attention_paged  # noqa: E402

FULL_DECODE = {
    "batch": [1, 8, 32, 128],
    "ctx": [128, 1024, 4096, 16384],
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "page_size": 16,
}
FULL_PREFILL = {
    "seq_len": [512, 1024, 2048, 4096],
    "batch": 1,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "page_size": 16,
}
# Deliberately smaller than the GPU grid; these are CPU reference timings, not
# kernel throughput. Kept small enough to finish in seconds on a laptop.
CPU_DECODE = {
    "batch": [1, 8],
    "ctx": [128, 1024, 4096],
    "num_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 64,
    "page_size": 16,
}
CPU_PREFILL = {
    "seq_len": [128, 512, 1024],
    "batch": 1,
    "num_heads": 8,
    "num_kv_heads": 2,
    "head_dim": 64,
    "page_size": 16,
}
OURS_GPU = "flashattn_tile_triton"
OURS_CPU = "flashattn_tile_cpu_reference"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


# ---------------------------------------------------------------------------
# Optional backends
# ---------------------------------------------------------------------------
def _version(module: str) -> str | None:
    try:
        return importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_baselines() -> dict:
    available = {}
    try:
        import flash_attn  # noqa: F401

        available["flash_attn"] = {"available": True, "version": _version("flash_attn")}
    except Exception as exc:  # pragma: no cover - host dependent
        available["flash_attn"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        import flashinfer  # noqa: F401

        available["flashinfer"] = {"available": True, "version": _version("flashinfer")}
    except Exception as exc:  # pragma: no cover
        available["flashinfer"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    available["sdpa"] = {"available": hasattr(F, "scaled_dot_product_attention"),
                         "version": torch.__version__}
    available["triton"] = {"available": triton_available(), "version": _version("triton")}
    available["native_cuda"] = {"available": False, "reason": "not probed"}
    available["ncu"] = {"available": shutil.which("ncu") is not None,
                        "path": shutil.which("ncu")}
    try:
        from flashattn_tile import native

        available["native_cuda"] = {
            "available": native.is_available(),
            "reason": native.unavailable_reason(),
        }
    except Exception as exc:  # pragma: no cover
        available["native_cuda"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return available


def _measure(fn, warmup: int, iters: int) -> dict:
    with torch.inference_mode():
        times = time_callable(fn, warmup=warmup, iters=iters)
    return summarize_times(times)


def _row(mode, path, device, shape, flops, nbytes, stats, dtype, **extra):
    # CPU-only hosts here are shared and heavily loaded, so the headline CPU
    # number is the minimum over iterations (least interference). GPU numbers
    # use the median, the usual benchmark convention. Both are recorded.
    aggregate = extra.pop("aggregate", "min" if device == "cpu" else "median")
    latency = stats["min_ms"] if aggregate == "min" else stats["median_ms"]
    row = {
        "mode": mode,
        "path": path,
        "device": device,
        "batch": shape["batch"],
        "seq_len": shape["seq_len"],
        "num_heads": shape["num_heads"],
        "num_kv_heads": shape["num_kv_heads"],
        "head_dim": shape["head_dim"],
        "page_size": shape.get("page_size"),
        "dtype": dtype,
        "latency_ms": latency,
        "latency_aggregation": aggregate,
        "mean_ms": stats["mean_ms"],
        "median_ms": stats["median_ms"],
        "min_ms": stats["min_ms"],
        "p90_ms": stats["p90_ms"],
        "iters": stats["iters"],
        "tflops": flops / (latency * 1e-3) / 1e12,
        "gbytes_per_s": nbytes / (latency * 1e-3) / 1e9,
        "arithmetic_intensity": arithmetic_intensity(flops, nbytes),
        "flops": int(flops),
        "bytes": int(nbytes),
        "pct_measured_hbm": None,
        "gather_ms": None,
        "speedup_vs_sdpa": None,
        "max_abs_vs_ref": None,
        "notes": "",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# CUDA paths
# ---------------------------------------------------------------------------
def _fill_cache(batch, ctx, page_size, kv_heads, head_dim, dtype, device):
    cache = allocate(batch, ctx, kv_heads, head_dim, page_size, dtype=dtype, device=device)
    with torch.no_grad():
        cache.k_cache.normal_(0.0, 0.2)
        cache.v_cache.normal_(0.0, 0.2)
    cache.seq_lens[:] = ctx
    return cache


def _sdpa_decode_inputs(cache, q):
    k, v = gather_paged(cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens,
                        cache.page_size)
    group = q.shape[1] // k.shape[2]
    k = k.permute(0, 2, 1, 3).repeat_interleave(group, dim=1).contiguous()
    v = v.permute(0, 2, 1, 3).repeat_interleave(group, dim=1).contiguous()
    return q.unsqueeze(2), k, v


def _flashinfer_decode(q, cache):
    import flashinfer

    if q.shape[0] != 1:
        raise RuntimeError("flashinfer single_decode baseline implemented for batch=1 only")
    if not hasattr(flashinfer, "single_decode_with_kv_cache"):
        raise RuntimeError("installed flashinfer exposes no single_decode_with_kv_cache")
    k, v = gather_paged(cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens,
                        cache.page_size)
    group = q.shape[1] // k.shape[2]
    k = k[0].repeat_interleave(group, dim=1).contiguous()
    v = v[0].repeat_interleave(group, dim=1).contiguous()
    return flashinfer.single_decode_with_kv_cache(q[0], k, v).unsqueeze(0)


def _flashinfer_prefill(q, k, v):
    import flashinfer

    if q.shape[0] != 1:
        raise RuntimeError("flashinfer single_prefill baseline implemented for batch=1 only")
    if not hasattr(flashinfer, "single_prefill_with_kv_cache"):
        raise RuntimeError("installed flashinfer exposes no single_prefill_with_kv_cache")
    return flashinfer.single_prefill_with_kv_cache(
        q[0], k[0], v[0], causal=True, pos_encoding_mode="NONE"
    ).unsqueeze(0)


def benchmark_decode_gpu(dtype: torch.dtype, dtype_name: str, hbm_gbs: float | None,
                         baselines: dict, warmup: int, iters: int):
    rows, skips, tune_rows = [], [], []
    autotune_summaries = []
    device = torch.cuda.get_device_name(0)
    for batch in FULL_DECODE["batch"]:
        for ctx in FULL_DECODE["ctx"]:
            H, Hkv, D = FULL_DECODE["num_heads"], FULL_DECODE["num_kv_heads"], FULL_DECODE["head_dim"]
            ps = FULL_DECODE["page_size"]
            shape = {"batch": batch, "seq_len": ctx, "num_heads": H, "num_kv_heads": Hkv,
                     "head_dim": D, "page_size": ps}
            cache = _fill_cache(batch, ctx, ps, Hkv, D, dtype, "cuda")
            q = (torch.randn(batch, H, D, device="cuda") * 0.2).to(dtype)
            flops = flops_attention(H, D, 1, ctx, causal=True, batch=batch)
            nbytes = bytes_attention("decode", batch, 1, ctx, H, Hkv, D, dtype.itemsize)

            signature = f"B{batch}_H{H}_Hkv{Hkv}_D{D}_P{ps}_L{ctx}_sk1"
            space = build_config_space("decode", D, dtype.itemsize, ps)
            problem = {
                "kind": "decode",
                "signature": signature,
                "runner": lambda cfg: paged_decode_attention(q, cache, config=cfg),
                "flops": flops,
                "bytes": nbytes,
            }
            df = autotune(problem, space, warmup=5, iters=20, hbm_gbs=hbm_gbs, save=True)
            tune_rows.append(df.assign(batch=batch, seq_len=ctx, num_heads=H, head_dim=D))
            best = best_config("decode", signature)
            autotune_summaries.append({
                "signature": signature,
                "enumerated": len(space),
                "feasible": len(space.feasible()),
                "winner": best,
                "best_ms": float(df.iloc[0].latency_ms),
                "worst_feasible_ms": float(df.iloc[-1].latency_ms),
                "spread_x": float(df.iloc[-1].latency_ms / df.iloc[0].latency_ms),
            })
            stats = _measure(lambda: paged_decode_attention(q, cache), warmup, iters)
            ours = _row("decode", OURS_GPU, device, shape, flops, nbytes, stats, dtype_name,
                        notes=f"winner={best}")
            if hbm_gbs:
                ours["pct_measured_hbm"] = 100.0 * ours["gbytes_per_s"] / hbm_gbs
            rows.append(ours)

            q_sdpa, k_sdpa, v_sdpa = _sdpa_decode_inputs(cache, q)
            gather_stats = _measure(
                lambda: gather_paged(cache.k_cache, cache.v_cache, cache.page_table,
                                     cache.seq_lens, cache.page_size),
                max(1, warmup // 2), max(5, iters // 2),
            )
            # A single decode query sees the full cache: no causal mask (PyTorch's
            # is_causal=True uses top-left alignment for non-square shapes).
            sdpa_stats = _measure(
                lambda: F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False),
                warmup, iters,
            )
            nbytes_sdpa = bytes_attention("decode", batch, 1, ctx, H, H, D, dtype.itemsize)
            sdpa = _row("decode", "sdpa", device, shape, flops, nbytes_sdpa, sdpa_stats, dtype_name,
                        gather_ms=gather_stats["median_ms"],
                        notes="attention only; gather timed separately; K/V expanded for GQA")
            if hbm_gbs:
                sdpa["pct_measured_hbm"] = 100.0 * sdpa["gbytes_per_s"] / hbm_gbs
            rows.append(sdpa)

            if baselines["flash_attn"]["available"]:
                try:
                    from flash_attn import flash_attn_func

                    qf = q.unsqueeze(1)
                    kf, vf = gather_paged(cache.k_cache, cache.v_cache, cache.page_table,
                                          cache.seq_lens, cache.page_size)
                    st = _measure(lambda: flash_attn_func(qf, kf, vf, causal=True), warmup, iters)
                    row = _row("decode", "flash_attn", device, shape, flops, nbytes, st,
                               dtype_name, notes="flash_attn_func, gathered K/V")
                    if hbm_gbs:
                        row["pct_measured_hbm"] = 100.0 * row["gbytes_per_s"] / hbm_gbs
                    rows.append(row)
                except Exception as exc:
                    skips.append({"mode": "decode", "path": "flash_attn", "batch": batch,
                                  "seq_len": ctx, "reason": f"{type(exc).__name__}: {exc}"})

            if baselines["flashinfer"]["available"]:
                try:
                    st = _measure(lambda: _flashinfer_decode(q, cache), warmup, iters)
                    rows.append(_row("decode", "flashinfer", device, shape, flops, nbytes, st,
                                     dtype_name))
                except Exception as exc:
                    skips.append({"mode": "decode", "path": "flashinfer", "batch": batch,
                                  "seq_len": ctx, "reason": f"{type(exc).__name__}: {exc}"})

            if baselines["native_cuda"]["available"]:
                try:
                    from flashattn_tile import native

                    st = _measure(
                        lambda: native.paged_decode_attention(
                            q, cache.k_cache, cache.v_cache, cache.page_table,
                            cache.seq_lens, ps,
                        ),
                        warmup, iters,
                    )
                    rows.append(_row("decode", "native_cuda", device, shape, flops, nbytes, st,
                                     dtype_name, notes="csrc/paged_decode.cu baseline"))
                except Exception as exc:
                    skips.append({"mode": "decode", "path": "native_cuda", "batch": batch,
                                  "seq_len": ctx, "reason": f"{type(exc).__name__}: {exc}"})
    return rows, skips, tune_rows, autotune_summaries


def benchmark_prefill_gpu(dtype: torch.dtype, dtype_name: str, hbm_gbs: float | None,
                          baselines: dict, warmup: int, iters: int):
    rows, skips, tune_rows, autotune_summaries = [], [], [], []
    device = torch.cuda.get_device_name(0)
    H, Hkv, D = FULL_PREFILL["num_heads"], FULL_PREFILL["num_kv_heads"], FULL_PREFILL["head_dim"]
    for seq in FULL_PREFILL["seq_len"]:
        batch = FULL_PREFILL["batch"]
        shape = {"batch": batch, "seq_len": seq, "num_heads": H, "num_kv_heads": Hkv,
                 "head_dim": D, "page_size": None}
        q = (torch.randn(batch * seq, H, D, device="cuda") * 0.2).to(dtype)
        k = (torch.randn(batch * seq, Hkv, D, device="cuda") * 0.2).to(dtype)
        v = (torch.randn(batch * seq, Hkv, D, device="cuda") * 0.2).to(dtype)
        cu = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
        flops = flops_attention(H, D, seq, seq, causal=True, batch=batch)
        nbytes = bytes_attention("prefill", batch, seq, seq, H, Hkv, D, dtype.itemsize)

        signature = f"B{batch}_H{H}_Hkv{Hkv}_D{D}_Lq{seq}_Lk{seq}_c1"
        space = build_config_space("prefill", D, dtype.itemsize)
        problem = {
            "kind": "prefill",
            "signature": signature,
            "runner": lambda cfg: prefill_attention(q, k, v, cu, cu, seq, seq, causal=True,
                                                    config=cfg),
            "flops": flops,
            "bytes": nbytes,
        }
        df = autotune(problem, space, warmup=5, iters=20, hbm_gbs=hbm_gbs, save=True)
        tune_rows.append(df.assign(batch=batch, seq_len=seq, num_heads=H, head_dim=D))
        best = best_config("prefill", signature)
        autotune_summaries.append({
            "signature": signature,
            "enumerated": len(space),
            "feasible": len(space.feasible()),
            "winner": best,
            "best_ms": float(df.iloc[0].latency_ms),
            "worst_feasible_ms": float(df.iloc[-1].latency_ms),
            "spread_x": float(df.iloc[-1].latency_ms / df.iloc[0].latency_ms),
        })
        stats = _measure(
            lambda: prefill_attention(q, k, v, cu, cu, seq, seq, causal=True), warmup, iters
        )
        ours = _row("prefill", OURS_GPU, device, shape, flops, nbytes, stats, dtype_name,
                    notes=f"winner={best}")
        if hbm_gbs:
            ours["pct_measured_hbm"] = 100.0 * ours["gbytes_per_s"] / hbm_gbs
        rows.append(ours)

        qd = q.view(batch, seq, H, D).transpose(1, 2)
        kd = k.view(batch, seq, Hkv, D).transpose(1, 2)
        vd = v.view(batch, seq, Hkv, D).transpose(1, 2)
        group = H // Hkv
        kd = kd.repeat_interleave(group, dim=1).contiguous()
        vd = vd.repeat_interleave(group, dim=1).contiguous()
        sdpa_stats = _measure(
            lambda: F.scaled_dot_product_attention(qd, kd, vd, is_causal=True), warmup, iters
        )
        nbytes_sdpa = bytes_attention("prefill", batch, seq, seq, H, H, D, dtype.itemsize)
        sdpa = _row("prefill", "sdpa", device, shape, flops, nbytes_sdpa, sdpa_stats, dtype_name,
                    notes="K/V expanded for GQA")
        if hbm_gbs:
            sdpa["pct_measured_hbm"] = 100.0 * sdpa["gbytes_per_s"] / hbm_gbs
        rows.append(sdpa)

        if baselines["flash_attn"]["available"]:
            try:
                from flash_attn import flash_attn_func

                qf = q.view(batch, seq, H, D)
                kf = k.view(batch, seq, Hkv, D)
                vf = v.view(batch, seq, Hkv, D)
                st = _measure(lambda: flash_attn_func(qf, kf, vf, causal=True), warmup, iters)
                row = _row("prefill", "flash_attn", device, shape, flops, nbytes, st, dtype_name)
                if hbm_gbs:
                    row["pct_measured_hbm"] = 100.0 * row["gbytes_per_s"] / hbm_gbs
                rows.append(row)
            except Exception as exc:
                skips.append({"mode": "prefill", "path": "flash_attn", "batch": batch,
                              "seq_len": seq, "reason": f"{type(exc).__name__}: {exc}"})

        if baselines["flashinfer"]["available"]:
            try:
                qf = q.view(batch, seq, H, D)
                kf = k.view(batch, seq, Hkv, D)
                vf = v.view(batch, seq, Hkv, D)
                st = _measure(lambda: _flashinfer_prefill(qf, kf, vf), warmup, iters)
                row = _row("prefill", "flashinfer", device, shape, flops, nbytes, st, dtype_name)
                if hbm_gbs:
                    row["pct_measured_hbm"] = 100.0 * row["gbytes_per_s"] / hbm_gbs
                rows.append(row)
            except Exception as exc:
                skips.append({"mode": "prefill", "path": "flashinfer", "batch": batch,
                              "seq_len": seq, "reason": f"{type(exc).__name__}: {exc}"})
    return rows, skips, tune_rows, autotune_summaries


# ---------------------------------------------------------------------------
# CPU reference path
# ---------------------------------------------------------------------------
def benchmark_decode_cpu(warmup: int, iters: int):
    rows = []
    cfg = CPU_DECODE
    H, Hkv, D, ps = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    for batch in cfg["batch"]:
        for ctx in cfg["ctx"]:
            shape = {"batch": batch, "seq_len": ctx, "num_heads": H, "num_kv_heads": Hkv,
                     "head_dim": D, "page_size": ps}
            cache = _fill_cache(batch, ctx, ps, Hkv, D, torch.float32, "cpu")
            q = torch.randn(batch, H, D) * 0.2
            flops = flops_attention(H, D, 1, ctx, causal=True, batch=batch)
            nbytes = bytes_attention("decode", batch, 1, ctx, H, Hkv, D, 4)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats = _measure(lambda: paged_decode_attention(q, cache), warmup, iters)
                ref = ref_attention_paged(
                    q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens
                )
                ours = _row("decode", OURS_CPU, "cpu", shape, flops, nbytes, stats, "float32",
                            notes="PyTorch fp32 reference fallback; NOT kernel throughput")
                ours["max_abs_vs_ref"] = float(
                    (paged_decode_attention(q, cache) - ref).abs().max().item()
                )
            rows.append(ours)

            q_sdpa, k_sdpa, v_sdpa = _sdpa_decode_inputs(cache, q)
            gather_stats = _measure(
                lambda: gather_paged(cache.k_cache, cache.v_cache, cache.page_table,
                                     cache.seq_lens, cache.page_size),
                warmup, iters,
            )
            # Single query -> no mask (is_causal=True would only expose key 0).
            sdpa_stats = _measure(
                lambda: F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False),
                warmup, iters,
            )
            out_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
            nbytes_sdpa = bytes_attention("decode", batch, 1, ctx, H, H, D, 4)
            sdpa = _row("decode", "sdpa", "cpu", shape, flops, nbytes_sdpa, sdpa_stats, "float32",
                        gather_ms=gather_stats["median_ms"],
                        notes="CPU SDPA; attention only, gather timed separately; K/V expanded")
            sdpa["max_abs_vs_ref"] = float((out_sdpa.squeeze(2) - ref).abs().max().item())
            rows.append(sdpa)
    return rows


def benchmark_prefill_cpu(warmup: int, iters: int):
    rows = []
    cfg = CPU_PREFILL
    H, Hkv, D = cfg["num_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    for seq in cfg["seq_len"]:
        batch = cfg["batch"]
        shape = {"batch": batch, "seq_len": seq, "num_heads": H, "num_kv_heads": Hkv,
                 "head_dim": D, "page_size": None}
        q = torch.randn(batch * seq, H, D) * 0.2
        k = torch.randn(batch * seq, Hkv, D) * 0.2
        v = torch.randn(batch * seq, Hkv, D) * 0.2
        cu = torch.tensor([b * seq for b in range(batch + 1)], dtype=torch.int32)
        flops = flops_attention(H, D, seq, seq, causal=True, batch=batch)
        nbytes = bytes_attention("prefill", batch, seq, seq, H, Hkv, D, 4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stats = _measure(
                lambda: prefill_attention(q, k, v, cu, cu, seq, seq, causal=True), warmup, iters
            )
            ref = ref_attention_dense(
                q.view(batch, seq, H, D).transpose(1, 2),
                k.view(batch, seq, Hkv, D).transpose(1, 2),
                v.view(batch, seq, Hkv, D).transpose(1, 2),
                causal=True,
            )
            ours = _row("prefill", OURS_CPU, "cpu", shape, flops, nbytes, stats, "float32",
                        notes="PyTorch fp32 reference fallback; NOT kernel throughput")
            ours["max_abs_vs_ref"] = 0.0
        rows.append(ours)

        qd = q.view(batch, seq, H, D).transpose(1, 2)
        kd = k.view(batch, seq, Hkv, D).transpose(1, 2).repeat_interleave(H // Hkv, dim=1).contiguous()
        vd = v.view(batch, seq, Hkv, D).transpose(1, 2).repeat_interleave(H // Hkv, dim=1).contiguous()
        sdpa_stats = _measure(
            lambda: F.scaled_dot_product_attention(qd, kd, vd, is_causal=True), warmup, iters
        )
        out_sdpa = F.scaled_dot_product_attention(qd, kd, vd, is_causal=True)
        nbytes_sdpa = bytes_attention("prefill", batch, seq, seq, H, H, D, 4)
        sdpa = _row("prefill", "sdpa", "cpu", shape, flops, nbytes_sdpa, sdpa_stats, "float32",
                    notes="CPU SDPA; K/V expanded for GQA")
        sdpa["max_abs_vs_ref"] = float((out_sdpa - ref).abs().max().item())
        rows.append(sdpa)
    return rows


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------
def collect_occupancy(baselines: dict) -> dict:
    if baselines["ncu"]["available"] and torch.cuda.is_available():
        raw_dir = ROOT / "results" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = raw_dir / "ncu_decode.txt"
        metrics = (
            "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
            "launch__occupancy_limit_shared_mem,"
            "dram__throughput.avg.pct_of_peak_sustained_elapsed"
        )
        cmd = [
            baselines["ncu"]["path"], "--metrics", metrics, "--target-processes", "all",
            sys.executable, str(ROOT / "scripts" / "ncu_probe.py"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out_path.write_text(proc.stdout + "\n" + proc.stderr)
        parsed = {}
        for line in proc.stdout.splitlines():
            if "%" in line and any(key in line for key in ("sm__throughput", "dram__throughput",
                                                           "occupancy_limit_shared_mem")):
                parsed[line.split()[0]] = line
        return {"method": "ncu", "raw_path": str(out_path.relative_to(ROOT)),
                "metrics": parsed, "returncode": proc.returncode}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        smem_per_sm = int(getattr(props, "shared_memory_per_multiprocessor", 0) or 0)
        max_threads = int(getattr(props, "max_threads_per_multiprocessor", 0) or 2048)
        from flashattn_tile.autotune import decode_smem_bytes

        block_smem = decode_smem_bytes(16, 128, 2, 2)
        threads_per_block = 4 * 32
        blocks_smem = smem_per_sm // block_smem if block_smem else 0
        blocks_threads = max_threads // threads_per_block
        blocks = max(0, min(blocks_smem, blocks_threads))
        return {
            "method": "analytic",
            "note": "smem/warp upper bound from Triton launch params; register usage not "
                    "available, so this is NOT a profiled number",
            "smem_per_block": block_smem,
            "smem_per_sm": smem_per_sm,
            "occupancy_pct": 100.0 * blocks * threads_per_block / max_threads if max_threads else None,
        }
    return {
        "method": "unavailable",
        "reason": "no CUDA device and no ncu binary on this host; no occupancy was measured",
    }


# ---------------------------------------------------------------------------
# Post-processing and artifact writing
# ---------------------------------------------------------------------------
def _shape_key(row):
    return (row["mode"], row["batch"], row["seq_len"], row["num_heads"], row["num_kv_heads"],
            row["head_dim"], row["dtype"])


def add_speedups(rows):
    ours = {_shape_key(r): r for r in rows if r["path"] in (OURS_GPU, OURS_CPU)}
    sdpa = {_shape_key(r): r for r in rows if r["path"] == "sdpa"}
    for key, row in ours.items():
        if key in sdpa:
            row["speedup_vs_sdpa"] = sdpa[key]["latency_ms"] / row["latency_ms"]
    for row in rows:
        if row["path"] == "sdpa" and row["gather_ms"]:
            row["sdpa_e2e_ms"] = row["latency_ms"] + row["gather_ms"]
    return rows


def _write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _merge_payload(payload: dict, mode: str, mode_payload: dict, out_path: Path):
    status = payload["status"]
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing = None
        if existing and existing.get("status") == status:
            modes = existing.get("modes", {})
            modes[mode] = mode_payload
            existing["modes"] = modes
            existing["generated_utc"] = payload["generated_utc"]
            existing["environment"] = payload["environment"]
            existing["baselines"] = payload["baselines"]
            existing["executed_grid"] = {**existing.get("executed_grid", {}),
                                         mode: mode_payload["executed_grid"]}
            existing["autotuning"] = {**existing.get("autotuning", {}),
                                      mode: mode_payload["autotuning"]}
            existing["occupancy"] = payload["occupancy"]
            existing["skip_reasons"] = sorted(
                {**{s["mode"] + ":" + s["path"]: s for s in existing.get("skip_reasons", [])},
                 **{s["mode"] + ":" + s["path"]: s for s in payload["skip_reasons"]}}.values(),
                key=lambda s: (s["mode"], s["path"]),
            )
            out_path.write_text(json.dumps(existing, indent=2) + "\n")
            return existing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "benchmarks.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--cpu-warmup", type=int, default=3)
    parser.add_argument("--cpu-iters", type=int, default=21)
    args = parser.parse_args()

    torch.manual_seed(0)
    baselines = probe_baselines()
    env = probe()
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    hbm = env.get("measured_hbm_bandwidth_gbs")
    dtype = DTYPES[args.dtype]
    modes = ["decode", "prefill"] if args.mode == "both" else [args.mode]
    use_gpu = torch.cuda.is_available() and baselines["triton"]["available"]

    if use_gpu:
        status, reason = "ok", ""
    else:
        missing = []
        if not torch.cuda.is_available():
            missing.append("no CUDA device")
        if not baselines["triton"]["available"]:
            missing.append("triton not importable")
        status = "no_gpu"
        reason = "; ".join(missing)
        print(f"[benchmark] NO GPU PATH: {reason}.")
        print("[benchmark] Running the small CPU reference sweep and writing status=no_gpu.")
        initialize_store(
            "No GPU autotuning was performed on this host (" + reason + "). "
            "Kernel wrappers use their built-in defaults until tuned_configs.json is populated "
            "by scripts/benchmark.py on a CUDA machine."
        )

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "environment": env,
        "baselines": baselines,
        "requested_grid": {
            "decode": FULL_DECODE,
            "prefill": FULL_PREFILL,
            "note": "full grid is executed only on CUDA hosts; CPU hosts run a reduced sweep",
        },
        "executed_grid": {},
        "modes": {},
        "autotuning": {},
        "occupancy": collect_occupancy(baselines),
        "skip_reasons": [],
    }

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    all_tune_rows = []
    for mode in modes:
        if use_gpu:
            if mode == "decode":
                rows, skips, tune_rows, tunes = benchmark_decode_gpu(
                    dtype, args.dtype, hbm, baselines, args.warmup, args.iters
                )
                executed = {k: FULL_DECODE[k] for k in
                            ("batch", "ctx", "num_heads", "num_kv_heads", "head_dim", "page_size")}
            else:
                rows, skips, tune_rows, tunes = benchmark_prefill_gpu(
                    dtype, args.dtype, hbm, baselines, args.warmup, args.iters
                )
                executed = {k: FULL_PREFILL[k] for k in
                            ("seq_len", "batch", "num_heads", "num_kv_heads", "head_dim")}
        else:
            if mode == "decode":
                rows, skips, tunes = benchmark_decode_cpu(args.cpu_warmup, args.cpu_iters), [], []
                executed = {k: CPU_DECODE[k] for k in
                            ("batch", "ctx", "num_heads", "num_kv_heads", "head_dim", "page_size")}
            else:
                rows, skips, tunes = benchmark_prefill_cpu(args.cpu_warmup, args.cpu_iters), [], []
                executed = {k: CPU_PREFILL[k] for k in
                            ("seq_len", "batch", "num_heads", "num_kv_heads", "head_dim")}
            tune_rows = []

        rows = add_speedups(rows)
        all_tune_rows.extend(tune_rows)
        mode_payload = {
            "status": status,
            "device": "cuda" if use_gpu else "cpu",
            "dtype": args.dtype if use_gpu else "float32",
            "requested_dtype": args.dtype,
            "executed_grid": executed,
            "rows": rows,
            "notes": (
                "Kernel throughput measured with CUDA events."
                if use_gpu else
                "CPU reference fallback timings. These are not GPU kernel throughput numbers."
            ),
            "autotuning": {
                "performed": bool(tune_rows),
                "search_space": (
                    {
                        "decode": build_config_space("decode", 128, 2, 16).to_dict(),
                        "prefill": build_config_space("prefill", 128, 2).to_dict(),
                    }
                    if not use_gpu else
                    {
                        "note": "per-shape space recorded in the summaries below",
                        "summaries": tunes,
                    }
                ),
                "reason": None if use_gpu else "no CUDA GPU / Triton",
            },
        }
        payload["executed_grid"][mode] = executed
        payload["modes"][mode] = mode_payload
        payload["autotuning"][mode] = mode_payload["autotuning"]
        payload["skip_reasons"].extend(skips)

        raw = ROOT / "results" / "raw" / f"bench_{mode}.csv"
        committed = ROOT / "results" / f"bench_{mode}.csv"
        _write_csv(rows, raw)
        _write_csv(rows, committed)
        if tune_rows:
            import pandas as pd

            pd.concat(tune_rows, ignore_index=True).to_csv(
                ROOT / "results" / "raw" / f"autotune_{mode}.csv", index=False
            )

    merged = _merge_payload(payload, modes[0] if len(modes) == 1 else "decode", payload["modes"][modes[0]], args.out)
    if len(modes) == 2:
        for mode in modes:
            _merge_payload(payload, mode, payload["modes"][mode], args.out)
        merged = json.loads(args.out.read_text())

    print(f"[benchmark] status={merged['status']} path={'cuda' if use_gpu else 'cpu'}")
    for mode in modes:
        rows = merged["modes"].get(mode, {}).get("rows", [])
        if rows:
            import pandas as pd

            df = pd.DataFrame(rows)
            cols = [c for c in ("path", "batch", "seq_len", "latency_ms", "tflops",
                                "gbytes_per_s", "speedup_vs_sdpa", "gather_ms") if c in df]
            print(f"\n== {mode} ==")
            print(df[cols].to_string(index=False))
    print(f"\n[benchmark] wrote {args.out.relative_to(ROOT)} and per-mode CSVs")
    if not use_gpu:
        print("[benchmark] status=no_gpu: kernel TFLOPs/bandwidth numbers are absent by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
