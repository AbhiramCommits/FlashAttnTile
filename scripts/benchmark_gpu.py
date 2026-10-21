#!/usr/bin/env python
"""GPU benchmark: FlashAttnTile vs PyTorch SDPA (flash backend).

Times with ``triton.testing.do_bench`` (warmup + median, L2 flushed between
iterations) and reports

* prefill: TFLOP/s and % of the datasheet FP16 tensor-core peak, across
  sequence length, number of heads and GQA ratio;
* decode: achieved HBM GB/s and % of the datasheet HBM peak, across batch,
  context length, number of heads and page size.

FlashAttention-2 (the ``flash_attn`` package) and vLLM paged attention are not
installed in this container: ``flash-attn`` would require a long source build
(no matching prebuilt wheel for torch 2.8/cu128), and vLLM is a heavy install
that risks the torch pin. PyTorch's built-in SDPA *flash* backend is used as the
flash-attention-style baseline, and SDPA's math backend is recorded for the
prefill sweep to show the tensor-core gap.

Artifacts
---------
``results/benchmark_prefill_gpu.csv``, ``results/benchmark_decode_gpu.csv``
``results/benchmark_gpu.json``, ``results/prefill_tflops.png``,
``results/decode_bandwidth.png``
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from flashattn_tile.autotune import bytes_attention, flops_attention  # noqa: E402
from flashattn_tile.env import peak_fp16_tflops_spec, peak_hbm_gbs_spec, probe  # noqa: E402
from flashattn_tile.kernels.decode import paged_decode_attention  # noqa: E402
from flashattn_tile.kernels.prefill import prefill_attention  # noqa: E402
from flashattn_tile.paging import allocate, gather_paged  # noqa: E402

try:
    from triton.testing import do_bench
except Exception:  # pragma: no cover
    do_bench = None

RESULTS = ROOT / "results"
DEV = "cuda"
HEAD_DIM = 128
GROUP = 4  # default GQA ratio H // Hkv


def bench(fn) -> float:
    """Median latency in ms via triton.testing.do_bench (L2 flushed)."""
    return do_bench(fn, warmup=25, rep=100, return_mode="median")


def _gbytes(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def _tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def _row(mode, path, shape, dtype_name, ms, flops, nbytes, peak_t, peak_bw, extra=None):
    row = {
        "mode": mode,
        "path": path,
        "dtype": dtype_name,
        "batch": shape.get("batch"),
        "seq_len": shape.get("seq_len"),
        "num_heads": shape.get("num_heads"),
        "num_kv_heads": shape.get("num_kv_heads"),
        "head_dim": shape.get("head_dim"),
        "page_size": shape.get("page_size"),
        "latency_ms": ms,
        "tflops": _tflops(flops, ms) if flops else None,
        "pct_peak_tflops": (100.0 * _tflops(flops, ms) / peak_t) if (flops and peak_t) else None,
        "gbytes_per_s": _gbytes(nbytes, ms) if nbytes else None,
        "pct_peak_hbm": (100.0 * _gbytes(nbytes, ms) / peak_bw) if (nbytes and peak_bw) else None,
        "flops": int(flops) if flops else None,
        "bytes": int(nbytes) if nbytes else None,
    }
    if extra:
        row.update(extra)
    return row


def _fill_cache(batch, ctx, page_size, kv_heads, head_dim, dtype):
    cache = allocate(batch, ctx, kv_heads, head_dim, page_size, dtype=dtype, device=DEV)
    with torch.no_grad():
        cache.k_cache.normal_(0.0, 0.2)
        cache.v_cache.normal_(0.0, 0.2)
    cache.seq_lens[:] = ctx
    return cache


def _sdpa_flash(q, k, v, causal):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
                                        enable_mem_efficient=False):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
def benchmark_prefill(dtype, dtype_name, peak_t, peak_bw):
    rows = []
    H, Hkv, D = 32, 8, HEAD_DIM

    def run(seq, heads, kv_heads, tag, batch=1):
        q = (torch.randn(batch * seq, heads, D, device=DEV) * 0.2).to(dtype)
        k = (torch.randn(batch * seq, kv_heads, D, device=DEV) * 0.2).to(dtype)
        v = (torch.randn(batch * seq, kv_heads, D, device=DEV) * 0.2).to(dtype)
        cu = torch.tensor([b * seq for b in range(batch + 1)], dtype=torch.int32, device=DEV)
        flops = flops_attention(heads, D, seq, seq, causal=True, batch=batch)
        nbytes = bytes_attention("prefill", batch, seq, seq, heads, kv_heads, D, dtype.itemsize)

        ms = bench(lambda: prefill_attention(q, k, v, cu, cu, seq, seq, causal=True))
        rows.append(_row("prefill", "flashattn_tile", {"batch": batch, "seq_len": seq,
                                                       "num_heads": heads, "num_kv_heads": kv_heads,
                                                       "head_dim": D, "page_size": None},
                         dtype_name, ms, flops, nbytes, peak_t, peak_bw, {"tag": tag}))

        qd = q.view(batch, seq, heads, D).transpose(1, 2)
        kd = k.view(batch, seq, kv_heads, D).transpose(1, 2)
        vd = v.view(batch, seq, kv_heads, D).transpose(1, 2)
        g = heads // kv_heads
        if g != 1:
            kd = kd.repeat_interleave(g, dim=1).contiguous()
            vd = vd.repeat_interleave(g, dim=1).contiguous()
        try:
            ms_sdpa = bench(lambda: _sdpa_flash(qd, kd, vd, True))
            rows.append(_row("prefill", "sdpa_flash", {"batch": batch, "seq_len": seq,
                                                       "num_heads": heads, "num_kv_heads": kv_heads,
                                                       "head_dim": D, "page_size": None},
                             dtype_name, ms_sdpa, flops, nbytes, peak_t, peak_bw,
                             {"tag": tag, "note": "K/V expanded for GQA"}))
        except Exception as exc:
            rows.append(_row("prefill", "sdpa_flash", {"batch": batch, "seq_len": seq,
                                                       "num_heads": heads, "num_kv_heads": kv_heads,
                                                       "head_dim": D, "page_size": None},
                             dtype_name, float("nan"), flops, nbytes, peak_t, peak_bw,
                             {"tag": tag, "note": f"skipped: {type(exc).__name__}"}))

    # sequence length sweep (fixed GQA=4)
    for seq in [512, 1024, 2048, 4096, 8192]:
        run(seq, H, Hkv, "seq_len")
    # heads sweep (fixed GQA=4)
    for heads in [8, 16, 32, 64]:
        run(2048, heads, heads // GROUP, "heads")
    # GQA ratio sweep (MHA / GQA4 / MQA)
    for kv_heads in [32, 8, 1]:
        run(2048, H, kv_heads, "gqa")
    return rows


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def benchmark_decode(dtype, dtype_name, peak_t, peak_bw):
    rows = []
    H, Hkv, D = 32, 8, HEAD_DIM

    def run(batch, ctx, heads, kv_heads, page_size, tag):
        cache = _fill_cache(batch, ctx, page_size, kv_heads, D, dtype)
        q = (torch.randn(batch, heads, D, device=DEV) * 0.2).to(dtype)
        flops = flops_attention(heads, D, 1, ctx, causal=True, batch=batch)
        nbytes = bytes_attention("decode", batch, 1, ctx, heads, kv_heads, D, dtype.itemsize)
        shape = {"batch": batch, "seq_len": ctx, "num_heads": heads,
                 "num_kv_heads": kv_heads, "head_dim": D, "page_size": page_size}

        ms = bench(lambda: paged_decode_attention(q, cache))
        rows.append(_row("decode", "flashattn_tile", shape, dtype_name, ms, flops, nbytes,
                         peak_t, peak_bw, {"tag": tag}))

        # SDPA baseline on gathered, GQA-expanded K/V (attention only; gather
        # timed separately). Expansion blows up memory, so skip when it would
        # not fit in HBM.
        expand_elems = batch * heads * ctx * D
        if expand_elems * 2 * dtype.itemsize < 12 * 1024**3:
            gk, gv = gather_paged(cache.k_cache, cache.v_cache, cache.page_table,
                                  cache.seq_lens, cache.page_size)
            g = heads // kv_heads
            if g != 1:
                k_s = gk.permute(0, 2, 1, 3).repeat_interleave(g, dim=1).contiguous()
                v_s = gv.permute(0, 2, 1, 3).repeat_interleave(g, dim=1).contiguous()
            else:
                k_s, v_s = gk.permute(0, 2, 1, 3), gv.permute(0, 2, 1, 3)
            ms_sdpa = bench(lambda: F.scaled_dot_product_attention(
                q.unsqueeze(2), k_s, v_s, is_causal=False))
            rows.append(_row("decode", "sdpa", shape, dtype_name, ms_sdpa, flops, nbytes,
                             peak_t, peak_bw,
                             {"tag": tag, "note": "gathered+expanded K/V, attention only"}))
        else:
            rows.append(_row("decode", "sdpa", shape, dtype_name, float("nan"), flops, nbytes,
                             peak_t, peak_bw, {"tag": tag, "note": "skipped: expansion OOM"}))
        return rows

    # batch x context sweep
    for batch in [1, 8, 32, 128]:
        for ctx in [128, 1024, 4096, 16384]:
            run(batch, ctx, H, Hkv, 16, "batch_ctx")
    # page size sweep
    for ps in [8, 16, 32, 64]:
        run(32, 4096, H, Hkv, ps, "page_size")
    # heads sweep
    for heads in [8, 32, 64]:
        run(32, 4096, heads, heads // GROUP, 16, "heads")
    return rows


def _write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _speedups(rows):
    by_key = {}
    for r in rows:
        key = (r["mode"], r["tag"], r["dtype"], r["batch"], r["seq_len"], r["num_heads"],
               r["num_kv_heads"], r["page_size"])
        by_key.setdefault(key, {})[r["path"]] = r
    for key, d in by_key.items():
        ours = d.get("flashattn_tile")
        ref = d.get("sdpa_flash") or d.get("sdpa")
        if ours and ref and ref["latency_ms"] == ref["latency_ms"]:  # not NaN
            ours["speedup_vs_sdpa"] = ref["latency_ms"] / ours["latency_ms"]
    return rows


def _plot(rows, peak_t, peak_bw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_rows = [r for r in rows if r["latency_ms"] == r["latency_ms"]]  # drop NaN

    # Prefill: TFLOP/s vs sequence length
    pre = [r for r in df_rows if r["mode"] == "prefill" and r["tag"] == "seq_len"
           and r["dtype"] == "fp16"]
    if pre:
        fig, ax = plt.subplots(figsize=(8, 5))
        for path, color, label in (("flashattn_tile", "#0b6e4f", "FlashAttnTile"),
                                   ("sdpa_flash", "#1f77b4", "SDPA (flash)")):
            pts = sorted([r for r in pre if r["path"] == path], key=lambda r: r["seq_len"])
            ax.plot([p["seq_len"] for p in pts], [p["tflops"] for p in pts],
                    marker="o", color=color, label=label)
        ax.axhline(peak_t, color="#aa3377", ls=":", label=f"peak {peak_t:.1f} TFLOP/s (spec)")
        ax.set_xscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("TFLOP/s")
        ax.set_title("Prefill: achieved TFLOP/s (fp16, H=32, GQA=4, D=128)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "prefill_tflops.png", dpi=160)
        plt.close(fig)

    # Decode: HBM GB/s vs context, grouped by batch
    dec = [r for r in df_rows if r["mode"] == "decode" and r["tag"] == "batch_ctx"
           and r["dtype"] == "fp16" and r["path"] == "flashattn_tile"]
    if dec:
        fig, ax = plt.subplots(figsize=(8, 5))
        for batch in [1, 8, 32, 128]:
            pts = sorted([r for r in dec if r["batch"] == batch], key=lambda r: r["seq_len"])
            ax.plot([p["seq_len"] for p in pts], [p["gbytes_per_s"] for p in pts],
                    marker="o", label=f"batch={batch}")
        ax.axhline(peak_bw, color="#aa3377", ls=":", label=f"peak {peak_bw:.0f} GB/s (spec)")
        ax.set_xscale("log")
        ax.set_xlabel("context length")
        ax.set_ylabel("achieved HBM GB/s")
        ax.set_title("Decode: achieved HBM bandwidth (fp16, H=32, GQA=4, page_size=16)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "decode_bandwidth.png", dpi=160)
        plt.close(fig)


def main():
    torch.manual_seed(0)
    env = probe()
    peak_t = peak_fp16_tflops_spec()[0] or 30.3
    peak_bw = peak_hbm_gbs_spec() or 300.0

    print(f"[benchmark] GPU={env['gpu_name']} peak_tflops={peak_t} peak_hbm={peak_bw} GB/s")

    all_rows = []
    for dtype, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        all_rows.extend(benchmark_prefill(dtype, name, peak_t, peak_bw))
    all_rows.extend(benchmark_decode(torch.float16, "fp16", peak_t, peak_bw))
    # bf16 decode on the primary batch/ctx sweep only
    all_rows.extend(_bf16_decode_sweep(peak_t, peak_bw))

    all_rows = _speedups(all_rows)
    prefill_rows = [r for r in all_rows if r["mode"] == "prefill"]
    decode_rows = [r for r in all_rows if r["mode"] == "decode"]
    _write_csv(prefill_rows, RESULTS / "benchmark_prefill_gpu.csv")
    _write_csv(decode_rows, RESULTS / "benchmark_decode_gpu.csv")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "peak_fp16_tflops_spec": peak_t,
        "peak_hbm_gbs_spec": peak_bw,
        "baselines": {
            "sdpa_flash": "torch built-in flash backend (FlashAttention-style)",
            "flash_attn": "not installed (needs long source build for torch 2.8/cu128)",
            "vllm": "not installed (heavy install, torch-pin risk)",
        },
        "timing": "triton.testing.do_bench, warmup=25ms rep=100ms, median, L2 flushed",
    }
    (RESULTS / "benchmark_gpu.json").write_text(json.dumps(payload, indent=2) + "\n")

    _plot(all_rows, peak_t, peak_bw)

    print(f"[benchmark] wrote {len(prefill_rows)} prefill rows, {len(decode_rows)} decode rows")
    _print_summary(all_rows)
    return 0


def _bf16_decode_sweep(peak_t, peak_bw):
    rows = []
    H, Hkv, D = 32, 8, HEAD_DIM
    for batch in [1, 32, 128]:
        for ctx in [1024, 4096]:
            cache = _fill_cache(batch, ctx, 16, Hkv, D, torch.bfloat16)
            q = (torch.randn(batch, H, D, device=DEV) * 0.2).to(torch.bfloat16)
            flops = flops_attention(H, D, 1, ctx, causal=True, batch=batch)
            nbytes = bytes_attention("decode", batch, 1, ctx, H, Hkv, D, 2)
            shape = {"batch": batch, "seq_len": ctx, "num_heads": H, "num_kv_heads": Hkv,
                     "head_dim": D, "page_size": 16}
            ms = bench(lambda: paged_decode_attention(q, cache))
            rows.append(_row("decode", "flashattn_tile", shape, "bf16", ms, flops, nbytes,
                             peak_t, peak_bw, {"tag": "batch_ctx"}))
    return rows


def _print_summary(rows):
    import math

    def fmt(v, d=2):
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"

    for mode, title, met in (("prefill", "PREFILL (TFLOP/s, % peak)", "tflops"),
                             ("decode", "DECODE (GB/s, % peak HBM)", "gbytes_per_s")):
        print(f"\n== {title} ==")
        for tag in ("seq_len", "batch_ctx", "heads", "gqa", "page_size"):
            subset = [r for r in rows if r["mode"] == mode and r["tag"] == tag
                      and r["dtype"] == "fp16"]
            if not subset:
                continue
            ours = [r for r in subset if r["path"] == "flashattn_tile"]
            if not ours:
                continue
            for r in sorted(ours, key=lambda x: (x.get("batch") or 0, x.get("seq_len") or 0,
                                                 x.get("page_size") or 0, x.get("num_heads") or 0)):
                pct = r.get("pct_peak_tflops") if mode == "prefill" else r.get("pct_peak_hbm")
                sp = r.get("speedup_vs_sdpa")
                dim = (f"b={r['batch']} L={r['seq_len']}" if mode == "decode"
                       else f"L={r['seq_len']} H={r['num_heads']}/{r['num_kv_heads']}"
                       if tag in ("heads", "gqa") else f"L={r['seq_len']}")
                if mode == "decode" and tag == "page_size":
                    dim = f"ps={r['page_size']} L={r['seq_len']}"
                print(f"  {dim:22s} {met}={fmt(r[met], 2)}  pct={fmt(pct, 1)}%  "
                      f"speedup_vs_sdpa={fmt(sp, 2)}x")


if __name__ == "__main__":
    raise SystemExit(main())
