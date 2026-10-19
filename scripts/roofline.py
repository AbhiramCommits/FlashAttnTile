#!/usr/bin/env python
"""Roofline plot and markdown summary from ``results/benchmarks.json``.

The plot is generated from measured latencies only. On a CPU host the roof line
uses the *measured CPU copy bandwidth* and the scatter shows PyTorch reference
timings; the GPU memory/compute roofs are simply absent. On a CUDA host the roof
uses measured HBM bandwidth and the spec-sheet FP16 ceiling (labelled as spec).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _fmt(value, digits=3, missing="n/a"):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return missing
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load():
    path = RESULTS / "benchmarks.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/benchmark.py first")
    return json.loads(path.read_text())


def _environment(payload: dict) -> dict:
    """Prefer the dedicated env artifact so README and summary cite one file."""
    path = RESULTS / "env.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return payload["environment"]


def _plot(payload: dict) -> dict:
    env = _environment(payload)
    rows = [r for m in payload["modes"].values() for r in m.get("rows", [])]
    if not rows:
        raise SystemExit("benchmarks.json has no rows")
    df = pd.DataFrame(rows)
    status = payload["status"]
    fig, ax = plt.subplots(figsize=(9, 6))

    styles = {
        "flashattn_tile_triton": ("o", "#0b6e4f", "FlashAttnTile (Triton)"),
        "flashattn_tile_cpu_reference": ("o", "#0b6e4f", "FlashAttnTile CPU reference"),
        "sdpa": ("s", "#1f77b4", "PyTorch SDPA"),
        "flash_attn": ("^", "#d62728", "FlashAttention-2"),
        "flashinfer": ("D", "#9467bd", "FlashInfer"),
        "native_cuda": ("x", "#8c564b", "CUDA C++ baseline"),
    }
    for path, group in df.groupby("path"):
        marker, color, label = styles.get(path, ("o", "gray", path))
        ax.scatter(
            group["arithmetic_intensity"], group["tflops"] * 1000.0,
            marker=marker, color=color, label=label, alpha=0.75, s=46,
        )

    ai = df["arithmetic_intensity"].replace(0, float("nan")).dropna()
    if len(ai):
        xs = [ai.min() / 2, ai.max() * 2]
    else:
        xs = [0.1, 100]

    if status == "ok":
        bw = env.get("measured_hbm_bandwidth_gbs")
        label = f"measured HBM roof ({_fmt(bw, 1)} GB/s)"
    else:
        bw = env.get("measured_host_bandwidth_gbs")
        label = f"measured CPU copy-bandwidth roof ({_fmt(bw, 1)} GB/s)"
    if bw:
        ax.plot(xs, [bw * x for x in xs], "--", color="#444444", lw=1.6, label=label)

    if status == "ok":
        peak = env.get("peak_fp16_tflops_spec")
        if peak:
            ax.axhline(peak * 1000.0, color="#aa3377", ls=":", lw=1.6,
                       label=f"spec FP16 tensor-core roof ({_fmt(peak, 1)} TFLOP/s)")
        ax.set_ylabel("achieved GFLOP/s")
        title = "Roofline: measured GPU attention kernels"
    else:
        ax.set_ylabel("achieved GFLOP/s (CPU reference path)")
        title = "CPU-only: PyTorch reference timings, NO GPU kernel throughput"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (analytic FLOPs / byte)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = RESULTS / "roofline.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    data = RESULTS / "roofline_data.csv"
    df[[
        "mode", "path", "device", "batch", "seq_len", "num_heads", "num_kv_heads",
        "head_dim", "arithmetic_intensity", "tflops", "gbytes_per_s", "latency_ms",
    ]].to_csv(data, index=False)
    return {"png": out, "csv": data, "rows": len(df)}


def _path_pivot(rows):
    """Build one record per (batch, seq_len) with latency per backend."""
    table: dict[tuple, dict] = {}
    for row in rows:
        key = (row["batch"], row["seq_len"])
        entry = table.setdefault(
            key,
            {"batch": row["batch"], "seq_len": row["seq_len"], "paths": {}},
        )
        entry["paths"][row["path"]] = row
    return [table[k] for k in sorted(table)]


def _mode_table(payload: dict, mode: str) -> str:
    mode_payload = payload["modes"].get(mode)
    if not mode_payload:
        return f"_No `{mode}` results in the payload._\n"
    rows = mode_payload.get("rows", [])
    if not rows:
        return f"_No rows for `{mode}`._\n"
    status = payload["status"]
    ours_path = "flashattn_tile_triton" if status == "ok" else "flashattn_tile_cpu_reference"
    entries = _path_pivot(rows)
    if status == "ok":
        cols = ["batch", "ctx" if mode == "decode" else "seq", "ours p50 (ms)", "SDPA p50 (ms)"]
        if mode == "decode":
            cols.append("SDPA+gather (ms)")
        cols += ["speedup vs SDPA", "ours TFLOP/s", "% measured HBM", "max abs vs ref"]
    else:
        # CPU medians are shown next to the minimum so host noise is visible.
        cols = ["batch", "ctx" if mode == "decode" else "seq", "ours min (ms)",
                "ours median (ms)", "ours p90 (ms)", "SDPA min (ms)"]
        if mode == "decode":
            cols.append("SDPA+gather (ms)")
        cols += ["min-based speedup vs SDPA", "ours TFLOP/s", "max abs vs ref"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for entry in entries:
        ours = entry["paths"].get(ours_path)
        sdpa = entry["paths"].get("sdpa")
        if not ours:
            continue
        speedup = sdpa["latency_ms"] / ours["latency_ms"] if sdpa else None
        if status == "ok":
            cells = [
                str(entry["batch"]),
                str(entry["seq_len"]),
                _fmt(ours["latency_ms"]),
                _fmt(sdpa["latency_ms"]) if sdpa else "n/a",
            ]
            if mode == "decode":
                cells.append(_fmt(sdpa.get("sdpa_e2e_ms"), 3) if sdpa else "n/a")
            cells += [
                _fmt(speedup, 2) + "x" if speedup else "n/a",
                _fmt(ours["tflops"], 4),
                _fmt(ours.get("pct_measured_hbm"), 1) + "%"
                if ours.get("pct_measured_hbm") is not None else "n/a",
                _fmt(ours.get("max_abs_vs_ref"), 4, "not computed"),
            ]
        else:
            cells = [
                str(entry["batch"]),
                str(entry["seq_len"]),
                _fmt(ours["latency_ms"]),
                _fmt(ours.get("median_ms")),
                _fmt(ours.get("p90_ms")),
                _fmt(sdpa["latency_ms"]) if sdpa else "n/a",
            ]
            if mode == "decode":
                cells.append(_fmt(sdpa.get("sdpa_e2e_ms"), 3) if sdpa else "n/a")
            cells += [
                _fmt(speedup, 2) + "x" if speedup else "n/a",
                _fmt(ours["tflops"], 4),
                _fmt(ours.get("max_abs_vs_ref"), 4, "not computed"),
            ]
        lines.append("| " + " | ".join(cells) + " |")

    # Optional competitor rows that exist for at least one shape.
    for path, label in (("flash_attn", "FlashAttention-2"), ("flashinfer", "FlashInfer"),
                        ("native_cuda", "CUDA C++ baseline")):
        lat = [e["paths"][path]["latency_ms"] for e in entries if path in e["paths"]]
        if lat:
            lines.append("")
            lines.append(f"{label} latency (ms), where available: " + ", ".join(_fmt(x) for x in lat))
    return "\n".join(lines) + "\n"


def _numerics_table() -> str:
    path = RESULTS / "numerics.json"
    if not path.exists():
        return "_`results/numerics.json` not produced yet._\n"
    data = json.loads(path.read_text())
    rows = data.get("records", [])
    if not rows:
        return "_No numerics records._\n"
    lines = [
        "| path | dtype | max abs | mean abs | max rel | tolerance | passed | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {path} | {dtype} | {max_abs} | {mean_abs} | {max_rel} | {tol} | {passed} | {note} |".format(
                path=r.get("path"),
                dtype=r.get("dtype"),
                max_abs=_fmt(r.get("max_abs"), 6),
                mean_abs=_fmt(r.get("mean_abs"), 6),
                max_rel=_fmt(r.get("max_rel"), 6),
                tol=_fmt(r.get("tolerance"), 4),
                passed=("skipped" if r.get("status") == "skipped"
                        else ("yes" if r.get("passed") else "no")),
                note=r.get("note", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _summary(payload: dict, plot_info: dict) -> str:
    env = _environment(payload)
    status = payload["status"]
    lines = [
        "# FlashAttnTile benchmark summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}  ",
        f"Artifact generated: {payload.get('generated_utc', 'unknown')}  ",
        f"Status: **{payload['status']}**"
        + (f" — {payload['reason']}" if payload.get("reason") else ""),
        "",
    ]
    if status == "ok":
        lines += [
            "## Hardware",
            "",
            f"- GPU: **{env.get('gpu_name')}**, compute capability {env.get('compute_capability')}, "
            f"{env.get('sm_count')} SMs",
            f"- HBM: {env.get('total_hbm_gib')} GiB; measured copy bandwidth "
            f"**{_fmt(env.get('measured_hbm_bandwidth_gbs'), 1)} GB/s**",
            f"- FP16 tensor-core ceiling: {_fmt(env.get('peak_fp16_tflops_spec'), 1)} TFLOP/s "
            f"({env.get('peak_fp16_tflops_source')}) — *spec sheet, not measured*",
            "",
        ]
    else:
        lines += [
            "## Hardware",
            "",
            "**CPU-only, no CUDA device detected.** No GPU kernel throughput, HBM bandwidth, "
            "or FlashAttention/FlashInfer comparison was collected on this machine.",
            f"- CPU: {env.get('processor')} ({env.get('cpu_count')} hardware threads)",
            f"- Measured CPU copy bandwidth: "
            f"{_fmt(env.get('measured_host_bandwidth_gbs'), 1)} GB/s (host DRAM, measured)",
            f"- Torch {env.get('torch_version')}, Triton importable: {env.get('triton_available')}",
            f"- Host load average at probe time: {env.get('load_average')}",
            "- Timings: CPU headline latency is the **minimum** of the measured iterations "
            "(the build host was shared and heavily loaded); median/p90 are in the CSV. "
            "These are reference-path timings, not GPU kernel throughput.",
            "",
        ]

    lines += ["## Decode", "", _mode_table(payload, "decode"), ""]
    lines += ["## Prefill", "", _mode_table(payload, "prefill"), ""]

    lines += ["## Autotuning", ""]
    autotune = payload.get("autotuning", {})
    any_performed = any(v.get("performed") for v in autotune.values())
    if any_performed:
        for mode, info in autotune.items():
            if not info.get("performed"):
                continue
            space = info.get("search_space", {})
            for summary in space.get("summaries", []):
                lines.append(
                    f"- `{mode}` {summary['signature']}: {summary['enumerated']} enumerated, "
                    f"{summary['feasible']} feasible; winner `{summary['winner']}`; "
                    f"best {_fmt(summary['best_ms'], 4)} ms, worst feasible "
                    f"{_fmt(summary['worst_feasible_ms'], 4)} ms ({_fmt(summary['spread_x'], 2)}x)"
                )
    else:
        space = autotune.get("prefill", {}).get("search_space", {}).get("prefill", {})
        decode_space = autotune.get("decode", {}).get("search_space", {}).get("decode", {})
        lines.append(
            "No GPU autotuning was executed. The analytic config spaces contain "
            f"{space.get('enumerated', 0)} prefill and {decode_space.get('enumerated', 0)} decode "
            "candidates; shared-memory feasibility can only be evaluated against a real device. "
            "`flashattn_tile/tuned_configs.json` is therefore empty."
        )
    lines.append("")

    occ = payload.get("occupancy", {})
    lines += ["## Occupancy", ""]
    if occ.get("method") == "ncu":
        lines.append("**Profiled with ncu.** Raw output: `results/raw/ncu_decode.txt`.")
        for key, value in occ.get("metrics", {}).items():
            lines.append(f"- `{key}`: {value}")
    elif occ.get("method") == "analytic":
        lines.append(
            "**Analytic upper bound, not profiled** (ncu unavailable). "
            f"Estimated occupancy: {_fmt(occ.get('occupancy_pct'), 1)}%; "
            f"smem/block {occ.get('smem_per_block')} B, smem/SM {occ.get('smem_per_sm')} B. "
            "Register usage is unknown."
        )
    else:
        lines.append(f"**Not collected.** {occ.get('reason')}")
    lines.append("")

    lines += ["## Numerics", "", _numerics_table(), ""]
    test_file = RESULTS / "tests.txt"
    lines += ["## Test suite", ""]
    if test_file.exists():
        import re

        text = test_file.read_text()
        pytest_line = next(
            (ln.strip() for ln in reversed(text.splitlines())
             if re.search(r"\d+ passed", ln)),
            None,
        )
        collect_line = next(
            (ln.strip() for ln in reversed(text.splitlines())
             if "tests collected" in ln),
            None,
        )
        lines.append(f"- pytest: `{pytest_line or 'not recorded'}`")
        lines.append(f"- collection: `{collect_line or 'not recorded'}`")
        lines.append(f"- ruff: `{'clean' if 'All checks passed!' in text else 'see results/tests.txt'}`")
    else:
        lines.append("_No recorded test run._")
    lines.append("")
    lines += [
        "## Roofline",
        "",
        f"![roofline]({Path(plot_info['png']).name})",
        "",
        f"- Plot points: {plot_info['rows']} measured rows; data in `roofline_data.csv`.",
        "- Arithmetic intensity and FLOPs use the analytic formulas in "
        "`flashattn_tile/autotune.py`; GPU latency is the median of measured CUDA-event "
        "iterations, CPU latency is the minimum (see the hardware note).",
        "- The roof uses DRAM bandwidth. Points above it can occur for small working sets "
        "that fit in CPU/GPU cache; the DRAM roof is not a hard ceiling for cache-resident data.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    payload = _load()
    plot_info = _plot(payload)
    summary = _summary(payload, plot_info)
    out = RESULTS / "summary.md"
    out.write_text(summary + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"wrote {Path(plot_info['png']).relative_to(ROOT)}")
    print(f"wrote {Path(plot_info['csv']).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
