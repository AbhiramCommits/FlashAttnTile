"""Autotuning, analytic FLOP/byte accounting, and timing helpers.

Two principles keep the benchmark honest:

1. Configurations that would exceed the device's shared-memory budget are
   *filtered out analytically* (with the byte calculation exposed) instead of
   being launched to crash.
2. Every achieved-throughput number is ``analytic_work / measured_latency``.
   The FLOP and byte formulas live here so a reader can audit them.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

PACKAGE_DIR = Path(__file__).resolve().parent
TUNED_CONFIGS_PATH = PACKAGE_DIR / "tuned_configs.json"


# ---------------------------------------------------------------------------
# Analytic work
# ---------------------------------------------------------------------------
def _causal_pairs(q_len: int, kv_len: int) -> int:
    """Query/key pairs visible under bottom-right causal alignment."""
    if q_len <= 0 or kv_len <= 0:
        return 0
    if kv_len < q_len:
        raise ValueError("causal accounting requires kv_len >= q_len")
    return q_len * (kv_len - q_len) + q_len * (q_len + 1) // 2


def flops_attention(
    num_heads: int,
    head_dim: int,
    q_len: int,
    kv_len: int,
    causal: bool = False,
    batch: int = 1,
) -> int:
    """Analytic FLOPs for attention (QK^T + PV, 2 FLOPs per MAC).

    ``2 * H * D`` FLOPs per query/key pair for QK^T plus the same for PV, hence
    ``4 * H * D * pairs``. Softmax/exp work is excluded (small relative to the
    matmuls and not part of the roofline convention used here).
    """
    pairs = _causal_pairs(q_len, kv_len) if causal else q_len * kv_len
    return int(4 * batch * num_heads * head_dim * pairs)


def flops_attention_varlen(
    num_heads: int,
    head_dim: int,
    q_lens,
    kv_lens,
    causal: bool = False,
) -> int:
    """Sum :func:`flops_attention` over a ragged batch."""
    if len(q_lens) != len(kv_lens):
        raise ValueError("q_lens and kv_lens must have the same length")
    return int(
        sum(
            flops_attention(num_heads, head_dim, int(q), int(k), causal=causal)
            for q, k in zip(q_lens, kv_lens)
        )
    )


def bytes_attention(
    mode: str,
    batch: int,
    q_len: int,
    kv_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    elem_bytes: int = 2,
) -> int:
    """Analytic bytes moved through HBM by the fused kernel.

    ``decode`` reads ``q``, the whole K/V cache and writes ``o``; the page table
    is a few int32 per page and is included as ``4 * ceil(L / page_size) * 2``
    only when ``page_size`` is supplied through ``q_len``-independent terms.
    ``prefill`` reads packed ``q``, packed ``k``/``v`` and writes ``o``; scores
    never reach HBM.
    """
    if mode == "decode":
        if q_len != 1:
            raise ValueError("decode accounting assumes q_len == 1")
        q_bytes = batch * num_heads * head_dim * elem_bytes
        kv_bytes = batch * kv_len * num_kv_heads * head_dim * 2 * elem_bytes
        out_bytes = q_bytes
        return int(q_bytes + kv_bytes + out_bytes)
    if mode == "prefill":
        q_bytes = batch * q_len * num_heads * head_dim * elem_bytes
        kv_bytes = batch * kv_len * num_kv_heads * head_dim * 2 * elem_bytes
        out_bytes = q_bytes
        return int(q_bytes + kv_bytes + out_bytes)
    raise ValueError(f"unknown mode {mode!r}")


def arithmetic_intensity(flops: float, nbytes: float) -> float:
    if nbytes <= 0:
        return float("inf")
    return float(flops) / float(nbytes)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Config space
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TuneConfig:
    """One launch configuration, including its analytic shared-memory cost."""

    BLOCK_M: int
    BLOCK_N: int
    num_warps: int
    num_stages: int
    split_k: int
    smem_bytes: int

    def kernel_dict(self) -> dict:
        return {
            "BLOCK_M": self.BLOCK_M,
            "BLOCK_N": self.BLOCK_N,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "split_k": self.split_k,
        }


def prefill_smem_bytes(block_m: int, block_n: int, head_dim: int, elem_bytes: int,
                       num_stages: int) -> int:
    """Shared-memory estimate for the streaming prefill kernel.

    Q tile + pipelined K/V tiles + fp32 score tile + fp32 output accumulator.
    The estimate is conservative (it ignores register-resident pieces).
    """
    per_stage = (block_m * head_dim + 2 * block_n * head_dim) * elem_bytes
    scores = block_m * block_n * 4
    accumulator = block_m * head_dim * 4
    return int(num_stages * per_stage + scores + accumulator)


def decode_smem_bytes(page_size: int, head_dim: int, elem_bytes: int,
                      num_stages: int) -> int:
    """Shared-memory estimate for the paged decode kernel (one page per tile)."""
    per_stage = 2 * page_size * head_dim * elem_bytes
    scores = page_size * 4
    return int(num_stages * per_stage + scores)


@dataclass
class ConfigSpace:
    kind: str
    candidates: list[TuneConfig]
    smem_limit: int | None
    head_dim: int

    def __len__(self) -> int:
        return len(self.candidates)

    def feasible(self) -> list[TuneConfig]:
        if self.smem_limit is None:
            return list(self.candidates)
        return [c for c in self.candidates if c.smem_bytes <= self.smem_limit]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "head_dim": self.head_dim,
            "smem_limit": self.smem_limit,
            "enumerated": len(self.candidates),
            "feasible": len(self.feasible()),
            "candidates": [asdict(c) for c in self.candidates],
        }


def _device_smem_limit() -> int | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return int(
        getattr(props, "shared_memory_per_block_optin", None)
        or getattr(props, "shared_memory_per_block", 48 * 1024)
    )


def build_config_space(
    kind: str,
    head_dim: int,
    elem_bytes: int = 2,
    page_size: int = 16,
    smem_limit: int | None = None,
    block_m_choices=(32, 64, 128),
    block_n_choices=(32, 64, 128),
    num_warps_choices=(4, 8),
    num_stages_choices=(2, 3),
    split_k_choices=(1, 2, 4, 8),
) -> ConfigSpace:
    """Enumerate a candidate grid and tag each entry with its smem footprint."""
    if smem_limit is None:
        smem_limit = _device_smem_limit()
    configs: list[TuneConfig] = []
    if kind == "prefill":
        for bm in block_m_choices:
            for bn in block_n_choices:
                for warps in num_warps_choices:
                    for stages in num_stages_choices:
                        smem = prefill_smem_bytes(bm, bn, head_dim, elem_bytes, stages)
                        configs.append(TuneConfig(bm, bn, warps, stages, 1, smem))
    elif kind == "decode":
        if page_size & (page_size - 1):
            raise ValueError("page_size should be a power of two for decode tiling")
        for warps in num_warps_choices:
            for stages in num_stages_choices:
                for split_k in split_k_choices:
                    # Decode pages are fixed by the cache; BLOCK_M/BLOCK_N are
                    # not free parameters, so they carry the page geometry.
                    smem = decode_smem_bytes(page_size, head_dim, elem_bytes, stages)
                    configs.append(TuneConfig(1, page_size, warps, stages, split_k, smem))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return ConfigSpace(kind=kind, candidates=configs, smem_limit=smem_limit, head_dim=head_dim)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_callable(fn, warmup: int = 10, iters: int = 50) -> list[float]:
    """Return per-iteration latencies in milliseconds.

    CUDA events are used when the output of ``fn`` lives on a CUDA device; the
    CPU path uses ``time.perf_counter``.
    """
    if warmup < 1 or iters < 1:
        raise ValueError("warmup and iters must be >= 1")
    is_cuda = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing() is False
    times: list[float] = []
    if is_cuda:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
        return times

    for _ in range(warmup):
        fn()
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def summarize_times(times: list[float]) -> dict:
    if not times:
        raise ValueError("no timings")
    ordered = sorted(times)
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "min_ms": ordered[0],
        "p90_ms": ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)],
        "iters": len(times),
    }


# ---------------------------------------------------------------------------
# Autotuner
# ---------------------------------------------------------------------------
def autotune(
    problem: dict,
    candidates: list[TuneConfig] | ConfigSpace,
    warmup: int = 10,
    iters: int = 50,
    hbm_gbs: float | None = None,
    save: bool = True,
) -> "object":
    """Time every candidate and return a ranked :class:`pandas.DataFrame`.

    ``problem`` keys
        ``kind`` ("prefill"|"decode"), ``signature`` (stable shape string),
        ``runner`` (callable accepting a config dict), ``flops``, ``bytes``,
        optionally ``notes``.
    """
    import pandas as pd

    required = {"kind", "signature", "runner", "flops", "bytes"}
    missing = required - set(problem)
    if missing:
        raise ValueError(f"problem is missing keys: {sorted(missing)}")
    if isinstance(candidates, ConfigSpace):
        configs = candidates.feasible()
    else:
        configs = list(candidates)
    if not configs:
        raise ValueError("no feasible candidates")

    if hbm_gbs is None:
        from .env import measured_hbm_bandwidth_gbs

        hbm_gbs = measured_hbm_bandwidth_gbs()

    rows = []
    for cfg in configs:
        runner_cfg = cfg.kernel_dict()
        times = time_callable(lambda: problem["runner"](runner_cfg), warmup=warmup, iters=iters)
        stats = summarize_times(times)
        latency = stats["median_ms"]
        tflops = problem["flops"] / (latency * 1e-3) / 1e12
        gbs = problem["bytes"] / (latency * 1e-3) / 1e9
        rows.append(
            {
                "BLOCK_M": cfg.BLOCK_M,
                "BLOCK_N": cfg.BLOCK_N,
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
                "split_k": cfg.split_k,
                "smem_bytes": cfg.smem_bytes,
                "latency_ms": latency,
                "min_ms": stats["min_ms"],
                "p90_ms": stats["p90_ms"],
                "tflops": tflops,
                "gbytes_per_s": gbs,
                "pct_measured_hbm": (100.0 * gbs / hbm_gbs) if hbm_gbs else None,
                "arithmetic_intensity": arithmetic_intensity(problem["flops"], problem["bytes"]),
                "flops": problem["flops"],
                "bytes": problem["bytes"],
            }
        )

    df = pd.DataFrame(rows).sort_values("latency_ms", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    if save and len(df):
        winner = df.iloc[0]
        save_best(
            problem["kind"],
            problem["signature"],
            {
                "BLOCK_M": int(winner.BLOCK_M),
                "BLOCK_N": int(winner.BLOCK_N),
                "num_warps": int(winner.num_warps),
                "num_stages": int(winner.num_stages),
                "split_k": int(winner.split_k),
            },
            {
                "latency_ms": float(winner.latency_ms),
                "tflops": float(winner.tflops),
                "gbytes_per_s": float(winner.gbytes_per_s),
                "smem_bytes": int(winner.smem_bytes),
            },
        )
    return df


def _load_store(path: Path | None = None) -> dict:
    path = path or TUNED_CONFIGS_PATH
    if not path.exists():
        return {"_meta": {"schema_version": 1}, "configs": {}}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"_meta": {"schema_version": 1}, "configs": {}}
    data.setdefault("configs", {})
    data.setdefault("_meta", {"schema_version": 1})
    return data


def save_best(kind: str, signature: str, config: dict, metrics: dict | None = None) -> None:
    """Persist the winning config for one problem shape."""
    data = _load_store()
    data["configs"][f"{kind}:{signature}"] = {
        "kind": kind,
        "signature": signature,
        "config": config,
        "metrics": metrics or {},
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
    }
    data["_meta"]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    TUNED_CONFIGS_PATH.write_text(json.dumps(data, indent=2) + "\n")


def best_config(kind: str, signature: str, path: Path | None = None) -> dict | None:
    """Look up a persisted winning config; ``None`` when nothing was tuned."""
    entry = _load_store(path)["configs"].get(f"{kind}:{signature}")
    return entry["config"] if entry else None


def initialize_store(note: str) -> None:
    """Create/refresh the tuned-config store with an explanatory meta block."""
    data = _load_store()
    data["_meta"].update(
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "note": note,
        }
    )
    TUNED_CONFIGS_PATH.write_text(json.dumps(data, indent=2) + "\n")
