#!/usr/bin/env python
"""Autotune every benchmarked shape on the GPU and persist the winners.

For each (kind, signature) that ``scripts/benchmark_gpu.py`` measured, this
times the built-in default config and every shared-memory-feasible candidate in
the analytic config space with ``triton.testing.do_bench`` (warmup + median,
L2 flushed), then writes the winning config into ``tuned_configs.json`` via
``save_best`` and reports the speedup of tuned vs default.

The kernel wrappers read ``tuned_configs.json`` automatically on later calls
with ``config=None``, so running this script is what makes the persisted
"winning config" claim true.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triton.testing import do_bench  # noqa: E402

from flashattn_tile.autotune import build_config_space, initialize_store, save_best  # noqa: E402
from flashattn_tile.kernels import decode, prefill  # noqa: E402
from flashattn_tile.paging import allocate  # noqa: E402

RESULTS = ROOT / "results"
DEV = "cuda"
D = 128


def bench(fn) -> float:
    return do_bench(fn, warmup=25, rep=100, return_mode="median")


def _fill_cache(batch, ctx, page_size, kv_heads, dtype):
    cache = allocate(batch, ctx, kv_heads, D, page_size, dtype=dtype, device=DEV)
    with torch.no_grad():
        cache.k_cache.normal_(0.0, 0.2)
        cache.v_cache.normal_(0.0, 0.2)
    cache.seq_lens[:] = ctx
    return cache


# ---------------------------------------------------------------------------
# Shape grids (dtype-independent signatures, matching scripts/benchmark_gpu.py)
# ---------------------------------------------------------------------------
def prefill_shapes():
    shapes = []
    for seq in [512, 1024, 2048, 4096, 8192]:
        shapes.append((f"B1_H32_Hkv8_D128_Lq{seq}_Lk{seq}_c1", seq, 32, 8))
    for heads in [8, 16, 64]:  # 32 already covered
        shapes.append((f"B1_H{heads}_Hkv{heads // 4}_D128_Lq2048_Lk2048_c1", 2048, heads, heads // 4))
    for kv in [32, 1]:  # 8 already covered
        shapes.append((f"B1_H32_Hkv{kv}_D128_Lq2048_Lk2048_c1", 2048, 32, kv))
    return shapes


def decode_shapes():
    shapes = []
    for batch in [1, 8, 32, 128]:
        for ctx in [128, 1024, 4096, 16384]:
            shapes.append((f"B{batch}_H32_Hkv8_D128_P16_L{ctx}_sk1", batch, ctx, 32, 8, 16))
    for ps in [8, 32, 64]:  # 16 already covered
        shapes.append((f"B32_H32_Hkv8_D128_P{ps}_L4096_sk1", 32, 4096, 32, 8, ps))
    for heads in [8, 64]:  # 32 already covered
        shapes.append((f"B32_H{heads}_Hkv{heads // 4}_D128_P16_L4096_sk1", 32, 4096, heads, heads // 4, 16))
    return shapes


def _pick(rows):
    return min(rows, key=lambda r: r["ms"])


def autotune_prefill(sig, seq, heads, kv_heads):
    q = (torch.randn(seq, heads, D, device=DEV) * 0.2).to(torch.float16)
    k = (torch.randn(seq, kv_heads, D, device=DEV) * 0.2).to(torch.float16)
    v = (torch.randn(seq, kv_heads, D, device=DEV) * 0.2).to(torch.float16)
    cu = torch.tensor([0, seq], dtype=torch.int32, device=DEV)

    def run(cfg):
        return prefill.prefill_attention(
            q, k, v, cu, cu, seq, seq, causal=True,
            config={kk: cfg[kk] for kk in ("BLOCK_M", "BLOCK_N", "num_warps", "num_stages")},
        )

    default_ms = bench(lambda: run(prefill.DEFAULT_CONFIG))
    space = build_config_space("prefill", D, 2).feasible()
    rows = [{"cfg": c.kernel_dict(), "ms": bench(lambda: run(c.kernel_dict()))} for c in space]
    winner = _pick(rows)
    save_best("prefill", sig, winner["cfg"], {"latency_ms": winner["ms"],
                                              "default_ms": default_ms})
    return default_ms, winner, len(space)


def autotune_decode(sig, batch, ctx, heads, kv_heads, page_size):
    cache = _fill_cache(batch, ctx, page_size, kv_heads, torch.float16)
    q = (torch.randn(batch, heads, D, device=DEV) * 0.2).to(torch.float16)

    def run(cfg):
        return decode.paged_decode_attention(
            q, cache,
            config={kk: cfg[kk] for kk in ("split_k", "num_warps", "num_stages")},
        )

    default_ms = bench(lambda: run(decode.DEFAULT_CONFIG))
    space = build_config_space("decode", D, 2, page_size).feasible()
    rows = [{"cfg": c.kernel_dict(), "ms": bench(lambda: run(c.kernel_dict()))} for c in space]
    winner = _pick(rows)
    save_best("decode", sig, winner["cfg"], {"latency_ms": winner["ms"],
                                             "default_ms": default_ms})
    return default_ms, winner, len(space)


def main():
    torch.manual_seed(0)
    import torch as _torch

    gpu = _torch.cuda.get_device_name(0)
    initialize_store(
        f"Autotuned on {gpu} with triton.testing.do_bench (warmup=25ms rep=100ms, "
        "median, L2 flushed). Winners are keyed by the kernel signature strings in "
        "flashattn_tile/kernels/*.py; wrappers read them automatically."
    )
    records = []

    for sig, seq, heads, kv_heads in prefill_shapes():
        default_ms, winner, n = autotune_prefill(sig, seq, heads, kv_heads)
        gain = default_ms / winner["ms"]
        records.append({
            "kind": "prefill", "signature": sig, "default_ms": default_ms,
            "winner_ms": winner["ms"], "gain_vs_default": gain,
            "num_candidates": n, "winner": winner["cfg"],
        })
        print(f"prefill {sig:40s} default={default_ms:.3f}ms winner={winner['ms']:.3f}ms "
              f"gain={gain:.2f}x cfg={winner['cfg']}")

    for sig, batch, ctx, heads, kv_heads, ps in decode_shapes():
        default_ms, winner, n = autotune_decode(sig, batch, ctx, heads, kv_heads, ps)
        gain = default_ms / winner["ms"]
        records.append({
            "kind": "decode", "signature": sig, "default_ms": default_ms,
            "winner_ms": winner["ms"], "gain_vs_default": gain,
            "num_candidates": n, "winner": winner["cfg"],
        })
        print(f"decode {sig:40s} default={default_ms:.3f}ms winner={winner['ms']:.3f}ms "
              f"gain={gain:.2f}x cfg={winner['cfg']}")

    # Persist a human-readable report alongside the config store.
    fieldnames = ["kind", "signature", "default_ms", "winner_ms", "gain_vs_default",
                  "num_candidates", "winner"]
    with (RESULTS / "autotune_gpu.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()})

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "timing": "triton.testing.do_bench, warmup=25ms rep=100ms, median, L2 flushed",
        "records": [{k: (v if not isinstance(v, dict) else v) for k, v in r.items()}
                    for r in records],
    }
    (RESULTS / "autotune_gpu.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Summary of gain ranges
    p_gain = [r["gain_vs_default"] for r in records if r["kind"] == "prefill"]
    d_gain = [r["gain_vs_default"] for r in records if r["kind"] == "decode"]
    print(f"\nprefill: {len(p_gain)} shapes, gain {min(p_gain):.2f}x..{max(p_gain):.2f}x "
          f"(geomean {_geomean(p_gain):.2f}x)")
    print(f"decode : {len(d_gain)} shapes, gain {min(d_gain):.2f}x..{max(d_gain):.2f}x "
          f"(geomean {_geomean(d_gain):.2f}x)")
    print("wrote results/autotune_gpu.csv, results/autotune_gpu.json, updated tuned_configs.json")
    return 0


def _geomean(xs):
    import math
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


if __name__ == "__main__":
    raise SystemExit(main())
