#!/usr/bin/env python
"""Probe the host and write ``results/env.json``.

This is the only script CI needs to run to prove the package entrypoint works.
It never fabricates GPU numbers: on CPU-only hosts the CUDA fields are null and
``has_gpu`` is false.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashattn_tile.env import probe  # noqa: E402


def main() -> int:
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    info = probe()
    out = results / "env.json"
    out.write_text(json.dumps(info, indent=2) + "\n")

    print(f"torch          : {info['torch_version']} (CUDA build: {info['torch_cuda_version']})")
    print(f"python         : {info['python_version']} on {info['platform']}")
    print(f"triton         : {info['triton_version'] or 'not installed'} (importable: {info['triton_available']})")
    print(f"CUDA available : {info['cuda_available']}")
    if info["has_gpu"]:
        print(f"GPU            : {info['gpu_name']}")
        print(f"compute cap.   : {info['compute_capability']}")
        print(f"SMs            : {info['sm_count']}")
        print(f"HBM            : {info['total_hbm_gib']} GiB")
        print(f"L2             : {info['l2_cache_bytes']} bytes")
        print(f"max smem/block : {info['max_shared_memory_per_block']} bytes")
        print(f"measured HBM   : {info['measured_hbm_bandwidth_gbs']:.1f} GB/s")
        print(
            f"spec FP16 TC   : {info['peak_fp16_tflops_spec']} TFLOP/s "
            f"[{info['peak_fp16_tflops_source']}]"
        )
    else:
        print(f"GPU            : none (CPU: {info['processor']}, {info['cpu_physical_count']} threads)")
        print("HBM bandwidth  : not measured (no CUDA device)")
        print(
            "spec FP16 TC   : not applicable "
            f"[{info['peak_fp16_tflops_source']}]"
        )
    print(f"host mem BW    : {info['measured_host_bandwidth_gbs']:.1f} GB/s (CPU copy, measured)")
    print(f"wrote          : {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
