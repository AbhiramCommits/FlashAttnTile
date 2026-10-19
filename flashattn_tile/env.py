"""Environment probing and microbenchmarks.

Two kinds of numbers live here and they are deliberately kept apart:

``spec`` numbers
    Vendor datasheet values from :data:`PEAK_FP16_SPEC_TFLOPS`. Every value
    returned by :func:`peak_fp16_tflops_spec` carries a source string so the
    README can label it as a *spec sheet* number rather than a measurement.

``measured`` numbers
    :func:`measured_hbm_bandwidth_gbs` (device-to-device copy, CUDA events) and
    :func:`measured_host_bandwidth_gbs` (CPU-to-CPU copy, wall clock). These are
    measured on the machine that runs this module and must never be replaced by
    a marketing number.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
import time
from datetime import datetime, timezone

import torch

# Dense FP16 tensor-core throughput from vendor product pages, *not* measured.
# Compute capability granularity is coarser than the product stack (e.g. H100
# SXM vs PCIe share 9.0), so the source string says so explicitly.
PEAK_FP16_SPEC_TFLOPS: dict[tuple[int, int], tuple[float, str]] = {
    (7, 0): (125.0, "NVIDIA V100 datasheet, dense FP16 tensor core, no sparsity"),
    (8, 0): (312.0, "NVIDIA A100 SXM 80GB datasheet, dense FP16 tensor core, no sparsity"),
    (8, 6): (125.0, "NVIDIA A10 datasheet, dense FP16 tensor core, no sparsity"),
    (8, 9): (362.05, "NVIDIA L40S datasheet, dense FP16 tensor core, no sparsity"),
    (9, 0): (989.4, "NVIDIA H100 SXM datasheet, dense FP16 tensor core, no sparsity; "
                    "SXM variant assumed for compute capability 9.0"),
    (10, 0): (1650.0, "NVIDIA H200 SXM datasheet, dense FP16 tensor core, no sparsity"),
}


def _triton_version() -> str | None:
    try:
        return importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return None


def triton_available() -> bool:
    """True when ``triton`` can actually be imported on this machine."""
    try:
        import triton  # noqa: F401
    except Exception:  # pragma: no cover - depends on host
        return False
    return True


def device_info() -> dict:
    """Return a JSON-serialisable description of the compute environment.

    CUDA-specific fields are ``None`` when no GPU is present; ``has_gpu`` is the
    single flag downstream code should branch on.
    """
    cuda = torch.cuda.is_available()
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "triton_version": _triton_version(),
        "triton_available": triton_available(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cuda_available": cuda,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "has_gpu": bool(cuda),
        "device_count": torch.cuda.device_count() if cuda else 0,
        "gpu_name": None,
        "compute_capability": None,
        "sm_count": None,
        "total_hbm_bytes": None,
        "total_hbm_gib": None,
        "l2_cache_bytes": None,
        "max_shared_memory_per_block": None,
        "max_shared_memory_per_multiprocessor": None,
        "regs_per_multiprocessor": None,
        "max_threads_per_multiprocessor": None,
        "cpu_count": torch.get_num_threads(),
        "cpu_physical_count": __import__("os").cpu_count(),
    }
    if cuda:
        p = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{p.major}.{p.minor}",
                "sm_count": int(p.multi_processor_count),
                "total_hbm_bytes": int(p.total_memory),
                "total_hbm_gib": round(p.total_memory / 1024**3, 3),
                "l2_cache_bytes": int(getattr(p, "L2_cache_size", 0)),
                "max_shared_memory_per_block": int(getattr(p, "shared_memory_per_block", 0)),
                "max_shared_memory_per_multiprocessor": int(
                    getattr(p, "shared_memory_per_multiprocessor", 0)
                ),
                "regs_per_multiprocessor": int(getattr(p, "regs_per_multiprocessor", 0)),
                "max_threads_per_multiprocessor": int(
                    getattr(p, "max_threads_per_multiprocessor", 0)
                ),
            }
        )
    return info


def _bandwidth_gbs(num_bytes_moved: int, seconds: float) -> float:
    return (num_bytes_moved / seconds) / 1e9


def measured_hbm_bandwidth_gbs(size_mb: int = 512, iters: int = 20) -> float | None:
    """Measure device-to-device copy bandwidth with CUDA events.

    Returns ``None`` when no CUDA device exists -- callers must not substitute a
    spec-sheet number. Each iteration copies ``size_mb`` MiB and therefore moves
    ``2 * size_mb`` MiB (one read + one write).
    """
    if not torch.cuda.is_available():
        return None
    if size_mb <= 0 or iters <= 0:
        raise ValueError("size_mb and iters must be positive")

    element_bytes = 4
    n_elems = (size_mb * 1024 * 1024) // element_bytes
    src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    src.normal_()

    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()

    total_bytes = 0
    elapsed = 0.0
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src)
        end.record()
        end.synchronize()
        elapsed += start.elapsed_time(end) / 1000.0
        total_bytes += 2 * n_elems * element_bytes

    del src, dst
    torch.cuda.empty_cache()
    return _bandwidth_gbs(total_bytes, elapsed)


def measured_host_bandwidth_gbs(size_mb: int = 256, iters: int = 10) -> float | None:
    """Measure CPU copy bandwidth (``dst.copy_(src)``) with a wall clock.

    This is a *host* number. It is useful for CPU-only roofline plots and must be
    labelled as host memory bandwidth, never as HBM bandwidth.
    """
    if size_mb <= 0 or iters <= 0:
        raise ValueError("size_mb and iters must be positive")
    n_elems = (size_mb * 1024 * 1024) // 4
    src = torch.empty(n_elems, dtype=torch.float32)
    dst = torch.empty_like(src)
    src.normal_()

    for _ in range(2):
        dst.copy_(src)

    total_bytes = 0
    start = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
        total_bytes += 2 * n_elems * 4
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return None
    return _bandwidth_gbs(total_bytes, elapsed)


def peak_fp16_tflops_spec() -> tuple[float | None, str]:
    """Return ``(tflops, source)`` from the spec table for the current GPU.

    On a CPU-only host this returns ``(None, reason)``. Never use the returned
    value as if it were measured; it is a datasheet ceiling.
    """
    if not torch.cuda.is_available():
        return None, "no CUDA device detected; spec table not applicable"
    p = torch.cuda.get_device_properties(0)
    key = (int(p.major), int(p.minor))
    if key not in PEAK_FP16_SPEC_TFLOPS:
        return None, (
            f"compute capability {key[0]}.{key[1]} ({torch.cuda.get_device_name(0)}) "
            "is not in the spec table; no value reported"
        )
    value, source = PEAK_FP16_SPEC_TFLOPS[key]
    return value, source


def probe(include_bandwidth: bool = True) -> dict:
    """Full environment probe used by ``scripts/probe_env.py`` and the README."""
    info = device_info()
    if include_bandwidth:
        info["measured_hbm_bandwidth_gbs"] = measured_hbm_bandwidth_gbs()
        info["measured_host_bandwidth_gbs"] = measured_host_bandwidth_gbs()
        tflops, source = peak_fp16_tflops_spec()
        info["peak_fp16_tflops_spec"] = tflops
        info["peak_fp16_tflops_source"] = source
    return info


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(probe(), indent=2))
