#!/usr/bin/env python
"""End-to-end autoregressive decode loop through the compiled custom op.

Builds a :class:`PagedKVCache`, prefills a prompt, then runs N decode steps. Each
step calls ``torch.compile``d ``flashattn_tile::paged_decode`` and checks the
output against the fp32 reference on the *same* cache state before appending the
next token. Writes ``results/demo_decode_loop.txt``.

On a CPU-only host the custom op executes the reference fallback; the timings
are therefore reference-path timings and are labelled as such in the artifact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashattn_tile.env import probe  # noqa: E402
from flashattn_tile.ops import ops_available  # noqa: E402
from flashattn_tile.paging import allocate, append_tokens  # noqa: E402
from flashattn_tile.reference import ref_attention_paged  # noqa: E402

OPS_AVAILABLE = ops_available()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "demo_decode_loop.txt")
    args = parser.parse_args()

    torch.manual_seed(7)
    env = probe(include_bandwidth=False)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    scale = args.head_dim ** -0.5

    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    log("FlashAttnTile decode-loop demo")
    log(f"generated_utc: {datetime.now(timezone.utc).isoformat()}")
    log(f"torch {env['torch_version']} | device={device} | dtype={dtype}")
    log(
        "GPU: " + (env["gpu_name"] if env["has_gpu"] else
                    "none (CUDA unavailable; reference fallback)")
    )
    log(f"custom ops registered: {OPS_AVAILABLE}")
    if not OPS_AVAILABLE:
        log("ERROR: custom ops did not register")
        return 1

    cache = allocate(
        batch=args.batch,
        max_seq_len=args.prompt_len + args.steps + 1,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        dtype=dtype,
        device=device,
    )
    for _ in range(args.prompt_len):
        append_tokens(
            cache,
            torch.randn(args.batch, args.num_kv_heads, args.head_dim, device=device, dtype=dtype),
            torch.randn(args.batch, args.num_kv_heads, args.head_dim, device=device, dtype=dtype),
        )
    log(
        f"cache: {cache.num_pages} pages x {cache.page_size} slots, "
        f"KV heads={cache.num_kv_heads}, head_dim={cache.head_dim}, "
        f"prompt_len={args.prompt_len}, batch={args.batch}, decode_steps={args.steps}"
    )

    k_cache, v_cache = cache.k_cache, cache.v_cache
    page_table, seq_lens = cache.page_table, cache.seq_lens

    def decode_step(q: torch.Tensor) -> torch.Tensor:
        return torch.ops.flashattn_tile.paged_decode(
            q, k_cache, v_cache, page_table, seq_lens, cache.page_size, scale, 1
        )

    q_probe = torch.randn(args.batch, args.num_heads, args.head_dim, device=device, dtype=dtype)
    compile_ms = None
    compiled_backend = "inductor"
    t0 = time.perf_counter()
    try:
        compiled = torch.compile(decode_step, fullgraph=True)
        compiled(q_probe)
        _sync(device)
    except Exception as exc:  # pragma: no cover - backend dependent
        log(f"inductor compile failed ({type(exc).__name__}: {exc}); retrying backend='eager'")
        compiled = torch.compile(decode_step, fullgraph=True, backend="eager")
        compiled(q_probe)
        _sync(device)
        compiled_backend = "eager"
    compile_ms = (time.perf_counter() - t0) * 1e3
    log(f"torch.compile: fullgraph=True, backend={compiled_backend}, "
        f"compile+first-call={compile_ms:.1f} ms")

    step_times: list[float] = []
    max_abs = 0.0
    mean_abs = 0.0
    final_error = None
    for step in range(args.steps):
        q = torch.randn(args.batch, args.num_heads, args.head_dim, device=device, dtype=dtype)
        _sync(device)
        t0 = time.perf_counter()
        out = compiled(q)
        _sync(device)
        step_times.append((time.perf_counter() - t0) * 1e3)

        ref = ref_attention_paged(
            q, cache.k_cache, cache.v_cache, cache.page_table, cache.seq_lens,
            softmax_scale=scale,
        )
        diff = (out.float() - ref.float()).abs()
        max_abs = max(max_abs, diff.max().item())
        mean_abs += diff.mean().item() / args.steps
        final_error = diff.max().item()

        new_k = torch.randn(args.batch, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        new_v = torch.randn(args.batch, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
        append_tokens(cache, new_k, new_v)

    tokens = args.batch * args.steps
    total_s = sum(step_times) / 1e3
    per_step = sorted(step_times)
    p50 = statistics.median(step_times)
    p90 = per_step[min(len(per_step) - 1, int(0.9 * len(per_step)))]
    tolerance = 2e-2 if dtype == torch.float16 else 1e-6

    log("")
    log("correctness (compiled custom op vs fp32 reference, same cache state each step)")
    log(f"  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  final_step_max_abs={final_error:.3e} "
        f"(tolerance {tolerance:g})")
    if max_abs > tolerance:
        log("  FAILED tolerance")
        return 1
    log("")
    log("timing")
    log(f"  steps={args.steps}  tokens={tokens}")
    log(f"  per-step p50={p50:.3f} ms  p90={p90:.3f} ms  min={per_step[0]:.3f} ms  "
        f"max={per_step[-1]:.3f} ms")
    log(f"  total decode time={total_s * 1e3:.3f} ms  tokens/sec={tokens / total_s:.2f}")
    log(f"  final cache length: {int(cache.seq_lens.min().item())}..{int(cache.seq_lens.max().item())}")
    if not use_cuda:
        log("")
        log("NOTE: no CUDA device. These are CPU fp32 reference-path timings, not GPU kernel")
        log("throughput. The custom op was still traced and executed under torch.compile.")
    log("")
    log(f"environment: {json.dumps({'load_average': env.get('load_average'), 'cpu_count': env.get('cpu_count')})}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
