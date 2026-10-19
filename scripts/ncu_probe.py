#!/usr/bin/env python
"""Tiny decode launch used as the target of an ``ncu`` profile.

Only runs on CUDA; invoked by ``scripts/benchmark.py`` when ``ncu`` exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashattn_tile.kernels.decode import paged_decode_attention  # noqa: E402
from flashattn_tile.paging import allocate  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 1
    batch, ctx, page_size, kv_heads, head_dim = 8, 1024, 16, 8, 128
    cache = allocate(batch, ctx, kv_heads, head_dim, page_size,
                     dtype=torch.float16, device="cuda")
    cache.k_cache.normal_(0.0, 0.2)
    cache.v_cache.normal_(0.0, 0.2)
    cache.seq_lens[:] = ctx
    q = (torch.randn(batch, 32, head_dim, device="cuda") * 0.2).to(torch.float16)
    out = paged_decode_attention(q, cache)
    torch.cuda.synchronize()
    print(f"decode out shape={tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
