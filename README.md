# FlashAttnTile

Fused paged-attention kernels for LLM inference, written in Triton with an optional CUDA C++ reference path. It implements varlen prefill (online softmax, GQA/MQA, causal masking) and single-query paged decode (page-table indirection, split-K with a log-sum-exp reduction, INT8/FP8 KV cache), exposed as `torch.library` custom ops and benchmarked against PyTorch SDPA. The numbers below come from `results/` and were measured on an NVIDIA L4.

## Environment

NVIDIA L4 (compute 8.9, 58 SMs, 22 GiB, 300 GB/s HBM spec, 30.3 TFLOP/s FP16 spec) · torch 2.8.0+cu128 · triton 3.4.0 — full probe in `results/env.json`.

## Results

### Prefill vs SDPA (flash backend)

fp16, batch 1, 32 query heads / 8 KV heads (GQA=4), head_dim 128, causal. "Ours" is the Triton varlen kernel; SDPA is `scaled_dot_product_attention` on the flash backend with K/V expanded for GQA. Source: `results/benchmark_prefill_gpu.csv`.

| seq_len | ours TFLOP/s | SDPA TFLOP/s | ours / SDPA |
|---:|---:|---:|---:|
| 512   | 6.4  | 21.6 | 0.30× |
| 1024  | 16.7 | 39.5 | 0.42× |
| 2048  | 27.6 | 54.7 | 0.50× |
| 4096  | 31.1 | 52.4 | 0.59× |
| 8192  | 32.5 | 51.8 | 0.63× |

The kernel saturates near the FP32-accumulate tensor-core ceiling (~30 TFLOP/s); SDPA flash runs 1.6–3.3× faster because it uses FP16 accumulation (≈60 TFLOP/s). bf16 tracks fp16 closely (same table in the CSV).

### Decode bandwidth

fp16, 32 query heads / 8 KV heads (GQA=4), head_dim 128, page_size 16. "% peak" is against the 300 GB/s HBM spec. Source: `results/benchmark_decode_gpu.csv`.

| batch | ctx | ours GB/s | % peak | vs SDPA |
|---:|---:|---:|---:|---:|
| 1   | 128   | 2.2   | 0.7%  | 0.08× |
| 1   | 1024  | 11.7  | 3.9%  | 0.28× |
| 1   | 4096  | 25.2  | 8.4%  | 0.55× |
| 1   | 16384 | 34.5  | 11.5% | 0.60× |
| 8   | 128   | 15.8  | 5.3%  | 0.37× |
| 8   | 1024  | 69.3  | 23.1% | 1.43× |
| 8   | 4096  | 105.7 | 35.2% | 1.89× |
| 8   | 16384 | 103.0 | 34.3% | 1.85× |
| 32  | 128   | 49.1  | 16.4% | 1.03× |
| 32  | 1024  | 115.5 | 38.5% | 1.88× |
| 32  | 4096  | 108.5 | 36.2% | 1.67× |
| 32  | 16384 | 93.1  | 31.0% | 1.44× |
| 128 | 128   | 90.7  | 30.2% | 1.55× |
| 128 | 1024  | 106.1 | 35.4% | 1.64× |
| 128 | 4096  | 73.3  | 24.4% | 1.20× |
| 128 | 16384 | 68.1  | 22.7% | — (SDPA OOM) |

Best decode is 148 GB/s (49% of peak) at page_size 32/64. Loses to SDPA at batch 1 (few blocks on 58 SMs) and wins at batch ≥ 8 with long context, because it never materializes SDPA's 4×-expanded K/V. Larger pages help: page_size 8 → 22.5%, 16 → 35.5%, 32 → 49.3% of peak.

### Autotuning

`scripts/autotune_gpu.py` tuned 31 shapes. Prefill gain was ~1.0× (0.94–1.05×) — tile size is irrelevant at the FP32-accumulate ceiling. Decode gain was up to 2.59× (batch 1, ctx 16384, `split_k=8`), 1.24× geomean. Winners are in `flashattn_tile/tuned_configs.json`; per-shape rows in `results/autotune_gpu.csv`.

## Validated vs not validated

Validated on an NVIDIA L4:

- Triton prefill, decode and quantised kernels match the fp32 reference and SDPA across fp16/bf16, head_dim 64/128, MHA/GQA/MQA, causal on/off, odd sequence lengths, page sizes 16/32 and mixed varlen batches (`tests/`).
- The CUDA C++ baseline (`csrc/`) builds with nvcc and matches Triton and the reference to ≤2e-5 (`tests/test_decode.py::test_native_decode_matches_triton_and_reference`).
- The FP8 quantised decode kernel compiles and runs (the masked-load `other=0` fill was fixed).

Not validated:

- No `ncu` hardware-counter profile — the container exposes no `/dev/nvidia-caps`, so `ncu` returns `ERR_NVGPUCTRPERM`.
- No FlashAttention-2 (`flash_attn`) or vLLM baseline — neither installs cleanly for torch 2.8/cu128 in this container.
- No backward pass, no multi-token decode, no end-to-end serving integration.

## How to run

```bash
pip install -e ".[dev]"
python scripts/probe_env.py        # results/env.json
pytest -q                          # 210 tests; GPU tests skip without CUDA/Triton
python scripts/benchmark_gpu.py    # results/benchmark_{prefill,decode}_gpu.csv + PNG plots
python scripts/autotune_gpu.py     # writes flashattn_tile/tuned_configs.json
```

## Limitations

- Inference only; no backward pass.
- Decode assumes one query per sequence.
- FP8/INT8 scales are per `(page, KV-head)`; no per-token or per-channel scales.
- The CUDA C++ baseline is intentionally simple: one block per `(batch, head)`, `page_size ≤ 32`, `head_dim ∈ {64,128}`.
- The backend is a vLLM-shaped shim, not a vLLM plugin.

## What I learned

- Prefill is compute-bound at the FP32-accumulate tensor-core rate; the gap to SDPA flash is a precision tradeoff (FP16 vs FP32 accumulation), not a correctness bug, so closing it means giving up numerical headroom.
- Decode is won by scheduling, not tiling: split-K (up to 2.6×) and reading each K/V page once per GQA group matter far more than any launch-parameter sweep, and the naive grouped kernel only paid off at high batch.
- Autotuning the obvious knob (prefill tile size) changed almost nothing, while the non-obvious knob (decode split-K) changed everything — profile first, then tune.
