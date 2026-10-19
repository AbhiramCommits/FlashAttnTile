# FlashAttnTile

Fused paged-attention kernels for LLM inference, written in Triton with an optional CUDA C++ reference path. The project implements varlen prefill attention (online softmax, GQA/MQA, causal masking) and single-query paged decode attention (page-table indirection, split-K with a log-sum-exp reduction, and an INT8/FP8 KV-cache variant), exposes both as `torch.library` custom ops with fake kernels, autotunes launch configurations against an analytic shared-memory budget, and benchmarks the result against PyTorch SDPA (plus FlashAttention-2/FlashInfer when importable). A small vLLM-shaped backend shim and an autoregressive decode-loop demo show how the pieces fit together.

**This build host has no CUDA device and no Triton wheel.** Every kernel therefore executed its numerically identical PyTorch reference fallback; the GPU kernel throughput numbers requested by the project brief were **not collected** and are not claimed anywhere below. The measured CPU results, numerics, and end-to-end demo timings are real and trace to `results/`.

## Architecture

```
FlashAttnTile/
  flashattn_tile/
    env.py              environment probe, measured HBM/host copy bandwidth, spec-table lookup
    reference.py        fp32-accumulating varlen + paged attention (ground truth)
    paging.py           PagedKVCache, page-table contract, allocate/append/gather
    quant.py            INT8 / FP8 E4M3 per-(page, KV-head) KV quantisation
    autotune.py         analytic FLOPs/bytes, config space + smem filter, CUDA-event tuner
    ops.py              torch.library custom ops + register_fake implementations
    backend.py          vLLM-shaped FlashAttnTileBackend (prefill/decode routing)
    native.py           lazy build of the optional CUDA C++ extension
    kernels/
      prefill.py        _fwd_varlen_kernel (FlashAttention-style streaming)
      decode.py         _decode_paged_kernel, split-K + reduce, quantised-KV kernel
    tuned_configs.json  winning config per shape (no GPU tuning performed here)
  csrc/
    paged_decode.cu     readable CUDA C++ decode baseline (shared-memory tiles, warp reductions)
    bindings.cpp
  scripts/              probe_env, benchmark, roofline, demo_decode_loop, ncu_probe
  tests/                100 collected tests (60 run on CPU, 40 GPU-gated)
  docs/kernel_design.md tiling, online-softmax math, page indexing, roofline reasoning
  results/              measured artifacts (JSON/CSV/PNG/Markdown/text)
  .github/workflows/ci.yml
```

Dataflow for decode: the page table maps each logical block to a physical page; the decode kernel walks one page at a time, applies online softmax in fp32, and optionally writes `(acc, m, l)` partials for `split_k` programs. A small reduce kernel merges the partials with `m* = max(m_s)`, `w_s = exp(m_s - m*)`, `l* = sum(l_s w_s)`, `o = sum(acc_s w_s) / l*`. The result is returned through a `torch.library` custom op, which the backend calls after routing `is_prefill` to the varlen kernel or to paged decode. The Triton path is the primary implementation; `csrc/paged_decode.cu` is a readable CUDA C++ baseline that is built lazily as a PyTorch extension only when CUDA and `nvcc` are available.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # Triton installs on Linux; macOS has no Triton wheel
python scripts/probe_env.py      # writes results/env.json
pytest -q                        # GPU tests skip cleanly when CUDA/Triton are absent
python scripts/benchmark.py --mode decode
python scripts/benchmark.py --mode prefill
python scripts/roofline.py       # results/roofline.png + results/summary.md
python scripts/demo_decode_loop.py
```

On a CUDA host, `scripts/benchmark.py` sweeps the full grid (decode: batch 1/8/32/128 × context 128/1024/4096/16384; prefill: sequence 512/1024/2048/4096), autotunes each shape, and compares against SDPA, FlashAttention-2, FlashInfer, and the native baseline when each is available. On a CPU host it prints a `no_gpu` status and records a reduced CPU-reference sweep instead, which is what is committed here.

## Results

### Hardware

From `results/env.json`:

- **CPU-only, no CUDA device detected.** No GPU name, compute capability, SM count, or HBM bandwidth exists to report.
- Apple Silicon (`arm64`, `machine=arm64`), 8 hardware threads.
- Measured host copy bandwidth: **75.2 GB/s** (`dst.copy_(src)`, wall clock). This is host DRAM bandwidth, not HBM.
- PyTorch 2.14.1, Python 3.13.11. Triton importable: **false**.
- Host load average at probe time: **28.5 / 64.5 / 46.1** (1/5/15 min). The machine was shared and heavily loaded; CPU headline latencies below are the **minimum** of 21 iterations, with medians and p90 in `results/bench_*.csv`. They are indicative reference-path timings, not kernel throughput.

The GPU entries requested by the brief — GPU name, compute capability, SM count, HBM bandwidth, FP16 tensor-core ceiling — are **not available** on this machine.

### Decode: paged decode, batch × context sweep

“Ours” is the PyTorch fp32 reference fallback (the Triton kernel did not run). `speedup` is SDPA latency divided by ours; values below 1 mean the reference is slower. FlashAttention-2 and FlashInfer are not importable here; their columns are therefore absent. Achieved TFLOP/s is computed from the analytic FLOP count and the measured latency. `% of measured HBM` is **n/a** because no HBM exists/measured on this host.

| batch | ctx | ours min (ms) | ours median (ms) | SDPA min (ms) | gather (ms) | SDPA+gather (ms) | speedup vs SDPA | ours TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.209 | 0.239 | 0.012 | 0.034 | 0.046 | 0.06× | 0.0013 |
| 1 | 1024 | 0.449 | 0.484 | 0.029 | 0.142 | 0.170 | 0.06× | 0.0047 |
| 1 | 4096 | 2.703 | 3.184 | 0.164 | 0.365 | 0.530 | 0.06× | 0.0031 |
| 8 | 128 | 1.341 | 2.560 | 0.056 | 0.352 | 0.408 | 0.04× | 0.0016 |
| 8 | 1024 | 5.041 | 5.529 | 0.540 | 1.752 | 2.292 | 0.11× | 0.0033 |
| 8 | 4096 | 26.728 | 33.386 | 2.241 | 11.802 | 14.042 | 0.08× | 0.0025 |

Source: `results/bench_decode.csv` and `results/benchmarks.json` (mode `decode`). SDPA outputs match the fp32 reference to ≤3e-8 max abs; our path is the reference itself (0.0 error).

### Prefill: varlen prefill, sequence-length sweep

Again, “ours” is the fp32 reference fallback, batch 1, 8 query heads / 2 KV heads, head_dim 64. FlashAttention-2 and FlashInfer were not importable.

| seq | ours min (ms) | ours median (ms) | SDPA min (ms) | speedup vs SDPA | ours TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.385 | 0.454 | 0.121 | 0.31× | 0.0439 |
| 512 | 4.631 | 4.909 | 1.092 | 0.24× | 0.0581 |
| 1024 | 18.266 | 24.872 | 3.553 | 0.19× | 0.0588 |

Source: `results/bench_prefill.csv` and `results/benchmarks.json` (mode `prefill`). SDPA matches the fp32 reference to ≤6e-8 max abs.

### Autotuning

`flashattn_tile/autotune.py` enumerates **36 prefill** candidates (`BLOCK_M`/`BLOCK_N` ∈ {32,64,128} × `num_warps` ∈ {4,8} × `num_stages` ∈ {2,3}) and **16 decode** candidates (`num_warps` × `num_stages` × `split_k` ∈ {1,2,4,8}). Each candidate carries an analytic shared-memory estimate; candidates above the device limit are filtered before launch.

**No autotuning was executed** because there is no CUDA device. There are no winning configs, no measured best-vs-worst latency spread, and `flashattn_tile/tuned_configs.json` contains an empty `configs` map with a note explaining this. The counts above are enumerated candidates from `results/benchmarks.json`, not measurements.

### Numerics vs the fp32 reference

Every number below comes from `results/numerics.json`, produced by `tests/test_numerics.py` on this CPU host. “Max rel” divides by `|ref| + 1e-3`, so near-zero outputs can produce large relative values; the pass criterion is max absolute error.

| path | dtype | max abs | mean abs | max rel | tolerance | passed |
|---|---|---:|---:|---:|---:|---|
| paged reference sanity | float32 | 0.000000 | 0.000000 | 0.000000 | 0.0 | yes |
| paged, 16-bit storage | float16 | 0.000120 | 0.000013 | 0.0098 | 0.02 | yes |
| paged, 16-bit storage | bfloat16 | 0.000955 | 0.000106 | 0.0664 | 0.05 | yes |
| prefill, 16-bit inputs | float16 | 0.000237 | 0.000010 | 0.0436 | 0.02 | yes |
| prefill, 16-bit inputs | bfloat16 | 0.001871 | 0.000080 | 0.2962 | 0.05 | yes |
| INT8 KV cache | int8 | 0.003039 | 0.000503 | 0.4817 | 0.05 | yes |
| FP8 E4M3 KV cache | float8_e4m3fn | 0.022083 | 0.001416 | 1.2380 | 0.05 | yes |
| split-K=4 CPU fallback | float32 | 0.000000 | 0.000000 | 0.000000 | 1e-6 | yes |

These rows measure 16-bit storage rounding and KV quantisation through the **fp32 reference**, not the Triton kernel. Triton decode/prefill and the CUDA C++ baseline are recorded as `skipped` in the same JSON because no GPU/Triton/nvcc is present.

### INT8 / FP8 KV cache

From `results/numerics.json` (`kv_bytes_per_token`, analytic, includes two fp32 scales per page per KV head), for 8 KV heads, head_dim 128, page size 16:

- FP16 / BF16: **4100 bytes/token** for K+V plus scales.
- INT8 / FP8: **2052 bytes/token** — **1.998×** smaller.
- Accuracy versus the fp32 reference: INT8 max abs **0.003039** / mean abs **0.000503**; FP8 E4M3 max abs **0.022083** / mean abs **0.001416**. FP8’s max error is about 7× INT8’s on this data.

The quantised cache was exercised through the reference/dequantise path; the Triton FP8/int8 decode kernel was not run.

### Occupancy

**Not collected.** `results/benchmarks.json` records `method: "unavailable"`: there is no CUDA device and no `ncu` binary, so neither a profiled occupancy nor an analytic shared-memory occupancy estimate against a real device exists. No occupancy percentage is claimed.

### Demo decode loop

`results/demo_decode_loop.txt`, generated by `scripts/demo_decode_loop.py`:

- Batch 4, prompt 24 tokens, 8 decode steps = 32 tokens, page size 16, 8 heads / 2 KV heads, head_dim 64.
- `torch.compile(fullgraph=True)` with the **inductor** backend succeeded; compile + first call took 1966.9 ms. The compiled custom op matched the eager fp32 reference exactly (max abs 0.0) at every step.
- Per-step p50 **1.436 ms**, p90 **2.301 ms**, min 0.768 ms; total 11.622 ms; **2753.47 tokens/s**.
- These are CPU fp32 reference-path timings, not GPU kernel throughput.

### Test suite and CI

- **100 tests collected**, **60 passed**, **40 skipped** (all skips are GPU/Triton-gated), ruff clean. Source: `results/tests.txt`.
- `.github/workflows/ci.yml` runs ruff, pytest, and `scripts/probe_env.py` on Python 3.10/3.11 CPU runners, with an opt-in self-hosted CUDA job guarded by `vars.ENABLE_CUDA_CI`. The workflow was **not executed on GitHub** from this machine; the local results above are the only test evidence.

### Roofline

![Roofline: CPU reference timings and measured host-bandwidth roof](results/roofline.png)

`results/roofline.png` and `results/roofline_data.csv` contain 18 measured rows. With no GPU, the roof is the measured **CPU host copy bandwidth (75.2 GB/s)** and the points are reference-path timings; there is no GPU compute roof. Points above the DRAM roof occur for small working sets that fit in cache — the DRAM roof is not a hard ceiling for cache-resident data.

### What is explicitly absent

No GPU was available, so this repository contains **no measured** kernel latency, kernel TFLOP/s, achieved HBM bandwidth, GPU speedup versus FlashAttention-2/FlashInfer, autotuned winner/spread, occupancy/profile metric, Triton-kernel numerics, FP8 kernel numerics, or CUDA C++ baseline timing. Those require a CUDA GPU with Triton (and `nvcc` for the native path). The CPU reference timings and quantisation numerics above are the complete set of measurements taken on this machine.

## Limitations and next steps

- **GPU paths unexecuted.** The Triton prefill/decode/quantised kernels and the CUDA C++ baseline have not been compiled or validated on this machine. GPU-gated tests are written but skipped; treat GPU correctness as unverified until run on hardware.
- No backward pass; this is inference-only.
- Decode assumes one query per sequence; speculative/multi-token decode is not implemented.
- FP8/INT8 scales are per `(page, KV-head)` only; no per-token or per-channel scales.
- The CUDA C++ baseline is intentionally simple (one block per `(batch, head)`, `page_size ≤ 32`, `head_dim ∈ {64,128}`) and was not built here.
- The backend is a vLLM-shaped shim, not a vLLM plugin; there is no end-to-end serving integration test.
- CPU timings were taken on a heavily loaded shared host and use min-of-iterations; they are noisy and not a substitute for GPU kernel measurements.
- Architectures other than the host CPU were not validated. Autotuning and occupancy require a real device.

## References

1. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
2. Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. arXiv:2307.08691.
3. Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180.
4. Zihao Ye et al. *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving.* 2025. arXiv:2501.01005.
5. Philippe Tillet, H. T. Kung, David Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019. arXiv:1907.12491.
6. PyTorch documentation: `torch.library.custom_op`, `torch.compile`, and `scaled_dot_product_attention`.
7. Triton documentation: https://triton-lang.org/
