# FlashAttnTile benchmark summary

Generated: 2026-10-19T08:55:44.265457+00:00  
Artifact generated: 2026-10-19T08:55:04.722635+00:00  
Status: **no_gpu** — no CUDA device; triton not importable

## Hardware

**CPU-only, no CUDA device detected.** No GPU kernel throughput, HBM bandwidth, or FlashAttention/FlashInfer comparison was collected on this machine.
- CPU: arm (8 hardware threads)
- Measured CPU copy bandwidth: 75.2 GB/s (host DRAM, measured)
- Torch 2.14.1, Triton importable: False
- Host load average at probe time: {'1m': 28.490234375, '5m': 64.49658203125, '15m': 46.0673828125}
- Timings: CPU headline latency is the **minimum** of the measured iterations (the build host was shared and heavily loaded); median/p90 are in the CSV. These are reference-path timings, not GPU kernel throughput.

## Decode

| batch | ctx | ours min (ms) | ours median (ms) | ours p90 (ms) | SDPA min (ms) | SDPA+gather (ms) | min-based speedup vs SDPA | ours TFLOP/s | max abs vs ref |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 0.209 | 0.239 | 0.268 | 0.012 | 0.046 | 0.06x | 0.0013 | 0.0000 |
| 1 | 1024 | 0.449 | 0.484 | 0.595 | 0.029 | 0.170 | 0.06x | 0.0047 | 0.0000 |
| 1 | 4096 | 2.703 | 3.184 | 3.665 | 0.164 | 0.530 | 0.06x | 0.0031 | 0.0000 |
| 8 | 128 | 1.341 | 2.560 | 6.636 | 0.056 | 0.408 | 0.04x | 0.0016 | 0.0000 |
| 8 | 1024 | 5.041 | 5.529 | 7.088 | 0.540 | 2.292 | 0.11x | 0.0033 | 0.0000 |
| 8 | 4096 | 26.728 | 33.386 | 47.738 | 2.241 | 14.042 | 0.08x | 0.0025 | 0.0000 |


## Prefill

| batch | seq | ours min (ms) | ours median (ms) | ours p90 (ms) | SDPA min (ms) | min-based speedup vs SDPA | ours TFLOP/s | max abs vs ref |
|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 0.385 | 0.454 | 0.523 | 0.121 | 0.31x | 0.0439 | 0.0000 |
| 1 | 512 | 4.631 | 4.909 | 5.371 | 1.092 | 0.24x | 0.0581 | 0.0000 |
| 1 | 1024 | 18.266 | 24.872 | 27.029 | 3.553 | 0.19x | 0.0588 | 0.0000 |


## Autotuning

No GPU autotuning was executed. The analytic config spaces contain 36 prefill and 16 decode candidates; shared-memory feasibility can only be evaluated against a real device. `flashattn_tile/tuned_configs.json` is therefore empty.

## Occupancy

**Not collected.** no CUDA device and no ncu binary on this host; no occupancy was measured

## Numerics

| path | dtype | max abs | mean abs | max rel | tolerance | passed | note |
|---|---|---|---|---|---|---|---|
| paged_reference_fp32 | float32 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | yes | fp32 reference vs itself (sanity) |
| paged_storage_fp16 | float16 | 0.000120 | 0.000013 | 0.009819 | 0.0200 | yes | fp32 reference with 16-bit cache/query storage; output cast back to 16-bit |
| paged_storage_bf16 | bfloat16 | 0.000955 | 0.000106 | 0.066395 | 0.0500 | yes | fp32 reference with 16-bit cache/query storage; output cast back to 16-bit |
| prefill_storage_fp16 | float16 | 0.000237 | 0.000010 | 0.043586 | 0.0200 | yes | fp32 reference prefill with 16-bit inputs; output cast back to 16-bit |
| prefill_storage_bf16 | bfloat16 | 0.001871 | 0.000080 | 0.296199 | 0.0500 | yes | fp32 reference prefill with 16-bit inputs; output cast back to 16-bit |
| kv_int8 | int8 | 0.003039 | 0.000503 | 0.481686 | 0.0500 | yes | KV cache quantised per (page, KV-head), dequantised to fp32, then fp32 reference attention |
| kv_fp8_e4m3fn | fp8_e4m3fn | 0.022083 | 0.001416 | 1.237964 | 0.0500 | yes | KV cache quantised per (page, KV-head), dequantised to fp32, then fp32 reference attention |
| split_k_4_cpu_fallback | float32 | 0.000000 | 0.000000 | 0.000000 | 0.0000 | yes | CPU fallback ignores split_k (identical reference path); the GPU split-K LSE merge is validated by tests/test_decode.py |
| triton_decode | float16/bf16 | n/a | n/a | n/a | n/a | skipped | not run here (no CUDA device; Triton not importable); validated by the GPU test suite |
| triton_prefill | float16/bf16 | n/a | n/a | n/a | n/a | skipped | not run here (no CUDA device; Triton not importable); validated by the GPU test suite |
| native_cuda_decode | float16/bf16 | n/a | n/a | n/a | n/a | skipped | CUDA C++ baseline not built (no CUDA device; Triton not importable) |


## Test suite

- pytest: `60 passed, 40 skipped, 3 warnings in 3.84s`
- collection: `100 tests collected in 0.24s`
- ruff: `clean`

## Roofline

![roofline](roofline.png)

- Plot points: 18 measured rows; data in `roofline_data.csv`.
- Arithmetic intensity and FLOPs use the analytic formulas in `flashattn_tile/autotune.py`; GPU latency is the median of measured CUDA-event iterations, CPU latency is the minimum (see the hardware note).
- The roof uses DRAM bandwidth. Points above it can occur for small working sets that fit in CPU/GPU cache; the DRAM roof is not a hard ceiling for cache-resident data.

