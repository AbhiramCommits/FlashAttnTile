# Kernel design

This document describes the tiling, online-softmax math, page-table addressing,
memory-access reasoning, and autotuning search space of FlashAttnTile. Numbers
about achieved performance are deliberately **not** repeated here; they live in
`results/` and are summarized in the README.

## Dataflow

```
PagedKVCache (page_table, seq_lens, K/V pages)
        |
        v
 decode kernel: grid (batch, head, split)        prefill kernel: grid (q-block, head, seq)
        |                                               |
 online softmax per page tile                   online softmax per K/V block
        |                                               |
 split-K partials (acc, m, l)  ->  LSE reduce   fp32 accumulator -> output
        |                                               |
        +----------------> torch custom op <------------+
                                  |
                    vLLM-style backend.forward()
```

## Prefill: streaming varlen attention

**Grid.** `(ceil(max_seqlen_q / BLOCK_M), num_heads, batch)`. Each program owns
`BLOCK_M` packed query rows for one head of one sequence. Sequences shorter than
`BLOCK_M` exit early; the packed `cu_seqlens` offsets give the token base of Q, K
and V.

**Tiles.** Q is loaded once as a `[BLOCK_M, HEAD_DIM]` tile. K and V are streamed
in `[BLOCK_N, HEAD_DIM]` tiles. No `[Lq, Lk]` score matrix is ever written to
global memory; `tl.dot(q, k^T)` produces a `[BLOCK_M, BLOCK_N]` fp32 score tile
that stays on chip.

**Online softmax.** For each K/V block `j`:

```
S_j      = Q K_j^T * scale
m_new    = max(m_i, rowmax(S_j))
alpha    = exp(m_i - m_new)
P_j      = exp(S_j - m_new)
l_i      = l_i * alpha + rowsum(P_j)
acc      = acc * alpha + P_j V_j          # acc is fp32
m_i      = m_new
```

At the end `out = acc / l_i`. The rescale keeps every exponent argument <= 0, so
`P_j` cannot overflow. Inputs are fp16/bf16, both `tl.dot`s accumulate in fp32,
and the probability tile is cast back to the input dtype before the PV product.

**Causal masking.** Query `i` (0-based in its sequence) may see key `j` when
`j <= i + (kv_len - q_len)`. For a query block the kernel stops at
`hi = min(kv_len, (pid_m + 1) * BLOCK_M + kv_len - q_len)`, so causal prefill
skips all K/V blocks strictly in the future. The diagonal block is masked
element-wise.

**GQA.** Query head `h` reads KV head `h // (num_heads / num_kv_heads)`; no K/V
replication is materialised.

## Decode: one query, paged KV, split-K

**Grid.** `(batch, num_heads, split_k)`. Each program reads one query vector,
walks the logical pages `0 .. ceil(seq_len / page_size)` for its sequence, and
maintains the same `(m_i, l_i, acc)` triple as prefill. For a page `p`:

```
phys_page = page_table[b, p]
logical_t = p * page_size + t
score_t   = (q . K[phys_page, t, kv_head]) * scale      for logical_t < seq_len
```

The tail page is masked with `logical_t < seq_len`, and `exp(-inf) = 0`.

**No tensor cores for decode.** With one query, the decode inner product is a
reduction over `head_dim`, not a matrix multiply with a useful second dimension.
The kernel uses fp32 FMA-style vector reductions. This is intentional: decode is
bandwidth-bound and tensor-core utilisation is not the limiting factor.

**Split-K.** Logical pages are partitioned across `split_k` programs. Each writes
`acc_s` (`[head_dim]` fp32), `m_s`, and `l_s`. The reduce kernel computes

```
m* = max_s m_s
w_s = exp(m_s - m*)
l* = sum_s l_s * w_s
o  = (sum_s acc_s * w_s) / l*
```

This is numerically stable because the global max is subtracted before
exponentiating; it is the same merge used by split-K FlashAttention. Empty
splits (when `split_k` exceeds the page count) store `m_s = -inf`, `l_s = 0`, so
`w_s = 0` and they contribute nothing.

**Quantised KV.** INT8/FP8 caches store one scale per `(page, KV head)`. The
kernel loads the page scale once, dequantises each element to fp32 inside the
page tile, and runs the same online softmax. The scale layout matches the decode
tiling: one page for one KV head is contiguous along `head_dim`.

## Page-table indexing and memory access

`PagedKVCache` uses NHD layout `[num_pages, page_size, num_kv_heads, head_dim]`.
A token at logical position `t` is at

```
physical_page = page_table[b, t // page_size]
slot          = t % page_size
address       = physical_page * page_stride
              + slot          * page_size_stride   # stride(1)
              + kv_head       * kv_head_stride     # stride(2)
              + d             * 1                  # stride(3), contiguous
```

The innermost `head_dim` axis is contiguous, so a warp loading consecutive `d`
values issues coalesced 128-byte transactions. Prefill's `[BLOCK_N, HEAD_DIM]`
tile is likewise contiguous along `d`; `head_dim` 64/128 keeps each row aligned.
Pages are read through the table rather than assumed contiguous, so the same
kernel works with a permuted allocator. The table entry is uniform across the
program and is served from cache/broadcast.

## Why decode is bandwidth-bound and prefill compute-bound

Analytic intensity from `flashattn_tile/autotune.py`:

* Decode, one query, GQA group `G = num_heads / num_kv_heads`:
  `4 * G` FLOPs per KV pair (2 for QK, 2 for PV) over `4` bytes per fp16 KV pair,
  so intensity is about `G` FLOPs/byte, independent of context length. For the
  benchmark geometry (`G=4`) that is ~4 FLOPs/byte, far below the machine balance
  of a modern GPU. Reading the KV cache dominates.
* Prefill, `q_len = kv_len = L`, causal:
  `2 * H * D * L^2` FLOPs over roughly `4 * D * L * (H + 2 H_kv)` bytes, so
  intensity grows linearly with `L` (about `H * L / (2 * (H + 2 H_kv))`). At long
  sequence lengths the matmuls dominate and the kernel is compute-bound.

This is why the decode benchmark reports a large fraction of measured HBM
bandwidth while prefill is compared against the FP16 tensor-core roofline.

## Autotuner

`build_config_space` enumerates:

* prefill: `BLOCK_M ∈ {32,64,128}`, `BLOCK_N ∈ {32,64,128}`, `num_warps ∈ {4,8}`,
  `num_stages ∈ {2,3}` → 36 candidates.
* decode: `num_warps ∈ {4,8}`, `num_stages ∈ {2,3}`, `split_k ∈ {1,2,4,8}` →
  16 candidates. `BLOCK_N` is the page size and `BLOCK_M=1`.

Each candidate carries an analytic shared-memory estimate:

```
prefill_smem = num_stages * (BLOCK_M*D + 2*BLOCK_N*D) * elem_bytes
             + BLOCK_M*BLOCK_N*4 + BLOCK_M*D*4
decode_smem  = num_stages * (2*page_size*D) * elem_bytes + page_size*4
```

Candidates whose estimate exceeds the device's opt-in shared-memory limit are
filtered before any launch, so an infeasible config cannot crash the tuner.
Surviving configs are timed with CUDA events (median latency), ranked, and the
winner is written to `flashattn_tile/tuned_configs.json` keyed by a shape
signature. Wrappers load that file by default.

## Optional CUDA C++ baseline

`csrc/paged_decode.cu` is a readable baseline rather than an optimised kernel:

* one thread block per `(batch, head)`, 128 threads;
* the current K/V page is staged into dynamic shared memory;
* one thread per cached token computes `q·k` with a fully unrolled `head_dim`
  loop; the page max and sum use `__shfl_down_sync` warp reductions;
* each thread owns one or more output dimensions and keeps fp32 accumulators
  across pages;
* constraints: `page_size <= 32` (one warp reduces the page) and
  `head_dim ∈ {64, 128}`.

It is built lazily by `flashattn_tile/native.py` through
`torch.utils.cpp_extension.load`, and is only enabled when a CUDA device and
`nvcc` are present. Otherwise `native.is_available()` is false and the benchmark
records the baseline as unavailable.

## Limitations

* No backward pass; this is an inference kernel.
* Decode assumes one query per sequence; multi-token speculative decode would
  need the prefill kernel or a separate speculative-decode variant.
* The Triton kernels are written for fp16/bf16 on CUDA; fp32 inputs fall back to
  the reference.
* The quantised kernel path supports per-(page, KV-head) scales only.
* The CUDA C++ baseline is intentionally simple and was not built on the
  CPU-only host used for this repository's artifacts.
