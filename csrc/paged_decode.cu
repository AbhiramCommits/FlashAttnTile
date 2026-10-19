// Readable CUDA C++ paged-decode baseline.
//
// One thread block per (batch, head). K/V pages are staged through shared
// memory; the online-softmax page max/sum use warp shuffles; the output
// accumulator is fp32 and per-thread (each thread owns one or more head dims).
//
// This is intentionally straightforward -- it exists so the Triton kernel has a
// transparent reference implementation to be compared against. It is only built
// when a CUDA toolchain and GPU are present (see flashattn_tile/native.py).

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kBlockThreads = 128;

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* ptr) {
  return *ptr;
}

template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* ptr) {
  return __half2float(*ptr);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* ptr) {
  return __bfloat162float(*ptr);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* ptr, float value);

template <>
__device__ __forceinline__ void store_from_float<float>(float* ptr, float value) {
  *ptr = value;
}

template <>
__device__ __forceinline__ void store_from_float<__half>(__half* ptr, float value) {
  *ptr = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* ptr, float value) {
  *ptr = __float2bfloat16(value);
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// head_dim is templated so the dot-product loop can be unrolled; page_size is
// runtime but constrained to <= 32 by the host wrapper so one warp can reduce
// the whole page score vector.
template <typename T, int HEAD_DIM>
__global__ void paged_decode_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    const int* __restrict__ page_table,
    const int* __restrict__ seq_lens,
    T* __restrict__ out,
    int num_heads,
    int num_kv_heads,
    int page_size,
    int64_t stride_qb,
    int64_t stride_qh,
    int64_t stride_kp,
    int64_t stride_kt,
    int64_t stride_kh,
    int64_t stride_vp,
    int64_t stride_vt,
    int64_t stride_vh,
    int64_t stride_pb,
    int64_t stride_ob,
    int64_t stride_oh,
    float softmax_scale) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int group = num_heads / num_kv_heads;
  const int kv_head = h / group;
  const int seq_len = seq_lens[b];

  constexpr int kDimsPerThread = (HEAD_DIM + kBlockThreads - 1) / kBlockThreads;

  extern __shared__ float smem[];
  float* sk = smem;                              // [page_size, HEAD_DIM]
  float* sv = sk + page_size * HEAD_DIM;         // [page_size, HEAD_DIM]
  float* sq = sv + page_size * HEAD_DIM;         // [HEAD_DIM]
  float* ss = sq + HEAD_DIM;                     // [page_size] scores/probs

  __shared__ float s_m;
  __shared__ float s_l;
  __shared__ float s_alpha;
  __shared__ float s_m_new;

  if (tid == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
    s_alpha = 1.0f;
    s_m_new = -CUDART_INF_F;
  }

  for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
    sq[d] = load_as_float(q + b * stride_qb + h * stride_qh + d);
  }
  __syncthreads();

  if (seq_len <= 0) {
    for (int i = 0; i < kDimsPerThread; ++i) {
      const int d = tid + i * blockDim.x;
      if (d < HEAD_DIM) {
        store_from_float(out + b * stride_ob + h * stride_oh + d, 0.0f);
      }
    }
    return;
  }

  // Each thread owns kDimsPerThread output dimensions.
  float acc[kDimsPerThread];
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    acc[i] = 0.0f;
  }

  const int num_pages = (seq_len + page_size - 1) / page_size;
  for (int page = 0; page < num_pages; ++page) {
    const int phys = page_table[b * stride_pb + page];

    // Cooperative load of the K/V page into shared memory.
    for (int idx = tid; idx < page_size * HEAD_DIM; idx += blockDim.x) {
      const int token = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      const int logical = page * page_size + token;
      float kval = 0.0f;
      float vval = 0.0f;
      if (logical < seq_len) {
        const int64_t offset =
            static_cast<int64_t>(phys) * stride_kp + token * stride_kt + kv_head * stride_kh + d;
        kval = load_as_float(k_cache + offset);
        const int64_t voffset =
            static_cast<int64_t>(phys) * stride_vp + token * stride_vt + kv_head * stride_vh + d;
        vval = load_as_float(v_cache + voffset);
      }
      sk[idx] = kval;
      sv[idx] = vval;
    }
    __syncthreads();

    // One thread per cached token computes q . k. head_dim is compile-time.
    if (tid < page_size) {
      const int logical = page * page_size + tid;
      float score = -CUDART_INF_F;
      if (logical < seq_len) {
        score = 0.0f;
#pragma unroll 4
        for (int d = 0; d < HEAD_DIM; ++d) {
          score = fmaf(sq[d], sk[tid * HEAD_DIM + d], score);
        }
        score *= softmax_scale;
      }
      ss[tid] = score;
    }
    __syncthreads();

    // Online-softmax page max + rescale factor (warp 0, shuffle reduction).
    if (tid < 32) {
      const float value = (tid < page_size) ? ss[tid] : -CUDART_INF_F;
      const float page_max = warp_reduce_max(value);
      if (tid == 0) {
        const float m_new = fmaxf(s_m, page_max);
        s_alpha = (s_m == -CUDART_INF_F) ? 0.0f : expf(s_m - m_new);
        s_m_new = m_new;
        s_m = m_new;
      }
    }
    __syncthreads();

    // probabilities exp(score - m_new); masked lanes contribute 0.
    if (tid < page_size) {
      ss[tid] = (tid < page_size) ? expf(ss[tid] - s_m_new) : 0.0f;
    }
    __syncthreads();

    if (tid < 32) {
      const float p = (tid < page_size) ? ss[tid] : 0.0f;
      const float psum = warp_reduce_sum(p);
      if (tid == 0) {
        s_l = s_l * s_alpha + psum;
      }
    }
    __syncthreads();

    const float alpha = s_alpha;
#pragma unroll
    for (int i = 0; i < kDimsPerThread; ++i) {
      const int d = tid + i * blockDim.x;
      float value = acc[i] * alpha;
      if (d < HEAD_DIM) {
#pragma unroll 4
        for (int token = 0; token < 32; ++token) {
          if (token < page_size) {
            value = fmaf(ss[token], sv[token * HEAD_DIM + d], value);
          }
        }
      }
      acc[i] = value;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / s_l;
#pragma unroll
  for (int i = 0; i < kDimsPerThread; ++i) {
    const int d = tid + i * blockDim.x;
    if (d < HEAD_DIM) {
      store_from_float(out + b * stride_ob + h * stride_oh + d, acc[i] * inv_l);
    }
  }
}

template <typename T>
void launch_decode(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& page_table,
    const torch::Tensor& seq_lens,
    torch::Tensor& out,
    int64_t page_size,
    double softmax_scale,
    int head_dim,
    cudaStream_t stream) {
  const int64_t batch = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t num_kv_heads = k_cache.size(2);
  const int64_t smem_bytes =
      (2 * page_size * head_dim + head_dim + page_size) * static_cast<int64_t>(sizeof(float));
  const dim3 grid(static_cast<unsigned int>(batch), static_cast<unsigned int>(num_heads));

#define LAUNCH_WITH_DIM(DIM)                                                                    \
  paged_decode_kernel<T, DIM><<<grid, kBlockThreads, smem_bytes, stream>>>(                     \
      reinterpret_cast<const T*>(q.data_ptr()),                                                 \
      reinterpret_cast<const T*>(k_cache.data_ptr()),                                           \
      reinterpret_cast<const T*>(v_cache.data_ptr()),                                           \
      page_table.data_ptr<int>(),                                                               \
      seq_lens.data_ptr<int>(),                                                                 \
      reinterpret_cast<T*>(out.data_ptr()),                                                     \
      static_cast<int>(num_heads), static_cast<int>(num_kv_heads),                              \
      static_cast<int>(page_size),                                                              \
      q.stride(0), q.stride(1),                                                                 \
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),                                  \
      v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),                                  \
      page_table.stride(0),                                                                     \
      out.stride(0), out.stride(1),                                                             \
      static_cast<float>(softmax_scale))

  if (head_dim == 64) {
    LAUNCH_WITH_DIM(64);
  } else if (head_dim == 128) {
    LAUNCH_WITH_DIM(128);
  } else {
    TORCH_CHECK(false, "native paged decode supports head_dim 64 or 128");
  }
#undef LAUNCH_WITH_DIM
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor paged_decode_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor page_table,
    torch::Tensor seq_lens,
    int64_t page_size,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "KV cache must be CUDA tensors");
  TORCH_CHECK(page_table.is_cuda() && seq_lens.is_cuda(), "metadata must be CUDA tensors");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4, "cache must be [pages, page, kv, dim]");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(), "cache must be contiguous");
  TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table must be int32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type(), "q and cache dtypes must match");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k/v cache shapes must match");
  TORCH_CHECK(q.size(2) == k_cache.size(3), "q head_dim must match cache head_dim");
  TORCH_CHECK(page_size > 0 && page_size <= 32, "page_size must be in [1, 32]");
  TORCH_CHECK(page_table.size(0) == q.size(0), "page_table batch must match q");
  TORCH_CHECK(seq_lens.size(0) == q.size(0), "seq_lens batch must match q");
  TORCH_CHECK(q.size(1) % k_cache.size(2) == 0, "heads must be divisible by KV heads");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  auto out = torch::empty_like(q);
  const int head_dim = static_cast<int>(q.size(2));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (q.scalar_type()) {
    case at::kHalf:
      launch_decode<__half>(q, k_cache, v_cache, page_table, seq_lens, out, page_size,
                            softmax_scale, head_dim, stream);
      break;
    case at::kBFloat16:
      launch_decode<__nv_bfloat16>(q, k_cache, v_cache, page_table, seq_lens, out, page_size,
                                   softmax_scale, head_dim, stream);
      break;
    case at::kFloat:
      launch_decode<float>(q, k_cache, v_cache, page_table, seq_lens, out, page_size,
                           softmax_scale, head_dim, stream);
      break;
    default:
      TORCH_CHECK(false, "native paged decode supports fp16, bf16 and fp32");
  }
  return out;
}
