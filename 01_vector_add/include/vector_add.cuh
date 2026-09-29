#pragma once
// CUDA vector-add kernels: naive, grid-stride, and a vectorized float4 path with a scalar tail so it is
// correct for every N (not only multiples of 4).

#include <cuda_runtime.h>

#include <cstddef>

namespace portfolio::vecadd::cuda {

__global__ void add_naive(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                          std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void add_grid_stride(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = a[i] + b[i];
}

// Processes 4 floats per thread via one 128-bit load/store (`LDG.128`/`STG.128`); `n_vec` is the number
// of float4 groups, i.e. floor(n / 4). The caller is responsible for the scalar tail (see below).
__global__ void add_vectorized(const float4* __restrict__ a, const float4* __restrict__ b,
                               float4* __restrict__ c, std::size_t n_vec) {
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
    const float4 va = a[i], vb = b[i];
    c[i] = float4{va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};
  }
}

// Scalar epilogue for the `n % 4` elements the float4 kernel above cannot cover, starting at `offset`.
__global__ void add_tail(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c,
                         std::size_t offset, std::size_t n) {
  const std::size_t i = offset + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

inline int grid_size(std::size_t work_items, int block) {
  return static_cast<int>((work_items + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

// Aligned prefix through the float4 path, last 0-3 elements through a scalar tail kernel.
inline void launch_vectorized(const float* d_a, const float* d_b, float* d_c, std::size_t n, int block = 256) {
  const std::size_t n_vec = n / 4;
  if (n_vec > 0) {
    add_vectorized<<<grid_size(n_vec, block), block>>>(reinterpret_cast<const float4*>(d_a),
                                                        reinterpret_cast<const float4*>(d_b),
                                                        reinterpret_cast<float4*>(d_c), n_vec);
  }
  const std::size_t offset = n_vec * 4, remainder = n - offset;
  if (remainder > 0) add_tail<<<grid_size(remainder, block), block>>>(d_a, d_b, d_c, offset, n);
}

}  // namespace portfolio::vecadd::cuda
