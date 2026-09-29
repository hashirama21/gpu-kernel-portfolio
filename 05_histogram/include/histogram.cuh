#pragma once
// Histogram over 256 uint8 bins: v0 global atomics, v1 shared-memory privatization, v2 adds 128-bit
// (uint4) vectorized reads plus a scalar tail so it is correct for every N (not only multiples of 16).
// The vectorized/tail split is verified in tests/test_histogram_cpu_sim.cpp.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace portfolio::histogram::cuda {

constexpr int kNumBins = 256;

__global__ void histogram_v0_global_atomic(const std::uint8_t* __restrict__ data, unsigned* __restrict__ bins,
                                           std::size_t n) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = idx; i < n; i += stride) atomicAdd(&bins[data[i]], 1u);
}

__global__ void histogram_v1_shared(const std::uint8_t* __restrict__ data, unsigned* __restrict__ bins, std::size_t n) {
  __shared__ unsigned s_bins[kNumBins];
  for (int b = threadIdx.x; b < kNumBins; b += blockDim.x) s_bins[b] = 0;
  __syncthreads();

  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = idx; i < n; i += stride) atomicAdd(&s_bins[data[i]], 1u);
  __syncthreads();

  for (int b = threadIdx.x; b < kNumBins; b += blockDim.x) atomicAdd(&bins[b], s_bins[b]);
}

// Vectorized pass over the aligned prefix only: `n_v` = n / 16 groups of 16 bytes each, read as one
// uint4 (128-bit) load per group.
__global__ void histogram_v2_vectorized(const std::uint8_t* __restrict__ data, unsigned* __restrict__ bins,
                                        std::size_t n_v) {
  __shared__ unsigned s_bins[kNumBins];
  for (int b = threadIdx.x; b < kNumBins; b += blockDim.x) s_bins[b] = 0;
  __syncthreads();

  const uint4* data_v = reinterpret_cast<const uint4*>(data);
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = idx; i < n_v; i += stride) {
    const uint4 v = data_v[i];
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&v);
#pragma unroll
    for (int k = 0; k < 16; ++k) atomicAdd(&s_bins[bytes[k]], 1u);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < kNumBins; b += blockDim.x) atomicAdd(&bins[b], s_bins[b]);
}

// Scalar pass over the `n - 16*n_v` trailing bytes the vectorized kernel above cannot cover.
__global__ void histogram_v2_tail(const std::uint8_t* __restrict__ data, unsigned* __restrict__ bins,
                                  std::size_t offset, std::size_t n) {
  const std::size_t i = offset + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bins[data[i]], 1u);
}

inline int grid_size(std::size_t work_items, int block) {
  return static_cast<int>((work_items + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

// Aligned prefix through the vectorized kernel, last 0-15 bytes through the scalar tail. `bins` must
// already be zeroed by the caller (both kernels atomicAdd into it).
inline void launch_v2(const std::uint8_t* d_data, unsigned* d_bins, std::size_t n, int block = 256) {
  const std::size_t n_v = n / 16;
  if (n_v > 0) histogram_v2_vectorized<<<grid_size(n_v, block), block>>>(d_data, d_bins, n_v);
  const std::size_t offset = n_v * 16, remainder = n - offset;
  if (remainder > 0) histogram_v2_tail<<<grid_size(remainder, block), block>>>(d_data, d_bins, offset, n);
}

}  // namespace portfolio::histogram::cuda
