#pragma once
// Parallel reduction (sum): v0 naive, v1 sequential addressing, v2 warp-shuffle (Mark Harris' classic
// progression). v2's warp/block structure is verified in tests/test_reduction_cpu_sim.cpp.

#include <cuda_runtime.h>

#include <cstddef>

namespace portfolio::reduction::cuda {

constexpr int kBlockSize = 256;

// v0: naive. `tid % (2*s) == 0` diverges within every warp once s > 0, and threads access sdata with a
// stride of 2*s, which also causes shared-memory bank conflicts for several early values of s.
__global__ void reduce_v0_naive(const float* __restrict__ g_idata, float* __restrict__ g_odata, std::size_t n) {
  __shared__ float sdata[kBlockSize];
  const unsigned tid = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
  __syncthreads();

  for (unsigned s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// v1: sequential addressing. Active threads are always a contiguous prefix [0, s) of the block, so no
// warp is ever partially active until s < 32, and shared-memory accesses are contiguous (no bank
// conflicts).
__global__ void reduce_v1_sequential(const float* __restrict__ g_idata, float* __restrict__ g_odata, std::size_t n) {
  __shared__ float sdata[kBlockSize];
  const unsigned tid = threadIdx.x;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
  __syncthreads();

  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(g_odata, sdata[0]);
}

// v2: warp-shuffle. Verified structure (see file header): each thread accumulates a grid-stride partial
// sum, each warp reduces its 32 partial sums with __shfl_down_sync (no shared memory, no __syncthreads),
// the per-warp results go into a tiny shared array, and warp 0 does one more shuffle-reduction over
// those. blockDim.x must be a multiple of 32 (enforced by kBlockSize = 256) so every warp is full.
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(0xffffffffu, val, offset);
  return val;
}

__global__ void reduce_v2_warp_shuffle(const float* __restrict__ g_idata, float* __restrict__ g_odata, std::size_t n) {
  __shared__ float shared[32];  // one slot per warp (blockDim.x / 32 <= 32 for any block size <= 1024)
  float sum = 0.0f;

  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t i = idx; i < n; i += stride) sum += g_idata[i];

  sum = warp_reduce_sum(sum);

  const int lane = static_cast<int>(threadIdx.x % 32), warp_id = static_cast<int>(threadIdx.x / 32);
  if (lane == 0) shared[warp_id] = sum;
  __syncthreads();

  sum = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (warp_id == 0) {
    sum = warp_reduce_sum(sum);
    if (lane == 0) atomicAdd(g_odata, sum);
  }
}

}  // namespace portfolio::reduction::cuda
