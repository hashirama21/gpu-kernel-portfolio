#pragma once
// GEMM: v0 naive, v1 shared-memory tiling. The tiling/boundary logic (including non-tile-aligned M/N/K)
// is verified in tests/test_gemm_cpu_sim.cpp.

#include <cuda_runtime.h>

namespace portfolio::gemm::cuda {

constexpr int kTileSize = 32;

__global__ void gemm_v0_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                              int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float sum = 0.0f;
  for (int i = 0; i < K; ++i) sum += A[row * K + i] * B[i * N + col];
  C[row * N + col] = sum;
}

__global__ void gemm_v1_shared(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                               int M, int N, int K) {
  __shared__ float s_A[kTileSize][kTileSize];
  __shared__ float s_B[kTileSize][kTileSize];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * kTileSize + ty, col = blockIdx.x * kTileSize + tx;

  float sum = 0.0f;
  const int num_phases = (K + kTileSize - 1) / kTileSize;
  for (int p = 0; p < num_phases; ++p) {
    s_A[ty][tx] = (row < M && (p * kTileSize + tx) < K) ? A[row * K + (p * kTileSize + tx)] : 0.0f;
    s_B[ty][tx] = (col < N && (p * kTileSize + ty) < K) ? B[(p * kTileSize + ty) * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kTileSize; ++i) sum += s_A[ty][i] * s_B[i][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = sum;
}

}  // namespace portfolio::gemm::cuda
