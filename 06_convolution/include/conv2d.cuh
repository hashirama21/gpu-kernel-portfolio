#pragma once
// 2D convolution, clamp-to-edge boundary: v0 naive, v1 filter in constant memory, v2 shared-memory
// tiling with a halo. v2 loads the whole (TILE + 2R)^2 tile (center + halo, corners included) with a
// cooperative strided loop; the halo-load logic is verified in tests/test_conv2d_cpu_sim.cpp.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace portfolio::conv2d::cuda {

constexpr int kMaxFilterSize = 15;  // supports radius up to 7; c_Filter is sized for the largest radius used
__constant__ float c_Filter[kMaxFilterSize * kMaxFilterSize];

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return max(lo, min(v, hi)); }

__global__ void conv2d_v0_naive(const float* __restrict__ input, const float* __restrict__ filter,
                                float* __restrict__ output, int width, int height, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int fsize = 2 * radius + 1;
  float sum = 0.f;
  for (int r = -radius; r <= radius; ++r)
    for (int c = -radius; c <= radius; ++c) {
      const int iy = clampi(y + r, 0, height - 1), ix = clampi(x + c, 0, width - 1);
      sum += input[iy * width + ix] * filter[(r + radius) * fsize + (c + radius)];
    }
  output[y * width + x] = sum;
}

__global__ void conv2d_v1_constant(const float* __restrict__ input, float* __restrict__ output, int width,
                                   int height, int radius) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int fsize = 2 * radius + 1;
  float sum = 0.f;
  for (int r = -radius; r <= radius; ++r)
    for (int c = -radius; c <= radius; ++c) {
      const int iy = clampi(y + r, 0, height - 1), ix = clampi(x + c, 0, width - 1);
      sum += input[iy * width + ix] * c_Filter[(r + radius) * fsize + (c + radius)];
    }
  output[y * width + x] = sum;
}

// TILE_WIDTH x TILE_WIDTH threads per block; shared tile is (TILE_WIDTH + 2*radius) on each side. Must be
// launched with dim3(TILE_WIDTH, TILE_WIDTH) threads; radius must not exceed (kMaxFilterSize-1)/2.
template <int TILE_WIDTH>
__global__ void conv2d_v2_shared(const float* __restrict__ input, float* __restrict__ output, int width,
                                 int height, int radius) {
  extern __shared__ float s_tile[];  // (TILE_WIDTH + 2*radius)^2 floats
  const int shared_w = TILE_WIDTH + 2 * radius;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * TILE_WIDTH + tx;             // linear thread id within the block
  const int num_threads = TILE_WIDTH * TILE_WIDTH;
  const int total_shared = shared_w * shared_w;
  const int block_x = blockIdx.x * TILE_WIDTH, block_y = blockIdx.y * TILE_WIDTH;

  // Cooperative load covering the whole shared tile (center + halo, corners included): every linear
  // shared-memory index in [0, total_shared) is written by exactly one thread.
  for (int lin = tid; lin < total_shared; lin += num_threads) {
    const int sy = lin / shared_w, sx = lin % shared_w;
    const int gy = block_y + sy - radius, gx = block_x + sx - radius;
    const int cy = clampi(gy, 0, height - 1), cx = clampi(gx, 0, width - 1);
    s_tile[sy * shared_w + sx] = input[cy * width + cx];
  }
  __syncthreads();

  const int out_x = block_x + tx, out_y = block_y + ty;
  if (out_x >= width || out_y >= height) return;
  const int fsize = 2 * radius + 1;
  float sum = 0.f;
  for (int r = 0; r < fsize; ++r)
    for (int c = 0; c < fsize; ++c) sum += s_tile[(ty + r) * shared_w + (tx + c)] * c_Filter[r * fsize + c];
  output[out_y * width + out_x] = sum;
}

inline void upload_filter(const float* h_filter, int radius) {
  const int fsize = 2 * radius + 1;
  if (radius < 0 || fsize > kMaxFilterSize)
    throw std::invalid_argument("upload_filter: radius must be in [0, (kMaxFilterSize-1)/2]");
  cudaMemcpyToSymbol(c_Filter, h_filter, static_cast<std::size_t>(fsize) * fsize * sizeof(float));
}

// Launches v2 with the shared-memory size the kernel needs; TILE_WIDTH is fixed at 16 (a template
// parameter on the kernel) to keep the launch site simple.
inline void launch_v2(const float* d_input, float* d_output, int width, int height, int radius) {
  constexpr int TILE = 16;
  const dim3 threads(TILE, TILE);
  const dim3 blocks((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
  const int shared_w = TILE + 2 * radius;
  const std::size_t shared_bytes = static_cast<std::size_t>(shared_w) * shared_w * sizeof(float);
  conv2d_v2_shared<TILE><<<blocks, threads, shared_bytes>>>(d_input, d_output, width, height, radius);
}

}  // namespace portfolio::conv2d::cuda
