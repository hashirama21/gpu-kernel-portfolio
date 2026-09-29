#pragma once
// Row-wise softmax: v0 safe 3-pass, v1 online 1-pass (Milakov & Gimelshein), one warp (32 threads) per
// row. blockDim.x must be 32 (enforced by check_launch_config). Verified in tests/test_softmax_cpu_sim.cpp.

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace portfolio::softmax::cuda {

struct OnlineState {
  float max_val, sum;
};

__device__ __forceinline__ OnlineState combine_online_states(OnlineState a, OnlineState b) {
  if (a.max_val > b.max_val) return {a.max_val, a.sum + b.sum * __expf(b.max_val - a.max_val)};
  return {b.max_val, b.sum + a.sum * __expf(a.max_val - b.max_val)};
}

__device__ __forceinline__ OnlineState warp_reduce_online_softmax(OnlineState state) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    OnlineState other;
    other.max_val = __shfl_down_sync(0xffffffffu, state.max_val, offset);
    other.sum = __shfl_down_sync(0xffffffffu, state.sum, offset);
    state = combine_online_states(state, other);
  }
  return state;
}

__global__ void softmax_v0_safe_3pass(const float* __restrict__ input, float* __restrict__ output, int rows,
                                      int cols) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const float* row_in = input + static_cast<std::size_t>(row) * cols;
  float* row_out = output + static_cast<std::size_t>(row) * cols;

  float local_max = -1e30f;
  for (int col = lane; col < cols; col += 32) local_max = fmaxf(local_max, row_in[col]);
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
  const float global_max = __shfl_sync(0xffffffffu, local_max, 0);

  float local_sum = 0.0f;
  for (int col = lane; col < cols; col += 32) local_sum += __expf(row_in[col] - global_max);
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) local_sum += __shfl_down_sync(0xffffffffu, local_sum, offset);
  const float global_sum = __shfl_sync(0xffffffffu, local_sum, 0);

  const float inv_sum = 1.0f / global_sum;
  for (int col = lane; col < cols; col += 32) row_out[col] = __expf(row_in[col] - global_max) * inv_sum;
}

__global__ void softmax_v1_online(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  const int lane = threadIdx.x;
  const float* row_in = input + static_cast<std::size_t>(row) * cols;
  float* row_out = output + static_cast<std::size_t>(row) * cols;

  OnlineState state{-1e30f, 0.0f};
  for (int col = lane; col < cols; col += 32) {
    const float val = row_in[col];
    if (val > state.max_val) {
      state.sum = state.sum * __expf(state.max_val - val) + 1.0f;
      state.max_val = val;
    } else {
      state.sum += __expf(val - state.max_val);
    }
  }
  state = warp_reduce_online_softmax(state);
  const float final_max = __shfl_sync(0xffffffffu, state.max_val, 0);
  const float final_sum = __shfl_sync(0xffffffffu, state.sum, 0);
  const float inv_sum = 1.0f / final_sum;

  for (int col = lane; col < cols; col += 32) row_out[col] = __expf(row_in[col] - final_max) * inv_sum;
}

// Enforces the one-warp-per-row invariant the kernels rely on.
inline void check_launch_config(int threads_x, int threads_y) {
  if (threads_x != 32) throw std::invalid_argument("softmax kernels require blockDim.x == 32 (one warp per row)");
  if (threads_y <= 0) throw std::invalid_argument("blockDim.y must be positive");
}

}  // namespace portfolio::softmax::cuda
