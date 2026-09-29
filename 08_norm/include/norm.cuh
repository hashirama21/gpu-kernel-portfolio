#pragma once
// LayerNorm and RMSNorm. block_reduce_sum does a warp reduce + shared-memory cross-warp reduce +
// broadcast, so the kernels are correct for any blockDim.x that is a multiple of 32 (up to 1024).
// eps is added inside the sqrt (rsqrt(mean(x^2) + eps)), matching the LLaMA/RMSNorm convention.
// Verified in tests/test_norm_cpu_sim.cpp.

#include <cuda_runtime.h>

#include <cstddef>

namespace portfolio::norm::cuda {

constexpr int kMaxWarpsPerBlock = 32;  // supports blockDim.x up to 1024

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) val += __shfl_down_sync(0xffffffffu, val, offset);
  return val;
}

// Full block reduction: every thread's `val` is combined into one total and every thread receives it.
// `scratch` must point to at least kMaxWarpsPerBlock floats of __shared__ memory.
__device__ __forceinline__ float block_reduce_sum(float val, float* scratch) {
  const int lane = threadIdx.x % 32, warp_id = threadIdx.x / 32;
  val = warp_reduce_sum(val);
  if (lane == 0) scratch[warp_id] = val;
  __syncthreads();
  const int n_warps = (blockDim.x + 31) / 32;
  float total = (threadIdx.x < static_cast<unsigned>(n_warps)) ? scratch[threadIdx.x] : 0.0f;
  if (warp_id == 0) total = warp_reduce_sum(total);
  if (threadIdx.x == 0) scratch[0] = total;
  __syncthreads();
  return scratch[0];
}

__global__ void layernorm(const float* __restrict__ input, const float* __restrict__ gamma,
                          const float* __restrict__ beta, float* __restrict__ output, int D, float epsilon) {
  __shared__ float scratch[kMaxWarpsPerBlock];
  const int row = blockIdx.x, tid = threadIdx.x;
  const float* x = input + static_cast<std::size_t>(row) * D;
  float* y = output + static_cast<std::size_t>(row) * D;

  float local_sum = 0.0f;
  for (int i = tid; i < D; i += blockDim.x) local_sum += x[i];
  const float mean = block_reduce_sum(local_sum, scratch) / D;
  __syncthreads();  // scratch is reused by the next reduction below

  float local_var = 0.0f;
  for (int i = tid; i < D; i += blockDim.x) {
    const float d = x[i] - mean;
    local_var += d * d;
  }
  const float variance = block_reduce_sum(local_var, scratch) / D;
  const float inv_stddev = rsqrtf(variance + epsilon);

  for (int i = tid; i < D; i += blockDim.x) y[i] = (x[i] - mean) * inv_stddev * gamma[i] + beta[i];
}

__global__ void rmsnorm(const float* __restrict__ input, const float* __restrict__ gamma,
                        float* __restrict__ output, int D, float epsilon) {
  __shared__ float scratch[kMaxWarpsPerBlock];
  const int row = blockIdx.x, tid = threadIdx.x;
  const float* x = input + static_cast<std::size_t>(row) * D;
  float* y = output + static_cast<std::size_t>(row) * D;

  float local_sq_sum = 0.0f;
  for (int i = tid; i < D; i += blockDim.x) {
    const float v = x[i];
    local_sq_sum += v * v;
  }
  const float sq_sum = block_reduce_sum(local_sq_sum, scratch);
  const float rms = rsqrtf(sq_sum / D + epsilon);

  for (int i = tid; i < D; i += blockDim.x) y[i] = x[i] * rms * gamma[i];
}

// Vectorized RMSNorm: float4 fast path over the aligned prefix, scalar tail for the remainder.
__global__ void rmsnorm_vectorized(const float* __restrict__ input, const float* __restrict__ gamma,
                                   float* __restrict__ output, int D, float epsilon) {
  __shared__ float scratch[kMaxWarpsPerBlock];
  const int row = blockIdx.x, tid = threadIdx.x;
  const float* x = input + static_cast<std::size_t>(row) * D;
  float* y = output + static_cast<std::size_t>(row) * D;
  const float4* x_vec = reinterpret_cast<const float4*>(x);
  const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
  float4* y_vec = reinterpret_cast<float4*>(y);
  // float4 loads require a 16-byte-aligned base; the per-row base `input + row*D` is only aligned when
  // D % 4 == 0, so fall back to the scalar loop for the whole row otherwise (misaligned 128-bit loads
  // are a hard runtime error on the GPU, which the element-wise CPU simulation cannot surface).
  const int d_vec = (D % 4 == 0) ? D / 4 : 0, tail_start = d_vec * 4;

  float local_sq_sum = 0.0f;
  for (int i = tid; i < d_vec; i += blockDim.x) {
    const float4 v = x_vec[i];
    local_sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (int i = tail_start + tid; i < D; i += blockDim.x) {  // the fix: scalar tail
    const float v = x[i];
    local_sq_sum += v * v;
  }
  const float sq_sum = block_reduce_sum(local_sq_sum, scratch);
  const float rms = rsqrtf(sq_sum / D + epsilon);

  for (int i = tid; i < d_vec; i += blockDim.x) {
    const float4 v = x_vec[i], g = gamma_vec[i];
    y_vec[i] = float4{v.x * rms * g.x, v.y * rms * g.y, v.z * rms * g.z, v.w * rms * g.w};
  }
  for (int i = tail_start + tid; i < D; i += blockDim.x) y[i] = x[i] * rms * gamma[i];
}

}  // namespace portfolio::norm::cuda
