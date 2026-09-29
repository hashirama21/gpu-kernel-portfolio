#pragma once
// FP16-input / FP32-accumulate GEMM using the WMMA (Tensor Core) API. launch_wmma_gemm pads A, B, and C
// up to the tile alignment (M to a multiple of 64, N and K to a multiple of 16) into zero-filled device
// buffers, runs the kernel over the fully tile-aligned problem, and copies back the true M x N region —
// so every warp's tile is either all real data or all zero padding, the grid covers every output, and
// wmma::load/store_matrix_sync (which have no bounds checking) never touch out-of-bounds memory.
// The padding premise is verified in tests/test_wmma_padding_sim.cpp. Requires a Tensor Core GPU
// (compute capability >= 70) and a CUDA toolkit; not built here.

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.cuh"

namespace portfolio::wmma_gemm::cuda {

using namespace nvcuda;

constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 16;
constexpr int kWarpsPerBlockY = 4;  // dim3 threadsPerBlock(32, 4): 4 warps stacked on the M axis

__global__ void wmma_gemm_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
                                 int M, int N, int K) {
  const int warpM = (blockIdx.y * blockDim.y + threadIdx.y) * kWmmaM;
  const int warpN = blockIdx.x * kWmmaN;  // one warp column per block: (blockIdx.x*32 + tx)/32 == blockIdx.x
  if (warpM >= M || warpN >= N) return;

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < K; k += kWmmaK) {
    wmma::load_matrix_sync(a_frag, A + static_cast<std::size_t>(warpM) * K + k, K);
    wmma::load_matrix_sync(b_frag, B + static_cast<std::size_t>(k) * N + warpN, N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(C + static_cast<std::size_t>(warpM) * N + warpN, c_frag, N, wmma::mem_row_major);
}

inline int round_up(int v, int multiple) { return ((v + multiple - 1) / multiple) * multiple; }

// h_A, h_B: row-major, M*K and K*N floats. h_C: caller-provided output buffer, M*N floats.
inline void launch_wmma_gemm(const float* h_A, const float* h_B, float* h_C, int M, int K, int N) {
  const int Mp = round_up(M, kWarpsPerBlockY * kWmmaM), Kp = round_up(K, kWmmaK), Np = round_up(N, kWmmaN);

  std::vector<half> h_Ap(static_cast<std::size_t>(Mp) * Kp, __float2half(0.0f));
  std::vector<half> h_Bp(static_cast<std::size_t>(Kp) * Np, __float2half(0.0f));
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) h_Ap[static_cast<std::size_t>(i) * Kp + k] = __float2half(h_A[static_cast<std::size_t>(i) * K + k]);
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j) h_Bp[static_cast<std::size_t>(k) * Np + j] = __float2half(h_B[static_cast<std::size_t>(k) * N + j]);

  DeviceBuffer<half> d_Ap(h_Ap.size()), d_Bp(h_Bp.size());
  DeviceBuffer<float> d_Cp(static_cast<std::size_t>(Mp) * Np);
  d_Ap.upload(h_Ap);
  d_Bp.upload(h_Bp);
  d_Cp.zero();

  const dim3 threads(32, kWarpsPerBlockY);
  const dim3 blocks(static_cast<unsigned>(Np / kWmmaN), static_cast<unsigned>(Mp / (kWarpsPerBlockY * kWmmaM)));
  wmma_gemm_kernel<<<blocks, threads>>>(d_Ap.get(), d_Bp.get(), d_Cp.get(), Mp, Np, Kp);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(std::string("wmma_gemm_kernel launch failed: ") + cudaGetErrorString(err));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_Cp(static_cast<std::size_t>(Mp) * Np);
  d_Cp.download(h_Cp);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) h_C[static_cast<std::size_t>(i) * N + j] = h_Cp[static_cast<std::size_t>(i) * Np + j];
}

}  // namespace portfolio::wmma_gemm::cuda
