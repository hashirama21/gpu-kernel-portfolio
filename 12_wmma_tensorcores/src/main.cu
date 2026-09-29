// Benchmarks and validates the padded WMMA GEMM against a CPU reference, using M/N/K not multiples of the
// tile alignment. Requires a Tensor Core GPU (compute capability >= 70); not built without a CUDA toolkit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "timer.hpp"
#include "wmma_gemm.cuh"

using namespace portfolio;

static std::vector<float> cpu_matmul(const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N) {
  std::vector<float> C(static_cast<std::size_t>(M) * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += static_cast<double>(A[static_cast<std::size_t>(i) * K + k]) * B[static_cast<std::size_t>(k) * N + j];
      C[static_cast<std::size_t>(i) * N + j] = static_cast<float>(acc);
    }
  return C;
}

int main(int argc, char** argv) {
  // Not multiples of 64 (M) / 16 (N, K): exactly what the original grid-sizing bug could not handle.
  const int M = (argc > 1) ? std::atoi(argv[1]) : 100;
  const int K = (argc > 2) ? std::atoi(argv[2]) : 80;
  const int N = (argc > 3) ? std::atoi(argv[3]) : 50;
  std::printf("--- Project 12: WMMA Tensor Core GEMM (M=%d, K=%d, N=%d, none tile-aligned) ---\n", M, K, N);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> h_A(static_cast<std::size_t>(M) * K), h_B(static_cast<std::size_t>(K) * N), h_C(static_cast<std::size_t>(M) * N);
  for (float& x : h_A) x = dist(rng);
  for (float& x : h_B) x = dist(rng);
  const auto ref = cpu_matmul(h_A, h_B, M, K, N);

  const double ms = benchmark_ms([&] { wmma_gemm::cuda::launch_wmma_gemm(h_A.data(), h_B.data(), h_C.data(), M, K, N); }, 1, 3);

  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(h_C[i] - ref[i])));
  // fp16 inputs: a much looser tolerance than the fp32 GEMM projects is expected and correct here.
  const bool ok = worst < 0.5;
  std::printf("  max abs error vs CPU (fp32) reference: %.4e -> %s\n", worst, ok ? "OK (within fp16 precision)" : "MISMATCH");
  report("wmma gemm (padded)", ms, 2.0 * M * N * K / (ms / 1e3) / 1e12, "TFLOPS");

  return ok ? 0 : 1;
}
