// Benchmarks and validates the two GEMM kernels against a CPU reference (every element), reporting TFLOPS
// only after validating. Requires a CUDA GPU; not built without a CUDA toolkit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "gemm.cuh"
#include "timer.hpp"

using namespace portfolio;
using namespace portfolio::gemm::cuda;

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
  const int M = (argc > 1) ? std::atoi(argv[1]) : 1024;  // smaller than the 2048^3 draft default, so the
  const int K = (argc > 2) ? std::atoi(argv[2]) : 1024;  // CPU reference here finishes in reasonable time
  const int N = (argc > 3) ? std::atoi(argv[3]) : 1024;
  std::printf("--- Project 9: GEMM (M=%d, K=%d, N=%d) ---\n", M, K, N);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> h_A(static_cast<std::size_t>(M) * K), h_B(static_cast<std::size_t>(K) * N);
  for (float& x : h_A) x = dist(rng);
  for (float& x : h_B) x = dist(rng);
  std::printf("  computing CPU reference (O(M*N*K), please wait)...\n");
  const auto ref = cpu_matmul(h_A, h_B, M, K, N);

  DeviceBuffer<float> d_A(h_A.size()), d_B(h_B.size()), d_C(ref.size());
  d_A.upload(h_A);
  d_B.upload(h_B);

  const dim3 threads(kTileSize, kTileSize);
  const dim3 blocks(static_cast<unsigned>((N + kTileSize - 1) / kTileSize), static_cast<unsigned>((M + kTileSize - 1) / kTileSize));
  const double flops = 2.0 * M * N * K;

  std::vector<float> h_C(ref.size());
  auto validate = [&](const char* name) -> bool {
    d_C.download(h_C);
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(h_C[i] - ref[i])));
    const bool ok = worst < 1e-1;  // fp32 accumulation over K terms: tolerance scales loosely with K
    std::printf("  %-16s max abs error: %.4e -> %s\n", name, worst, ok ? "OK" : "MISMATCH");
    return ok;
  };

  bool all_ok = true;
  double ms = gpu_benchmark_ms([&] { gemm_v0_naive<<<blocks, threads>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
  all_ok &= validate("v0 naive");
  report("v0 naive", ms, flops / (ms / 1e3) / 1e12, "TFLOPS");

  d_C.zero();
  ms = gpu_benchmark_ms([&] { gemm_v1_shared<<<blocks, threads>>>(d_A.get(), d_B.get(), d_C.get(), M, N, K); });
  all_ok &= validate("v1 shared tiling");
  report("v1 shared tiling", ms, flops / (ms / 1e3) / 1e12, "TFLOPS");

  std::printf(all_ok ? "\nBoth kernels match the CPU reference.\n" : "\nMISMATCH.\n");
  return all_ok ? 0 : 1;
}
