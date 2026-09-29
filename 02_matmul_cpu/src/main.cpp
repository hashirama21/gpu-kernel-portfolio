#include <cstdio>
#include <random>
#include <vector>

#include "matmul.hpp"
#include "timer.hpp"

using namespace portfolio;
using namespace portfolio::matmul;

int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 512;
  const Dims d{n, n, n};
  std::printf("--- Project 2: Matrix Multiplication (%d x %d x %d, fp32) ---\n", n, n, n);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> A(static_cast<std::size_t>(n) * n), B(static_cast<std::size_t>(n) * n), C(static_cast<std::size_t>(n) * n);
  for (auto& x : A) x = dist(rng);
  for (auto& x : B) x = dist(rng);

  const double flops = 2.0 * n * n * n;
  auto bench = [&](const char* name, auto&& fn) {
    const double ms = benchmark_ms([&] { fn(A, B, C, d); }, 1, 3);
    report(name, ms, flops / (ms / 1e3) / 1e9, "GFLOP/s");
  };

  bench("naive (i-j-k)", [](auto&... a) { naive(a...); });
  bench("ikj", [](auto&... a) { ikj(a...); });
  bench("tiled (64)", [](auto&... a) { tiled(a..., 64); });
#if defined(__AVX2__)
  bench("tiled + AVX2", [](auto&... a) { tiled_avx2(a..., 64); });
  bench("tiled + AVX2 + OpenMP", [](auto&... a) { tiled_avx2_omp(a..., 64); });
#else
  std::printf("  (AVX2 not enabled by the compiler on this build)\n");
#endif
  return 0;
}
