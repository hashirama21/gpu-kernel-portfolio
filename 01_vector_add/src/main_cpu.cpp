// CPU-only benchmark: naive vs AVX2 vs OpenMP.

#include <cstdio>
#include <random>
#include <vector>

#include "timer.hpp"
#include "vector_add.hpp"

using namespace portfolio;

int main(int argc, char** argv) {
  const std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::atoll(argv[1])) : (std::size_t{1} << 26);
  std::printf("--- Project 1 (CPU-only): Vector Add (N = %zu elements) ---\n", n);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-10.f, 10.f);
  std::vector<float> a(n), b(n), c(n);
  for (auto& x : a) x = d(rng);
  for (auto& x : b) x = d(rng);

  const double bytes = 3.0 * n * sizeof(float);
  auto bench = [&](const char* name, auto&& fn) {
    const double ms = benchmark_ms([&] { fn(a, b, c); });
    report(name, ms, bytes / (ms / 1e3) / 1e9, "GB/s");
  };

  bench("CPU naive", vecadd::add_naive);
#if defined(__AVX2__)
  bench("CPU AVX2", vecadd::add_avx2);
#else
  std::printf("  (AVX2 not enabled by the compiler on this build)\n");
#endif
  bench("CPU OpenMP", vecadd::add_openmp);
  return 0;
}
