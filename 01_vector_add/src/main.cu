// Benchmarks and validates the CUDA vector-add kernels against the CPU reference (every element checked,
// with a reported max error). Requires a CUDA GPU; not built without a CUDA toolkit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "timer.hpp"
#include "vector_add.cuh"
#include "vector_add.hpp"

using namespace portfolio;

static std::vector<float> random_vec(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-100.f, 100.f);
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

// Returns the number of mismatching elements (0 = pass) and prints the worst absolute error found.
static int validate(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
  double worst = 0.0;
  int bad = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
    worst = std::max(worst, err);
    if (err > 1e-4) ++bad;
  }
  std::printf("  %-20s max abs error %.3e over %zu elements -> %s\n", name, worst, ref.size(),
              bad == 0 ? "OK" : "MISMATCH");
  return bad;
}

int main(int argc, char** argv) {
  const std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::atoll(argv[1])) : (std::size_t{1} << 27) + 3;
  const std::size_t bytes = n * sizeof(float);
  std::printf("--- Project 1: Vector Add (N = %zu elements, %.1f MB per array) ---\n", n,
              static_cast<double>(bytes) / (1024.0 * 1024.0));

  const auto h_a = random_vec(n, 1), h_b = random_vec(n, 2);
  std::vector<float> h_ref(n);
  vecadd::add_naive(h_a, h_b, h_ref);

  DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
  d_a.upload(h_a);
  d_b.upload(h_b);

  const int block = 256;
  int fails = 0;
  std::vector<float> h_c(n);

  // Naive
  d_c.zero();
  const double ms_naive = gpu_benchmark_ms([&] { vecadd::cuda::add_naive<<<vecadd::cuda::grid_size(n, block), block>>>(d_a.get(), d_b.get(), d_c.get(), n); });
  d_c.download(h_c);
  fails += validate("naive", h_c, h_ref);
  report("GPU naive", ms_naive, 3.0 * bytes / (ms_naive / 1e3) / 1e9, "GB/s");

  // Grid-stride
  d_c.zero();
  const int num_sm = [] { int v = 0; cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, 0); return v; }();
  const int stride_grid = std::max(1, num_sm) * 8;
  const double ms_stride = gpu_benchmark_ms([&] { vecadd::cuda::add_grid_stride<<<stride_grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n); });
  d_c.download(h_c);
  fails += validate("grid-stride", h_c, h_ref);
  report("GPU grid-stride", ms_stride, 3.0 * bytes / (ms_stride / 1e3) / 1e9, "GB/s");

  // Vectorized float4 + scalar tail (correct now for any n, see vector_add.cuh)
  d_c.zero();
  const double ms_vec = gpu_benchmark_ms([&] { vecadd::cuda::launch_vectorized(d_a.get(), d_b.get(), d_c.get(), n, block); });
  d_c.download(h_c);
  fails += validate("vectorized (float4)", h_c, h_ref);
  report("GPU vectorized", ms_vec, 3.0 * bytes / (ms_vec / 1e3) / 1e9, "GB/s");

  std::printf(fails == 0 ? "\nAll GPU results match the CPU reference.\n" : "\n%d validation failure(s).\n", fails);
  return fails == 0 ? 0 : 1;
}
