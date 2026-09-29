// Benchmarks and validates the softmax kernels against a CPU reference (every row's every element).
// Requires a CUDA GPU; not built without a CUDA toolkit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "softmax.cuh"
#include "timer.hpp"

using namespace portfolio;
using namespace portfolio::softmax::cuda;

static std::vector<float> cpu_reference(const std::vector<float>& in, int rows, int cols) {
  std::vector<float> out(in.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = in.data() + static_cast<std::size_t>(r) * cols;
    float* orow = out.data() + static_cast<std::size_t>(r) * cols;
    float m = row[0];
    for (int c = 1; c < cols; ++c) m = std::max(m, row[c]);
    double z = 0;
    for (int c = 0; c < cols; ++c) { orow[c] = std::exp(row[c] - m); z += orow[c]; }
    for (int c = 0; c < cols; ++c) orow[c] = static_cast<float>(orow[c] / z);
  }
  return out;
}

int main(int argc, char** argv) {
  const int rows = (argc > 1) ? std::atoi(argv[1]) : 8192;
  const int cols = (argc > 2) ? std::atoi(argv[2]) : 4096;
  std::printf("--- Project 7: Softmax (%d x %d, %.1f MB) ---\n", rows, cols,
              static_cast<double>(rows) * cols * sizeof(float) / (1024.0 * 1024.0));

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-10.f, 10.f);
  std::vector<float> h_in(static_cast<std::size_t>(rows) * cols);
  for (float& x : h_in) x = dist(rng);
  const auto ref = cpu_reference(h_in, rows, cols);

  DeviceBuffer<float> d_in(h_in.size()), d_out(h_in.size());
  d_in.upload(h_in);

  softmax::cuda::check_launch_config(32, 8);
  const dim3 threads(32, 8);
  const dim3 blocks(static_cast<unsigned>((rows + 7) / 8));

  std::vector<float> h_out(h_in.size());
  auto validate = [&](const char* name) -> bool {
    d_out.download(h_out);
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(h_out[i] - ref[i])));
    const bool ok = worst < 1e-4;
    std::printf("  %-14s max abs error over all %d rows: %.3e -> %s\n", name, rows, worst, ok ? "OK" : "MISMATCH");
    return ok;
  };

  bool all_ok = true;
  double ms = gpu_benchmark_ms([&] { softmax_v0_safe_3pass<<<blocks, threads>>>(d_in.get(), d_out.get(), rows, cols); });
  all_ok &= validate("v0 3-pass");
  report("v0 3-pass", ms, 2.0 * h_in.size() * sizeof(float) / (ms / 1e3) / 1e9, "GB/s");

  ms = gpu_benchmark_ms([&] { softmax_v1_online<<<blocks, threads>>>(d_in.get(), d_out.get(), rows, cols); });
  all_ok &= validate("v1 online");
  report("v1 online", ms, 2.0 * h_in.size() * sizeof(float) / (ms / 1e3) / 1e9, "GB/s");

  std::printf(all_ok ? "\nAll GPU results match the CPU reference, every row.\n" : "\nMISMATCH.\n");
  return all_ok ? 0 : 1;
}
