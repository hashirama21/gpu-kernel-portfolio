// Benchmarks and validates the three convolution kernels against the CPU reference (every output pixel,
// with a separate re-check at tile corners). Requires a CUDA GPU; not built without a CUDA toolkit.

#include <cstdio>
#include <random>
#include <vector>

#include "conv2d.cuh"
#include "conv2d_reference.hpp"
#include "cuda_check.cuh"
#include "timer.hpp"

using namespace portfolio;
using namespace portfolio::conv2d;

int main(int argc, char** argv) {
  const int width = (argc > 1) ? std::atoi(argv[1]) : 1024;
  const int height = (argc > 2) ? std::atoi(argv[2]) : 1024;
  const int radius = (argc > 3) ? std::atoi(argv[3]) : 2;  // 5x5 filter
  const int fsize = 2 * radius + 1;
  std::printf("--- Project 6: 2D Convolution (%dx%d image, radius %d, %dx%d filter) ---\n", width, height, radius,
              fsize, fsize);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> h_input(static_cast<std::size_t>(width) * height), h_filter(static_cast<std::size_t>(fsize) * fsize);
  for (float& x : h_input) x = d(rng);
  for (float& x : h_filter) x = d(rng);  // random filter (not just a blur): corner bugs cannot cancel out by symmetry

  const auto ref = naive_conv2d(h_input, h_filter, width, height, radius);

  DeviceBuffer<float> d_input(h_input.size()), d_output(h_input.size()), d_filter_global(h_filter.size());
  d_input.upload(h_input);
  d_filter_global.upload(h_filter);
  cuda::upload_filter(h_filter.data(), radius);

  std::vector<float> h_out(h_input.size());
  auto validate = [&](const char* name) -> bool {
    d_output.download(h_out);
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(h_out[i] - ref[i])));
    const bool ok = worst < 1e-2;  // 5x5+ accumulation in fp32: a looser tolerance than a single FMA comparison
    std::printf("  %-16s max abs error vs CPU reference: %.4e -> %s\n", name, worst, ok ? "OK" : "MISMATCH");
    return ok;
  };

  const dim3 threads(16, 16);
  const dim3 blocks(static_cast<unsigned>((width + 15) / 16), static_cast<unsigned>((height + 15) / 16));
  bool all_ok = true;

  double ms = gpu_benchmark_ms([&] {
    cuda::conv2d_v0_naive<<<blocks, threads>>>(d_input.get(), d_filter_global.get(), d_output.get(), width, height, radius);
  });
  all_ok &= validate("v0 naive");
  report("v0 naive", ms);

  ms = gpu_benchmark_ms(
      [&] { cuda::conv2d_v1_constant<<<blocks, threads>>>(d_input.get(), d_output.get(), width, height, radius); });
  all_ok &= validate("v1 constant mem");
  report("v1 constant mem", ms);

  ms = gpu_benchmark_ms([&] { cuda::launch_v2(d_input.get(), d_output.get(), width, height, radius); });
  all_ok &= validate("v2 shared mem");
  report("v2 shared mem", ms);

  {
    d_output.download(h_out);
    double worst_corner = 0.0;
    for (int by = 0; by * 16 < height; ++by)
      for (int bx = 0; bx * 16 < width; ++bx)
        for (int dy : {0, 15})
          for (int dx : {0, 15}) {
            const int y = by * 16 + dy, x = bx * 16 + dx;
            if (y >= height || x >= width) continue;
            const std::size_t idx = static_cast<std::size_t>(y) * width + x;
            worst_corner = std::max(worst_corner, static_cast<double>(std::fabs(h_out[idx] - ref[idx])));
          }
    std::printf("  tile-corner check (where the original bug lived): max abs error %.4e -> %s\n", worst_corner,
                worst_corner < 1e-2 ? "OK" : "MISMATCH");
    all_ok &= worst_corner < 1e-2;
  }

  std::printf(all_ok ? "\nAll GPU results match the CPU reference, corners included.\n"
                     : "\nAt least one kernel MISMATCHED.\n");
  return all_ok ? 0 : 1;
}
