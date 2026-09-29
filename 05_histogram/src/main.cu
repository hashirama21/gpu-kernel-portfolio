// Benchmarks and validates the three histogram kernels against a CPU reference.
// Requires a CUDA GPU; not built without a CUDA toolkit.

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "histogram.cuh"
#include "timer.hpp"

using namespace portfolio;
using namespace portfolio::histogram::cuda;

static std::vector<unsigned> cpu_reference(const std::vector<std::uint8_t>& data) {
  std::vector<unsigned> bins(kNumBins, 0);
  for (std::uint8_t b : data) bins[b]++;
  return bins;
}

int main(int argc, char** argv) {
  // Deliberately NOT a multiple of 16, to actually exercise the tail-handling fix in v2.
  const std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::atoll(argv[1])) : (std::size_t{1} << 28) + 7;
  std::printf("--- Project 5: Histogram (N = %zu bytes, 256 bins) ---\n", n);

  std::mt19937 rng(1);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<std::uint8_t> h_data(n);
  for (auto& b : h_data) b = static_cast<std::uint8_t>(dist(rng));
  const auto ref = cpu_reference(h_data);

  DeviceBuffer<std::uint8_t> d_data(n);
  d_data.upload(h_data);
  DeviceBuffer<unsigned> d_bins(kNumBins);

  const int threads = 256;
  int num_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
  const int blocks = std::max(1, num_sm) * 8;

  auto validate = [&](const char* name, const std::vector<unsigned>& got) -> bool {
    unsigned total = 0;
    bool ok = true;
    for (int i = 0; i < kNumBins; ++i) {
      total += got[static_cast<std::size_t>(i)];
      if (got[static_cast<std::size_t>(i)] != ref[static_cast<std::size_t>(i)]) ok = false;
    }
    std::printf("  %-12s total counted %u / %zu  -> %s\n", name, total, n, ok && total == n ? "OK" : "MISMATCH");
    return ok && total == n;
  };

  std::vector<unsigned> h_bins(kNumBins);
  bool all_ok = true;

  d_bins.zero();
  double ms = gpu_benchmark_ms([&] { d_bins.zero(); histogram_v0_global_atomic<<<blocks, threads>>>(d_data.get(), d_bins.get(), n); });
  d_bins.download(h_bins);
  all_ok &= validate("v0 global", h_bins);
  report("v0 global atomic", ms, static_cast<double>(n) / (ms / 1e3) / 1e9, "Gbytes/s");

  ms = gpu_benchmark_ms([&] { d_bins.zero(); histogram_v1_shared<<<blocks, threads>>>(d_data.get(), d_bins.get(), n); });
  d_bins.download(h_bins);
  all_ok &= validate("v1 shared", h_bins);
  report("v1 shared mem", ms, static_cast<double>(n) / (ms / 1e3) / 1e9, "Gbytes/s");

  ms = gpu_benchmark_ms([&] { d_bins.zero(); launch_v2(d_data.get(), d_bins.get(), n, threads); });
  d_bins.download(h_bins);
  all_ok &= validate("v2 vectorized", h_bins);
  report("v2 vectorized+tail", ms, static_cast<double>(n) / (ms / 1e3) / 1e9, "Gbytes/s");

  std::printf(all_ok ? "\nAll GPU results match the CPU reference (including the N %% 16 != 0 tail).\n"
                     : "\nAt least one kernel MISMATCHED.\n");
  return all_ok ? 0 : 1;
}
