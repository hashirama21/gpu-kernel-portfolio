// Benchmarks and validates the three reduction kernels against a double-precision CPU reference.
// Requires a CUDA GPU; not built without a CUDA toolkit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "reduction.cuh"
#include "timer.hpp"

using namespace portfolio;
using namespace portfolio::reduction::cuda;

int main(int argc, char** argv) {
  const std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::atoll(argv[1])) : (std::size_t{1} << 26);
  std::printf("--- Project 4: Parallel Reduction (N = %zu float32, %.1f MB) ---\n", n,
              static_cast<double>(n * sizeof(float)) / (1024.0 * 1024.0));

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> h_data(n);
  for (float& x : h_data) x = dist(rng);

  // Double-precision reference: for a sum of ~1e7-1e8 terms, plain float accumulation on the CPU would
  // itself accumulate meaningful rounding error, so it would be the wrong thing to compare the GPU
  // result against. Summing in double keeps the reference accurate to roughly float epsilon.
  double ref = 0.0;
  for (float x : h_data) ref += x;

  DeviceBuffer<float> d_data(n);
  d_data.upload(h_data);
  DeviceBuffer<float> d_out(1);

  const int threads = kBlockSize;
  int num_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
  const int grid_stride_blocks = std::max(1, num_sm) * 8;
  const int grid_direct = static_cast<int>((n + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads));

  // v0 and v1 launch one thread per element directly (grid_direct); v2 uses a grid-stride loop over a
  // fixed, SM-count-sized grid, which is the standard way to size a reduction kernel that internally
  // loops (oversubscribing the SMs would just add scheduling overhead with no benefit here).
  auto run = [&](const char* name, auto&& launch) -> bool {
    // `launch` itself zeroes d_out before every call (warm-up and timed alike), so the value downloaded
    // below is always the correct fresh result of the final timed launch, not a stale accumulation.
    const double ms = gpu_benchmark_ms(launch);
    float got = 0.f;
    d_out.download(std::span<float>(&got, 1));
    const double bytes = static_cast<double>(n) * sizeof(float);
    const double rel_err = std::fabs(static_cast<double>(got) - ref) / std::max(1.0, std::fabs(ref));
    // float32 accumulation error grows roughly with sqrt(N) * epsilon in the worst realistic case;
    // 1e-4 comfortably covers that for N up to ~1e8 while still catching a genuinely wrong algorithm.
    const bool ok = rel_err < 1e-4;
    std::printf("  %-18s got %.6f  ref %.6f  rel.err %.2e  -> %s\n", name, static_cast<double>(got), ref, rel_err,
                ok ? "OK" : "MISMATCH");
    report(name, ms, bytes / (ms / 1e3) / 1e9, "GB/s");
    return ok;
  };

  bool ok = true;
  ok &= run("v0 naive", [&] { d_out.zero(); reduce_v0_naive<<<grid_direct, threads>>>(d_data.get(), d_out.get(), n); });
  ok &= run("v1 sequential", [&] { d_out.zero(); reduce_v1_sequential<<<grid_direct, threads>>>(d_data.get(), d_out.get(), n); });
  ok &= run("v2 warp shuffle", [&] { d_out.zero(); reduce_v2_warp_shuffle<<<grid_stride_blocks, threads>>>(d_data.get(), d_out.get(), n); });

  std::printf(ok ? "\nAll GPU results match the CPU reference.\n" : "\nAt least one kernel MISMATCHED.\n");
  return ok ? 0 : 1;
}
