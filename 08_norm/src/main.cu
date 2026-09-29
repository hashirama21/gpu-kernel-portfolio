// Benchmarks and validates layernorm/rmsnorm against a CPU reference, launched with 128 threads/block
// (4 warps). Requires a CUDA GPU; not built without a CUDA toolkit.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "norm.cuh"
#include "timer.hpp"

using namespace portfolio;
using namespace portfolio::norm::cuda;

static void cpu_layernorm(const std::vector<float>& in, const std::vector<float>& gamma,
                          const std::vector<float>& beta, std::vector<float>& out, int N, int D, float eps) {
  for (int r = 0; r < N; ++r) {
    const float* x = in.data() + static_cast<std::size_t>(r) * D;
    float* y = out.data() + static_cast<std::size_t>(r) * D;
    double mean = 0;
    for (int i = 0; i < D; ++i) mean += x[i];
    mean /= D;
    double var = 0;
    for (int i = 0; i < D; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= D;
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int i = 0; i < D; ++i) y[i] = static_cast<float>((x[i] - mean) * inv * gamma[i] + beta[i]);
  }
}

static void cpu_rmsnorm(const std::vector<float>& in, const std::vector<float>& gamma, std::vector<float>& out,
                        int N, int D, float eps) {
  for (int r = 0; r < N; ++r) {
    const float* x = in.data() + static_cast<std::size_t>(r) * D;
    float* y = out.data() + static_cast<std::size_t>(r) * D;
    double sq = 0;
    for (int i = 0; i < D; ++i) sq += static_cast<double>(x[i]) * x[i];
    const double rms = 1.0 / std::sqrt(sq / D + eps);
    for (int i = 0; i < D; ++i) y[i] = static_cast<float>(x[i] * rms * gamma[i]);
  }
}

int main(int argc, char** argv) {
  const int N = (argc > 1) ? std::atoi(argv[1]) : 8192;
  const int D = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const float eps = 1e-5f;
  std::printf("--- Project 8: LayerNorm / RMSNorm (N=%d, D=%d, 128 threads/block) ---\n", N, D);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  std::vector<float> h_in(static_cast<std::size_t>(N) * D), h_gamma(static_cast<std::size_t>(D), 1.f),
      h_beta(static_cast<std::size_t>(D), 0.f);
  for (float& x : h_in) x = dist(rng);

  std::vector<float> ref_ln(h_in.size()), ref_rms(h_in.size());
  cpu_layernorm(h_in, h_gamma, h_beta, ref_ln, N, D, eps);
  cpu_rmsnorm(h_in, h_gamma, ref_rms, N, D, eps);

  DeviceBuffer<float> d_in(h_in.size()), d_gamma(h_gamma.size()), d_beta(h_beta.size()), d_out(h_in.size());
  d_in.upload(h_in);
  d_gamma.upload(h_gamma);
  d_beta.upload(h_beta);

  const int threads = 128;  // 4 warps
  std::vector<float> h_out(h_in.size());
  auto validate = [&](const char* name, const std::vector<float>& ref) -> bool {
    d_out.download(h_out);
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(h_out[i] - ref[i])));
    const bool ok = worst < 1e-3;
    std::printf("  %-20s max abs error: %.3e -> %s\n", name, worst, ok ? "OK" : "MISMATCH");
    return ok;
  };

  bool all_ok = true;
  double ms = gpu_benchmark_ms([&] { layernorm<<<N, threads>>>(d_in.get(), d_gamma.get(), d_beta.get(), d_out.get(), D, eps); });
  all_ok &= validate("layernorm", ref_ln);
  report("layernorm", ms, 2.0 * h_in.size() * sizeof(float) / (ms / 1e3) / 1e9, "GB/s");

  ms = gpu_benchmark_ms([&] { rmsnorm<<<N, threads>>>(d_in.get(), d_gamma.get(), d_out.get(), D, eps); });
  all_ok &= validate("rmsnorm", ref_rms);
  report("rmsnorm", ms, 2.0 * h_in.size() * sizeof(float) / (ms / 1e3) / 1e9, "GB/s");

  ms = gpu_benchmark_ms([&] { rmsnorm_vectorized<<<N, threads>>>(d_in.get(), d_gamma.get(), d_out.get(), D, eps); });
  all_ok &= validate("rmsnorm (float4)", ref_rms);
  report("rmsnorm (float4)", ms, 2.0 * h_in.size() * sizeof(float) / (ms / 1e3) / 1e9, "GB/s");

  std::printf(all_ok ? "\nAll GPU results match the CPU reference at 128 threads/block.\n" : "\nMISMATCH.\n");
  return all_ok ? 0 : 1;
}
