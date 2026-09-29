#pragma once
// Minimal CPU-side benchmarking helper shared by every project. Header-only, no dependencies.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace portfolio {

// Runs `fn` `warmup` times (discarded), then `iters` times; returns the median time in milliseconds
// (median, not mean, to resist occasional outliers).
inline double benchmark_ms(const std::function<void()>& fn, int warmup = 2, int iters = 5) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// Prints one benchmark row in a consistent format across every project, with an optional
// derived-quantity column (bandwidth in GB/s, throughput in GFLOP/s, ...).
inline void report(const std::string& label, double ms, double derived = -1.0, const char* unit = "") {
  if (derived >= 0.0)
    std::printf("  %-28s %9.4f ms   %10.3f %s\n", label.c_str(), ms, derived, unit);
  else
    std::printf("  %-28s %9.4f ms\n", label.c_str(), ms);
}

}  // namespace portfolio
