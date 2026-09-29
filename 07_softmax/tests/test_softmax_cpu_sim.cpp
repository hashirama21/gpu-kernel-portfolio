// CPU simulation of softmax_v0_safe_3pass and softmax_v1_online at warp level (32 lanes, same
// combine/reduction logic), checked against a double-precision softmax. Covers the online-softmax
// recurrence, D not a multiple of 32, large-magnitude inputs, and D < 32. No GPU needed.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

constexpr int kWarp = 32;

// Trusted reference: double-precision, textbook 3-pass softmax.
static std::vector<double> reference_softmax(const std::vector<float>& row) {
  double m = -1e300;
  for (float x : row) m = std::max(m, static_cast<double>(x));
  double z = 0;
  std::vector<double> out(row.size());
  for (std::size_t i = 0; i < row.size(); ++i) {
    out[i] = std::exp(static_cast<double>(row[i]) - m);
    z += out[i];
  }
  for (double& v : out) v /= z;
  return out;
}

// Mirrors softmax_v0_safe_3pass for one row: each of the 32 "lanes" strides over the row, then a
// max-reduce and a sum-reduce (both broadcasting lane 0's result), exactly as the CUDA warp does.
static std::vector<float> simulate_v0(const std::vector<float>& row) {
  const int D = static_cast<int>(row.size());
  std::vector<float> local_max(kWarp, -1e30f);
  for (int lane = 0; lane < kWarp; ++lane)
    for (int col = lane; col < D; col += kWarp) local_max[lane] = std::max(local_max[lane], row[col]);
  float global_max = -1e30f;
  for (float v : local_max) global_max = std::max(global_max, v);  // equivalent to the shfl_down tree

  std::vector<float> local_sum(kWarp, 0.f);
  for (int lane = 0; lane < kWarp; ++lane)
    for (int col = lane; col < D; col += kWarp) local_sum[lane] += std::exp(row[col] - global_max);
  float global_sum = 0.f;
  for (float v : local_sum) global_sum += v;

  std::vector<float> out(D);
  for (int i = 0; i < D; ++i) out[i] = std::exp(row[i] - global_max) / global_sum;
  return out;
}

struct OnlineState {
  float max_val, sum;
};
static OnlineState combine(OnlineState a, OnlineState b) {
  if (a.max_val > b.max_val) return {a.max_val, a.sum + b.sum * std::exp(b.max_val - a.max_val)};
  return {b.max_val, b.sum + a.sum * std::exp(a.max_val - b.max_val)};
}

// Mirrors softmax_v1_online for one row: each lane keeps a running (max, sum) online-softmax state
// while striding over the row, then all 32 lanes' states are combined pairwise (mirroring the
// shfl_down-based butterfly reduction, order-independent since combine is associative+commutative).
static std::vector<float> simulate_v1(const std::vector<float>& row) {
  const int D = static_cast<int>(row.size());
  std::vector<OnlineState> lanes(kWarp, OnlineState{-1e30f, 0.f});
  for (int lane = 0; lane < kWarp; ++lane) {
    OnlineState& s = lanes[lane];
    for (int col = lane; col < D; col += kWarp) {
      const float val = row[col];
      if (val > s.max_val) {
        s.sum = s.sum * std::exp(s.max_val - val) + 1.0f;
        s.max_val = val;
      } else {
        s.sum += std::exp(val - s.max_val);
      }
    }
  }
  OnlineState final_state = lanes[0];
  for (int lane = 1; lane < kWarp; ++lane) final_state = combine(final_state, lanes[lane]);

  std::vector<float> out(D);
  for (int i = 0; i < D; ++i) out[i] = std::exp(row[i] - final_state.max_val) / final_state.sum;
  return out;
}

// NaN/Inf is treated as a maximal failure: std::max(m, NaN) returns m (comparisons with NaN are false),
// which would otherwise let an algorithm that outputs NaN pass undetected.
static double max_abs_diff(const std::vector<float>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double v = static_cast<double>(a[i]);
    if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
    m = std::max(m, std::fabs(v - b[i]));
  }
  return m;
}

static void check_row(int D, unsigned seed, float scale, float offset) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> row(static_cast<std::size_t>(D));
  for (float& x : row) x = dist(rng) * scale + offset;

  const auto ref = reference_softmax(row);
  CHECK(max_abs_diff(simulate_v0(row), ref) < 1e-5);
  CHECK(max_abs_diff(simulate_v1(row), ref) < 1e-5);

  double sum_ref = 0;
  for (double v : ref) sum_ref += v;
  CHECK(std::fabs(sum_ref - 1.0) < 1e-9);
}

int main() {
  // D spans below/above/at the warp width and non-multiples of 32; scale/offset stress numerical
  // stability (large positive values that would overflow a naive exp(), and very negative ones).
  for (int D : {1, 5, 31, 32, 33, 63, 64, 65, 100, 4096, 4097}) {
    check_row(D, static_cast<unsigned>(D) + 1, 1.0f, 0.0f);
    check_row(D, static_cast<unsigned>(D) + 2, 50.0f, 80.0f);    // near float exp() overflow territory
    check_row(D, static_cast<unsigned>(D) + 3, 0.001f, -500.0f);  // all deeply negative
  }
  // Degenerate: every element identical (max == every value; must not divide by zero or produce NaN).
  for (int D : {1, 32, 100}) {
    std::vector<float> row(static_cast<std::size_t>(D), 3.5f);
    const auto ref = reference_softmax(row);
    CHECK(max_abs_diff(simulate_v0(row), ref) < 1e-5);
    CHECK(max_abs_diff(simulate_v1(row), ref) < 1e-5);
  }
  std::printf("softmax CPU simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
