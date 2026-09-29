// CPU simulation of the block-reduction algorithm used by layernorm/rmsnorm (per-thread partial sum,
// per-warp reduction, cross-warp reduction via a small shared array, broadcast), checked against naive
// mean/variance/sum-of-squares. Covers blockDim.x from one warp to several and D not a multiple of 4
// (the vectorized RMSNorm tail). No GPU needed.

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// Simulates the fixed block_reduce_sum: `contrib[t]` is thread t's local partial sum (already summed
// over its strided share of the row); returns the value every thread in the block would see after the
// full reduction (all threads get the same, fully-reduced value — this mirrors the real kernel writing
// to shared memory and every thread reading it back).
static float simulate_block_reduce_sum(const std::vector<float>& contrib, int block_threads) {
  const int n_warps = block_threads / 32;
  std::vector<float> warp_sums(static_cast<std::size_t>(n_warps), 0.f);
  for (int w = 0; w < n_warps; ++w) {
    float s = 0.f;
    for (int lane = 0; lane < 32; ++lane) s += contrib[static_cast<std::size_t>(w * 32 + lane)];  // warp reduce
    warp_sums[static_cast<std::size_t>(w)] = s;                                                    // lane 0 -> shared[w]
  }
  float total = 0.f;
  for (float s : warp_sums) total += s;  // first warp reduces the (<=32, since max 1024 threads/block) partials
  return total;                          // broadcast to the whole block
}

static void check_layernorm(int D, int block_threads, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-5.f, 5.f);
  std::vector<float> x(static_cast<std::size_t>(D));
  for (float& v : x) v = dist(rng);

  double ref_mean = 0;
  for (float v : x) ref_mean += v;
  ref_mean /= D;
  double ref_var = 0;
  for (float v : x) ref_var += (v - ref_mean) * (v - ref_mean);
  ref_var /= D;

  std::vector<float> contrib(static_cast<std::size_t>(block_threads), 0.f);
  for (int t = 0; t < block_threads; ++t)
    for (int i = t; i < D; i += block_threads) contrib[static_cast<std::size_t>(t)] += x[static_cast<std::size_t>(i)];
  const float mean = simulate_block_reduce_sum(contrib, block_threads) / D;
  CHECK(std::fabs(mean - ref_mean) < 1e-3 * std::max(1.0, std::fabs(ref_mean)));

  std::fill(contrib.begin(), contrib.end(), 0.f);
  for (int t = 0; t < block_threads; ++t)
    for (int i = t; i < D; i += block_threads) {
      const float d = x[static_cast<std::size_t>(i)] - mean;
      contrib[static_cast<std::size_t>(t)] += d * d;
    }
  const float var = simulate_block_reduce_sum(contrib, block_threads) / D;
  CHECK(std::fabs(var - ref_var) < 1e-2 * std::max(1.0, ref_var));
}

// Mirrors the FIXED rmsnorm_v2: float4 groups over the aligned prefix, scalar tail for D % 4.
static void check_rmsnorm_vectorized(int D, int block_threads, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-3.f, 3.f);
  std::vector<float> x(static_cast<std::size_t>(D));
  for (float& v : x) v = dist(rng);

  double ref_sq = 0;
  for (float v : x) ref_sq += static_cast<double>(v) * v;

  const int d_vec = D / 4;
  std::vector<float> contrib(static_cast<std::size_t>(block_threads), 0.f);
  for (int t = 0; t < block_threads; ++t) {
    for (int g = t; g < d_vec; g += block_threads) {
      for (int lane = 0; lane < 4; ++lane) {
        const float v = x[static_cast<std::size_t>(g * 4 + lane)];
        contrib[static_cast<std::size_t>(t)] += v * v;
      }
    }
    for (int i = d_vec * 4 + t; i < D; i += block_threads) {  // the fix: scalar tail, striped the same way
      const float v = x[static_cast<std::size_t>(i)];
      contrib[static_cast<std::size_t>(t)] += v * v;
    }
  }
  const float sq_sum = simulate_block_reduce_sum(contrib, block_threads);
  CHECK(std::fabs(sq_sum - ref_sq) < 1e-2 * std::max(1.0, ref_sq));

  // Full RMSNorm output check against x * rsqrt(mean(x^2) + eps) * gamma (eps inside the sqrt).
  const float eps = 1e-5f;
  const float rms = 1.0f / std::sqrt(static_cast<float>(sq_sum / D) + eps);
  std::vector<float> gamma(static_cast<std::size_t>(D), 1.0f);
  for (int i = 0; i < D; ++i) {
    const float got = x[static_cast<std::size_t>(i)] * rms * gamma[static_cast<std::size_t>(i)];
    const float want = static_cast<float>(x[static_cast<std::size_t>(i)] / std::sqrt(ref_sq / D + eps));
    CHECK(std::fabs(got - want) < 1e-3f * std::max(1.0f, std::fabs(want)));
  }
}

int main() {
  // Block sizes spanning 1 warp to several; D both a multiple of 128 (matching real config) and
  // deliberately NOT a multiple of 4 (exercises the tail fix) or of any block size.
  for (int threads : {32, 64, 128, 256}) {
    for (int D : {31, 32, 33, 127, 128, 129, 4096, 4099}) {
      check_layernorm(D, threads, static_cast<unsigned>(D * 1000 + threads));
      check_rmsnorm_vectorized(D, threads, static_cast<unsigned>(D * 2000 + threads));
    }
  }
  std::printf("layernorm/rmsnorm CPU simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
