// CPU simulation of reduce_v2_warp_shuffle's block/warp structure (grid/block/warp/lane as nested loops),
// checked against std::accumulate. Verifies the warp-shuffle tree and cross-warp hand-off without a GPU.

#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
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

constexpr int kWarpSize = 32;

// Mirrors warp_reduce_sum(): a butterfly reduction over 32 lanes using only "shuffle" (no shared memory).
static float warp_reduce_sum(std::array<float, kWarpSize> lanes) {
  for (int offset = 16; offset > 0; offset /= 2)
    for (int lane = 0; lane < kWarpSize; ++lane)
      if (lane + offset < kWarpSize) lanes[static_cast<std::size_t>(lane)] += lanes[static_cast<std::size_t>(lane + offset)];
      // (lane >= kWarpSize - offset keeps its own value this round, exactly like __shfl_down_sync
      // returning an unspecified value that is never used because those lanes stop mattering)
  return lanes[0];
}

// Simulates reduce_v2_warp_shuffle for one thread block: `data` is this block's share of the input
// (already distributed by the grid-stride loop, simulated by the caller), `threads_per_block` mirrors
// blockDim.x. Returns the single float this block would atomicAdd into the output.
static float simulate_block_reduce(const std::vector<float>& data, int threads_per_block) {
  const int n_warps = threads_per_block / kWarpSize;
  std::vector<float> per_thread_sum(static_cast<std::size_t>(threads_per_block), 0.f);
  for (std::size_t i = 0; i < data.size(); ++i)
    per_thread_sum[i % static_cast<std::size_t>(threads_per_block)] += data[i];  // grid-stride assignment

  std::vector<float> shared(kWarpSize, 0.f);  // mirrors __shared__ float shared[32]
  for (int w = 0; w < n_warps; ++w) {
    std::array<float, kWarpSize> lanes{};
    for (int lane = 0; lane < kWarpSize; ++lane) lanes[static_cast<std::size_t>(lane)] = per_thread_sum[static_cast<std::size_t>(w * kWarpSize + lane)];
    shared[static_cast<std::size_t>(w)] = warp_reduce_sum(lanes);  // "if (lane == 0) shared[warp_id] = sum"
  }
  // "sum = (threadIdx.x < blockDim.x/32) ? shared[lane] : 0" then warp_reduce_sum, for warp 0 only.
  std::array<float, kWarpSize> final_lanes{};
  for (int lane = 0; lane < kWarpSize; ++lane) final_lanes[static_cast<std::size_t>(lane)] = (lane < n_warps) ? shared[static_cast<std::size_t>(lane)] : 0.f;
  return warp_reduce_sum(final_lanes);
}

static void test_warp_reduce_matches_naive_sum() {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-5.f, 5.f);
  for (int trial = 0; trial < 200; ++trial) {
    std::array<float, kWarpSize> lanes{};
    double naive_sum = 0.0;
    for (float& x : lanes) { x = d(rng); naive_sum += x; }
    const float got = warp_reduce_sum(lanes);
    CHECK(std::fabs(got - naive_sum) < 1e-3);
  }
}

static void test_block_reduce_matches_naive_sum() {
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> d(-3.f, 3.f);
  for (int threads : {32, 64, 128, 256}) {
    for (std::size_t n : {std::size_t{1}, static_cast<std::size_t>(threads / 2), static_cast<std::size_t>(threads),
                          static_cast<std::size_t>(threads * 5 + 7)}) {
      if (n == 0) continue;
      std::vector<float> data(n);
      double naive_sum = 0.0;
      for (float& x : data) { x = d(rng); naive_sum += x; }
      const float got = simulate_block_reduce(data, threads);
      CHECK(std::fabs(got - naive_sum) < 1e-2 * std::max(1.0, std::fabs(naive_sum)));
    }
  }
}

// Simulates the FULL kernel (grid of blocks, each atomicAdd-ing into one accumulator) for a range of N
// and grid/block configurations, including N not evenly divisible by the block or grid size.
static void test_full_grid_matches_naive_sum() {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> d(-2.f, 2.f);
  for (int threads : {32, 128, 256}) {
    for (int blocks : {1, 4, 17}) {
      for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{1000}, std::size_t{1'234'567}}) {
        std::vector<float> data(n);
        double naive_sum = 0.0;
        for (float& x : data) { x = d(rng); naive_sum += x; }

        // Grid-stride distribution across (blocks * threads) global thread indices, exactly as
        // reduce_v2_warp_shuffle's `for (i = idx; i < n; i += stride)` does.
        std::vector<std::vector<float>> per_block(static_cast<std::size_t>(blocks));
        const std::size_t global_threads = static_cast<std::size_t>(blocks) * threads;
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t global_tid = i % global_threads;
          const std::size_t block_id = global_tid / static_cast<std::size_t>(threads);
          per_block[block_id].push_back(data[i]);
        }
        double total = 0.0;
        for (const auto& blk : per_block) total += simulate_block_reduce(blk, threads);
        CHECK(std::fabs(total - naive_sum) < 1e-1 * std::max(1.0, std::fabs(naive_sum)));
      }
    }
  }
}

int main() {
  test_warp_reduce_matches_naive_sum();
  test_block_reduce_matches_naive_sum();
  test_full_grid_matches_naive_sum();
  std::printf("reduction CPU simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
