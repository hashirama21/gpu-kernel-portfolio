// CPU simulation of gemm_v1_shared's phase-by-phase tiling (shared-memory tile load with zero-padding
// out of bounds, tile-local accumulation, phase advance), checked against a naive triple loop for
// non-tile-aligned M/N/K. No GPU needed.

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

constexpr int kTile = 32;

static std::vector<float> naive_matmul(const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N) {
  std::vector<float> C(static_cast<std::size_t>(M) * N, 0.f);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += static_cast<double>(A[static_cast<std::size_t>(i) * K + k]) * B[static_cast<std::size_t>(k) * N + j];
      C[static_cast<std::size_t>(i) * N + j] = static_cast<float>(acc);
    }
  return C;
}

// Simulates gemm_v1_shared for ONE (block_row, block_col) output tile: phases over K, zero-padding
// exactly as the CUDA kernel's boundary checks do.
static void simulate_tile(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M,
                          int K, int N, int block_row, int block_col) {
  const int num_phases = (K + kTile - 1) / kTile;
  std::vector<std::vector<float>> sum(kTile, std::vector<float>(kTile, 0.f));

  for (int p = 0; p < num_phases; ++p) {
    std::vector<std::vector<float>> sA(kTile, std::vector<float>(kTile, 0.f)), sB(kTile, std::vector<float>(kTile, 0.f));
    for (int ty = 0; ty < kTile; ++ty)
      for (int tx = 0; tx < kTile; ++tx) {
        const int row = block_row * kTile + ty, col = block_col * kTile + tx;
        if (row < M && (p * kTile + tx) < K) sA[static_cast<std::size_t>(ty)][static_cast<std::size_t>(tx)] = A[static_cast<std::size_t>(row) * K + (p * kTile + tx)];
        if (col < N && (p * kTile + ty) < K) sB[static_cast<std::size_t>(ty)][static_cast<std::size_t>(tx)] = B[static_cast<std::size_t>(p * kTile + ty) * N + col];
      }
    for (int ty = 0; ty < kTile; ++ty)
      for (int tx = 0; tx < kTile; ++tx)
        for (int i = 0; i < kTile; ++i) sum[static_cast<std::size_t>(ty)][static_cast<std::size_t>(tx)] += sA[static_cast<std::size_t>(ty)][static_cast<std::size_t>(i)] * sB[static_cast<std::size_t>(i)][static_cast<std::size_t>(tx)];
  }
  for (int ty = 0; ty < kTile; ++ty)
    for (int tx = 0; tx < kTile; ++tx) {
      const int row = block_row * kTile + ty, col = block_col * kTile + tx;
      if (row < M && col < N) C[static_cast<std::size_t>(row) * N + col] = sum[static_cast<std::size_t>(ty)][static_cast<std::size_t>(tx)];
    }
}

static std::vector<float> simulate_gemm(const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N) {
  std::vector<float> C(static_cast<std::size_t>(M) * N, -999.f);  // poison: every element must be overwritten
  const int blocks_m = (M + kTile - 1) / kTile, blocks_n = (N + kTile - 1) / kTile;
  for (int br = 0; br < blocks_m; ++br)
    for (int bc = 0; bc < blocks_n; ++bc) simulate_tile(A, B, C, M, K, N, br, bc);
  return C;
}

static void check_shape(int M, int K, int N, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(K) * N);
  for (float& x : A) x = dist(rng);
  for (float& x : B) x = dist(rng);

  const auto ref = naive_matmul(A, B, M, K, N);
  const auto got = simulate_gemm(A, B, M, K, N);
  // NaN/Inf treated as a maximal failure (std::max(m, NaN) returns m, so NaN would otherwise pass).
  double worst = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double v = static_cast<double>(got[i]);
    if (!std::isfinite(v)) { worst = std::numeric_limits<double>::infinity(); break; }
    worst = std::max(worst, std::fabs(v - ref[i]));
  }
  CHECK(worst < 1e-2);
}

int main() {
  // M, N, K deliberately not multiples of kTile=32, including sizes smaller than one tile.
  check_shape(1, 1, 1, 1);
  check_shape(5, 7, 3, 2);
  check_shape(31, 31, 31, 3);
  check_shape(33, 33, 33, 4);
  check_shape(50, 70, 20, 5);
  check_shape(65, 65, 65, 6);          // just over 2 tiles on every axis
  check_shape(96, 33, 129, 7);         // one axis exactly tile-aligned, others not
  check_shape(100, 200, 50, 8);
  std::printf("gemm CPU simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
