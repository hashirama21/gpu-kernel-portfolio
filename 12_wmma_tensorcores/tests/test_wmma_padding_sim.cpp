// Verifies the mathematical premise of launch_wmma_gemm's padding fix: zero-padding A/B up to the tile
// alignment, running a naive matmul, and slicing back to M x N reproduces the unpadded result exactly.
// Does not simulate Tensor Core FP16 numerics. No GPU needed.

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

static int round_up(int v, int multiple) { return ((v + multiple - 1) / multiple) * multiple; }

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

// Mirrors what the fixed launch_wmma_gemm() does on the host side: pad, compute (naive stand-in for the
// tiled/WMMA kernel, since the point here is the padding logic, not the tiling), slice.
static std::vector<float> padded_matmul(const std::vector<float>& A, const std::vector<float>& B, int M, int K,
                                        int N, int m_align, int n_align, int k_align) {
  const int Mp = round_up(M, m_align), Kp = round_up(K, k_align), Np = round_up(N, n_align);
  std::vector<float> Ap(static_cast<std::size_t>(Mp) * Kp, 0.f), Bp(static_cast<std::size_t>(Kp) * Np, 0.f);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) Ap[static_cast<std::size_t>(i) * Kp + k] = A[static_cast<std::size_t>(i) * K + k];
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j) Bp[static_cast<std::size_t>(k) * Np + j] = B[static_cast<std::size_t>(k) * N + j];

  const auto Cp = naive_matmul(Ap, Bp, Mp, Kp, Np);
  std::vector<float> C(static_cast<std::size_t>(M) * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) C[static_cast<std::size_t>(i) * N + j] = Cp[static_cast<std::size_t>(i) * Np + j];
  return C;
}

static void check_shape(int M, int K, int N, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<float> A(static_cast<std::size_t>(M) * K), B(static_cast<std::size_t>(K) * N);
  for (float& x : A) x = dist(rng);
  for (float& x : B) x = dist(rng);

  const auto ref = naive_matmul(A, B, M, K, N);
  // WMMA_M=WMMA_N=WMMA_K=16, 4 warps stacked on the M axis per block -> M rounds to a multiple of 64,
  // N and K round to multiples of 16 (matching the kernel's tile shape).
  const auto got = padded_matmul(A, B, M, K, N, 64, 16, 16);

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
  // Every dimension deliberately NOT a multiple of the tile alignment -- exactly the case the original
  // kernel could not handle safely.
  check_shape(1, 1, 1, 1);
  check_shape(17, 17, 17, 2);      // just over one 16-tile on every axis
  check_shape(63, 15, 31, 3);      // M just under 64, K just under 16, N just over one tile
  check_shape(65, 33, 17, 4);      // just over the alignment on every axis
  check_shape(100, 200, 50, 5);
  check_shape(128, 128, 128, 6);   // exactly tile-aligned; kept small (O(n^3) naive reference)
  std::printf("wmma padding simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
