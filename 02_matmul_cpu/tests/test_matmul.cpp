#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "matmul.hpp"

using namespace portfolio::matmul;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

static std::vector<float> random_mat(int r, int c, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> v(static_cast<std::size_t>(r) * c);
  for (float& x : v) x = d(rng);
  return v;
}

// Independent double-precision reference, written directly from the definition of matrix product.
static std::vector<double> reference(const std::vector<float>& A, const std::vector<float>& B, Dims d) {
  const auto [M, K, N] = d;
  std::vector<double> C(static_cast<std::size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += static_cast<double>(A[static_cast<std::size_t>(i) * K + k]) * B[static_cast<std::size_t>(k) * N + j];
      C[static_cast<std::size_t>(i) * N + j] = acc;
    }
  return C;
}

static double max_rel_error(const std::vector<float>& got, const std::vector<double>& ref) {
  double worst = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
    const double scale = std::max(1.0, std::fabs(ref[i]));
    worst = std::max(worst, err / scale);
  }
  return worst;
}

static void check_shape(int M, int K, int N, unsigned seed) {
  const Dims d{M, K, N};
  const auto A = random_mat(M, K, seed), B = random_mat(K, N, seed + 1);
  const auto ref = reference(A, B, d);
  std::vector<float> C(static_cast<std::size_t>(M) * N);

  naive(A, B, C, d);
  CHECK(max_rel_error(C, ref) < 1e-4);

  ikj(A, B, C, d);
  CHECK(max_rel_error(C, ref) < 1e-4);

  for (int tile : {8, 16, 64}) {  // tile sizes both smaller and larger than typical dimensions
    tiled(A, B, C, d, tile);
    CHECK(max_rel_error(C, ref) < 1e-4);
#if defined(__AVX2__)
    tiled_avx2(A, B, C, d, tile);
    CHECK(max_rel_error(C, ref) < 1e-4);
    tiled_avx2_omp(A, B, C, d, tile);
    CHECK(max_rel_error(C, ref) < 1e-4);
#endif
  }
}

int main() {
  // Square, tall, wide, tiny, and deliberately NOT multiples of any tile size (8, 16, 64) or of the
  // AVX2 width (8) — this is exactly where a fencepost error in the tiling or the vectorized cleanup
  // loop would show up.
  struct Shape { int M, K, N; };
  const Shape shapes[] = {{1, 1, 1},  {1, 5, 1},   {5, 1, 5},   {7, 7, 7},     {9, 13, 5},
                          {17, 3, 17}, {33, 65, 20}, {64, 64, 64}, {70, 70, 70}, {129, 10, 71}};
  unsigned seed = 100;
  for (const Shape& s : shapes) check_shape(s.M, s.K, s.N, seed++);

  std::printf("matmul CPU: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
