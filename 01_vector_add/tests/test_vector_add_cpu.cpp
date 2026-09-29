// Validates every CPU vector-add variant against the naive reference, including sizes that are not
// multiples of 32/8 (the AVX2 tail path).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "vector_add.hpp"

using namespace portfolio::vecadd;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

static std::vector<float> random_vec(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-100.f, 100.f);
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

static void check_matches_naive(std::size_t n) {
  const auto a = random_vec(n, static_cast<unsigned>(n) * 2 + 1), b = random_vec(n, static_cast<unsigned>(n) * 2 + 2);
  std::vector<float> ref(n), got(n);
  add_naive(a, b, ref);

#if defined(__AVX2__)
  std::fill(got.begin(), got.end(), -1.f);
  add_avx2(a, b, got);
  for (std::size_t i = 0; i < n; ++i) CHECK(got[i] == ref[i]);  // pure add, bit-exact
#endif

  std::fill(got.begin(), got.end(), -1.f);
  add_openmp(a, b, got);
  for (std::size_t i = 0; i < n; ++i) CHECK(got[i] == ref[i]);
}

int main() {
  // 0, 1, and sizes straddling every unroll boundary (8, 32) the AVX2 path uses internally.
  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9},
                        std::size_t{31}, std::size_t{32}, std::size_t{33}, std::size_t{100}, std::size_t{100003}}) {
    check_matches_naive(n);
  }
  std::printf("vector_add CPU: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
