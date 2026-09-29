// Re-simulates the index arithmetic of launch_vectorized() (grid-stride bounds and the
// aligned-prefix/scalar-tail split) in plain C++, checking that for every n each index in [0, n) is
// covered by exactly one kernel. This proves the launch-side split is exhaustive and non-overlapping;
// it does not run the kernels themselves.

#include <cstddef>
#include <cstdio>
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

static int grid_size(std::size_t work_items, int block) {
  return static_cast<int>((work_items + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

// Mirrors add_vectorized's grid-stride loop: which float4-group indices [0, n_vec) get touched.
static std::vector<char> simulate_vectorized_coverage(std::size_t n_vec, int block, int grid) {
  std::vector<char> touched(n_vec, 0);
  const std::size_t stride = static_cast<std::size_t>(block) * grid;
  for (int b = 0; b < grid; ++b)
    for (int t = 0; t < block; ++t)
      for (std::size_t i = static_cast<std::size_t>(b) * block + t; i < n_vec; i += stride) {
        CHECK(touched[i] == 0);  // no index should be visited by two different (block, start) pairs
        touched[i] = 1;
      }
  return touched;
}

// Mirrors add_tail's bounds-checked single pass: which absolute indices [offset, n) get touched.
static std::vector<char> simulate_tail_coverage(std::size_t offset, std::size_t n, int block, int grid) {
  std::vector<char> touched(n, 0);
  for (int b = 0; b < grid; ++b)
    for (int t = 0; t < block; ++t) {
      const std::size_t i = offset + static_cast<std::size_t>(b) * block + t;
      if (i < n) {
        CHECK(touched[i] == 0);
        touched[i] = 1;
      }
    }
  return touched;
}

static void check_full_coverage(std::size_t n, int block) {
  const std::size_t n_vec = n / 4;
  const std::size_t offset = n_vec * 4;

  std::vector<char> covered(n, 0);
  if (n_vec > 0) {
    const auto v = simulate_vectorized_coverage(n_vec, block, grid_size(n_vec, block));
    for (std::size_t g = 0; g < n_vec; ++g) {
      CHECK(v[g] == 1);  // launch_vectorized's grid must cover every float4 group
      for (int lane = 0; lane < 4; ++lane) covered[g * 4 + lane] = 1;  // one float4 covers 4 scalar slots
    }
  }
  const std::size_t remainder = n - offset;
  if (remainder > 0) {
    const auto t = simulate_tail_coverage(offset, n, block, grid_size(remainder, block));
    for (std::size_t i = offset; i < n; ++i) {
      CHECK(t[i] == 1);
      covered[i] = 1;
    }
  }
  for (std::size_t i = 0; i < n; ++i) CHECK(covered[i] == 1);  // every scalar index in [0, n) is covered exactly once
}

int main() {
  for (int block : {32, 128, 256}) {
    // n = 0, 1, exactly-divisible-by-4, and every remainder 1..3, both small and large.
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{4},
                          std::size_t{5}, std::size_t{127}, std::size_t{128}, std::size_t{129},
                          std::size_t{130}, std::size_t{131}, std::size_t{1'000'003}, std::size_t{(1u << 20) + 1}}) {
      check_full_coverage(n, block);
    }
  }
  std::printf("vector_add CUDA index-math simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
