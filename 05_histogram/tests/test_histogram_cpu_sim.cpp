// CPU simulation of histogram_v2's uint4 vectorized reads plus the scalar tail, checking every byte in
// [0, n) lands in exactly one bin (including every tail length 0-15). No GPU needed.

#include <cstdint>
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

constexpr int kNumBins = 256;

// Mirrors the FIXED histogram_v2_vectorized + histogram_v2_tail: every group of 16 bytes in the aligned
// prefix is "read as a uint4" (here just 16 consecutive bytes) and binned, then whatever remains (0-15
// bytes) is binned one at a time by the scalar tail kernel.
static std::vector<unsigned> simulate_v2(const std::vector<std::uint8_t>& data) {
  std::vector<unsigned> bins(kNumBins, 0);
  const std::size_t n = data.size();
  const std::size_t n_v = n / 16;
  for (std::size_t g = 0; g < n_v; ++g)
    for (int k = 0; k < 16; ++k) bins[data[g * 16 + static_cast<std::size_t>(k)]]++;
  for (std::size_t i = n_v * 16; i < n; ++i) bins[data[i]]++;  // the fix: scalar tail
  return bins;
}

static std::vector<unsigned> naive_histogram(const std::vector<std::uint8_t>& data) {
  std::vector<unsigned> bins(kNumBins, 0);
  for (std::uint8_t b : data) bins[b]++;
  return bins;
}

static void check_size(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<std::uint8_t> data(n);
  for (auto& b : data) b = static_cast<std::uint8_t>(dist(rng));

  const auto ref = naive_histogram(data);
  const auto got = simulate_v2(data);
  CHECK(ref.size() == got.size());
  for (int i = 0; i < kNumBins; ++i) CHECK(ref[static_cast<std::size_t>(i)] == got[static_cast<std::size_t>(i)]);

  unsigned total = 0;
  for (unsigned c : got) total += c;
  CHECK(total == n);  // every byte counted exactly once — the property the original bug violated
}

int main() {
  // 0..16 covers every possible tail length on its own; the rest are spot checks at various alignments.
  for (std::size_t n = 0; n <= 16; ++n) check_size(n, static_cast<unsigned>(n) + 1);
  for (std::size_t n : {std::size_t{17}, std::size_t{31}, std::size_t{32}, std::size_t{33}, std::size_t{1000},
                        std::size_t{1'000'003}, std::size_t{(1u << 20) + 7}}) {
    check_size(n, static_cast<unsigned>(n));
  }
  std::printf("histogram CPU simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
