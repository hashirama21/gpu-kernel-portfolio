// CPU simulation of flash_attention_forward_kernel's per-row algorithm: the running (m_i, l_i, O_i)
// online-softmax state, updated block-by-block over K/V tiles with the rescaling formula
// O_new = (O_old * l_old * alpha + P @ V_block) / l_new. Checked against naive full-matrix attention
// across shapes with N not a multiple of the KV block size and several head dimensions. No GPU needed.

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

using Vec = std::vector<float>;

// Trusted reference: materialize the full N x N score matrix, safe (max-subtracted) softmax per row,
// then multiply by V. No tiling, no online update -- the textbook definition of attention.
static Vec naive_attention(const Vec& Q, const Vec& K, const Vec& V, int N, int d) {
  Vec O(static_cast<std::size_t>(N) * d, 0.f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  for (int i = 0; i < N; ++i) {
    std::vector<double> scores(static_cast<std::size_t>(N));
    double mx = -1e300;
    for (int j = 0; j < N; ++j) {
      double s = 0;
      for (int c = 0; c < d; ++c) s += static_cast<double>(Q[static_cast<std::size_t>(i) * d + c]) * K[static_cast<std::size_t>(j) * d + c];
      s *= scale;
      scores[static_cast<std::size_t>(j)] = s;
      mx = std::max(mx, s);
    }
    double z = 0;
    for (double& s : scores) { s = std::exp(s - mx); z += s; }
    for (int c = 0; c < d; ++c) {
      double acc = 0;
      for (int j = 0; j < N; ++j) acc += scores[static_cast<std::size_t>(j)] * V[static_cast<std::size_t>(j) * d + c];
      O[static_cast<std::size_t>(i) * d + c] = static_cast<float>(acc / z);
    }
  }
  return O;
}

// Mirrors flash_attention_forward_kernel for one Q row: block-by-block over K/V tiles, online (m_i, l_i)
// update, and the same O-rescaling formula. K/V rows past the end of the sequence are masked out of
// max/sum, matching the kernel's `if (kv_row_start + j < N)` guards.
static Vec simulate_flash_row(const Vec& Q, const Vec& K, const Vec& V, int N, int d, int row, int block_kv) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  float m_i = -std::numeric_limits<float>::max();
  float l_i = 0.0f;
  Vec O_i(static_cast<std::size_t>(d), 0.0f);

  const int num_blocks_kv = (N + block_kv - 1) / block_kv;
  for (int by = 0; by < num_blocks_kv; ++by) {
    const int kv_start = by * block_kv;
    std::vector<float> S(static_cast<std::size_t>(block_kv)), P(static_cast<std::size_t>(block_kv));
    float m_new = m_i;
    for (int j = 0; j < block_kv; ++j) {
      double s = 0;
      for (int c = 0; c < d; ++c) s += static_cast<double>(Q[static_cast<std::size_t>(row) * d + c]) * K[static_cast<std::size_t>(kv_start + j < N ? kv_start + j : 0) * d + c];
      s *= scale;
      S[static_cast<std::size_t>(j)] = static_cast<float>(s);
      if (kv_start + j < N) m_new = std::max(m_new, S[static_cast<std::size_t>(j)]);
    }
    float l_new_partial = 0.0f;
    for (int j = 0; j < block_kv; ++j) {
      if (kv_start + j < N) {
        P[static_cast<std::size_t>(j)] = std::exp(S[static_cast<std::size_t>(j)] - m_new);
        l_new_partial += P[static_cast<std::size_t>(j)];
      } else {
        P[static_cast<std::size_t>(j)] = 0.0f;
      }
    }
    const float alpha = std::exp(m_i - m_new);
    const float l_new = l_i * alpha + l_new_partial;

    for (int c = 0; c < d; ++c) {
      float v_sum = 0.0f;
      for (int j = 0; j < block_kv; ++j)
        if (kv_start + j < N) v_sum += P[static_cast<std::size_t>(j)] * V[static_cast<std::size_t>(kv_start + j) * d + c];
      O_i[static_cast<std::size_t>(c)] = (O_i[static_cast<std::size_t>(c)] * l_i * alpha + v_sum) / l_new;
    }
    m_i = m_new;
    l_i = l_new;
  }
  return O_i;
}

// NaN/Inf is treated as a maximal failure: std::max(m, NaN) returns m (comparisons with NaN are false),
// so a NaN would otherwise vanish inside std::max and report a passing diff. Each element is checked for
// finiteness explicitly.
static double max_abs_diff_row(const Vec& sim, const Vec& ref, int row, int d) {
  double m = 0;
  for (int c = 0; c < d; ++c) {
    const double got = sim[static_cast<std::size_t>(c)];
    if (!std::isfinite(got)) return std::numeric_limits<double>::infinity();  // NaN/Inf: an unambiguous failure
    m = std::max(m, std::fabs(got - ref[static_cast<std::size_t>(row) * d + c]));
  }
  return m;
}

static void check_case(int N, int d, int block_kv, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  Vec Q(static_cast<std::size_t>(N) * d), K(static_cast<std::size_t>(N) * d), V(static_cast<std::size_t>(N) * d);
  for (float& x : Q) x = dist(rng);
  for (float& x : K) x = dist(rng);
  for (float& x : V) x = dist(rng);

  const auto ref = naive_attention(Q, K, V, N, d);
  double worst = 0;
  for (int row = 0; row < N; ++row) {
    const auto O_row = simulate_flash_row(Q, K, V, N, d, row, block_kv);
    worst = std::max(worst, max_abs_diff_row(O_row, ref, row, d));
  }
  CHECK(worst < 1e-3);
}

// Adversarial case: Q and K are built so the row-wise max strictly increases every block, forcing the
// online update to rescale on almost every block. This makes the check sensitive to a flipped-sign alpha
// bug, which the random-data cases do not reliably catch (there the max is mostly set by the first block
// and rarely updated, so exp(m_i - m_new) = exp(0) = 1 regardless of a sign error).
static void check_ramp_case(int N, int d, int block_kv, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
  Vec Q(static_cast<std::size_t>(N) * d), K(static_cast<std::size_t>(N) * d), V(static_cast<std::size_t>(N) * d);
  for (int i = 0; i < N; ++i) {
    // V correlates with position so which block dominates the weighted average is visible in the output
    // (mean-zero random V would look the same under almost any block weighting).
    const float v = static_cast<float>(i) / N * 10.0f;
    for (int c = 0; c < d; ++c) V[static_cast<std::size_t>(i) * d + c] = v;
  }
  for (int i = 0; i < N; ++i)
    for (int c = 0; c < d; ++c) {
      // K[i] grows with i, so later blocks contain the largest dot products with any fixed Q row.
      Q[static_cast<std::size_t>(i) * d + c] = 1.0f + noise(rng);
      K[static_cast<std::size_t>(i) * d + c] = static_cast<float>(i) / N * 8.0f + noise(rng);
    }
  const auto ref = naive_attention(Q, K, V, N, d);
  double worst = 0;
  for (int row = 0; row < N; ++row) {
    const auto O_row = simulate_flash_row(Q, K, V, N, d, row, block_kv);
    worst = std::max(worst, max_abs_diff_row(O_row, ref, row, d));
  }
  CHECK(worst < 1e-2);
}

int main() {
  check_ramp_case(128, 16, 16, 1);
  check_ramp_case(200, 8, 16, 2);
  check_ramp_case(257, 32, 32, 3);

  // N not a multiple of block_kv (exercises the masked tail block), block_kv both smaller and larger
  // than N, and d spanning small to the paper's typical head dimension (64).
  check_case(1, 8, 32, 1);
  check_case(10, 8, 32, 2);       // N < block_kv: a single, partially-masked block
  check_case(32, 8, 32, 3);       // exactly one full block
  check_case(33, 8, 32, 4);       // one full block + one almost-empty block
  check_case(100, 16, 32, 5);     // N not a multiple of block_kv
  check_case(257, 64, 32, 6);     // the paper's typical head dim, several full blocks + a tail
  check_case(200, 32, 16, 7);     // smaller block_kv than the others, more blocks

  std::printf("flash attention CPU simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
