// Verifies the SiLU forward/backward formulas (analytic derivative sigmoid(x)*(1 + x*(1 - sigmoid(x))))
// against central finite differences over a wide range of x. Tests the math the CUDA extension
// implements, not the extension itself.

#include <cmath>
#include <cstdio>
#include <random>

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double silu(double x) { return x * sigmoid(x); }

// Exactly silu_backward_kernel's formula, transcribed: local_grad = sig * (1 + x * (1 - sig)).
static double silu_grad_analytic(double x) {
  const double sig = sigmoid(x);
  return sig * (1.0 + x * (1.0 - sig));
}

static void check_gradient_at(double x) {
  const double h = 1e-5;
  const double numeric = (silu(x + h) - silu(x - h)) / (2 * h);
  const double analytic = silu_grad_analytic(x);
  const double tol = 1e-4 * std::max(1.0, std::fabs(numeric));
  CHECK(std::fabs(numeric - analytic) < tol);
}

int main() {
  // Regular grid across a wide range, plus the boundary/limit cases a wrong formula is most likely to
  // get visibly wrong: x = 0 (sig = 0.5 exactly), large positive x (sig -> 1, silu -> x, grad -> 1),
  // large negative x (sig -> 0, silu -> 0, grad -> 0).
  for (double x = -20.0; x <= 20.0; x += 0.01) check_gradient_at(x);
  for (double x : {0.0, -0.0, 1e-6, -1e-6, 30.0, -30.0, 60.0, -60.0}) check_gradient_at(x);

  // Random sampling as an independent check on top of the regular grid.
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> dist(-50.0, 50.0);
  for (int i = 0; i < 5000; ++i) check_gradient_at(dist(rng));

  // Known closed-form values as an additional sanity anchor, independent of the finite-difference check.
  CHECK(std::fabs(silu(0.0) - 0.0) < 1e-12);
  CHECK(std::fabs(silu_grad_analytic(0.0) - 0.5) < 1e-12);  // silu'(0) = sigmoid(0) = 0.5
  CHECK(silu(30.0) > 29.999);                               // silu(x) -> x as x -> +inf
  CHECK(std::fabs(silu(-30.0)) < 1e-6);                      // silu(x) -> 0 as x -> -inf
  CHECK(silu_grad_analytic(30.0) > 0.999);                   // silu'(x) -> 1 as x -> +inf
  CHECK(std::fabs(silu_grad_analytic(-30.0)) < 1e-6);        // silu'(x) -> 0 as x -> -inf

  std::printf("SiLU forward/backward math: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
