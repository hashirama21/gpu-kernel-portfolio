#pragma once
// Vector addition: naive scalar, hand-written AVX2, and an OpenMP-parallel scalar version, for
// comparison against the CUDA kernels in vector_add.cu.
//
// Kept in a header so the correctness tests and the benchmark binary share one implementation.

#include <cstddef>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace portfolio::vecadd {

inline void add_naive(std::span<const float> a, std::span<const float> b, std::span<float> c) {
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

#if defined(__AVX2__)
// 8 floats per __m256 register, unrolled x4 (32 floats / 128 bytes per loop iteration) to give the
// out-of-order engine enough independent loads in flight to hide memory latency; a tail loop handles
// whatever is left when n is not a multiple of 32.
inline void add_avx2(std::span<const float> a, std::span<const float> b, std::span<float> c) {
  const std::size_t n = a.size();
  std::size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    for (int u = 0; u < 4; ++u) {
      const __m256 va = _mm256_loadu_ps(a.data() + i + 8 * u);
      const __m256 vb = _mm256_loadu_ps(b.data() + i + 8 * u);
      _mm256_storeu_ps(c.data() + i + 8 * u, _mm256_add_ps(va, vb));
    }
  }
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a.data() + i);
    const __m256 vb = _mm256_loadu_ps(b.data() + i);
    _mm256_storeu_ps(c.data() + i, _mm256_add_ps(va, vb));
  }
  for (; i < n; ++i) c[i] = a[i] + b[i];
}
#endif  // __AVX2__

inline void add_openmp(std::span<const float> a, std::span<const float> b, std::span<float> c) {
  const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(a.size());
#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t i = 0; i < n; ++i) c[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] + b[static_cast<std::size_t>(i)];
}

}  // namespace portfolio::vecadd
