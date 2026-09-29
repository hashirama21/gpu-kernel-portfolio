#pragma once
// Matrix multiplication, C[M x N] = A[M x K] * B[K x N], row-major, fp32. Increasing sophistication:
//   naive          : textbook i-j-k triple loop (strides through B's column)
//   ikj            : loops reordered i-k-j so B is read row-wise (cache-friendly)
//   tiled          : ikj blocked over all three dimensions to fit each block's working set in cache
//   tiled_avx2     : innermost j loop vectorized with AVX2 + FMA, 8 columns at a time
//   tiled_avx2_omp : the same kernel, outer loop over row-tiles parallelized with OpenMP

#include <algorithm>
#include <cstddef>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace portfolio::matmul {

struct Dims {
  int M, K, N;
};

using Matrix = std::span<float>;
using ConstMatrix = std::span<const float>;

inline void naive(ConstMatrix A, ConstMatrix B, Matrix C, Dims d) {
  const auto [M, K, N] = d;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[static_cast<std::size_t>(i) * K + k] * B[static_cast<std::size_t>(k) * N + j];
      C[static_cast<std::size_t>(i) * N + j] = acc;
    }
}

// Loops reordered so the innermost loop walks B and C contiguously instead of striding through B.
inline void ikj(ConstMatrix A, ConstMatrix B, Matrix C, Dims d) {
  const auto [M, K, N] = d;
  std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M) * N, 0.f);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      const float a = A[static_cast<std::size_t>(i) * K + k];
      const float* brow = B.data() + static_cast<std::size_t>(k) * N;
      float* crow = C.data() + static_cast<std::size_t>(i) * N;
      for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
    }
}

// ikj, additionally blocked over (i, k, j) with block size `tile` so that the A/B/C sub-blocks touched
// by one group of iterations fit in L2 cache and are reused before being evicted.
inline void tiled(ConstMatrix A, ConstMatrix B, Matrix C, Dims d, int tile = 64) {
  const auto [M, K, N] = d;
  std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M) * N, 0.f);
  for (int i0 = 0; i0 < M; i0 += tile)
    for (int k0 = 0; k0 < K; k0 += tile)
      for (int j0 = 0; j0 < N; j0 += tile) {
        const int i_max = std::min(i0 + tile, M), k_max = std::min(k0 + tile, K), j_max = std::min(j0 + tile, N);
        for (int i = i0; i < i_max; ++i)
          for (int k = k0; k < k_max; ++k) {
            const float a = A[static_cast<std::size_t>(i) * K + k];
            const float* brow = B.data() + static_cast<std::size_t>(k) * N;
            float* crow = C.data() + static_cast<std::size_t>(i) * N;
            for (int j = j0; j < j_max; ++j) crow[j] += a * brow[j];
          }
      }
}

#if defined(__AVX2__)
// Same tiling, innermost j loop vectorized: 8 columns of C updated per FMA instruction. `N` need not be
// a multiple of 8 — a scalar cleanup loop handles the last (j_max - j0) % 8 columns of each tile row.
inline void tiled_avx2(ConstMatrix A, ConstMatrix B, Matrix C, Dims d, int tile = 64) {
  const auto [M, K, N] = d;
  std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M) * N, 0.f);
  for (int i0 = 0; i0 < M; i0 += tile)
    for (int k0 = 0; k0 < K; k0 += tile)
      for (int j0 = 0; j0 < N; j0 += tile) {
        const int i_max = std::min(i0 + tile, M), k_max = std::min(k0 + tile, K), j_max = std::min(j0 + tile, N);
        for (int i = i0; i < i_max; ++i)
          for (int k = k0; k < k_max; ++k) {
            const __m256 va = _mm256_set1_ps(A[static_cast<std::size_t>(i) * K + k]);
            const float* brow = B.data() + static_cast<std::size_t>(k) * N;
            float* crow = C.data() + static_cast<std::size_t>(i) * N;
            int j = j0;
            for (; j + 8 <= j_max; j += 8) {
              const __m256 vb = _mm256_loadu_ps(brow + j);
              __m256 vc = _mm256_loadu_ps(crow + j);
              vc = _mm256_fmadd_ps(va, vb, vc);
              _mm256_storeu_ps(crow + j, vc);
            }
            const float a = A[static_cast<std::size_t>(i) * K + k];
            for (; j < j_max; ++j) crow[j] += a * brow[j];
          }
      }
}

// tiled_avx2, with the outer i-tile loop parallelized. Each thread owns disjoint row-tiles of C, so no
// synchronization is needed inside the parallel region.
inline void tiled_avx2_omp(ConstMatrix A, ConstMatrix B, Matrix C, Dims d, int tile = 64) {
  const auto [M, K, N] = d;
  std::fill(C.begin(), C.begin() + static_cast<std::ptrdiff_t>(M) * N, 0.f);
  const int n_tiles_i = (M + tile - 1) / tile;
#pragma omp parallel for schedule(dynamic)
  for (int ti = 0; ti < n_tiles_i; ++ti) {
    const int i0 = ti * tile, i_max = std::min(i0 + tile, M);
    for (int k0 = 0; k0 < K; k0 += tile)
      for (int j0 = 0; j0 < N; j0 += tile) {
        const int k_max = std::min(k0 + tile, K), j_max = std::min(j0 + tile, N);
        for (int i = i0; i < i_max; ++i)
          for (int k = k0; k < k_max; ++k) {
            const __m256 va = _mm256_set1_ps(A[static_cast<std::size_t>(i) * K + k]);
            const float* brow = B.data() + static_cast<std::size_t>(k) * N;
            float* crow = C.data() + static_cast<std::size_t>(i) * N;
            int j = j0;
            for (; j + 8 <= j_max; j += 8) {
              __m256 vc = _mm256_loadu_ps(crow + j);
              vc = _mm256_fmadd_ps(va, _mm256_loadu_ps(brow + j), vc);
              _mm256_storeu_ps(crow + j, vc);
            }
            const float a = A[static_cast<std::size_t>(i) * K + k];
            for (; j < j_max; ++j) crow[j] += a * brow[j];
          }
      }
  }
}
#endif  // __AVX2__

}  // namespace portfolio::matmul
