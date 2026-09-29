// CPU simulation of conv2d_v2_shared's block/thread/shared-memory structure, checked against a naive
// clamp-to-edge convolution (with an explicit per-corner check). The cooperative strided load covers the
// whole (TILE + 2R)^2 tile by construction, corners included. No GPU needed.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "conv2d_reference.hpp"

using portfolio::conv2d::clampi;
using portfolio::conv2d::naive_conv2d;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

// (clampi and naive_conv2d now come from conv2d_reference.hpp, shared with src/main.cu)


// Simulates one thread block of the FIXED conv2d_v2_shared: builds the shared tile via the cooperative
// strided-loop load (corners included), then computes the tile's output pixels from shared memory only,
// exactly like the CUDA kernel does.
static void simulate_block(const std::vector<float>& in, const std::vector<float>& filter, int width, int height,
                           int radius, int tile_width, int block_x, int block_y, std::vector<float>& out) {
  const int fsize = 2 * radius + 1;
  const int shared_w = tile_width + 2 * radius;
  std::vector<float> s_tile(static_cast<std::size_t>(shared_w) * shared_w);

  const int num_threads = tile_width * tile_width;
  const int total_shared = shared_w * shared_w;
  // Cooperative load: every shared-memory cell [0, total_shared) is written by exactly one (tid + k*num_threads)
  // iteration for some thread tid in [0, num_threads) -- this is what guarantees the corners are covered.
  for (int tid = 0; tid < num_threads; ++tid) {
    for (int lin = tid; lin < total_shared; lin += num_threads) {
      const int sy = lin / shared_w, sx = lin % shared_w;
      const int gy = block_y * tile_width + sy - radius;
      const int gx = block_x * tile_width + sx - radius;
      const int cy = clampi(gy, 0, height - 1), cx = clampi(gx, 0, width - 1);
      s_tile[static_cast<std::size_t>(sy) * shared_w + sx] = in[static_cast<std::size_t>(cy) * width + cx];
    }
  }
  // Every thread in the block now reads only from s_tile (never re-touches global/`in` directly), exactly
  // mirroring the real kernel's second phase.
  for (int ty = 0; ty < tile_width; ++ty)
    for (int tx = 0; tx < tile_width; ++tx) {
      const int out_y = block_y * tile_width + ty, out_x = block_x * tile_width + tx;
      if (out_y >= height || out_x >= width) continue;
      float sum = 0.f;
      for (int r = 0; r < fsize; ++r)
        for (int c = 0; c < fsize; ++c) sum += s_tile[static_cast<std::size_t>(ty + r) * shared_w + (tx + c)] * filter[static_cast<std::size_t>(r) * fsize + c];
      out[static_cast<std::size_t>(out_y) * width + out_x] = sum;
    }
}

static std::vector<float> simulate_conv2d_v2(const std::vector<float>& in, const std::vector<float>& filter, int width,
                                             int height, int radius, int tile_width) {
  std::vector<float> out(static_cast<std::size_t>(width) * height, -12345.f);  // poison value: every pixel must be overwritten
  const int blocks_x = (width + tile_width - 1) / tile_width, blocks_y = (height + tile_width - 1) / tile_width;
  for (int by = 0; by < blocks_y; ++by)
    for (int bx = 0; bx < blocks_x; ++bx) simulate_block(in, filter, width, height, radius, tile_width, bx, by, out);
  return out;
}

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
  return m;
}

static void check_case(int width, int height, int radius, int tile_width, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::vector<float> in(static_cast<std::size_t>(width) * height);
  for (float& x : in) x = d(rng);
  const int fsize = 2 * radius + 1;
  std::vector<float> filt(static_cast<std::size_t>(fsize) * fsize);
  for (float& x : filt) x = d(rng);  // random (not just a blur) so corner errors cannot cancel out by symmetry

  const auto ref = naive_conv2d(in, filt, width, height, radius);
  const auto got = simulate_conv2d_v2(in, filt, width, height, radius, tile_width);
  CHECK(max_abs_diff(ref, got) < 1e-3);

  // Explicitly re-check the four tile corners of every block, where the original bug lived.
  const int blocks_x = (width + tile_width - 1) / tile_width, blocks_y = (height + tile_width - 1) / tile_width;
  for (int by = 0; by < blocks_y; ++by)
    for (int bx = 0; bx < blocks_x; ++bx) {
      for (int dy : {0, tile_width - 1})
        for (int dx : {0, tile_width - 1}) {
          const int y = by * tile_width + dy, x = bx * tile_width + dx;
          if (y >= height || x >= width) continue;
          CHECK(std::fabs(got[static_cast<std::size_t>(y) * width + x] - ref[static_cast<std::size_t>(y) * width + x]) < 1e-3);
        }
    }
}

int main() {
  // Sizes deliberately not multiples of tile_width, and radius up to 3, both larger than the original
  // fixed 5x5/R=2 filter, to stress the halo logic generally rather than one specific configuration.
  for (int radius : {1, 2, 3}) {
    for (int tile : {8, 16}) {
      check_case(37, 29, radius, tile, static_cast<unsigned>(radius * 100 + tile));   // small, ragged edges
      check_case(64, 64, radius, tile, static_cast<unsigned>(radius * 100 + tile + 1));  // exactly tile-aligned
      check_case(65, 63, radius, tile, static_cast<unsigned>(radius * 100 + tile + 2));  // off by one both ways
      check_case(100, 40, radius, tile, static_cast<unsigned>(radius * 100 + tile + 3)); // rectangular
    }
  }
  std::printf("conv2d CPU simulation: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
