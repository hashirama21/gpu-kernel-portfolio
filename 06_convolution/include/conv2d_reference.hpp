#pragma once
// Naive, obviously-correct clamp-to-edge 2D convolution: the trusted reference every kernel in this
// project is checked against. Used by both tests/test_conv2d_cpu_sim.cpp and src/main.cu, so the two
// cannot silently drift into checking against different definitions of "correct".

#include <algorithm>
#include <vector>

namespace portfolio::conv2d {

inline int clampi(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }

inline std::vector<float> naive_conv2d(const std::vector<float>& in, const std::vector<float>& filter, int width,
                                       int height, int radius) {
  const int fsize = 2 * radius + 1;
  std::vector<float> out(static_cast<std::size_t>(width) * height, 0.f);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      float sum = 0.f;
      for (int r = -radius; r <= radius; ++r)
        for (int c = -radius; c <= radius; ++c) {
          const int iy = clampi(y + r, 0, height - 1), ix = clampi(x + c, 0, width - 1);
          sum += in[static_cast<std::size_t>(iy) * width + ix] * filter[static_cast<std::size_t>(r + radius) * fsize + (c + radius)];
        }
      out[static_cast<std::size_t>(y) * width + x] = sum;
    }
  return out;
}

}  // namespace portfolio::conv2d
