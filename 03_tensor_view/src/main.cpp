#include <cstdio>
#include <numeric>
#include <vector>

#include "tensor_view.hpp"

using namespace portfolio::tensor;

int main() {
  std::printf("--- Project 3: TensorView (N-D, strides, broadcasting) ---\n\n");

  std::vector<float> storage(12);
  std::iota(storage.begin(), storage.end(), 1.0f);
  TensorView<float> t3d(storage.data(), {2, 2, 3});
  std::printf("Strides of a [2,2,3] tensor: (%zu, %zu, %zu)\n", t3d.strides()[0], t3d.strides()[1], t3d.strides()[2]);
  std::printf("t3d[1, 0, 2] = %.1f (expected 9.0)\n\n", t3d[1, 0, 2]);

  TensorView<float> s = t3d.slice(1);
  std::printf("slice(1) shape: (%zu, %zu)\n", s.shape()[0], s.shape()[1]);
  s[0, 2] = 99.0f;
  std::printf("after s[0,2] = 99, storage[8] = %.1f\n\n", storage[8]);

  std::vector<float> vec_storage{100.f, 200.f, 300.f};
  TensorView<float> vec(vec_storage.data(), {1, 3});
  TensorView<float> b = broadcast_to(vec, {2, 2, 3});
  std::printf("broadcast [1,3] -> [2,2,3] strides: (%zu, %zu, %zu)\n", b.strides()[0], b.strides()[1], b.strides()[2]);
  std::printf("b[0,0,1] = %.1f, b[1,1,1] = %.1f (same: stride 0 on axes 0,1)\n\n", b[0, 0, 1], b[1, 1, 1]);

  TensorView<float> r = t3d.reshape({4, 3});  // t3d is contiguous: allowed
  std::printf("reshape [2,2,3] -> [4,3], r[3,2] = %.1f (expected %.1f)\n", r[3, 2], storage[11]);

  try {
    TensorView<float> row(vec_storage.data(), {1, 3});
    [[maybe_unused]] auto bad = broadcast_to(row, {2, 3, 3}).reshape({18});  // non-contiguous: must throw
    std::printf("ERROR: reshape of a broadcast view should have thrown\n");
  } catch (const std::logic_error& e) {
    std::printf("reshape correctly rejected a non-contiguous view: %s\n", e.what());
  }

  std::printf("\nOK.\n");
  return 0;
}
