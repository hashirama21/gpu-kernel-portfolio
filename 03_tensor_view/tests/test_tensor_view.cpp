#include <cstdio>
#include <numeric>
#include <vector>

#include "tensor_view.hpp"

using namespace portfolio::tensor;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

// C++23 gotcha: the preprocessor does not treat [] as argument-grouping, so CHECK(t[i, j, k] == x) is
// parsed as three macro arguments and fails to compile. Wrap such calls in an extra pair of parentheses:
// CHECK((t[i, j, k] == x)).
#define CHECK_THROWS(expr, ExcType)                                                                \
  do {                                                                                             \
    ++g_checks;                                                                                    \
    bool threw = false;                                                                            \
    try {                                                                                          \
      (void)(expr);                                                                                \
    } catch (const ExcType&) {                                                                     \
      threw = true;                                                                                \
    } catch (...) {                                                                                \
    }                                                                                               \
    if (!threw) {                                                                                  \
      ++g_fail;                                                                                    \
      std::fprintf(stderr, "  FAIL %s:%d  expected %s: %s\n", __FILE__, __LINE__, #ExcType, #expr); \
    }                                                                                               \
  } while (0)

static void test_strides_and_indexing() {
  std::vector<float> storage(24);
  std::iota(storage.begin(), storage.end(), 1.0f);  // [1..24]
  TensorView<float> t(storage.data(), {2, 3, 4});
  CHECK((t.strides() == std::vector<std::size_t>{12, 4, 1}));
  CHECK(t.numel() == 24);
  CHECK((t[0, 0, 0] == 1.0f));
  CHECK((t[1, 2, 3] == 24.0f));
  CHECK(t(1, 0, 2) == 15.0f);          // operator() is a synonym for operator[]
  CHECK((&t[1, 0, 2] == &t(1, 0, 2)));   // same element, same address
}

static void test_scalar_is_one_element() {
  float x = 42.0f;
  TensorView<float> scalar(&x, {});  // rank 0: a scalar
  CHECK(scalar.numel() == 1);        // bug fix: used to be 0
  CHECK(scalar.rank() == 0);
  CHECK(scalar() == 42.0f);  // operator[] cannot take zero arguments (t[] is not valid C++ syntax); operator() can
  CHECK(scalar.is_contiguous());
}

static void test_bounds_checking() {
  std::vector<float> storage(6);
  TensorView<float> t(storage.data(), {2, 3});
  CHECK_THROWS((t[2, 0]), std::out_of_range);   // axis 0 out of range
  CHECK_THROWS((t[0, 3]), std::out_of_range);   // axis 1 out of range
  CHECK_THROWS(t.slice(2), std::out_of_range);
  TensorView<float> scalar(storage.data(), {});
  CHECK_THROWS(scalar.slice(0), std::out_of_range);  // cannot slice a rank-0 view
}

static void test_slicing_is_zero_copy_and_mutates_storage() {
  std::vector<float> storage(12);
  std::iota(storage.begin(), storage.end(), 1.0f);
  TensorView<float> t(storage.data(), {2, 2, 3});

  TensorView<float> s = t.slice(1);
  CHECK((s.shape() == std::vector<std::size_t>{2, 3}));
  CHECK(s.data() == storage.data() + 6);  // points straight into the same buffer, no copy
  CHECK((s[0, 2] == 9.0f));

  s[0, 2] = 99.0f;                 // mutate through the slice
  CHECK(storage[8] == 99.0f);      // ... and the underlying storage changes
  CHECK((t[1, 0, 2] == 99.0f));      // ... visible through the original view too
}

static void test_broadcast() {
  std::vector<float> vec_storage{100.f, 200.f, 300.f};
  TensorView<float> vec(vec_storage.data(), {1, 3});

  TensorView<float> b = broadcast_to(vec, {2, 2, 3});
  CHECK((b.strides() == std::vector<std::size_t>{0, 0, 1}));  // both new/expanded axes get stride 0
  CHECK(b.numel() == 12);                                     // logical size grows...
  CHECK(!b.is_contiguous());                                  // ...but no data was copied: not contiguous

  // Every one of the 12 logical positions aliases one of the 3 physical floats, correctly.
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      for (std::size_t k = 0; k < 3; ++k) CHECK((b[i, j, k] == vec_storage[k]));

  // Mutating through the broadcast view changes the single backing element for every broadcast position.
  b[1, 1, 1] = -5.0f;
  CHECK(vec_storage[1] == -5.0f);
  CHECK((b[0, 0, 1] == -5.0f));

  TensorView<float> row(vec_storage.data(), {3});     // rank grows: a new leading axis
  TensorView<float> b2 = broadcast_to(row, {4, 3});
  CHECK((b2.strides() == std::vector<std::size_t>{0, 1}));
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t k = 0; k < 3; ++k) CHECK((b2[i, k] == vec_storage[k]));

  CHECK_THROWS(broadcast_to(vec, {2, 5}), std::invalid_argument);  // 3 cannot broadcast to 5
  CHECK_THROWS(broadcast_to(TensorView<float>(vec_storage.data(), {2, 3}), {1, 3}), std::invalid_argument);  // target rank < source rank
}

static void test_reshape() {
  std::vector<float> storage(12);
  std::iota(storage.begin(), storage.end(), 1.0f);
  TensorView<float> t(storage.data(), {3, 4});

  TensorView<float> r = t.reshape({2, 6});
  CHECK(r.data() == storage.data());  // zero-copy
  CHECK((r[1, 5] == 12.0f));            // flat index 11, same as t[2, 3]
  CHECK((r[0, 0] == 1.0f));

  CHECK_THROWS(t.reshape({5, 5}), std::invalid_argument);  // wrong element count

  // Bug-fix regression test: reshaping a NON-contiguous view (here, a broadcast) must be rejected, not
  // silently produce a view whose flat-index-to-offset mapping is wrong.
  TensorView<float> vec(storage.data(), {1, 4});
  TensorView<float> bcast = broadcast_to(vec, {3, 4});  // stride {0, 1}: not contiguous
  CHECK(!bcast.is_contiguous());
  CHECK_THROWS(bcast.reshape({12}), std::logic_error);

  // A slice of a contiguous tensor is itself contiguous (it's a contiguous sub-block starting partway
  // through the buffer) and CAN be reshaped.
  TensorView<float> t3(storage.data(), {3, 4});
  TensorView<float> sl = t3.slice(1);  // shape {4}, contiguous
  CHECK(sl.is_contiguous());
  TensorView<float> sl_reshaped = sl.reshape({2, 2});
  CHECK((sl_reshaped[1, 1] == storage[7]));
}

static void test_rank_mismatch_indexing() {
  std::vector<float> storage(6);
  TensorView<float> t(storage.data(), {2, 3});
  CHECK_THROWS(t[0], std::out_of_range);        // too few indices for rank 2
  CHECK_THROWS((t[0, 0, 0]), std::out_of_range);  // too many
}

int main() {
  struct T { const char* name; void (*fn)(); };
  const T tests[] = {
      {"strides and multi-arg operator[] / operator()", test_strides_and_indexing},
      {"scalar (rank 0) has numel() == 1 [bug fix]", test_scalar_is_one_element},
      {"bounds checking throws (not a silent assert)", test_bounds_checking},
      {"slice is zero-copy and mutation is visible through both views", test_slicing_is_zero_copy_and_mutates_storage},
      {"broadcast_to: strides, aliasing, mutation, invalid shapes", test_broadcast},
      {"reshape: correctness, and rejection when non-contiguous [bug fix]", test_reshape},
      {"indexing with the wrong rank throws", test_rank_mismatch_indexing},
  };
  for (const T& t : tests) {
    const int before = g_fail;
    try {
      t.fn();
    } catch (const std::exception& e) {
      ++g_fail;
      std::fprintf(stderr, "  unexpected exception: %s\n", e.what());
    }
    std::printf("[%s] %s\n", g_fail == before ? "PASS" : "FAIL", t.name);
  }
  std::printf("\ntensor_view: %d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
