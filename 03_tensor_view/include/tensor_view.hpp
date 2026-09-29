#pragma once
// N-dimensional strided tensor view over externally-owned memory (no allocation, no ownership).
// Zero-copy slicing, broadcasting, and contiguous reshape; every access is bounds-checked and throws
// (not assert). Uses C++23's multi-argument subscript, so t[i, j, k] and t(i, j, k) are equivalent.

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace portfolio::tensor {

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <Numeric T>
class TensorView {
 public:
  // Contiguous row-major view over `ptr`, `shape.size()` dimensions.
  TensorView(T* ptr, std::vector<std::size_t> shape) : data_ptr_(ptr), shape_(std::move(shape)) {
    strides_ = contiguous_strides(shape_);
  }
  // Arbitrary strides (used internally by slice(), reshape(), and broadcast_to()).
  TensorView(T* ptr, std::vector<std::size_t> shape, std::vector<std::size_t> strides)
      : data_ptr_(ptr), shape_(std::move(shape)), strides_(std::move(strides)) {
    if (shape_.size() != strides_.size()) throw std::invalid_argument("TensorView: shape/strides rank mismatch");
  }

  const std::vector<std::size_t>& shape() const { return shape_; }
  const std::vector<std::size_t>& strides() const { return strides_; }
  std::size_t rank() const { return shape_.size(); }
  T* data() const { return data_ptr_; }

  // Total element count. An empty shape (rank 0) is a scalar: one element (NumPy/ATen convention).
  [[nodiscard]] std::size_t numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), std::size_t{1}, std::multiplies<>());
  }

  // True iff the strides are the default row-major strides for this shape — the condition reshape() needs
  // to reinterpret a flat index directly as the new shape's row-major offset. A broadcast view (stride 0)
  // is not contiguous.
  [[nodiscard]] bool is_contiguous() const { return strides_ == contiguous_strides(shape_); }

  // C++23 multi-argument subscript: t[i, j, k]. Throws std::out_of_range on a bad index or rank.
  template <std::integral... Idx>
  T& operator[](Idx... idx) const {
    return at(std::array<std::size_t, sizeof...(Idx)>{static_cast<std::size_t>(idx)...});
  }
  // operator() is a synonym for operator[], sharing the same bounds-checked implementation.
  template <std::integral... Idx>
  T& operator()(Idx... idx) const {
    return (*this)[idx...];
  }

  // Zero-copy sub-view fixing index `i` on axis 0 (rank decreases by 1).
  [[nodiscard]] TensorView slice(std::size_t index) const {
    if (shape_.empty()) throw std::out_of_range("slice: cannot slice a scalar (rank 0)");
    if (index >= shape_[0]) throw std::out_of_range("slice: index out of range");
    std::vector<std::size_t> new_shape(shape_.begin() + 1, shape_.end());
    std::vector<std::size_t> new_strides(strides_.begin() + 1, strides_.end());
    return TensorView(data_ptr_ + index * strides_[0], std::move(new_shape), std::move(new_strides));
  }

  // Zero-copy reshape. Requires a contiguous view.
  [[nodiscard]] TensorView reshape(std::vector<std::size_t> new_shape) const {
    if (!is_contiguous())
      throw std::logic_error("reshape: view is not contiguous (make a copy first, or use a compatible slice)");
    const std::size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), std::size_t{1}, std::multiplies<>());
    if (new_numel != numel()) throw std::invalid_argument("reshape: element count mismatch");
    return TensorView(data_ptr_, std::move(new_shape));
  }

  static std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> s(shape.size());
    std::size_t acc = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
      s[i] = acc;
      acc *= shape[i];
    }
    return s;
  }

 private:
  template <std::size_t R>
  T& at(const std::array<std::size_t, R>& idx) const {
    if (idx.size() != shape_.size()) throw std::out_of_range("TensorView: index rank does not match tensor rank");
    std::size_t offset = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) {
      if (idx[i] >= shape_[i]) throw std::out_of_range("TensorView: index out of bounds");
      offset += idx[i] * strides_[i];
    }
    return data_ptr_[offset];
  }

  T* data_ptr_;
  std::vector<std::size_t> shape_;
  std::vector<std::size_t> strides_;
};

// Broadcasts `src` to `target_shape` with zero-copy (every broadcast axis gets stride 0), following
// NumPy/PyTorch broadcasting rules: shapes are right-aligned, and each axis must either already match or
// have size 1 in `src`.
template <Numeric T>
TensorView<T> broadcast_to(const TensorView<T>& src, const std::vector<std::size_t>& target_shape) {
  const std::size_t src_rank = src.rank(), target_rank = target_shape.size();
  if (target_rank < src_rank) throw std::invalid_argument("broadcast_to: target rank must be >= source rank");

  std::vector<std::size_t> new_strides(target_rank, 0);
  const std::size_t rank_diff = target_rank - src_rank;
  for (std::size_t i = target_rank; i-- > 0;) {
    if (i < rank_diff) {
      new_strides[i] = 0;  // a brand-new leading axis: every position reads the same data
      continue;
    }
    const std::size_t src_idx = i - rank_diff;
    const std::size_t src_dim = src.shape()[src_idx], target_dim = target_shape[i];
    if (src_dim == target_dim) {
      new_strides[i] = src.strides()[src_idx];
    } else if (src_dim == 1) {
      new_strides[i] = 0;  // broadcast axis: stride 0 makes every index alias the single element
    } else {
      throw std::invalid_argument("broadcast_to: incompatible dimension (" + std::to_string(src_dim) + " vs " +
                                  std::to_string(target_dim) + ")");
    }
  }
  return TensorView<T>(src.data(), target_shape, std::move(new_strides));
}

}  // namespace portfolio::tensor
