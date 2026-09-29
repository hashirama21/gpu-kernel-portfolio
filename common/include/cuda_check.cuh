#pragma once
// Shared CUDA plumbing: an error-checking macro (CUDA_CHECK), an RAII device buffer (DeviceBuffer), and
// an event-based timer (GpuTimer). Compiled only into the CUDA targets.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace portfolio {

// Aborts with a clear message on the first CUDA error. A benchmark harness that silently produced
// wrong numbers because a kernel launch failed is worse than one that stops immediately.
#define CUDA_CHECK(expr)                                                                             \
  do {                                                                                               \
    const cudaError_t _err = (expr);                                                                 \
    if (_err != cudaSuccess) {                                                                       \
      std::fprintf(stderr, "CUDA error: %s\n  at %s:%d in `%s`\n", cudaGetErrorString(_err), __FILE__, \
                    __LINE__, #expr);                                                                \
      std::exit(EXIT_FAILURE);                                                                       \
    }                                                                                                \
  } while (0)

// A single owned device allocation of `T[n]`, non-copyable, movable. Frees itself in the destructor,
// including when an exception or an early `return` skips the matching cudaFree the code would otherwise
// need to write by hand.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) : n_(n) {
    if (n_ > 0) CUDA_CHECK(cudaMalloc(&ptr_, n_ * sizeof(T)));
  }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
    o.ptr_ = nullptr;
    o.n_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      n_ = o.n_;
      o.ptr_ = nullptr;
      o.n_ = 0;
    }
    return *this;
  }

  T* get() const { return ptr_; }
  std::size_t size() const { return n_; }
  std::size_t bytes() const { return n_ * sizeof(T); }

  void upload(std::span<const T> host) {
    if (host.size() != n_) throw std::invalid_argument("DeviceBuffer::upload: size mismatch");
    if (n_ > 0) CUDA_CHECK(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice));
  }
  void download(std::span<T> host) const {
    if (host.size() != n_) throw std::invalid_argument("DeviceBuffer::download: size mismatch");
    if (n_ > 0) CUDA_CHECK(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
  }
  void zero() {
    if (n_ > 0) CUDA_CHECK(cudaMemset(ptr_, 0, bytes()));
  }

 private:
  void reset() {
    if (ptr_) cudaFree(ptr_);  // destructor: ignore the error, there is nothing left to report it to
    ptr_ = nullptr;
    n_ = 0;
  }
  T* ptr_ = nullptr;
  std::size_t n_ = 0;
};

// RAII pair of CUDA events for timing one kernel launch, in milliseconds.
class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  // Synchronizes on the stop event and returns the elapsed time. Call this only after start().
  float stop_ms() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

 private:
  cudaEvent_t start_{}, stop_{};
};

// Runs `launch` (the kernel launch statement) `warmup` times, then `iters` times; returns the median
// time in milliseconds. The warm-up hides one-time context/JIT initialization on the first launch.
template <typename Launch>
double gpu_benchmark_ms(Launch&& launch, int warmup = 3, int iters = 10) {
  for (int i = 0; i < warmup; ++i) launch();
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    GpuTimer t;
    t.start();
    launch();
    samples.push_back(t.stop_ms());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace portfolio
