// PyTorch CUDA extension: fused SiLU forward + backward. The forward/backward formulas are verified in
// ../tests/test_silu_math.cpp. Requires PyTorch/LibTorch and a CUDA toolkit; not built here.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void silu_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = x[idx];
        y[idx] = val * sigmoid(val);
    }
}

__global__ void silu_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ x,
    float* __restrict__ grad_x,
    int num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = x[idx];
        float sig = sigmoid(val);
        float local_grad = sig * (1.0f + val * (1.0f - sig));
        grad_x[idx] = grad_output[idx] * local_grad;
    }
}

void launch_silu_forward(const torch::Tensor& x, torch::Tensor& y) {
    int num_elements = x.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    silu_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        num_elements
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "silu_forward_kernel launch failed: ", cudaGetErrorString(err));
}

void launch_silu_backward(const torch::Tensor& grad_output, const torch::Tensor& x, torch::Tensor& grad_x) {
    int num_elements = x.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    silu_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        x.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        num_elements
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "silu_backward_kernel launch failed: ", cudaGetErrorString(err));
}
