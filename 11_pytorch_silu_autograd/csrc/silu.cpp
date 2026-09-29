// PyTorch bindings for the fused SiLU kernels, with explicit dtype/shape/contiguity checks.
// Requires PyTorch/LibTorch; not built here.

#include <torch/extension.h>

void launch_silu_forward(const torch::Tensor& x, torch::Tensor& y);
void launch_silu_backward(const torch::Tensor& grad_output, const torch::Tensor& x, torch::Tensor& grad_x);

torch::Tensor silu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "silu: x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "silu: x must be float32, got ", x.scalar_type());
    TORCH_CHECK(x.is_contiguous(), "silu: x must be contiguous");

    auto y = torch::empty_like(x);
    launch_silu_forward(x, y);
    return y;
}

torch::Tensor silu_backward(torch::Tensor grad_output, torch::Tensor x) {
    TORCH_CHECK(grad_output.is_cuda() && x.is_cuda(), "silu: tensors must be CUDA tensors");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32 && x.scalar_type() == torch::kFloat32,
                "silu: tensors must be float32");
    TORCH_CHECK(grad_output.sizes() == x.sizes(), "silu: grad_output and x must have the same shape");
    TORCH_CHECK(grad_output.is_contiguous() && x.is_contiguous(), "silu: tensors must be contiguous");

    auto grad_x = torch::empty_like(x);
    launch_silu_backward(grad_output, x, grad_x);
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &silu_forward, "Fused SiLU Forward");
    m.def("backward", &silu_backward, "Fused SiLU Backward");
}
