// PyTorch binding for the flash-attention forward kernel, with explicit dtype/shape/contiguity checks.
// Requires PyTorch/LibTorch; not built here.

#include <torch/extension.h>

void launch_flash_attention(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V, torch::Tensor& O);

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "flash_attention: Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32 && K.scalar_type() == torch::kFloat32 && V.scalar_type() == torch::kFloat32,
                "flash_attention: Q, K, V must be float32");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "flash_attention: Q, K, V must be contiguous");
    TORCH_CHECK(Q.dim() == 2 && K.dim() == 2 && V.dim() == 2, "flash_attention: expected [sequence, head_dim] tensors");
    TORCH_CHECK(Q.size(0) == K.size(0) && K.size(0) == V.size(0), "flash_attention: sequence lengths must match");
    TORCH_CHECK(Q.size(1) == K.size(1) && K.size(1) == V.size(1), "flash_attention: head dimensions must match");

    auto O = torch::empty_like(Q);
    launch_flash_attention(Q, K, V, O);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "Flash Attention Forward (simplified)");
}
