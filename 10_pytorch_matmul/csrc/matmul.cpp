// PyTorch binding for the tiled SGEMM kernel, with explicit dtype/shape/contiguity checks that give
// clearer errors than data_ptr<float>() alone. Requires PyTorch/LibTorch; not built here.

#include <torch/extension.h>

void launch_matmul_cuda(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C);

torch::Tensor matmul_cuda_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "matmul: A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "matmul: B must be a CUDA tensor");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "matmul: A must be float32, got ", A.scalar_type());
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "matmul: B must be float32, got ", B.scalar_type());
    TORCH_CHECK(A.is_contiguous(), "matmul: A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "matmul: B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul: A and B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "matmul: inner dimensions mismatch: A is ", A.sizes(), ", B is ", B.sizes());

    int M = A.size(0);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    launch_matmul_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda_forward, "Tiled SGEMM Matrix Multiplication Forward (CUDA)");
}
