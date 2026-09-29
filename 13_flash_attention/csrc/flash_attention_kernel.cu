// Tiled attention with online softmax (simplified FlashAttention forward pass). The online-softmax
// rescaling recurrence is verified in ../tests/test_flash_attention_cpu_sim.cpp against naive attention,
// over N not a multiple of the KV block size and several head dimensions. launch_flash_attention requests
// the exact dynamic shared memory the launch needs (opting in beyond the default 48 KB via
// cudaFuncSetAttribute when d is large) and checks for launch errors.
// Limitations (by design): no batch/multi-head dimension, no causal masking, Q*K^T on CUDA cores.
// Requires PyTorch/LibTorch and a CUDA toolkit; not built here.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <float.h>

#define BLOCK_SIZE_Q 32
#define BLOCK_SIZE_KV 32

__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d
) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int q_offset = bx * BLOCK_SIZE_Q * d;
    int row_idx = bx * BLOCK_SIZE_Q + tx;

    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = &s_Q[BLOCK_SIZE_Q * d];
    float* s_V = &s_K[BLOCK_SIZE_KV * d];
    float* s_O = &s_V[BLOCK_SIZE_KV * d];

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    if (row_idx < N) {
        for (int i = 0; i < d; ++i) {
            s_Q[tx * d + i] = Q[q_offset + tx * d + i];
        }
    }
    for (int i = 0; i < d; ++i) s_O[tx * d + i] = 0.0f;

    __syncthreads();

    int num_blocks_kv = (N + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;
    for (int by = 0; by < num_blocks_kv; ++by) {

        int kv_row_start = by * BLOCK_SIZE_KV;

        if (kv_row_start + tx < N) {
             for (int i = 0; i < d; ++i) {
                s_K[tx * d + i] = K[(kv_row_start + tx) * d + i];
                s_V[tx * d + i] = V[(kv_row_start + tx) * d + i];
             }
        } else {
             for (int i = 0; i < d; ++i) {
                s_K[tx * d + i] = 0.0f;
                s_V[tx * d + i] = 0.0f;
             }
        }

        __syncthreads();

        if (row_idx < N) {
            float m_i_new = m_i;
            float S_ij[BLOCK_SIZE_KV];

            for (int j = 0; j < BLOCK_SIZE_KV; ++j) {
                float sum = 0.0f;
                for (int c = 0; c < d; ++c) {
                    sum += s_Q[tx * d + c] * s_K[j * d + c];
                }
                sum *= (1.0f / sqrtf((float)d));
                S_ij[j] = sum;
                if (kv_row_start + j < N) {
                   m_i_new = max(m_i_new, sum);
                }
            }

            float l_i_new = 0.0f;
            float P_ij[BLOCK_SIZE_KV];

            for (int j = 0; j < BLOCK_SIZE_KV; ++j) {
                if (kv_row_start + j < N) {
                    P_ij[j] = expf(S_ij[j] - m_i_new);
                    l_i_new += P_ij[j];
                } else {
                    P_ij[j] = 0.0f;
                }
            }

            float alpha = expf(m_i - m_i_new);
            l_i_new = l_i * alpha + l_i_new;

            for (int c = 0; c < d; ++c) {
                float v_sum = 0.0f;
                for (int j = 0; j < BLOCK_SIZE_KV; ++j) {
                    v_sum += P_ij[j] * s_V[j * d + c];
                }
                s_O[tx * d + c] = (s_O[tx * d + c] * l_i * alpha + v_sum) / l_i_new;
            }

            m_i = m_i_new;
            l_i = l_i_new;
        }
        __syncthreads();
    }

    if (row_idx < N) {
        for (int c = 0; c < d; ++c) {
            O[row_idx * d + c] = s_O[tx * d + c];
        }
    }
}

void launch_flash_attention(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V, torch::Tensor& O) {
    int N = Q.size(0);
    int d = Q.size(1);

    int num_blocks = (N + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE_Q);

    size_t shared_mem_size = (BLOCK_SIZE_Q * d + 2 * BLOCK_SIZE_KV * d + BLOCK_SIZE_Q * d) * sizeof(float);

    // Request extended dynamic shared memory when the launch needs more than the default 48 KB
    // (true for d >= 97 with this layout); a no-op otherwise.
    if (shared_mem_size > 48 * 1024) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            flash_attention_forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_mem_size));
        TORCH_CHECK(attr_err == cudaSuccess,
                    "flash_attention: requested ", shared_mem_size,
                    " bytes of shared memory per block, which this GPU cannot provide even with the "
                    "extended opt-in (", cudaGetErrorString(attr_err), "). Reduce the head dimension.");
    }

    flash_attention_forward_kernel<<<grid, block, shared_mem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, d
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flash_attention_forward_kernel launch failed: ", cudaGetErrorString(err));
}
