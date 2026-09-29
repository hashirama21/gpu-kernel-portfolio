# Correctness (including non-tile-aligned shapes) and benchmark vs cuBLAS. Requires PyTorch + CUDA;
# not run here.

import torch
import custom_matmul


def run_correctness():
    print("=== Correctness (including shapes NOT a multiple of TILE_SIZE=32) ===")
    shapes = [(2048, 2048, 2048), (100, 200, 50), (33, 33, 33), (1, 64, 1), (65, 31, 129)]
    all_ok = True
    for M, K, N in shapes:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        C_custom = custom_matmul.forward(A, B)
        C_torch = torch.matmul(A, B)
        ok = torch.allclose(C_custom, C_torch, atol=1e-2, rtol=1e-2)
        diff = torch.max(torch.abs(C_custom - C_torch)).item()
        print(f"  M={M:5d} K={K:5d} N={N:5d}  max err {diff:.2e}  -> {'OK' if ok else 'MISMATCH'}")
        all_ok &= ok
    if not all_ok:
        raise SystemExit("correctness check failed")


def run_benchmark(M=2048, K=2048, N=2048, warmup=10, iters=100):
    print(f"\n=== Benchmark A({M}x{K}) * B({K}x{N}) ===")
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    for _ in range(warmup):
        custom_matmul.forward(A, B)
        torch.matmul(A, B)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iters):
        _ = custom_matmul.forward(A, B)
    end_evt.record()
    torch.cuda.synchronize()
    custom_ms = start_evt.elapsed_time(end_evt) / iters
    gflops_custom = (2.0 * M * N * K) / (custom_ms * 1e6)

    start_evt.record()
    for _ in range(iters):
        _ = torch.matmul(A, B)
    end_evt.record()
    torch.cuda.synchronize()
    torch_ms = start_evt.elapsed_time(end_evt) / iters
    gflops_torch = (2.0 * M * N * K) / (torch_ms * 1e6)

    print(f"Custom CUDA Tiled : {custom_ms:.3f} ms | {gflops_custom:.2f} GFLOPS")
    print(f"PyTorch cuBLAS    : {torch_ms:.3f} ms | {gflops_torch:.2f} GFLOPS")
    print(f"vs cuBLAS         : {gflops_custom / gflops_torch * 100:.2f}%")


if __name__ == '__main__':
    run_correctness()
    run_benchmark()
