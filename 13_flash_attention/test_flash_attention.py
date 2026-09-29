# Custom flash-attention vs a PyTorch reference, over shapes with N not a multiple of 32 (the masked tail
# KV block), checking for NaN/Inf explicitly with torch.isfinite. Requires PyTorch + CUDA; not run here.

import torch
import math
import torch.nn.functional as F
import custom_flash_attention


def torch_attention(q, k, v):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def run(N, d, seed):
    torch.manual_seed(seed)
    Q = torch.randn(N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(N, d, device='cuda', dtype=torch.float32)

    O_torch = torch_attention(Q, K, V)
    O_custom = custom_flash_attention.forward(Q, K, V)

    finite = torch.isfinite(O_custom).all().item()
    is_close = finite and torch.allclose(O_custom, O_torch, atol=1e-3, rtol=1e-3)
    diff = torch.max(torch.abs(O_custom - O_torch)).item() if finite else float('nan')
    print(f"  N={N:5d} d={d:4d}  finite={finite}  max err={diff:.6f}  -> {'OK' if is_close else 'MISMATCH'}")
    return is_close


def test():
    print("=== Flash Attention (simplified): custom CUDA vs PyTorch reference ===")
    cases = [
        (2048, 64, 1),   # baseline aligned case
        (2000, 64, 2),   # N not a multiple of 32: exercises the masked tail KV block
        (33, 64, 3),     # one full block + a nearly-empty tail block
        (17, 32, 4),     # N smaller than one KV block entirely
        (4096, 128, 5),  # larger head dim: exercises the extended shared-memory request (fix #2)
    ]
    all_ok = all(run(N, d, seed) for N, d, seed in cases)
    if not all_ok:
        raise SystemExit("flash attention check failed")


if __name__ == '__main__':
    test()
