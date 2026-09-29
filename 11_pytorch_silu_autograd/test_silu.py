# Autograd forward/backward check vs torch.nn.functional.silu, using a random grad_output so the
# chain-rule multiplication is actually exercised. Requires PyTorch + CUDA; not run here.

import torch
import torch.nn.functional as F
import custom_silu


class FusedSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return custom_silu.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        return custom_silu.backward(grad_output, x)


fused_silu = FusedSiLUFunction.apply


def test_autograd():
    print("=== Fused SiLU: forward and backward vs torch.nn.functional.silu ===")

    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32, requires_grad=True)

    y_custom = fused_silu(x)
    y_torch = F.silu(x)
    fwd_ok = torch.allclose(y_custom, y_torch, atol=1e-5)
    print(f"Forward matches  : {fwd_ok}")

    # Fix: random grad_out, not all-ones (see file header).
    grad_out = torch.randn_like(y_custom)

    y_custom.backward(grad_out, retain_graph=True)
    grad_x_custom = x.grad.clone()
    x.grad.zero_()

    y_torch.backward(grad_out)
    grad_x_torch = x.grad.clone()

    bwd_ok = torch.allclose(grad_x_custom, grad_x_torch, atol=1e-4, rtol=1e-4)
    diff = torch.max(torch.abs(grad_x_custom - grad_x_torch)).item()
    print(f"Backward matches : {bwd_ok} (max err: {diff:.6e})")

    if not (fwd_ok and bwd_ok):
        raise SystemExit("SiLU autograd check failed")


if __name__ == '__main__':
    test_autograd()
