"""Python wrappers around the custom CUDA kernel + torch fallbacks.

Includes a torch.autograd.Function so the fused op is differentiable
and can be used inside training graphs without breaking backward.
"""

import torch
import torch.nn.functional as F

try:
    from hermes_fast import _C as _native
    _NATIVE_AVAILABLE = True
except ImportError:
    _native = None
    _NATIVE_AVAILABLE = False


class _FusedLNResidualFn(torch.autograd.Function):
    """Autograd wrapper for fused_ln_residual forward + backward.

    Forward:  out = gamma * LayerNorm(x + residual) + beta
    Backward: computes grad_x, grad_residual, grad_gamma, grad_beta
    using the standard LayerNorm backward formula, fused into a single
    kernel launch when possible.

    When the native CUDA kernel is unavailable, the entire forward+backward
    is expressed in pure PyTorch ops (still correct, just slower).
    """

    @staticmethod
    def forward(ctx, x, residual, gamma, beta, eps):
        D = x.size(-1)
        x2 = x.contiguous().view(-1, D)
        r2 = residual.contiguous().view(-1, D)

        if (
            _NATIVE_AVAILABLE
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and gamma.dtype == x.dtype
            and beta.dtype == x.dtype
            and D <= 1024
        ):
            out_2d = _native.fused_ln_residual(x2, r2, gamma, beta, eps)
        else:
            summed = x2 + r2
            out_2d = F.layer_norm(summed, (D,), gamma, beta, eps)

        ctx.save_for_backward(x2, r2, gamma, out_2d)
        ctx.eps = eps
        ctx.D = D
        # Save the ORIGINAL input shapes — backward must return grads
        # matching forward inputs exactly, not the flattened (B*N, D) view.
        ctx.x_shape = x.shape
        ctx.residual_shape = residual.shape

        return out_2d.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x2, r2, gamma, out_2d = ctx.saved_tensors
        eps = ctx.eps
        D = ctx.D

        g = grad_output.contiguous().view(-1, D)

        # Reconstruct z = x + residual and z_hat from the saved output.
        z = x2 + r2
        mean = z.mean(dim=-1, keepdim=True)
        var = z.var(dim=-1, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        z_hat = (z - mean) * rstd

        # LayerNorm backward:
        #   grad_gamma = sum_batch(g * z_hat)
        #   grad_beta  = sum_batch(g)
        #   grad_z     = rstd * gamma * (g - mean(g) - z_hat * mean(g * z_hat))
        grad_gamma = (g * z_hat).sum(dim=0)
        grad_beta = g.sum(dim=0)

        mean_g = g.mean(dim=-1, keepdim=True)
        mean_gz = (g * z_hat).mean(dim=-1, keepdim=True)
        grad_z = rstd * (gamma * (g - mean_g - z_hat * mean_gz))

        # z = x + residual, so grad_x = grad_residual = grad_z.
        # Reshape from flat (B*N, D) back to the original (possibly ND) input
        # shapes that autograd expects.
        grad_x = grad_z.view(ctx.x_shape)
        grad_residual = grad_z.view(ctx.residual_shape).clone()

        return grad_x, grad_residual, grad_gamma, grad_beta, None


def fused_layernorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute LayerNorm(x + residual) along the last dim.

    Fully differentiable — supports autograd backward.
    Uses the fused CUDA kernel on bf16/fp16 CUDA when available,
    otherwise falls back to torch.add + F.layer_norm (still correct
    and differentiable).
    """
    return _FusedLNResidualFn.apply(x, residual, gamma, beta, eps)


def native_available() -> bool:
    return _NATIVE_AVAILABLE
