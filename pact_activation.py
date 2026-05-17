from __future__ import annotations

import torch
import torch.nn as nn


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)


class EncoderPACTActivation(nn.Module):

    def __init__(self, nbits: int, alpha_init: float = 6.0):
        super().__init__()
        self.nbits = int(nbits)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nbits >= 32:
            return x

        alpha = self.alpha.abs().clamp(min=1e-4)
        x = torch.clamp(x, -alpha, alpha)

        if self.nbits <= 1:
            y = torch.sign(x) * alpha
            return x + (y - x).detach()

        qn = float(2 ** (self.nbits - 1) - 1)
        scale = qn / alpha
        y = round_ste(x * scale) / scale
        return y
