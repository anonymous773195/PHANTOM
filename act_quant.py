
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ActQuantConfig:
    nbits: int = 32
    mode: str = "global_pact"  # global_pact | channel_mean | channel_percentile | channel_minmax | channel_pact
    percentile: float = 0.95
    alpha_init: float = 6.0


class EncoderActivationQuantizer(nn.Module):

    def __init__(self, *, cfg: ActQuantConfig, channels: int | None = None):
        super().__init__()
        self.nbits = int(cfg.nbits)
        self.mode = str(cfg.mode)
        self.percentile = float(cfg.percentile)

        if self.mode not in {
            "global_pact",
            "channel_mean",
            "channel_percentile",
            "channel_minmax",
            "channel_pact",
        }:
            raise ValueError(f"Unknown activation quant mode: {self.mode!r}")

        if self.mode == "channel_pact":
            if channels is None:
                raise ValueError("channels must be provided for channel_pact mode")
            self.alpha = nn.Parameter(torch.full((int(channels),), float(cfg.alpha_init)))
        elif self.mode == "global_pact":
            self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)))
        else:
            self.register_parameter("alpha", None)

    def _get_channel_scale(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C) -> scale: (C,)
        if self.mode == "channel_mean":
            s = x.abs().mean(dim=(0, 1))
        elif self.mode == "channel_percentile":
            q = self.percentile
            q = min(max(q, 0.0), 1.0)
            # torch.quantile does not accept a tuple dim across versions; flatten (B,N)
            x_flat = x.abs().reshape(-1, x.shape[-1])  # (B*N, C)
            # torch.quantile requires float/double; AMP may feed fp16/bf16 activations.
            xq = x_flat
            if xq.dtype not in (torch.float32, torch.float64):
                xq = xq.float()
            s = torch.quantile(xq, q, dim=0).to(dtype=x.dtype)
        elif self.mode == "channel_minmax":
            xmin = x.amin(dim=(0, 1))
            xmax = x.amax(dim=(0, 1))
            s = torch.maximum(xmin.abs(), xmax.abs())
        else:
            raise RuntimeError("Not a channel-scale mode")

        # Treat as a per-forward statistic; do not backprop through s.
        return s.detach().clamp(min=1e-4)

    def _get_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "global_pact":
            return self.alpha.abs().clamp(min=1e-4)
        if self.mode == "channel_pact":
            return self.alpha.abs().clamp(min=1e-4)
        return self._get_channel_scale(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nbits >= 32:
            return x

        s = self._get_scale(x)
        if s.ndim == 1:
            s_view = s.view(1, 1, -1)
        else:
            s_view = s

        x_clip = torch.clamp(x, -s_view, s_view)

        if self.nbits <= 1:
            y = torch.sign(x_clip) * s_view
            return x + (y - x).detach()

        qn = float(2 ** (self.nbits - 1) - 1)
        scale = qn / s_view
        y = round_ste(x_clip * scale) / scale
        return y

