import torch
import torch.nn as nn
import math


USE_POW2_SCALES: bool = False


def set_pow2_scales(enabled: bool) -> None:
    global USE_POW2_SCALES
    USE_POW2_SCALES = bool(enabled)


def _maybe_make_pot_scale(stat: torch.Tensor) -> torch.Tensor:
    stat_clamped = stat.clamp(min=1e-8)

    if not USE_POW2_SCALES:
        return stat_clamped

    log2_s = torch.log2(stat_clamped)

    if torch.is_grad_enabled() and log2_s.requires_grad:
        with torch.no_grad():
            k_floor = torch.floor(log2_s)
            frac = (log2_s - k_floor).clamp(min=0.0, max=1.0)
            rnd = torch.rand_like(frac)
            k = torch.where(rnd < frac, k_floor + 1.0, k_floor)
        k_ste = log2_s + (k - log2_s).detach()
        return torch.pow(2.0, k_ste)

    k_det = torch.round(log2_s)
    return torch.pow(2.0, k_det)


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fwht(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if (n & (n - 1)) != 0:
        raise ValueError(f"Last dimension must be a power of 2, got {n}.")

    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2 * h)
        lo = x[..., :h]
        hi = x[..., h:]
        x = torch.cat([lo + hi, lo - hi], dim=-1)
        x = x.reshape(*x.shape[:-2], n)
        h *= 2
    return x



def make_hadamard_projections(
    out_features: int,
    in_features: int,
    seed: int = 42,
    device=None,
    dtype=None,
) -> torch.Tensor:
    
    n_pad = max(2, next_power_of_2(out_features))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = (torch.randint(0, 2, (n_pad,), generator=gen) * 2 - 1).float()

    kw = {}
    if device is not None:
        kw["device"] = device
    if dtype is not None:
        kw["dtype"] = dtype
    if kw:
        d = d.to(**kw)
    return d


def hadamard_embed(
    A: torch.Tensor,
    d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, M = A.shape
    n_pad = d.shape[0]

    scale = torch.quantile(A.abs(), 0.95, dim=1, keepdim=True)
    s = _maybe_make_pot_scale(scale)
    A_norm = A / s

    if n_pad > N:
        pad = A_norm.new_zeros(n_pad - N, M)
        A_pad = torch.cat([A_norm, pad], dim=0)
    else:
        A_pad = A_norm

    A_pad = d[:, None].to(A_pad.dtype) * A_pad

    Z_pad = fwht(A_pad.T).T / math.sqrt(n_pad)

    half = n_pad // 2
    Z_re = Z_pad[:half, :].contiguous()
    Z_im = Z_pad[half:, :].contiguous()

    return Z_re, Z_im, s


def hadamard_reconstruct(
    Z_re: torch.Tensor,
    Z_im: torch.Tensor,
    d: torch.Tensor,
    s: torch.Tensor | float = 1.0,
    original_size: int | None = None,
) -> torch.Tensor:
    
    n_pad = d.shape[0]

    Z_pad = torch.cat([Z_re, Z_im], dim=0)
    A_pad = fwht(Z_pad.T).T / math.sqrt(n_pad)
    A_pad = d[:, None].to(A_pad.dtype) * A_pad
    N = original_size if original_size is not None else n_pad
    A_norm_q = A_pad[:N, :]

    return A_norm_q * s


class HadamardPhaseQuantSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, d):
        N = A.shape[0]
        Z_re, Z_im, s = hadamard_embed(A, d)
        Z_re_q, Z_im_q = PhaseQuantSTE.apply(Z_re, Z_im)
        A_q = hadamard_reconstruct(Z_re_q, Z_im_q, d, s, original_size=N)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HadamardPhaseQuantSTE_V2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, d):
        N = A.shape[0]
        Z_re, Z_im, s = hadamard_embed(A, d)
        Z_re_q, Z_im_q = PhaseQuantSTE_V2.apply(Z_re, Z_im)
        A_q = hadamard_reconstruct(Z_re_q, Z_im_q, d, s, original_size=N)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HadamardPhaseQuantSTE_V3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, d):
        N = A.shape[0]
        Z_re, Z_im, s = hadamard_embed(A, d)
        Z_re_q, Z_im_q = PhaseQuantSTE_V3.apply(Z_re, Z_im)
        A_q = hadamard_reconstruct(Z_re_q, Z_im_q, d, s, original_size=N)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HadamardPhaseQuantSTE_V4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, d):
        N = A.shape[0]
        Z_re, Z_im, s = hadamard_embed(A, d)
        Z_re_q, Z_im_q = PhaseQuantSTE_V4.apply(Z_re, Z_im)
        A_q = hadamard_reconstruct(Z_re_q, Z_im_q, d, s, original_size=N)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_rkhs_projections(
    out_features: int,
    in_features: int,
    seed: int = 42,
    device=None,
    dtype=None,
) -> torch.Tensor:

    if out_features % 2 != 0:
        raise ValueError(
            f"out_features must be even, got {out_features}."
        )
    n = out_features

    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(n, n, generator=gen)
    Omega, _ = torch.linalg.qr(G)

    if device is not None or dtype is not None:
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        Omega = Omega.to(**kw)

    return Omega


def rkhs_embed(
    A: torch.Tensor,
    Omega: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    scale = torch.quantile(A.abs(), 0.95, dim=1, keepdim=True)
    s = scale.clamp(min=1e-8)
    A_norm = A / s

    Z = Omega @ A_norm

    n = A.shape[0] // 2
    Z_re = Z[:n, :].contiguous()
    Z_im = Z[n:, :].contiguous()
    return Z_re, Z_im, s


def rkhs_reconstruct(
    Z_re: torch.Tensor,
    Z_im: torch.Tensor,
    Omega: torch.Tensor,
    s: torch.Tensor | float = 1.0,
) -> torch.Tensor:

    Z_stacked = torch.cat([Z_re, Z_im], dim=0)
    A_norm_q = Omega.T @ Z_stacked
    return A_norm_q * s


class RKHSPhaseQuantSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, Omega):
        Z_re, Z_im, s = rkhs_embed(A, Omega)
        Z_re_q, Z_im_q = PhaseQuantSTE.apply(Z_re, Z_im)
        A_q = rkhs_reconstruct(Z_re_q, Z_im_q, Omega, s)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class RKHSPhaseQuantSTE_V2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, Omega):
        Z_re, Z_im, s = rkhs_embed(A, Omega)
        Z_re_q, Z_im_q = PhaseQuantSTE_V2.apply(Z_re, Z_im)
        A_q = rkhs_reconstruct(Z_re_q, Z_im_q, Omega, s)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class RKHSPhaseQuantSTE_V3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, Omega):
        Z_re, Z_im, s = rkhs_embed(A, Omega)
        Z_re_q, Z_im_q = PhaseQuantSTE_V3.apply(Z_re, Z_im)
        A_q = rkhs_reconstruct(Z_re_q, Z_im_q, Omega, s)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class RKHSPhaseQuantSTE_V4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, Omega):
        Z_re, Z_im, s = rkhs_embed(A, Omega)
        Z_re_q, Z_im_q = PhaseQuantSTE_V4.apply(Z_re, Z_im)
        A_q = rkhs_reconstruct(Z_re_q, Z_im_q, Omega, s)
        return A_q.to(A.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class BitNetQuantSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean()
        alpha = w.mean()
        centered_w = w - alpha
        binarized_w = torch.where(centered_w > 0, 1.0, -1.0).to(w.dtype)
        return binarized_w * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitNet1_58QuantSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w):
        gamma = w.abs().mean()
        w_normalized = w / (gamma + 1e-5)
        w_quantized = torch.clamp(torch.round(w_normalized), -1.0, 1.0)
        return (w_quantized * gamma).to(w.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PhaseQuantSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w_real, w_imag):
        phase = torch.angle(w_real + 1j * w_imag)

        real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
        real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
        imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
        imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

        mask_real = real_pos | real_neg
        mask_imag = imag_pos | imag_neg

        if w_real.ndim != 2:
            w_real_flat = w_real.reshape(-1, w_real.numel())
            w_imag_flat = w_imag.reshape_as(w_real_flat)
            real_pos_flat = real_pos.reshape_as(w_real_flat)
            real_neg_flat = real_neg.reshape_as(w_real_flat)
            imag_pos_flat = imag_pos.reshape_as(w_real_flat)
            imag_neg_flat = imag_neg.reshape_as(w_real_flat)
        else:
            w_real_flat = w_real
            w_imag_flat = w_imag
            real_pos_flat = real_pos
            real_neg_flat = real_neg
            imag_pos_flat = imag_pos
            imag_neg_flat = imag_neg

        mask_real_flat = real_pos_flat | real_neg_flat
        mask_imag_flat = imag_pos_flat | imag_neg_flat

        def _row_mean_amplitude(w_flat: torch.Tensor, mask_flat: torch.Tensor) -> torch.Tensor:
            if not mask_flat.any():
                return torch.zeros(w_flat.shape[0], 1, device=w_flat.device, dtype=w_flat.dtype)
            abs_w = w_flat.abs() * mask_flat
            counts = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)
            row_mean = abs_w.sum(dim=1, keepdim=True) / counts
            return row_mean

        s_re_row = _row_mean_amplitude(w_real_flat, mask_real_flat)
        s_im_row = _row_mean_amplitude(w_imag_flat, mask_imag_flat)

        s_re_row = _maybe_make_pot_scale(s_re_row).clamp(min=1e-6)
        s_im_row = _maybe_make_pot_scale(s_im_row).clamp(min=1e-6)

        qw_real = torch.zeros_like(w_real)
        qw_imag = torch.zeros_like(w_imag)

        qw_real[real_pos]  =  1.0
        qw_real[real_neg]  = -1.0
        qw_imag[imag_pos]  =  1.0
        qw_imag[imag_neg]  = -1.0

        if w_real.ndim == 2:
            s_re_broadcast = s_re_row
            s_im_broadcast = s_im_row
        else:
            s_re_broadcast = s_re_row.mean()
            s_im_broadcast = s_im_row.mean()

        return (qw_real * s_re_broadcast).to(w_real.dtype), (qw_imag * s_im_broadcast).to(w_imag.dtype)

    @staticmethod
    def backward(ctx, grad_w_real, grad_w_imag):
        return grad_w_real, grad_w_imag


class PhaseQuantSTE_V2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real = w_real - qw_real_o1
        error_imag = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real, error_imag)
        return qw_real_o1 + qw_real_o2, qw_imag_o1 + qw_imag_o2

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag


class PhaseQuantSTE_V3(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        er1, ei1 = w_real - qw_real_o1, w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(er1, ei1)
        er2, ei2 = er1 - qw_real_o2, ei1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(er2, ei2)
        return qw_real_o1 + qw_real_o2 + qw_real_o3, qw_imag_o1 + qw_imag_o2 + qw_imag_o3

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag


class PhaseQuantSTE_V4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        er1, ei1 = w_real - qw_real_o1, w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(er1, ei1)
        er2, ei2 = er1 - qw_real_o2, ei1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(er2, ei2)
        er3, ei3 = er2 - qw_real_o3, ei2 - qw_imag_o3
        qw_real_o4, qw_imag_o4 = PhaseQuantSTE.apply(er3, ei3)
        return (
            qw_real_o1 + qw_real_o2 + qw_real_o3 + qw_real_o4,
            qw_imag_o1 + qw_imag_o2 + qw_imag_o3 + qw_imag_o4,
        )

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag