import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from quantization import (
    BitNetQuantSTE,
    PhaseQuantSTE, PhaseQuantSTE_V2, PhaseQuantSTE_V3, PhaseQuantSTE_V4,

    make_rkhs_projections, rkhs_embed, rkhs_reconstruct,
    RKHSPhaseQuantSTE,
    RKHSPhaseQuantSTE_V2,
    RKHSPhaseQuantSTE_V3,
    RKHSPhaseQuantSTE_V4,

    make_hadamard_projections, hadamard_embed, hadamard_reconstruct,
    HadamardPhaseQuantSTE,
    HadamardPhaseQuantSTE_V2,
    HadamardPhaseQuantSTE_V3,
    HadamardPhaseQuantSTE_V4,
)


class QATLinearBitNet(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        quantized_weight = BitNetQuantSTE.apply(self.weight)
        return F.linear(x, quantized_weight, self.bias)


class QATLinearRKHSPhaseV1(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._rkhs_seed = seed
        Omega = make_rkhs_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("omega", Omega)

    def forward(self, x):
        omega = self.omega.to(self.weight.dtype)
        A_q = RKHSPhaseQuantSTE.apply(self.weight, omega)
        return F.linear(x, A_q, self.bias)


class QATLinearRKHSPhaseV2(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._rkhs_seed = seed
        Omega = make_rkhs_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("omega", Omega)

    def forward(self, x):
        omega = self.omega.to(self.weight.dtype)
        A_q = RKHSPhaseQuantSTE_V2.apply(self.weight, omega)
        return F.linear(x, A_q, self.bias)


class QATLinearRKHSPhaseV3(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._rkhs_seed = seed
        Omega = make_rkhs_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("omega", Omega)

    def forward(self, x):
        omega = self.omega.to(self.weight.dtype)
        A_q = RKHSPhaseQuantSTE_V3.apply(self.weight, omega)
        return F.linear(x, A_q, self.bias)


class QATLinearRKHSPhaseV4(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._rkhs_seed = seed
        Omega = make_rkhs_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("omega", Omega)

    def forward(self, x):
        omega = self.omega.to(self.weight.dtype)
        A_q = RKHSPhaseQuantSTE_V4.apply(self.weight, omega)
        return F.linear(x, A_q, self.bias)


class QATLinearHadamardPhaseV1(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._had_seed = seed
        d = make_hadamard_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("d", d)

    def forward(self, x):
        d = self.d.to(self.weight.dtype)
        A_q = HadamardPhaseQuantSTE.apply(self.weight, d)
        return F.linear(x, A_q, self.bias)


class QATLinearHadamardPhaseV2(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._had_seed = seed
        d = make_hadamard_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("d", d)

    def forward(self, x):
        d = self.d.to(self.weight.dtype)
        A_q = HadamardPhaseQuantSTE_V2.apply(self.weight, d)
        return F.linear(x, A_q, self.bias)


class QATLinearHadamardPhaseV3(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._had_seed = seed
        d = make_hadamard_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("d", d)

    def forward(self, x):
        d = self.d.to(self.weight.dtype)
        A_q = HadamardPhaseQuantSTE_V3.apply(self.weight, d)
        return F.linear(x, A_q, self.bias)


class QATLinearHadamardPhaseV4(nn.Linear):

    def __init__(self, *args, seed: int = 42, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._had_seed = seed
        d = make_hadamard_projections(
            self.out_features, self.in_features,
            seed=seed, device=self.weight.device,
        )
        self.register_buffer("d", d)

    def forward(self, x):
        d = self.d.to(self.weight.dtype)
        A_q = HadamardPhaseQuantSTE_V4.apply(self.weight, d)
        return F.linear(x, A_q, self.bias)


class InferenceOptimizedBitNet(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_quantized = False

    def _ensure_quantized(self):
        if not self._is_quantized:
            with torch.no_grad():
                w = self.weight
                scale = w.abs().mean()
                alpha = w.mean()
                centered_w = w - alpha
                binarized_w = torch.where(centered_w > 0, 1.0, -1.0).to(w.dtype)
                self.weight.data = binarized_w * scale
                self._is_quantized = True

    def forward(self, x):
        self._ensure_quantized()
        return F.linear(x, self.weight, self.bias)


class InferenceOptimizedComplexPhase(nn.Linear):

    def __init__(self, version="v1", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("PHANTOM requires even in/out features.")
        self._is_quantized = False
        self._version = version.lower()
        if self._version not in ["v1", "v2", "v3", "v4"]:
            raise ValueError(f"Unsupported version: {version}")

    def _ensure_quantized(self):
        if not self._is_quantized:
            with torch.no_grad():
                A = self.weight
                n, m = A.shape[0] // 2, A.shape[1] // 2
                A11, A12 = A[:n, :m], A[:n, m:]
                A21, A22 = A[n:, :m], A[n:, m:]

                U_re = 0.5 * (A11 + A22)
                U_im = 0.5 * (A21 - A12)
                W_re = 0.5 * (A11 - A22)
                W_im = 0.5 * (A12 + A21)

                quant_fn = {
                    "v1": self._phase_quant_v1,
                    "v2": self._phase_quant_v2,
                    "v3": self._phase_quant_v3,
                    "v4": self._phase_quant_v4,
                }[self._version]

                U_re_q, U_im_q = quant_fn(U_re, U_im)
                W_re_q, W_im_q = quant_fn(W_re, W_im)

                A11_q = W_re_q + U_re_q
                A12_q = W_im_q - U_im_q
                A21_q = W_im_q + U_im_q
                A22_q = -W_re_q + U_re_q

                self.weight.data = torch.cat(
                    [torch.cat([A11_q, A12_q], dim=1),
                     torch.cat([A21_q, A22_q], dim=1)], dim=0
                )
                self._is_quantized = True

    @staticmethod
    def _phase_quant_v1(w_real, w_imag):
        phase = torch.angle(w_real + 1j * w_imag)
        real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
        real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
        imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
        imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)
        mask_real = real_pos | real_neg
        mask_imag = imag_pos | imag_neg
        s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
        s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
        s_re = torch.clamp(s_re, min=1e-6)
        s_im = torch.clamp(s_im, min=1e-6)
        qw_real = torch.zeros_like(w_real)
        qw_imag = torch.zeros_like(w_imag)
        qw_real[real_pos] = 1.0
        qw_real[real_neg] = -1.0
        qw_imag[imag_pos] = 1.0
        qw_imag[imag_neg] = -1.0
        return qw_real * s_re, qw_imag * s_im

    @classmethod
    def _phase_quant_v2(cls, w_real, w_imag):
        qr1, qi1 = cls._phase_quant_v1(w_real, w_imag)
        qr2, qi2 = cls._phase_quant_v1(w_real - qr1, w_imag - qi1)
        return qr1 + qr2, qi1 + qi2

    @classmethod
    def _phase_quant_v3(cls, w_real, w_imag):
        qr1, qi1 = cls._phase_quant_v1(w_real, w_imag)
        er1, ei1 = w_real - qr1, w_imag - qi1
        qr2, qi2 = cls._phase_quant_v1(er1, ei1)
        er2, ei2 = er1 - qr2, ei1 - qi2
        qr3, qi3 = cls._phase_quant_v1(er2, ei2)
        return qr1 + qr2 + qr3, qi1 + qi2 + qi3

    @classmethod
    def _phase_quant_v4(cls, w_real, w_imag):
        qr1, qi1 = cls._phase_quant_v1(w_real, w_imag)
        er1, ei1 = w_real - qr1, w_imag - qi1
        qr2, qi2 = cls._phase_quant_v1(er1, ei1)
        er2, ei2 = er1 - qr2, ei1 - qi2
        qr3, qi3 = cls._phase_quant_v1(er2, ei2)
        er3, ei3 = er2 - qr3, ei2 - qi3
        qr4, qi4 = cls._phase_quant_v1(er3, ei3)
        return qr1 + qr2 + qr3 + qr4, qi1 + qi2 + qi3 + qi4

    def forward(self, x):
        self._ensure_quantized()
        return F.linear(x, self.weight, self.bias)


class InferenceOptimizedRKHSPhase(nn.Linear):

    def __init__(self, version: str = "v1", *args, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._is_quantized = False
        self._version = version.lower()
        if self._version not in ["v1", "v2", "v3", "v4"]:
            raise ValueError(f"Unsupported version: {version}")
        self.register_buffer("omega", None)

    def _ensure_quantized(self):
        if not self._is_quantized:
            if self.omega is None:
                raise RuntimeError(
                    "omega buffer not set. "
                    "Use convert_to_inference_mode() to convert from a QAT layer."
                )
            with torch.no_grad():
                A = self.weight
                omega = self.omega.to(A.dtype)
                Z_re, Z_im, s = rkhs_embed(A, omega)
                quant_fns = {
                    "v1": PhaseQuantSTE,
                    "v2": PhaseQuantSTE_V2,
                    "v3": PhaseQuantSTE_V3,
                    "v4": PhaseQuantSTE_V4,
                }
                Z_re_q, Z_im_q = quant_fns[self._version].apply(Z_re, Z_im)
                A_q = rkhs_reconstruct(Z_re_q, Z_im_q, omega, s)
                self.weight.data = A_q.to(A.dtype)
                self._is_quantized = True

    def forward(self, x):
        self._ensure_quantized()
        return F.linear(x, self.weight, self.bias)


class InferenceOptimizedHadamardPhase(nn.Linear):

    def __init__(self, version: str = "v1", *args, **kwargs):
        kwargs.pop("sigma", None)
        super().__init__(*args, **kwargs)
        self._is_quantized = False
        self._version = version.lower()
        if self._version not in ["v1", "v2", "v3", "v4"]:
            raise ValueError(f"Unsupported version: {version}")
        
        self.register_buffer("d", None)

    def _ensure_quantized(self):
        if not self._is_quantized:
            if self.d is None:
                raise RuntimeError(
                    "d buffer not set. "
                    "Use convert_to_inference_mode() to convert from a QAT layer."
                )
            with torch.no_grad():
                A = self.weight
                N = A.shape[0]
                d = self.d.to(A.dtype)

                Z_re, Z_im, s = hadamard_embed(A, d)

                quant_fns = {
                    "v1": PhaseQuantSTE,
                    "v2": PhaseQuantSTE_V2,
                    "v3": PhaseQuantSTE_V3,
                    "v4": PhaseQuantSTE_V4,
                }
                Z_re_q, Z_im_q = quant_fns[self._version].apply(Z_re, Z_im)

                A_q = hadamard_reconstruct(Z_re_q, Z_im_q, d, s, original_size=N)
                self.weight.data = A_q.to(A.dtype)

                self.d = None
                self._is_quantized = True

    def forward(self, x):
        self._ensure_quantized()
        return F.linear(x, self.weight, self.bias)


METHOD_MAP = {
    "bitnet": QATLinearBitNet,
    "rkhs_phase_v1": QATLinearRKHSPhaseV1,
    "rkhs_phase_v2": QATLinearRKHSPhaseV2,
    "rkhs_phase_v3": QATLinearRKHSPhaseV3,
    "rkhs_phase_v4": QATLinearRKHSPhaseV4,
    "hadamard_phase_v1": QATLinearHadamardPhaseV1,
    "hadamard_phase_v2": QATLinearHadamardPhaseV2,
    "hadamard_phase_v3": QATLinearHadamardPhaseV3,
    "hadamard_phase_v4": QATLinearHadamardPhaseV4,
}

_HEAD_NAMES = {"lm_head", "classifier", "head", "fc"}

_RKHS_METHODS = {
    "rkhs_phase_v1", "rkhs_phase_v2", "rkhs_phase_v3", "rkhs_phase_v4",
}

_HADAMARD_METHODS = {
    "hadamard_phase_v1", "hadamard_phase_v2",
    "hadamard_phase_v3", "hadamard_phase_v4",
}


def replace_modules_for_qat(
    model: nn.Module,
    method: str,
    skip_head: bool = False,
    skip_names: list[str] | None = None,
    sigma: float = 1.0,
    seed: int = 42,
):
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method!r}. Available: {sorted(METHOD_MAP)}")

    TargetQATClass = METHOD_MAP[method]
    names_to_skip = set(skip_names or [])
    if skip_head:
        names_to_skip |= _HEAD_NAMES

    is_rkhs = method in _RKHS_METHODS
    is_hadamard = method in _HADAMARD_METHODS

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_modules_for_qat(
                module, method, skip_head, list(names_to_skip),
                sigma=sigma, seed=seed,
            )

        if isinstance(module, nn.Linear):
            if name in names_to_skip:
                print(f"  -> Skipping layer: {name} (in skip list)")
                continue

            if is_rkhs and module.out_features % 2 != 0:
                print(
                    f"  -> Skipping RKHS layer: {name} "
                    f"(odd out_features={module.out_features})"
                )
                continue

            print(f"  -> Replacing: {name} ({module.out_features}x{module.in_features}) "
                  f"→ {TargetQATClass.__name__}")

            if is_rkhs or is_hadamard:
                layer_seed = seed ^ (module.out_features * 104729 + module.in_features * 15487)
                new_module = TargetQATClass(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                    device=module.weight.device,
                    seed=layer_seed,
                )
            else:
                new_module = TargetQATClass(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    dtype=module.weight.dtype,
                    device=module.weight.device,
                )

            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)

            setattr(model, name, new_module)


def convert_to_inference_mode(model: nn.Module) -> nn.Module:
    converted_count = 0

    _rkhs_qat_map = {
        QATLinearRKHSPhaseV1: "v1",
        QATLinearRKHSPhaseV2: "v2",
        QATLinearRKHSPhaseV3: "v3",
        QATLinearRKHSPhaseV4: "v4",
    }
    _hadamard_qat_map = {
        QATLinearHadamardPhaseV1: "v1",
        QATLinearHadamardPhaseV2: "v2",
        QATLinearHadamardPhaseV3: "v3",
        QATLinearHadamardPhaseV4: "v4",
    }

    def _convert_module(module: nn.Module, name_path: str = ""):
        nonlocal converted_count

        for name, child in list(module.named_children()):
            full_name = f"{name_path}.{name}" if name_path else name

            if isinstance(child, QATLinearBitNet):
                new_module = InferenceOptimizedBitNet(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_module.bias.data.copy_(child.bias.data)
                setattr(module, name, new_module)
                converted_count += 1
                print(f"  -> BitNet          : {full_name}")

            elif type(child) in _rkhs_qat_map:
                version = _rkhs_qat_map[type(child)]
                new_module = InferenceOptimizedRKHSPhase(
                    version=version,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_module.bias.data.copy_(child.bias.data)
                new_module.omega = child.omega.clone()
                setattr(module, name, new_module)
                converted_count += 1
                print(f"  -> RKHSPhase{version.upper()}       : {full_name}")

            elif type(child) in _hadamard_qat_map:
                version = _hadamard_qat_map[type(child)]
                new_module = InferenceOptimizedHadamardPhase(
                    version=version,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device, dtype=child.weight.dtype,
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_module.bias.data.copy_(child.bias.data)
                new_module.d = child.d.clone()
                setattr(module, name, new_module)
                converted_count += 1
                print(f"  -> HadamardPhase{version.upper()}   : {full_name}")

            else:
                _convert_module(child, full_name)

    _convert_module(model)
    print(f"\nConverted {converted_count} QAT layers to inference-optimized.")
    return model