import ssl
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
    vit_b_32,
    ViT_B_32_Weights,
    vit_l_16,
    ViT_L_16_Weights,
    vit_l_32,
    ViT_L_32_Weights,
    vit_h_14,
    ViT_H_14_Weights,
    swin_t,
    Swin_T_Weights,
    swin_s,
    Swin_S_Weights,
    swin_b,
    Swin_B_Weights,
)

from act_quant import ActQuantConfig, EncoderActivationQuantizer

ssl._create_default_https_context = ssl._create_unverified_context


def _make_encoder_act(config: dict, channels: int) -> nn.Module:
    nbits = int(config.get("encoder_activation_nbits", 32))
    if nbits >= 32:
        return nn.Identity()

    cfg = ActQuantConfig(
        nbits=nbits,
        mode=str(config.get("encoder_activation_mode", "global_pact")),
        percentile=float(config.get("act_scale_percentile", 0.95)),
        alpha_init=float(config.get("act_alpha_init", 6.0)),
    )
    return EncoderActivationQuantizer(cfg=cfg, channels=channels)


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_proj = nn.Conv2d(
            config["num_channels"],
            config["hidden_size"],
            kernel_size=config["patch_size"],
            stride=config["patch_size"],
        )
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2

    def forward(self, x):
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config["num_attention_heads"]
        self.hidden_size = config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])

        self.act_q = _make_encoder_act(config, channels=self.hidden_size)
        self.act_k = _make_encoder_act(config, channels=self.hidden_size)
        self.act_v = _make_encoder_act(config, channels=self.hidden_size)
        self.act_out = _make_encoder_act(config, channels=self.hidden_size)

    def forward(self, x):
        B, N, C = x.shape

        q = self.act_q(self.q_proj(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.act_k(self.k_proj(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.act_v(self.v_proj(x)).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, C)
        attn_output = self.act_out(self.out_proj(attn_output))
        return attn_output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(config["hidden_dropout_prob"])
        self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout2 = nn.Dropout(config["hidden_dropout_prob"])

        self.act_fc1 = _make_encoder_act(config, channels=config["intermediate_size"])
        self.act_fc2 = _make_encoder_act(config, channels=config["hidden_size"])

    def forward(self, x):
        x = self.act_fc1(self.fc1(x))
        x = self.act(x)
        x = self.dropout1(x)
        x = self.act_fc2(self.fc2(x))
        x = self.dropout2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.self_attention = MultiHeadSelfAttention(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.ln_2 = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.dropout(self.self_attention(self.ln_1(x)))
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(1, (config["image_size"] // config["patch_size"]) ** 2 + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.layers = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config["num_hidden_layers"])]
        )
        self.ln = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

    def forward(self, x, return_features=False):
        x = x + self.pos_embedding
        x = self.dropout(x)
        features = []
        for block in self.layers:
            x = block(x)
            if return_features:
                features.append(x)
        x = self.ln(x)
        if return_features:
            return x, features
        return x


class ViTForClassification(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)
        self.class_token = nn.Parameter(torch.empty(1, 1, config["hidden_size"]))
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(config["hidden_size"], num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.encoder.pos_embedding, std=0.02)

    def forward(self, x, return_features=False):
        B = x.shape[0]
        x = self.patch_embedding(x)
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        if return_features:
            x, features = self.encoder(x, return_features=True)
        else:
            x = self.encoder(x)
            features = None
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        if return_features:
            return logits, features
        return logits

    def get_layer_groups(self):
        groups = []
        groups.append(list(self.classifier.parameters()))
        for i in reversed(range(self.config["num_hidden_layers"])):
            groups.append(list(self.encoder.layers[i].parameters()))
        embedding_params = (
            list(self.patch_embedding.parameters())
            + [self.class_token, self.encoder.pos_embedding]
            + list(self.encoder.ln.parameters())
        )
        groups.append(embedding_params)
        return groups


SWIN_MODEL_NAMES = frozenset({"swin_t", "swin_s", "swin_b"})

_SWIN_REGISTRY = {
    "swin_t": (swin_t, Swin_T_Weights.IMAGENET1K_V1),
    "swin_s": (swin_s, Swin_S_Weights.IMAGENET1K_V1),
    "swin_b": (swin_b, Swin_B_Weights.IMAGENET1K_V1),
}


def _activation_quant_on_output(quant: nn.Module, output: torch.Tensor) -> torch.Tensor:
    if isinstance(quant, nn.Identity):
        return output
    orig_shape = output.shape
    if output.dim() == 4:
        b, h, w, c = output.shape
        x = output.reshape(b, h * w, c)
        y = quant(x)
        return y.reshape(orig_shape)
    if output.dim() == 3:
        return quant(output)
    if output.dim() == 2:
        return quant(output.unsqueeze(1)).squeeze(1)
    return quant(output.reshape(output.shape[0], -1, output.shape[-1])).reshape(orig_shape)


class SwinForClassification(nn.Module):
    def __init__(
        self,
        model_name: str = "swin_t",
        num_classes: int = 100,
        encoder_config: Optional[Dict] = None,
    ):
        super().__init__()
        if model_name not in _SWIN_REGISTRY:
            raise ValueError(
                f"Unknown Swin model: {model_name}. Choose from {sorted(_SWIN_REGISTRY)}"
            )

        self.model_name = model_name
        self._encoder_act_cfg: Dict = dict(encoder_config or {})

        constructor, weights = _SWIN_REGISTRY[model_name]
        self.swin = constructor(weights=weights)

        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.swin.head.weight, std=0.02)
        nn.init.zeros_(self.swin.head.bias)

        self._features: List = []
        self._hooks: List = []
        self.encoder_act_quants = nn.ModuleDict()
        self._encoder_act_handles: List = []

        self._create_encoder_activation_modules()

    def _clear_encoder_activation_hooks(self) -> None:
        for h in self._encoder_act_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._encoder_act_handles.clear()

    def _create_encoder_activation_modules(self) -> None:
        self.encoder_act_quants = nn.ModuleDict()
        nbits = int(self._encoder_act_cfg.get("encoder_activation_nbits", 32))
        if nbits >= 32:
            return

        act_cfg = {**self._encoder_act_cfg}
        block_idx = 0

        for layer in self.swin.features:
            if not isinstance(layer, nn.Sequential):
                continue
            for block in layer:
                attn = getattr(block, "attn", None)
                mlp = getattr(block, "mlp", None)
                if attn is None or mlp is None:
                    continue
                if not all(hasattr(attn, a) for a in ("qkv", "proj")) or not all(
                    hasattr(mlp, a) for a in ("fc1", "fc2")
                ):
                    continue

                c_qkv = attn.qkv.out_features
                c_proj = attn.proj.out_features
                c_fc1 = mlp.fc1.out_features
                c_fc2 = mlp.fc2.out_features
                b = block_idx

                for suffix, channels in (
                    ("qkv", c_qkv),
                    ("proj", c_proj),
                    ("mlp_fc1", c_fc1),
                    ("mlp_fc2", c_fc2),
                ):
                    key = f"blk{b}_{suffix}"
                    self.encoder_act_quants[key] = _make_encoder_act(act_cfg, channels=channels)

                block_idx += 1

    def attach_encoder_activation_hooks(self) -> None:
        self._clear_encoder_activation_hooks()
        nbits = int(self._encoder_act_cfg.get("encoder_activation_nbits", 32))
        if nbits >= 32 or not self.encoder_act_quants:
            return

        block_idx = 0

        for layer in self.swin.features:
            if not isinstance(layer, nn.Sequential):
                continue
            for block in layer:
                attn = getattr(block, "attn", None)
                mlp = getattr(block, "mlp", None)
                if attn is None or mlp is None:
                    continue
                if not all(hasattr(attn, a) for a in ("qkv", "proj")) or not all(
                    hasattr(mlp, a) for a in ("fc1", "fc2")
                ):
                    continue

                b = block_idx

                def make_hook(key: str):
                    def hook(module, inp, out):
                        qmod = self.encoder_act_quants[key]
                        return _activation_quant_on_output(qmod, out)

                    return hook

                for suffix, mod in (
                    ("qkv", attn.qkv),
                    ("proj", attn.proj),
                    ("mlp_fc1", mlp.fc1),
                    ("mlp_fc2", mlp.fc2),
                ):
                    key = f"blk{b}_{suffix}"
                    if key not in self.encoder_act_quants:
                        raise KeyError(f"encoder_act_quants missing {key}")
                    self._encoder_act_handles.append(
                        mod.register_forward_hook(make_hook(key))
                    )

                block_idx += 1

    def _register_feature_hooks(self) -> None:
        self._remove_hooks()
        self._features = []
        stage_indices = [1, 3, 5, 7]
        for idx in stage_indices:
            if idx < len(self.swin.features):
                hook = self.swin.features[idx].register_forward_hook(self._make_hook())
                self._hooks.append(hook)

    def _make_hook(self):
        def hook_fn(module, input, output):
            if output.dim() == 4:
                b, h, w, c = output.shape
                self._features.append(output.reshape(b, h * w, c))
            elif output.dim() == 3:
                self._features.append(output)
            else:
                self._features.append(output.flatten(1, -2))

        return hook_fn

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._features = []

    def forward(self, x, return_features: bool = False):
        if return_features:
            self._register_feature_hooks()

        logits = self.swin(x)

        if return_features:
            features = list(self._features)
            self._remove_hooks()
            return logits, features

        return logits

    def get_layer_groups(self):
        groups = []
        groups.append(list(self.swin.head.parameters()))
        stage_indices = [7, 5, 3, 1]
        merging_indices = [6, 4, 2, 0]
        for s_idx, m_idx in zip(stage_indices, merging_indices):
            params = []
            if s_idx < len(self.swin.features):
                params.extend(list(self.swin.features[s_idx].parameters()))
            if m_idx < len(self.swin.features):
                params.extend(list(self.swin.features[m_idx].parameters()))
            if params:
                groups.append(params)
        norm_params = list(self.swin.norm.parameters())
        if norm_params:
            groups.append(norm_params)
        return groups

    def extra_repr(self) -> str:
        n = self._encoder_act_cfg.get("encoder_activation_nbits", 32)
        return f"model_name={self.model_name}, encoder_activation_nbits={n}"


def load_pretrained_weights(model):
    if model.config["model_name"] == "vit_b16":
        pretrained = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).state_dict()
    elif model.config["model_name"] == "vit_b32":
        pretrained = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1).state_dict()
    elif model.config["model_name"] == "vit_l16":
        pretrained = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).state_dict()
    elif model.config["model_name"] == "vit_h14":
        pretrained = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1).state_dict()
    elif model.config["model_name"] == "vit_l32":
        pretrained = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1).state_dict()
    else:
        raise ValueError(f"Unknown model: {model.config['model_name']}")

    new_state = {}

    new_state["patch_embedding.conv_proj.weight"] = pretrained["conv_proj.weight"]
    new_state["patch_embedding.conv_proj.bias"] = pretrained["conv_proj.bias"]
    new_state["class_token"] = pretrained["class_token"]
    new_state["encoder.pos_embedding"] = pretrained["encoder.pos_embedding"]

    for i in range(model.config["num_hidden_layers"]):
        src = f"encoder.layers.encoder_layer_{i}"
        dst = f"encoder.layers.{i}"

        new_state[f"{dst}.ln_1.weight"] = pretrained[f"{src}.ln_1.weight"]
        new_state[f"{dst}.ln_1.bias"] = pretrained[f"{src}.ln_1.bias"]
        new_state[f"{dst}.ln_2.weight"] = pretrained[f"{src}.ln_2.weight"]
        new_state[f"{dst}.ln_2.bias"] = pretrained[f"{src}.ln_2.bias"]

        in_proj_weight = pretrained[f"{src}.self_attention.in_proj_weight"]
        in_proj_bias = pretrained[f"{src}.self_attention.in_proj_bias"]
        q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)

        new_state[f"{dst}.self_attention.q_proj.weight"] = q_w
        new_state[f"{dst}.self_attention.q_proj.bias"] = q_b
        new_state[f"{dst}.self_attention.k_proj.weight"] = k_w
        new_state[f"{dst}.self_attention.k_proj.bias"] = k_b
        new_state[f"{dst}.self_attention.v_proj.weight"] = v_w
        new_state[f"{dst}.self_attention.v_proj.bias"] = v_b

        new_state[f"{dst}.self_attention.out_proj.weight"] = pretrained[f"{src}.self_attention.out_proj.weight"]
        new_state[f"{dst}.self_attention.out_proj.bias"] = pretrained[f"{src}.self_attention.out_proj.bias"]

        new_state[f"{dst}.mlp.fc1.weight"] = pretrained[f"{src}.mlp.0.weight"]
        new_state[f"{dst}.mlp.fc1.bias"] = pretrained[f"{src}.mlp.0.bias"]
        new_state[f"{dst}.mlp.fc2.weight"] = pretrained[f"{src}.mlp.3.weight"]
        new_state[f"{dst}.mlp.fc2.bias"] = pretrained[f"{src}.mlp.3.bias"]

    new_state["encoder.ln.weight"] = pretrained["encoder.ln.weight"]
    new_state["encoder.ln.bias"] = pretrained["encoder.ln.bias"]

    pretrained_head_w = pretrained.get("heads.head.weight")
    if pretrained_head_w is not None and pretrained_head_w.shape[0] == model.classifier.out_features:
        new_state["classifier.weight"] = pretrained_head_w
        new_state["classifier.bias"] = pretrained["heads.head.bias"]

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Pretrained weights loaded. Missing keys (expected for different num_classes): {missing}")
    else:
        print("Pretrained weights loaded. All keys matched.")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    return model


if __name__ == "__main__":
    from config import vit_b16_config

    model = ViTForClassification(vit_b16_config, num_classes=10)
    model = load_pretrained_weights(model)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print("Logits shape:", logits.shape)
    print("Layer groups:", len(model.get_layer_groups()))
