#configurations for different transformer models
vit_b16_config = {
    "model_name":"vit_b16",
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "layer_norm_eps": 1e-6,
}

vit_b32_config = {
    "model_name":"vit_b32",
    "image_size": 224,
    "patch_size": 32,
    "num_channels": 3,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "layer_norm_eps": 1e-6,
}

vit_l16_config = {
    "model_name":"vit_l16",
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "layer_norm_eps": 1e-6,
}

vit_l32_config = {
    "model_name":"vit_l32",
    "image_size": 224,
    "patch_size": 32,
    "num_channels": 3,
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "layer_norm_eps": 1e-6,
}
vit_h14_config = {
    "model_name":"vit_h14",
    "image_size": 224,
    "patch_size": 14,
    "num_channels": 3,
    "hidden_size": 1280,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "intermediate_size": 5120,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "layer_norm_eps": 1e-6,
}

swin_t_config = {
    "model_name": "swin_t",
    "image_size": 224,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "num_stages": 4,
}

swin_s_config = {
    "model_name": "swin_s",
    "image_size": 224,
    "embed_dim": 96,
    "depths": [2, 2, 18, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "num_stages": 4,
}

swin_b_config = {
    "model_name": "swin_b",
    "image_size": 224,
    "embed_dim": 128,
    "depths": [2, 2, 18, 2],
    "num_heads": [4, 8, 16, 32],
    "window_size": 7,
    "num_stages": 4,
}

MODEL_CONFIGS = {
    "swin_t": swin_t_config,
    "swin_s": swin_s_config,
    "swin_b": swin_b_config,
}


def vit_config_by_name(model_name: str) -> dict:
    """Return ViT config dict for a known ``model_name``."""
    if model_name == "vit_b16":
        return vit_b16_config
    if model_name == "vit_b32":
        return vit_b32_config
    if model_name == "vit_l16":
        return vit_l16_config
    if model_name == "vit_h14":
        return vit_h14_config
    if model_name == "vit_l32":
        return vit_l32_config
    raise ValueError(f"Unknown ViT model: {model_name}")
