import os
import inspect
import argparse

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from config import vit_config_by_name
from model import ViTForClassification, SwinForClassification, SWIN_MODEL_NAMES
from data import (
    prepare_data,
    prepare_imagenet_data,
    prepare_imagenet100_data,
    get_num_classes,
    normalize_dataset_name,
)
from qat_modules import replace_modules_for_qat, convert_to_inference_mode, METHOD_MAP
from quantization import set_pow2_scales


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    init_kw = {"backend": "nccl", "init_method": "env://"}
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        init_kw["device_id"] = device
    dist.init_process_group(**init_kw)
    return local_rank, dist.get_world_size()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def _amp_autocast(enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _reduce_tensor(t, device):
    t = torch.tensor(t, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


@torch.no_grad()
def evaluate_model(model, loader, loss_fn, device, amp_enabled=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _amp_autocast(amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        if logits.size(1) >= 5:
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += (top5_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
        else:
            correct_top5 += correct
        total += images.size(0)

    if dist.is_initialized():
        loss_sum = _reduce_tensor(running_loss, device)
        correct_sum = _reduce_tensor(correct, device)
        correct5_sum = _reduce_tensor(correct_top5, device)
        total_sum = _reduce_tensor(total, device)
        avg_loss = (loss_sum / total_sum).item()
        top1 = (correct_sum / total_sum).item()
        top5 = (correct5_sum / total_sum).item()
    else:
        avg_loss = running_loss / max(total, 1)
        top1 = correct / max(total, 1)
        top5 = correct_top5 / max(total, 1)

    return avg_loss, top1, top5


def print_model_stats(model, quant_method=None):
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    num_linear = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear_params += m.weight.numel()
            num_linear += 1

    print(f"\n  Model statistics:")
    print(f"    Total parameters:  {total_params:>12,}")
    print(f"    Linear weight params: {linear_params:>12,}  ({num_linear} layers)")
    print(f"    Non-linear params: {total_params - linear_params:>12,}  (full-precision)")

    if quant_method and 'complex_phase' in quant_method:
        version = quant_method.split('_v')[-1]
        bits_per_param = int(version)
        quantized_bits = linear_params * bits_per_param
        full_bits = (total_params - linear_params) * 32
        total_bits = quantized_bits + full_bits
        effective_bpp = total_bits / total_params
        print(f"    Effective bits/param (approx): {effective_bpp:.2f}")
        print(f"    Quantized weight bits: {bits_per_param} bit/real-param")
    elif quant_method == 'bitnet':
        quantized_bits = linear_params * 1
        full_bits = (total_params - linear_params) * 32
        total_bits = quantized_bits + full_bits
        effective_bpp = total_bits / total_params
        print(f"    Effective bits/param (approx): {effective_bpp:.2f}")
        print(f"    Quantized weight bits: 1 bit/param")


def parse_args():
    _model_choices = [
        "vit_b16", "vit_b32", "vit_l16", "vit_h14", "vit_l32",
        "swin_t", "swin_s", "swin_b",
    ]
    p = argparse.ArgumentParser(description="Evaluate PHANTOM ViT/Swin checkpoint")

    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint .pt file")
    p.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "mnist", "imagenet", "imagenet100", "imagenet-100"])
    p.add_argument(
        "--model", type=str, default="vit_b16",
        choices=_model_choices,
        help="Architecture used for training (must match checkpoint).",
    )
    p.add_argument("--quant-method", type=str, default=None,
                    choices=[None] + list(METHOD_MAP.keys()),
                    help="If set, replace linear layers with QAT modules before loading weights")
    p.add_argument("--rkhs-sigma", type=float, default=1.0,
                    help="RKHS sigma (must match training if using rkhs_* methods).")
    p.add_argument("--rkhs-seed", type=int, default=42,
                    help="RKHS random seed (must match training if using rkhs_* methods).")
    p.add_argument("--inference-mode", action="store_true", default=False,
                    help="Convert QAT modules to inference-optimized")
    p.add_argument("--skip-classifier", action="store_true", default=False)

    p.add_argument(
        "--pow2-scales",
        action="store_true",
        help="If set, project Hadamard row scales and phase amplitude scales onto "
             "exact powers of two (must match training when evaluating Po2 checkpoints).",
    )

    p.add_argument(
        "--encoder-activation-nbits",
        type=int,
        default=32,
        help="Must match training: encoder activation bits (32 = disabled).",
    )
    p.add_argument(
        "--encoder-activation-mode",
        type=str,
        default="global_pact",
        choices=[
            "global_pact",
            "channel_mean",
            "channel_percentile",
            "channel_minmax",
            "channel_pact",
        ],
        help="Must match training: activation quant mode for encoder linear outputs.",
    )
    p.add_argument(
        "--act-scale-percentile",
        type=float,
        default=0.95,
        help="Must match training when mode=channel_percentile (0..1).",
    )
    p.add_argument(
        "--act-alpha-init",
        type=float,
        default=6.0,
        help="Unused for evaluation unless activation parameters are missing; kept for config symmetry.",
    )

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--imagenet100-val-split", type=float, default=0.1,
                    help="Validation split ratio if imagenet100 val/ folder is absent")
    p.add_argument("--imagenet100-split-seed", type=int, default=42,
                    help="Seed used for deterministic imagenet100 train/val split")
    p.add_argument("--amp", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    set_pow2_scales(args.pow2_scales)

    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    num_classes = get_num_classes(args.dataset)

    if args.dataset == "imagenet":
        _, testloader, _, num_classes = prepare_imagenet_data(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
        )
    elif args.dataset == "imagenet100":
        _, testloader, _, num_classes = prepare_imagenet100_data(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=True,
            val_split=args.imagenet100_val_split,
            seed=args.imagenet100_split_seed,
        )
    else:
        _, testloader, _, num_classes = prepare_data(
            args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_root=args.data_root,
            distributed=True,
        )

    enc_kw = {
        "encoder_activation_nbits": args.encoder_activation_nbits,
        "encoder_activation_mode": args.encoder_activation_mode,
        "act_scale_percentile": args.act_scale_percentile,
        "act_alpha_init": args.act_alpha_init,
    }

    if args.model in SWIN_MODEL_NAMES:
        model = SwinForClassification(
            args.model,
            num_classes=num_classes,
            encoder_config=enc_kw,
        )
    else:
        vit_cfg = vit_config_by_name(args.model)
        eval_cfg = {**vit_cfg, **enc_kw}
        model = ViTForClassification(eval_cfg, num_classes=num_classes)

    if args.quant_method:
        if is_main_process():
            print(f"Applying QAT module replacement: {args.quant_method}")
        replace_modules_for_qat(
            model,
            method=args.quant_method,
            skip_head=args.skip_classifier,
            sigma=args.rkhs_sigma,
            seed=args.rkhs_seed,
        )
    if isinstance(model, SwinForClassification):
        model.attach_encoder_activation_hooks()

    if is_main_process():
        print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)

    if args.inference_mode and args.quant_method:
        if is_main_process():
            print("Converting to inference mode...")
        model = convert_to_inference_mode(model)

    model_ddp = DDP(model, device_ids=[local_rank])

    if is_main_process():
        print_model_stats(model, args.quant_method)

    loss_fn = nn.CrossEntropyLoss()
    avg_loss, top1, top5 = evaluate_model(
        model_ddp, testloader, loss_fn, device, args.amp,
    )

    if is_main_process():
        print(f"\n  Results on {args.dataset} test set:")
        print(f"    Top-1 accuracy: {top1:.4f}  ({top1 * 100:.2f}%)")
        print(f"    Top-5 accuracy: {top5:.4f}  ({top5 * 100:.2f}%)")
        print(f"    Test loss:      {avg_loss:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
