import os
import copy
import inspect
import argparse
import time
import json

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import vit_config_by_name
from model import (
    ViTForClassification,
    SwinForClassification,
    load_pretrained_weights,
    SWIN_MODEL_NAMES,
)
from data import (
    prepare_data,
    prepare_imagenet_data,
    prepare_imagenet100_data,
    get_num_classes,
    normalize_dataset_name,
)
from qat_modules import replace_modules_for_qat, METHOD_MAP
from losses import PhasedKDLoss
from quantization import set_pow2_scales


def setup_ddp():
    """Initialise the NCCL process group and return (local_rank, world_size)."""
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


def barrier():
    if dist.is_initialized():
        b_kw = {}
        if (
            "device_ids" in inspect.signature(dist.barrier).parameters
            and "LOCAL_RANK" in os.environ
        ):
            b_kw["device_ids"] = [int(os.environ["LOCAL_RANK"])]
        dist.barrier(**b_kw)


def _reduce_tensor(t, device):
    """Sum a scalar tensor across all ranks."""
    t = torch.tensor(t, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def count_parameters(model):
    m = model.module if isinstance(model, DDP) else model
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def unwrap_model(model):
    """Return the base model, stripping DDP wrapper if present."""
    return model.module if isinstance(model, DDP) else model


def make_scheduler(optimizer, total_epochs, warmup_epochs, eta_min=1e-6):
    """CosineAnnealing with optional linear warmup."""
    if warmup_epochs > 0 and total_epochs > warmup_epochs:
        warmup = LinearLR(
            optimizer, start_factor=1e-2, total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min,
        )
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)


def _amp_autocast(enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _make_grad_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model, loader, optimizer, loss_fn, scaler, device,
    amp_enabled, grad_accum_steps, max_grad_norm,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _amp_autocast(amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        if step % grad_accum_steps == 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if total > 0 and step % grad_accum_steps != 0:
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if dist.is_initialized():
        loss_sum = _reduce_tensor(running_loss, device)
        correct_sum = _reduce_tensor(correct, device)
        total_sum = _reduce_tensor(total, device)
        avg_loss = (loss_sum / total_sum).item()
        accuracy = (correct_sum / total_sum).item()
    else:
        avg_loss = running_loss / max(total, 1)
        accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def train_one_epoch_kd(
    student, teacher, loader, optimizer, kd_loss_fn, scaler, device,
    amp_enabled, grad_accum_steps, max_grad_norm,
    current_epoch, total_epochs,
):
    student.train()
    teacher.eval()

    running_loss = 0.0
    running_ce = 0.0
    running_mmd = 0.0
    running_kl = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _amp_autocast(amp_enabled):
            s_logits, s_features = student(images, return_features=True)
            with torch.no_grad():
                t_logits, t_features = teacher(images, return_features=True)
            loss, loss_dict = kd_loss_fn(
                s_logits, t_logits, s_features, t_features,
                labels, current_epoch, total_epochs,
            )

        running_loss += loss.item() * images.size(0)
        running_ce += loss_dict["ce"] * images.size(0)
        running_mmd += loss_dict["mmd"] * images.size(0)
        running_kl += loss_dict["kl"] * images.size(0)
        correct += (s_logits.detach().argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()

        if step % grad_accum_steps == 0:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if total > 0 and step % grad_accum_steps != 0:
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if dist.is_initialized():
        loss_sum = _reduce_tensor(running_loss, device)
        ce_sum = _reduce_tensor(running_ce, device)
        mmd_sum = _reduce_tensor(running_mmd, device)
        kl_sum = _reduce_tensor(running_kl, device)
        correct_sum = _reduce_tensor(correct, device)
        total_sum = _reduce_tensor(total, device)
        n = total_sum.item()
        avg_loss = (loss_sum / total_sum).item()
        avg_ce = (ce_sum / total_sum).item()
        avg_mmd = (mmd_sum / total_sum).item()
        avg_kl = (kl_sum / total_sum).item()
        accuracy = (correct_sum / total_sum).item()
    else:
        n = max(total, 1)
        avg_loss = running_loss / n
        avg_ce = running_ce / n
        avg_mmd = running_mmd / n
        avg_kl = running_kl / n
        accuracy = correct / n

    loss_details = {
        "total": avg_loss, "ce": avg_ce, "mmd": avg_mmd, "kl": avg_kl,
        "alpha_ce": loss_dict["alpha_ce"],
        "alpha_mmd": loss_dict["alpha_mmd"],
        "alpha_kl": loss_dict["alpha_kl"],
    }
    return avg_loss, accuracy, loss_details


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_enabled=False):
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


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_model = unwrap_model(model)
    torch.save({
        "epoch": epoch,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }, path)
    print(f"  Checkpoint saved: {path}")


def run_phase(
    model, trainloader, testloader, optimizer, scheduler, loss_fn,
    scaler, device, num_epochs, phase_name, save_dir,
    amp_enabled, grad_accum_steps, max_grad_norm,
    start_epoch=0,
):
    history = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if hasattr(trainloader, "sampler") and hasattr(trainloader.sampler, "set_epoch"):
            trainloader.sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, trainloader, optimizer, loss_fn, scaler, device,
            amp_enabled, grad_accum_steps, max_grad_norm,
        )
        val_loss, val_top1, val_top5 = evaluate(
            model, testloader, loss_fn, device, amp_enabled,
        )
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        total_params, trainable_params = count_parameters(model)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        history.append(metrics)

        if is_main_process():
            print(
                f"[{phase_name}] Epoch {epoch + 1:3d} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Top-1: {val_top1:.4f}  Top-5: {val_top5:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Trainable: {trainable_params / total_params * 100:.1f}% | "
                f"Time: {elapsed:.1f}s"
            )

            if save_dir and (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(save_dir, f"{phase_name}_epoch{epoch + 1}.pt")
                save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, ckpt_path)

    if is_main_process() and save_dir:
        final_path = os.path.join(save_dir, f"{phase_name}_final.pt")
        save_checkpoint(model, optimizer, scheduler, start_epoch + num_epochs, metrics, final_path)
        hist_path = os.path.join(save_dir, f"{phase_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  History saved: {hist_path}")

    return metrics


def run_phase_kd(
    student, teacher, trainloader, testloader, optimizer, scheduler,
    kd_loss_fn, ce_loss_fn, scaler, device, num_epochs, phase_name,
    save_dir, amp_enabled, grad_accum_steps, max_grad_norm,
    start_epoch=0,
):
    history = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        if hasattr(trainloader, "sampler") and hasattr(trainloader.sampler, "set_epoch"):
            trainloader.sampler.set_epoch(epoch)

        kd_epoch = epoch - start_epoch

        t0 = time.time()
        train_loss, train_acc, loss_details = train_one_epoch_kd(
            student, teacher, trainloader, optimizer, kd_loss_fn, scaler,
            device, amp_enabled, grad_accum_steps, max_grad_norm,
            current_epoch=kd_epoch, total_epochs=num_epochs,
        )
        val_loss, val_top1, val_top5 = evaluate(
            student, testloader, ce_loss_fn, device, amp_enabled,
        )
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        total_params, trainable_params = count_parameters(student)

        metrics = {
            "epoch": epoch + 1,
            "kd_epoch": kd_epoch + 1,
            "train_loss": train_loss,
            "train_loss_ce": loss_details["ce"],
            "train_loss_mmd": loss_details["mmd"],
            "train_loss_kl": loss_details["kl"],
            "alpha_ce": loss_details["alpha_ce"],
            "alpha_mmd": loss_details["alpha_mmd"],
            "alpha_kl": loss_details["alpha_kl"],
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        history.append(metrics)

        if is_main_process():
            print(
                f"[{phase_name}] Epoch {epoch + 1:3d} (KD {kd_epoch + 1}/{num_epochs}) | "
                f"Loss: {train_loss:.4f} "
                f"(CE={loss_details['ce']:.4f} MMD={loss_details['mmd']:.4f} KL={loss_details['kl']:.4f}) | "
                f"w=[{loss_details['alpha_ce']:.2f},{loss_details['alpha_mmd']:.2f},{loss_details['alpha_kl']:.2f}] | "
                f"Acc: {train_acc:.4f} | "
                f"Val Top-1: {val_top1:.4f} Top-5: {val_top5:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            if save_dir and (epoch + 1) % 5 == 0:
                ckpt_path = os.path.join(save_dir, f"{phase_name}_epoch{epoch + 1}.pt")
                save_checkpoint(student, optimizer, scheduler, epoch + 1, metrics, ckpt_path)

    if is_main_process() and save_dir:
        final_path = os.path.join(save_dir, f"{phase_name}_final.pt")
        save_checkpoint(student, optimizer, scheduler, start_epoch + num_epochs, metrics, final_path)
        hist_path = os.path.join(save_dir, f"{phase_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  History saved: {hist_path}")

    return metrics


def _encoder_kwargs_from_args(args) -> dict:
    return {
        "encoder_activation_nbits": args.encoder_activation_nbits,
        "encoder_activation_mode": args.encoder_activation_mode,
        "act_scale_percentile": args.act_scale_percentile,
        "act_alpha_init": args.act_alpha_init,
    }


def parse_args():
    _model_choices = [
        "vit_b16", "vit_b32", "vit_l16", "vit_h14", "vit_l32",
        "swin_t", "swin_s", "swin_b",
    ]
    p = argparse.ArgumentParser(
        description="PHANTOM ViT/Swin: teacher training + QAT with Knowledge Distillation (DDP + AMP)",
    )

    p.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "mnist", "imagenet", "imagenet100", "imagenet-100"])
    p.add_argument("--quant-method", type=str, default="hadamard_phase_v2",
                    choices=list(METHOD_MAP.keys()),
                    help="QAT quantisation method for Phase 2")
    p.add_argument(
        "--model", type=str, default="vit_b16",
        choices=_model_choices,
        help="ViT variant or Swin-T/S/B",
    )

    p.add_argument("--rkhs-sigma", type=float, default=1.0,
                    help="RBF kernel bandwidth sigma for RKHS embedding. "
                         "Only used when --quant-method starts with 'rkhs_'.")
    p.add_argument("--rkhs-seed", type=int, default=42,
                    help="Random seed for generating the fixed RKHS projection "
                         "matrices. Only used when --quant-method starts with 'rkhs_'.")

    p.add_argument("--finetune-epochs", type=int, default=10,
                    help="Teacher fine-tuning epochs (Phase 1)")
    p.add_argument("--qat-epochs", type=int, default=30,
                    help="QAT + KD epochs (Phase 2)")

    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--qat-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=5,
                    help="Linear LR warmup epochs at start of each phase (0 = disable)")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--amp", dest="amp", action="store_true",
                    help="Enable mixed-precision training (recommended for ImageNet)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=False)

    p.add_argument("--grad-accum-steps", type=int, default=1,
                    help="Gradient accumulation steps (effective batch = batch_size * accum * gpus)")
    p.add_argument("--max-grad-norm", type=float, default=0.0,
                    help="Max gradient norm for clipping (0 = disabled)")

    p.add_argument("--skip-classifier", action="store_true", default=False)

    p.add_argument(
        "--pow2-scales",
        action="store_true",
        help="If set, supports internal scales "
             "to be projected onto exact powers of two."
    )

    p.add_argument(
        "--encoder-activation-nbits",
        type=int,
        default=32,
        help="Encoder activation quantization bit-width "
             "Use 32 to disable (default). Typical values: 4, 6, 8.",
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
        help="Activation quantization mode for encoder linear outputs.",
    )
    p.add_argument(
        "--act-scale-percentile",
        type=float,
        default=0.95,
        help="used when --encoder-activation-mode=channel_percentile.",
    )
    p.add_argument(
        "--act-alpha-init",
        type=float,
        default=6.0,
        help="Initial clip/scale value for PACT-based modes (global_pact/channel_pact).",
    )

    p.add_argument("--teacher-checkpoint", type=str, default=None,
                    help="Path to a pre-trained teacher checkpoint (skips Phase 1)")
    p.add_argument("--temperature", type=float, default=4.0,
                    help="Temperature for KL divergence softmax in KD")

    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--imagenet100-val-split", type=float, default=0.1,
                    help="Validation split ratio if imagenet100 val/ folder is absent")
    p.add_argument("--imagenet100-split-seed", type=int, default=42,
                    help="Seed used for deterministic imagenet100 train/val split")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    set_pow2_scales(args.pow2_scales)

    local_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    if is_main_process():
        print(
            f"Dataset: {args.dataset}  |  Model: {args.model}  |  "
            f"World size: {world_size}  |  AMP: {args.amp}  |  "
            f"Po2 scales: {args.pow2_scales}"
        )

    num_classes = get_num_classes(args.dataset)

    if args.dataset == "imagenet":
        trainloader, testloader, classes, num_classes = prepare_imagenet_data(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            world_size=world_size,
        )
    elif args.dataset == "imagenet100":
        trainloader, testloader, classes, num_classes = prepare_imagenet100_data(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=True,
            val_split=args.imagenet100_val_split,
            seed=args.imagenet100_split_seed,
        )
    else:
        trainloader, testloader, classes, num_classes = prepare_data(
            args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_root=args.data_root,
            distributed=True,
        )

    if is_main_process():
        print(f"  Classes: {num_classes}")

    vit_config = None
    teacher_encoder_cfg = {"encoder_activation_nbits": 32}

    if args.model in SWIN_MODEL_NAMES:
        if is_main_process():
            print(f"  Backbone: Swin ({args.model}, ImageNet-1K weights)")
        model = SwinForClassification(
            args.model,
            num_classes=num_classes,
            encoder_config=teacher_encoder_cfg,
        ).to(device)
    else:
        vit_config = vit_config_by_name(args.model)
        if is_main_process():
            print(f"  Backbone: ViT ({args.model})")
        model = ViTForClassification(vit_config, num_classes=num_classes)
        model = load_pretrained_weights(model)
        model = model.to(device)

    total_params, _ = count_parameters(model)
    if is_main_process():
        print(f"Model parameters: {total_params:,}")

    ce_loss_fn = nn.CrossEntropyLoss()
    scaler = _make_grad_scaler(args.amp)

    _rkhs_suffix = (
        f"_sigma{args.rkhs_sigma}" if args.quant_method.startswith("rkhs_") else ""
    )
    _act_suffix = (
        (
            f"_act{args.encoder_activation_nbits}b_{args.encoder_activation_mode}"
            + (f"_p{int(args.act_scale_percentile * 100)}" if args.encoder_activation_mode == "channel_percentile" else "")
        )
        if args.encoder_activation_nbits < 32
        else ""
    )
    _pow2_suffix = "_pow2" if args.pow2_scales else ""
    exp_name = (
        f"{args.dataset}_{args.model}_{args.quant_method}{_rkhs_suffix}{_act_suffix}"
        f"{_pow2_suffix}_KD_ft{args.finetune_epochs}_qat{args.qat_epochs}"
    )
    save_dir = os.path.join(args.save_dir, exp_name)
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    barrier()

    if args.teacher_checkpoint:
        if is_main_process():
            print(f"\nSkipping Phase 1 -- loading teacher: {args.teacher_checkpoint}")
        ckpt = torch.load(args.teacher_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        if is_main_process():
            print(f"\n{'='*60}")
            print(
                f"PHASE 1: Teacher training ({args.model}) on {args.dataset} "
                f"for {args.finetune_epochs} epochs (full precision)"
            )
            print(f"{'='*60}")

        model_ddp = DDP(model, device_ids=[local_rank])

        baseline_loss, baseline_top1, baseline_top5 = evaluate(
            model_ddp, testloader, ce_loss_fn, device, args.amp,
        )
        if is_main_process():
            print(f"  Pretrained baseline: "
                  f"Val Loss={baseline_loss:.4f}  "
                  f"Top-1={baseline_top1:.4f}  Top-5={baseline_top5:.4f}")

        optimizer_ft = optim.AdamW(
            model_ddp.parameters(), lr=args.finetune_lr,
            weight_decay=args.weight_decay,
        )
        scheduler_ft = make_scheduler(
            optimizer_ft, args.finetune_epochs, args.warmup_epochs, eta_min=1e-6,
        )

        run_phase(
            model_ddp, trainloader, testloader,
            optimizer_ft, scheduler_ft, ce_loss_fn, scaler, device,
            args.finetune_epochs, "phase1_teacher", save_dir,
            amp_enabled=args.amp,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
        )

        model = model_ddp.module
        del model_ddp

    teacher_tmp = DDP(model, device_ids=[local_rank])
    teacher_loss, teacher_top1, teacher_top5 = evaluate(
        teacher_tmp, testloader, ce_loss_fn, device, args.amp,
    )
    del teacher_tmp

    if is_main_process():
        teacher_path = os.path.join(save_dir, "teacher_final.pt")
        os.makedirs(os.path.dirname(teacher_path), exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, teacher_path)
        print(f"  Teacher saved: {teacher_path}")
        print(f"  Teacher accuracy: Top-1={teacher_top1:.4f}  Top-5={teacher_top5:.4f}")
    barrier()

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"PHASE 2: QAT + KD with {args.quant_method} "
              f"for {args.qat_epochs} epochs (T={args.temperature})")
        if args.encoder_activation_nbits < 32:
            print(
                f"  Encoder activation quantization: {args.encoder_activation_nbits} bits "
                f"(symmetric; backbone={args.model})"
            )
        print(f"{'='*60}")

    teacher = copy.deepcopy(model)
    teacher.eval()
    teacher.requires_grad_(False)
    teacher = teacher.to(device)

    enc_kw = _encoder_kwargs_from_args(args)
    if args.model in SWIN_MODEL_NAMES:
        student = SwinForClassification(
            args.model,
            num_classes=num_classes,
            encoder_config=enc_kw,
        ).to(device)
    else:
        student_cfg = {**vit_config, **enc_kw}
        student = ViTForClassification(student_cfg, num_classes=num_classes).to(device)

    load_sd = student.load_state_dict(model.state_dict(), strict=False)
    missing_keys = getattr(load_sd, "missing_keys", load_sd[0])
    unexpected_keys = getattr(load_sd, "unexpected_keys", load_sd[1])
    if is_main_process():
        print(f"  Replacing nn.Linear layers with {args.quant_method} QAT layers...")
        if missing_keys:
            print(f"    (Expected) missing keys after teacher load: {len(missing_keys)}")
        if unexpected_keys:
            print(f"    Unexpected keys: {unexpected_keys}")
    replace_modules_for_qat(
        student,
        method=args.quant_method,
        skip_head=args.skip_classifier,
        sigma=args.rkhs_sigma,
        seed=args.rkhs_seed,
    )
    if isinstance(student, SwinForClassification):
        student.attach_encoder_activation_hooks()
    barrier()

    student_ddp = DDP(student, device_ids=[local_rank])

    post_loss, post_top1, post_top5 = evaluate(
        student_ddp, testloader, ce_loss_fn, device, args.amp,
    )
    if is_main_process():
        print(f"  Post-conversion accuracy (before QAT+KD): "
              f"Top-1={post_top1:.4f}  Top-5={post_top5:.4f}")

    optimizer_qat = optim.AdamW(
        student_ddp.parameters(), lr=args.qat_lr,
        weight_decay=args.weight_decay,
    )
    scheduler_qat = make_scheduler(
        optimizer_qat, args.qat_epochs, args.warmup_epochs, eta_min=1e-7,
    )

    kd_loss_fn = PhasedKDLoss(
        temperature=args.temperature,
    )

    run_phase_kd(
        student_ddp, teacher, trainloader, testloader,
        optimizer_qat, scheduler_qat, kd_loss_fn, ce_loss_fn,
        scaler, device,
        args.qat_epochs, "phase2_qat_kd", save_dir,
        amp_enabled=args.amp,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        start_epoch=args.finetune_epochs,
    )

    final_loss, final_top1, final_top5 = evaluate(
        student_ddp, testloader, ce_loss_fn, device, args.amp,
    )
    if is_main_process():
        print(f"\n{'='*60}")
        print(
            f"FINAL RESULTS ({args.model} + {args.quant_method} + KD on {args.dataset})"
        )
        print(f"  Teacher  Top-1: {teacher_top1:.4f}  Top-5: {teacher_top5:.4f}")
        print(f"  Student  Top-1: {final_top1:.4f}  Top-5: {final_top5:.4f}")
        print(f"  Student  Test loss: {final_loss:.4f}")
        print(f"{'='*60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
