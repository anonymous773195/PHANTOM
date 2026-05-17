# PHANTOM

Training and evaluation code for **PHANTOM**: ViT/Swin vision transformers with weight and activation quantization along with power of two scales.

## Requirements

- Python 3.10+
- PyTorch and torchvision (CUDA recommended for training)
- `webdataset` — required only for full **ImageNet** (`--dataset imagenet`)

## Training pipeline

1. **Phase 1 (teacher):** Fine-tune a full-precision ViT or Swin backbone on the target dataset.
2. **Phase 2 (QAT + KD):** Replace linear layers with QAT modules, then train the student with KD from the teacher.


Typical artifacts:

- `phase1_teacher_final.pt` — teacher after Phase 1
- `teacher_final.pt` — teacher state dict used for KD
- `phase2_qat_kd_final.pt` — final quantized student (use for evaluation)

## Quick start (`run.sh`)

`run.sh` wraps distributed training via `torch.distributed.run`. Override settings with environment variables, then run:

```bash
DATA_ROOT=/path/to/imagenet NUM_GPUS=4 bash run.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `imagenet` | `cifar10`, `cifar100`, `mnist`, `imagenet`, `imagenet100` |
| `MODEL` | `swin_s|t|b, vit_b16, vit_b32, vit_l16, vit_l32`
| `QUANT_METHOD` | `hadamard_phase_v1|hadamard_phase_v2| hadamard_phase_v3| hadamard_phase_v4` (number of stages in the residual quantization)
| `DATA_ROOT` | *(placeholder)* | Dataset root; 
| `NUM_GPUS` | `1` | GPUs for DDP |
| `FT_EPOCHS` | `50` | Phase 1 teacher epochs |
| `QAT_EPOCHS` | `70` | Phase 2 QAT+KD epochs |
| `FT_LR` | `1e-5` | Phase 1 learning rate |
| `QAT_LR` | `3e-5` | Phase 2 learning rate (passed as `--qat-lr`) |
| `WARMUP_EPOCHS` | `5` | LR warmup per phase |
| `BATCH_SIZE` | `128` | Per-GPU batch size |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `ENCODER_ACTIVATION_NBITS` | `4` | Activation bits (`32` = disabled) |
| `ENCODER_ACTIVATION_MODE` | `channel_percentile` | `global_pact`, `channel_mean`, `channel_percentile`, `channel_minmax`, `channel_pact` |
| `ACT_SCALE_PERCENTILE` | `0.95` | Used when mode is `channel_percentile` |
| `ACT_ALPHA_INIT` | `6.0` | Initial PACT clip for PACT-based modes |
| `TEACHER_CKPT` | *(empty)* | If set, skips Phase 1 and loads this teacher checkpoint |
| `TEMPERATURE` | `4.0` | KD softmax temperature |
| `POW2_SCALES` | `0` | Set to `1` to enable power-of-two weight scales |
| `AMP_FLAG` | *(empty)* | Set to `--amp` to enable mixed precision |

Example (CIFAR-10, single GPU):

```bash
DATASET=cifar10 MODEL=vit_b16 NUM_GPUS=1 DATA_ROOT=./data bash run.sh
```

## `train.py`

Distributed training entry point. Use `torch.distributed.run` (as in `run.sh`):

```bash
python -m torch.distributed.run --nproc_per_node=4 train.py \
  --dataset imagenet \
  --model swin_s \
  --quant-method hadamard_phase_v2 \
  --data-root /path/to/imagenet \
  --encoder-activation-nbits 4 \
  --encoder-activation-mode channel_percentile \
  --finetune-epochs 2 \
  --qat-epochs 40 \
  --amp
```

### Main flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10`, `cifar100`, `mnist`, `imagenet`, `imagenet100`, `imagenet-100` |
| `--model` | `vit_b16` | ViT or Swin variant (see below) |
| `--quant-method` | `hadamard_phase_v2` | Weight QAT method |
| `--data-root` | `./data` | Dataset directory |
| `--finetune-epochs` | `10` | Phase 1 epochs |
| `--qat-epochs` | `30` | Phase 2 epochs |
| `--finetune-lr` | `1e-4` | Phase 1 LR |
| `--qat-lr` | `3e-5` | Phase 2 LR |
| `--warmup-epochs` | `5` | LR warmup (0 = off) |
| `--batch-size` | `64` | batch size |
| `--num-workers` | `8` | DataLoader workers |
| `--encoder-activation-nbits` | `32` | activation bits for encoder linears (`32` = off) |
| `--encoder-activation-mode` | `global_pact` | Activation scale mode |
| `--act-scale-percentile` | `0.95` | Percentile for `channel_percentile` |
| `--act-alpha-init` | `6.0` | Initial PACT alpha |
| `--pow2-scales` | off | Power-of-two Hadamard/phase scales |
| `--teacher-checkpoint` | — | Skip Phase 1; load teacher from file |
| `--temperature` | `4.0` | KD temperature |
| `--skip-classifier` | off | Do not quantize classifier head |
| `--save-dir` | `./checkpoints` | Root output directory |
| `--amp` / `--no-amp` | off | Mixed precision |
| `--grad-accum-steps` | `1` | Gradient accumulation |
| `--max-grad-norm` | `0` | Gradient clipping (0 = disabled) |
| `--seed` | `42` | Random seed |

## `evaluate.py`

Load a checkpoint and report validation/test accuracy. **Use the same** `--model`, `--quant-method`, `--encoder-activation-*`, and `--pow2-scales` as training.

```bash
python -m torch.distributed.run --nproc_per_node=1 evaluate.py \
  --checkpoint ./checkpoints/<exp_name>/phase2_qat_kd_final.pt \
  --dataset imagenet \
  --model swin_s \
  --quant-method hadamard_phase_v2 \
  --data-root /path/to/imagenet \
  --encoder-activation-nbits 4 \
  --encoder-activation-mode channel_percentile \
  --pow2-scales
```

| Flag | Required | Description |
|------|----------|-------------|
| `--checkpoint` | yes | Path to `.pt` checkpoint |
| `--dataset` | no | Must match training |
| `--model` | no | Must match training |
| `--quant-method` | no | Apply QAT modules before loading weights |
| `--encoder-activation-nbits` | no | Must match training |
| `--encoder-activation-mode` | no | Must match training |
| `--act-scale-percentile` | no | Must match training if using percentile mode |
| `--pow2-scales` | no | Enable if checkpoint was trained with Po2 scales |
| `--inference-mode` | no | Fuse QAT modules for inference |
| `--skip-classifier` | no | Must match training |
| `--data-root` | no | Dataset root |
| `--batch-size` | no | Default `128` |
| `--amp` | no | Mixed-precision evaluation |

## Models

**ViT:** `vit_b16`, `vit_b32`, `vit_l16`, `vit_h14`, `vit_l32`

**Swin:** `swin_t`, `swin_s`, `swin_b`




## Datasets

| Dataset | Layout under `--data-root` |
|---------|----------------------------|
| `cifar10`, `cifar100`, `mnist` | Downloaded automatically via torchvision |
| `imagenet` | WebDataset shards: `train_shards/*.tar`, `val_shards/*.tar` |
| `imagenet100` | ImageFolder: `train/` and optionally `val/`; if `val/` is missing, a random split is created using `--imagenet100-val-split` and `--imagenet100-split-seed` |



## File overview

| File | Role |
|------|------|
| `train.py` | Two-phase DDP training |
| `evaluate.py` | Checkpoint evaluation |
| `run.sh` | Example launcher with env-var configuration |
| `model.py` | ViT/Swin wrappers and pretrained loading |
| `qat_modules.py` | QAT linear replacements and `METHOD_MAP` |
| `quantization.py` | Hadamard quant ops and Po2 scales |
| `act_quant.py` | Encoder activation quantization |
| `data.py` | Dataloaders and ImageNet WebDataset pipeline |
| `config.py` | ViT architecture configs |
| `losses.py` | KD loss |

