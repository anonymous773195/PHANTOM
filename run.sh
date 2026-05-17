#!/bin/bash

DATASET="${DATASET:-imagenet}"
MODEL="${MODEL:-swin_s}"
QUANT_METHOD="${QUANT_METHOD:-hadamard_phase_v2}"
FT_EPOCHS="${FT_EPOCHS:-2}"
QAT_EPOCHS="${QAT_EPOCHS:-40}"
FT_LR="${FT_LR:-1e-5}"
QAT_LR="${QAT_LR:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_ROOT="${DATA_ROOT:-path to imagenet dataset}"
NUM_GPUS="${NUM_GPUS:-1}"
AMP_FLAG="${AMP_FLAG:-}"

ENCODER_ACTIVATION_NBITS="${ENCODER_ACTIVATION_NBITS:-4}" #activation quantization bitwidth
ENCODER_ACTIVATION_MODE="${ENCODER_ACTIVATION_MODE:-channel_percentile}" # options: global_pact, channel_mean, channel_percentile, channel_minmax, channel_pact
ACT_SCALE_PERCENTILE="${ACT_SCALE_PERCENTILE:-0.95}"
ACT_ALPHA_INIT="${ACT_ALPHA_INIT:-6.0}"                                   

RKHS_SIGMA="${RKHS_SIGMA:-1.0}"
RKHS_SEED="${RKHS_SEED:-42}"

# if you want to reuse a pre-trained teacher, set TEACHER_CKPT to the path to the teacher checkpoint
TEACHER_CKPT="${TEACHER_CKPT:-}"
TEMPERATURE="${TEMPERATURE:-4.0}"

# if you want to turn on power of two scales, set POW2_SCALES to 1
POW2_SCALES="${POW2_SCALES:-0}"


EXTRA_FLAGS=""
if [ -n "${TEACHER_CKPT}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --teacher-checkpoint ${TEACHER_CKPT}"
fi
if [ "${POW2_SCALES}" -eq 1 ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --pow2-scales"
fi


echo " PHANTOM"
echo " Dataset:      ${DATASET}"
echo " Model:        ${MODEL}"
echo " Quant method: ${QUANT_METHOD}"
echo " Act nbits:    ${ENCODER_ACTIVATION_NBITS}"
echo " Act mode:     ${ENCODER_ACTIVATION_MODE}"
echo " Act pct:      ${ACT_SCALE_PERCENTILE}"
echo " FT epochs:    ${FT_EPOCHS}"
echo " QAT epochs:   ${QAT_EPOCHS}"
echo " FT LR:        ${FT_LR}"
echo " Warmup:       ${WARMUP_EPOCHS} epochs"
echo " Batch size:   ${BATCH_SIZE}"
echo " Num workers:  ${NUM_WORKERS}"
echo " GPUs:         ${NUM_GPUS}"
echo " AMP:          ${AMP_FLAG:-disabled}"
echo " RKHS sigma:   ${RKHS_SIGMA}"
echo " RKHS seed:    ${RKHS_SEED}"
echo " Temperature:  ${TEMPERATURE}"
echo " Po2 scales:   ${POW2_SCALES} (Hadamard/phase weight scales; 0=off)"
echo " Teacher ckpt: ${TEACHER_CKPT:-train from scratch}"

python3 -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --quant-method "${QUANT_METHOD}" \
    --encoder-activation-nbits "${ENCODER_ACTIVATION_NBITS}" \
    --encoder-activation-mode "${ENCODER_ACTIVATION_MODE}" \
    --act-scale-percentile "${ACT_SCALE_PERCENTILE}" \
    --act-alpha-init "${ACT_ALPHA_INIT}" \
    --finetune-epochs "${FT_EPOCHS}" \
    --qat-epochs "${QAT_EPOCHS}" \
    --finetune-lr "${FT_LR}" \
    --qat-lr "${QAT_LR}" \
    --batch-size "${BATCH_SIZE}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --data-root "${DATA_ROOT}" \
    --save-dir "./checkpoints" \
    --rkhs-sigma "${RKHS_SIGMA}" \
    --rkhs-seed "${RKHS_SEED}" \
    --temperature "${TEMPERATURE}" \
    --num-workers "${NUM_WORKERS}" \
    ${EXTRA_FLAGS} \
    ${AMP_FLAG}

