#!/usr/bin/env bash
# Train MambaIRv2-Pola++ lightweight SR
# Usage: bash train.sh 3   or   bash train.sh 4
SCALE=${1:-3}
NPROC=${NPROC:-8}
PORT=${PORT:-1234}

python -m torch.distributed.launch --nproc_per_node=${NPROC} --master_port=${PORT} \
  basicsr/train.py \
  -opt options/train/mambairv2/train_MambaIRv2_lightSR_x${SCALE}.yml \
  --launcher pytorch
