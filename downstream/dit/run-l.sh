#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model DiT-L/2 \
    --data-path data/imagenet/train \
    --global-batch-size 256 \
    --epochs 90 \
    --image-size 256
