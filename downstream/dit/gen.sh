#!/bin/bash
# Example: DiT-B/2 FID sampling. Edit ckpt / sample-dir before running.
torchrun --nnodes=1 --nproc_per_node=8 --master_port 39600 sample_ddp.py \
    --model DiT-B/2 \
    --num-fid-samples 50000 \
    --cfg-scale 1.0 \
    --sample-dir results/samples_dit_b2 \
    --ckpt results/000-DiT-B-2/checkpoints/0450000.pt
