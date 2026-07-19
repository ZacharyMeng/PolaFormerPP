# DiT with PolaFormer++ (PolaPP)

Class-conditional ImageNet DiT where Softmax attention is replaced by **PolaPP** polarity-aware linear attention.

Based on the official [DiT](https://github.com/facebookresearch/DiT) codebase.

## Setup

```bash
conda env create -f environment.yml
conda activate DiT
# or: pip install torch torchvision timm diffusers accelerate einops
```

Prepare ImageNet under `data/imagenet/train` (ImageFolder layout).

## Train

```bash
# DiT-S/2
bash run-s.sh

# DiT-B/2
bash run-b.sh

# DiT-L/2
bash run-l.sh
```

Or directly:

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --model DiT-B/2 \
    --data-path data/imagenet/train \
    --global-batch-size 256 \
    --epochs 90 \
    --image-size 256
```

Supported models: `DiT-S/2`, `DiT-B/2`, `DiT-L/2`, and other DiT sizes in `models.py`.

## Sample / FID

```bash
# single-GPU sample
python sample.py --model DiT-B/2 --image-size 256 --ckpt /path/to/checkpoints/XXXXXXX.pt

# multi-GPU FID sampling (50k)
bash gen.sh
```

Edit `gen.sh` to set `--ckpt` and `--sample-dir`.
