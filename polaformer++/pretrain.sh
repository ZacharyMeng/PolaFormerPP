python -m torch.distributed.launch \
    --nproc_per_node=8 \
    main_ema.py \
    --cfg ./cfgs/xxx.yaml \
    --data-path xxx/imagenet \
    --output ./xxx \
    --model-ema \
    --model-ema-decay 0.99982