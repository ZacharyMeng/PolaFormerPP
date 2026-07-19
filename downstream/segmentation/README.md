## Usage

1. Install [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

2. Download ADE20K and set `data_root` in `configs/_base_/datasets/ade20k_repeat.py`. The directory structure should look like

   ```
   data/ade20k
   ├── annotations
   │   ├── training
   │   └── validation
   └── images
       ├── training
       └── validation
   ```

3. Install dependencies:

   ```bash
   bash install.sh
   ```

## Training

```bash
cd segmentation
python setup.py install
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.pretrained=<PRETRAIN_MODEL>
```

Example:

```bash
bash tools/dist_train.sh configs/PolaPP/polapp_b2.py 8 \
    --options model.pretrained=path/to/polapp_b2.pth
```

## Configs (ADE20K, 512x512, 160k)

| Model | Config |
| :---: | :---: |
| PolaPP-b1 | `configs/PolaPP/polapp_b1.py` |
| PolaPP-b2 | `configs/PolaPP/polapp_b2.py` |
| PolaPP-b3 | `configs/PolaPP/polapp_b3.py` |
