## Usage

1. Install [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)

2. Download COCO 2017 and set `data_root` in:
   - `configs/_base_/datasets/coco_instance.py`
   - `configs/_base_/datasets/coco_detection.py`

   ```
   data/coco
   ├── annotations
   ├── train2017
   └── val2017
   ```

3. Install dependencies:

   ```bash
   bash install.sh
   ```

## Training

```bash
cd detection
python setup.py install
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL>
```

Example:

```bash
bash tools/dist_train.sh configs/PolaPP/maskrcnn_b3_1x.py 8 \
    --cfg-options model.pretrained=path/to/polapp_b3.pth
```

## Configs

### RetinaNet 1x
| Model | Config |
| :---: | :---: |
| PolaPP-b2 | `configs/PolaPP/retinanet_b2_1x.py` |
| PolaPP-b3 | `configs/PolaPP/retinanet_b3_1x.py` |

### Mask R-CNN 1x
| Model | Config |
| :---: | :---: |
| PolaPP-b2 | `configs/PolaPP/maskrcnn_b2_1x.py` |
| PolaPP-b3 | `configs/PolaPP/maskrcnn_b3_1x.py` |
| PolaPP-b4 | `configs/PolaPP/maskrcnn_b4_1x.py` |
| PolaPP-b5 | `configs/PolaPP/maskrcnn_b5_1x.py` |

### Mask R-CNN 3x
| Model | Config |
| :---: | :---: |
| PolaPP-b2 | `configs/PolaPP/maskrcnn_b2_3x.py` |
| PolaPP-b3 | `configs/PolaPP/maskrcnn_b3_3x.py` |
| PolaPP-b4 | `configs/PolaPP/maskrcnn_b4_3x.py` |
