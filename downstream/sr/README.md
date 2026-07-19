# MambaIRv2-Pola++ (Lightweight SR, ×3 / ×4)

PolaFormer++ attention inside MambaIRv2-Light.

## Install

```bash
conda activate mambair
pip install -r requirements.txt
python setup.py develop
```

## Train / Test

Checkpoints are **not** shipped here. Place them under `tpami-re/sr_ckpt/`:

- `sr_ckpt/mambairv2_polapp_lightSR_x3.pth`
- `sr_ckpt/mambairv2_polapp_lightSR_x4.pth`

```bash
bash train.sh 3
bash train.sh 4
bash test.sh 3    # uses ../../sr_ckpt/mambairv2_polapp_lightSR_x3.pth
bash test.sh 4
# or: bash test.sh 3 /path/to/ckpt.pth
```

Configs: `options/train|test/mambairv2/*_x{3,4}.yml`
