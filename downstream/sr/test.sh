#!/usr/bin/env bash
# Test MambaIRv2-Pola++ lightweight SR
# Usage: bash test.sh 3 [/path/to/ckpt.pth]
#        bash test.sh 4 [/path/to/ckpt.pth]
SCALE=${1:-3}
CKPT=${2:-../../sr_ckpt/mambairv2_polapp_lightSR_x${SCALE}.pth}
OPT=options/test/mambairv2/test_MambaIRv2_lightSR_x${SCALE}.yml

python basicsr/test.py -opt "$OPT" --force_yml path:pretrain_network_g="${CKPT}" path:strict_load_g=true
