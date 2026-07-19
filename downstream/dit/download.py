# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Load DiT checkpoints for sampling.
"""
import os
import torch


def find_model(model_name):
    """
    Load a DiT checkpoint from a local path.
    Supports checkpoints saved by train.py (with optional "ema" key).
    """
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint
