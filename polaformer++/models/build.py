# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .pola import POLA_b0, POLA_b1, POLA_b2, POLA_b3, POLA_b4, POLA_b5


def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type in ['polab0']:
        model = eval('POLA_b0()')

    elif model_type in ['polab1']:
        model = eval('POLA_b1()')

    elif model_type in ['polab2']:
        model = eval('POLA_b2()')

    elif model_type in ['polab3']:
        model = eval('POLA_b3()')

    elif model_type in ['polab4']:
        model = eval('POLA_b4()')

    elif model_type in ['polab5']:
        model = eval('POLA_b5()')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
