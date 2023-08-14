""" ************************************************
* fileName: utils.py
* desc: Some model util functions.
* author: mingdeng_cao
* date: 2021/11/07 21:43
* last revised:
*     2023/08/14: fix the bug of printing model params.
************************************************ """


import torch.nn as nn
import numpy as np
import logging

loger = logging.getLogger(name="simdeblur")


def print_model_params(model: nn.Module):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        mul_value = np.prod(param.size())
        total_params += mul_value
        if param.requires_grad:
            trainable_params += mul_value
        else:
            non_trainable_params += mul_value

    total_params /= 1e6
    trainable_params /= 1e6
    non_trainable_params /= 1e6

    loger.info(f'Total params: {total_params} M.')
    loger.info(f'Trainable params: {trainable_params} M.')
    loger.info(f'Non-trainable params: {non_trainable_params} M.')
