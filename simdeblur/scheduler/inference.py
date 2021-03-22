"""
# _*_ coding: utf-8 _*_
# @Time    :   2021/02/08 10:31:58
# @FileName:   inference.py
# @Author  :   Minton Cao
# @Software:   VSCode
"""


import torch
import torch.nn as nn
from tqdm import tqdm

from ..utils.dist_utils import get_local_rank

def test_all(model, test_dataloader, out_dir=None):
    model.eval()
    results = []
    rank = get_local_rank()

    for batch_data in tqdm(test_dataloader):
        with torch.no_grad():
            result = model(batch_data["input_frames"])
