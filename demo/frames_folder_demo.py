""" ************************************************
* fileName: frames_folder_demo.py
* desc: The demo of inference only "a" video splited into consecutive framess
* author: mingdeng_cao
* date: 2021/07/09 11:31
* last revised: None
************************************************ """

import os
import sys
import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
from easydict import EasyDict as edict

from simdeblur.config import build_config
from simdeblur.model import build_backbone
from simdeblur.dataset.base import DatasetBase
from simdeblur.dataset.frames_folder import FramesFolder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="The config .yaml file of deblurring model. ")
    parser.add_argument("ckpt", type=str, help="The trained checkpoint of the selected deblurring model. ")
    parser.add_argument("--frames_folder_path", type=str, help="The video frames folder path. ")
    parser.add_argument("--save_path", type=str, help="The output deblurred path")

    args = parser.parse_args()

    return args


def frames_foler_demo():
    args = parse_args()
    config = build_config(args.config)
    config.args = args
    model = build_backbone(config.model).cuda()

    ckpt = torch.load(args.ckpt, map_location="cuda:0")

    model_ckpt = ckpt["model"]
    model_ckpt = {k[7:]: v for k, v in model_ckpt.items()}
    model.load_state_dict(model_ckpt)

    data_config = edict({
        "root_input": "/home/cmd/projects/simdeblur/datasets/DVD/qualitative_datasets/alley/input",
        "num_frames": 5,
        "overlapping": True,
        "sampling": "n_c"
    })
    frames_data = FramesFolder(data_config)
    frames_dataloader = torch.utils.data.DataLoader(frames_data, 1)

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(frames_dataloader):
            out = model(batch_data["input_frames"].cuda())
            print(batch_data["gt_names"], out.shape)


if __name__ == "__main__":
    frames_foler_demo()
