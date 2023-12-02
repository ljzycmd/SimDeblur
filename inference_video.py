""" ************************************************
* fileName: inference_video.py
* desc: inference a demo video
* author: mingdeng_cao
* date: 2023/12/2 15:36
* last revised: None
************************************************ """

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from easydict import EasyDict as edict

from simdeblur.config import build_config
from simdeblur.model import build_backbone, build_meta_arch
from simdeblur.dataset.frames_folder import FramesFolder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="The config .yaml file of deblurring model. ")
    parser.add_argument("ckpt", type=str, help="The trained checkpoint of the selected deblurring model. ")
    parser.add_argument("--frames_folder_path", type=str, help="The video frames folder path. ")
    parser.add_argument("--save_dir", type=str, help="The output deblurred path")

    args = parser.parse_args()

    return args


def frames_foler_demo():
    args = parse_args()
    cfg = build_config(args.config)
    cfg.args = edict(vars(args))

    if args.save_dir is None:
        args.save_dir = "./workdir/inference_video"
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = build_meta_arch(cfg)

    # load the trained checkpoint
    try:
        kwargs = {'map_location': lambda storage,
                    loc: storage.cuda(0)}
        ckpt = torch.load(os.path.abspath(cfg.args.ckpt), **kwargs)

        arch.load_ckpt(ckpt, strict=True)

        print(f"Using checkpoint loaded from {cfg.args.ckpt} for testing.")
    except Exception as e:
        print(e)
        print(f"Checkpoint loaded failed, cannot find ckpt file from {cfg.args.ckpt_file}.")

    data_config = edict({
        "root_input": "/group/30042/mingdengcao/datasets/DVD/test/720p_240fps_2/input",
        "num_frames": 5,
        "overlapping": True,
        "sampling": "n_c"
    })
    frames_data = FramesFolder(data_config)
    frames_dataloader = torch.utils.data.DataLoader(frames_data, 1)

    arch.model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(frames_dataloader):
            out_frame = arch.postprocess(arch(arch.preprocess(batch_data)))
            print(batch_data["gt_names"], out_frame.shape)
            save_frame_path = os.path.join(args.save_dir, f"deblurred_frame_{i}.png")
            save_image(out_frame, save_frame_path)


if __name__ == "__main__":
    frames_foler_demo()
