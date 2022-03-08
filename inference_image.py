""" ************************************************
* fileName: inference_image.py
* desc: inference a 
* author: mingdeng_cao
* date: 2022/03/08 19:31
* last revised: None
************************************************ """


import os
import time

import torch
import argparse
import cv2
import numpy as np

from easydict import EasyDict as edict

from torchvision.utils import save_image

from simdeblur.config.build import build_config
from simdeblur.model.build import build_backbone, build_meta_arch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters during inference of SimDeblur")
    parser.add_argument("config_file", default="", help="the path of config file")
    parser.add_argument("ckpt_file", default="", help="the path of checkpoint file")
    parser.add_argument("--img", help="the path of input blurry image")
    parser.add_argument("--save_path", default=None, help="the dir to save inference resutls")

    args = parser.parse_args()
    return args


def inference():
    # read arguments
    args = parse_arguments()
    cfg = build_config(args.config_file)
    cfg.args = edict(vars(args))

    # construct model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = build_meta_arch(cfg)

    # load the trained checkpoint
    try:
        kwargs = {'map_location': lambda storage,
                    loc: storage.cuda(0)}
        ckpt = torch.load(os.path.abspath(cfg.args.ckpt_file), **kwargs)

        arch.load_ckpt(ckpt, strict=True)

        print(f"Using checkpoint loaded from {cfg.args.ckpt_file} for testing.")
    except Exception as e:
        print(e)
        print(f"Checkpoint loaded failed, cannot find ckpt file from {cfg.args.ckpt_file}.")

    arch.model.eval()

    # read input image at RGB format, shape(1, 3, H, W)
    input_image = {"input_frames": torch.Tensor(np.ascontiguousarray(cv2.imread(args.img)[..., ::-1]/255.)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)}

    with torch.no_grad():
        if hasattr(arch, "inference"):
            outputs = arch.postprocess(arch.inference(arch.preprocess(input_image)))
        else:
            outputs = arch.postprocess(arch.model(arch.preprocess(input_image)))

    if args.save_path is None:
        save_path = "./inference_resutls"
        os.makedirs(save_path, exist_ok=True)
    else:
        os.makedirs(args.save_path)
    save_image(outputs.clamp(0, 1), os.path.join(save_path, "infer_output.png"))


if __name__ == "__main__":
    inference()
