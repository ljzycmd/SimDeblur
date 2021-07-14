""" ************************************************
* fileName: test.py
* desc: A simple test script of SimDeblur
* author: mingdeng_cao
* date: 2021/07/14 20:12
* last revised: None
************************************************ """


import argparse

from simdeblur.config import build_config
from simdeblur.engine.trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train image/video deblurring model")
    parser.add_argument("config_file", default="", help="the path of config file")
    parser.add_argument("ckpt_file", default="", help="the path of checkpoint file")
    parser.add_argument("--save_path", help="the dir to save logs and ckpts")

    parser.add_argument("--gpus", type=int, default=0, help="number of gpus per computing node")
    parser.add_argument("--nodes", type=int, default=1, help="number of total node")
    parser.add_argument("--local_rank", type=int, default=0, help="the local rank of current used gpu")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    cfg = build_config(args.config_file)
    cfg.args = args
    Trainer.test(cfg)


if __name__ == "__main__":
    main()
