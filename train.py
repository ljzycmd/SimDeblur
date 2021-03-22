"""
************************************************
* fileName: train.py
* desc: model training process
* author: minton_cao
* last revised: None
************************************************
"""

import torch

from simdeblur.config import build_config, merge_args
from simdeblur.engine.parse_arguments import parse_arguments
from simdeblur.engine.trainer import Trainer


def main():
    args = parse_arguments()

    cfg = build_config(args.config_file)
    cfg = merge_args(cfg, args)
    cfg.args = args

    trainer = Trainer(cfg)
    trainer.train()
    

if __name__ == "__main__":
    main()