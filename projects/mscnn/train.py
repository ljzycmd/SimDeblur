""" ************************************************
* fileName: train.py
* desc: The training process
* author: mingdeng_cao
* date: 2021/04/05 16:25
* last revised: None
************************************************ """

import torch

from simdeblur.config import build_config, merge_args
from simdeblur.engine.parse_arguments import parse_arguments

# the modified Trainer class
import mscnn_trainer

def main():
    args = parse_arguments()

    cfg = build_config(args.config_file)
    cfg = merge_args(cfg, args)
    cfg.args = args

    trainer = mscnn_trainer.MscnnTrainer(cfg)
    trainer.train()
    

if __name__ == "__main__":
    main()