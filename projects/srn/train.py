""" ************************************************
* fileName: train.py
* desc: 
* author: mingdeng_cao
* date: 2021/04/07 00:31
* last revised: None
************************************************ """

import torch

from simdeblur.config import build_config, merge_args
from simdeblur.engine.parse_arguments import parse_arguments

# the modified Trainer class
import srn_trainer

def main():
    args = parse_arguments()

    cfg = build_config(args.config_file)
    cfg = merge_args(cfg, args)
    cfg.args = args

    trainer = srn_trainer.SRNTrainer(cfg)
    trainer.train()
    

if __name__ == "__main__":
    main()