""" ************************************************
* fileName: train.py
* desc: The training file for SimDeblur,
        pay much attention to your constructed configs.
* author: mingdeng_cao
* date: 2021/07/14 17:26
* last revised: Reformat the file
************************************************ """


import os
import time
from simdeblur.config import build_config, merge_args
from simdeblur.engine.parse_arguments import parse_arguments
from simdeblur.engine.trainer import Trainer
from simdeblur.config import save_configs_to_yaml


def main():
    args = parse_arguments()

    cfg = build_config(args.config_file)
    cfg = merge_args(cfg, args)
    cfg.args = args
    cfg.experiment_time = time.strftime("%Y%m%d_%H%M%S")
    if args.local_rank == 0:
        save_path = os.path.join(cfg.work_dir, cfg.name, cfg.experiment_time)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_configs_to_yaml(cfg, os.path.join(save_path, cfg.name+".yaml"))

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()