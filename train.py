""" ************************************************
* fileName: train.py
* desc: The training file for SimDeblur,
        pay much attention to your constructed configs.
* author: mingdeng_cao
* date: 2021/07/14 17:26
* last revised: Reformat the file
************************************************ """


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
