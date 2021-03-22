"""
************************************************
* fileName: build.py
* desc: building the config wiht .yaml file
* author: minton_cao
* last revised: None
************************************************
"""

import os
import yaml
from easydict import EasyDict as edict


def build_config_from_file(cfg_path):
    """
    Get the config dict from cfg_path
    Args
        config_path(str): The path of config file
    Return
        cfg(edict): The config
    """
    assert os.path.isfile(cfg_path) and cfg_path.endswith(".yaml"), "The input cfg path '{}' is not a '.yaml' file. "
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        assert cfg, "The cfg is null from {}.".format(cfg_path)
    cfg = edict(cfg)
    
    return cfg


def build_config(cfg_path):
    """
    Get the config dict from cfg_path
    Args
        config_path(str): The path of config file
    Return
        cfg(edict): The config
    """
    assert os.path.isfile(cfg_path) and cfg_path.endswith(".yaml"), f"The input cfg path {cfg_path} is not a '.yaml' file. "
    cfg = build_config_from_file(os.path.abspath(cfg_path))
    if not cfg.get("__base_cfg__"):
        return cfg
    base_cfg = build_config(os.path.abspath(os.path.join(cfg_path, cfg.__base_cfg__)))
    if hasattr(base_cfg, "__base_cfg__"):
        base_cfg.pop("__base_cfg__")
    base_cfg.update(cfg)
    
    return base_cfg


def merge_from_file(cfg, base_cfg):
    """
    TODO merge the cfg to base cfg
    """
    return base_cfg


def merge_args(cfg: edict, args):
    """
    merge the args to the config
    """
    args_dict = vars(args)
    for k, v in args_dict.items():
        if k in cfg.keys() and v is not None:
            cfg[k] = v
    return cfg