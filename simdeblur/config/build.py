""" ************************************************
* fileName: build.py
* desc: The build script of config in SimDeblur
* author: mingdeng_cao
* date: 2021/03/29
* last revised: None
************************************************ """

import os
import yaml
from easydict import EasyDict as edict
import logging

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
    base_cfg = build_config(os.path.abspath(os.path.join(os.path.dirname(cfg_path), cfg.__base_cfg__)))
    if hasattr(base_cfg, "__base_cfg__"):
        base_cfg.pop("__base_cfg__")
    base_cfg = merge_from_cfg(cfg, base_cfg)
    return base_cfg


def merge_from_cfg(cfg, base_cfg):
    """
    merge the cfg to base cfg
    """
    for k, v in cfg.items():
        if base_cfg.get(k) and isinstance(v, dict):
            base_cfg.update({k: merge_from_cfg(v, base_cfg.get(k))})
        else:
            base_cfg.update({k: v})
    return base_cfg


def merge_args(cfg: edict, args):
    """
    merge the arguments to the config
    """
    args_dict = vars(args)
    for k, v in args_dict.items():
        if k in cfg.keys() and v is not None:
            cfg.update({k: v})
    return cfg


def save_configs_to_yaml(cfg: edict, save_path: str):
    """
    save the config file to a .yaml file
    """
    def edict_to_dict(cfg):
        if not (isinstance(cfg, dict) or isinstance(cfg, edict)):
            return cfg
        if isinstance(cfg, edict):
            cfg = dict(cfg)
        for k, v in cfg.items():
            cfg[k] = edict_to_dict(v)
        return cfg
    cfg = edict_to_dict(cfg)
    with open(save_path, "w") as f:
        yaml.dump(cfg, f)
