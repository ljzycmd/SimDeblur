"""
************************************************
* fileName: parse_arguments.py
* desc: default arguments parser
* author: minton_cao
* last revised: None
************************************************
"""


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train image/video deblurring model")
    parser.add_argument("config_file", default="", help="the path of config file")
    parser.add_argument("--work_dir", help="the dir to save logs and ckpts")
    
    parser.add_argument("--gpus", type=int, default=0, help="number of gpus per computing node")
    parser.add_argument("--gpu_ids", type=list)
    parser.add_argument("--nodes", type=int, default=1, help="number of total node")
    parser.add_argument("--local_rank", type=int, default=0, help="the local rank of current used gpu")

    args = parser.parse_args()
    return args