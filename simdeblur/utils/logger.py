""" ************************************************
* fileName: logger.py
* desc: The logger class of SimDeblur
* author: mingdeng_cao
* last revised: None
************************************************ """


import os
import time
import logging
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from .metrics import calculate_psnr, calculate_ssim
from .dist_utils import master_only


def init_logger(log_file_path="./", name="simdeblur"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    DATE_FORMAT = "%d %b %Y %H:%M:%S"
    BASIC_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    formatter = logging.Formatter(fmt=BASIC_FORMAT, datefmt=DATE_FORMAT)
    # add 2 handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_file_path, "logs.log"), mode="a")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info(f"logs are saved to {log_file_path}")


class LogBuffer:
    def __init__(self):
        self.var_history = OrderedDict()
        self.n_history = OrderedDict()

    def update(self, log_vars, count):
        assert isinstance(log_vars, dict), type(log_vars)
        for key, var in log_vars.items():
            if key not in self.var_history:
                self.var_history[key] = []
                self.n_history[key] = []
            self.var_history[key].append(var)
            self.n_history[key].append(count)

    def history(self, k):
        return self.var_history.get(k)[-1]

    def clear_vars(self):
        self.var_history.clear()
        self.n_history.clear()


class SimpleMetricPrinter:
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("simdeblur")

    def write(self, log_dict, mode="train"):
        if mode == "train":
            log_str = "epoch[{}/{}]--iter[{}/{}]".format(
                log_dict["epoch"], log_dict["total_epochs"],
                log_dict["iter"], log_dict["iters_per_epoch"])
            # lr
            if log_dict.get("lr"):
                if isinstance(log_dict["lr"], dict):
                    for k, v in log_dict["lr"].items():
                        log_str += f"--{k}:{v}"
                else:
                    log_str += "--lr:{}".format(log_dict.get("lr"))

            # loss
            if log_dict.get("loss"):
                if isinstance(log_dict["loss"], dict):
                    for k, v in log_dict["loss"].items():
                        log_str += f"--{k}:{v:.4f}"
                    log_str += "--loss:{:.4f}".format(
                        sum(log_dict["loss"].values()))
                else:
                    log_str += "--loss:{:.4f}".format(log_dict.get("loss"))
            # metrics
            if log_dict.get("metrics"):
                if isinstance(log_dict["metrics"], dict):
                    for k, v in log_dict["metrics"].items():
                        log_str += f"--{k}:{v:.4}"
                else:
                    log_str += "--psnr:{:.4f}".format(log_dict.get("psnr"))
                    log_str += "--ssim:{:.4f}".format(log_dict.get("ssim"))

            # time
            log_str += "--data_time:{:.4f}s--iter_time:{:.4f}s--eta:{}".format(
                log_dict["data_time"],
                log_dict["iter_time"],
                log_dict["eta_time"],
            )
            self.logger.info(log_str)

        elif mode == "val":
            self.logger.info("epoch[{}]--psnr: {:.2f}--ssim: {:.4f}".format(
                log_dict["epoch"],
                log_dict["psnr"],
                log_dict["ssim"],
            ))
        else:
            raise NotImplementedError


@master_only
class TensorboardWriter:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self._last_write = None

    def write(self, log_dict, mode="train"):
        if mode == "train":
            if "lr" in log_dict:
                if isinstance(log_dict.get("lr"), dict):
                    for k, v in log_dict.get("lr").items():
                        self.writer.add_scalar(f"lr/{k}", v, log_dict["epoch"])
                else:
                    self.writer.add_scalar(
                        "lr", log_dict["lr"], log_dict["epoch"])
            if log_dict.get("loss"):
                if isinstance(log_dict.get("loss"), dict):
                    for k, v in log_dict.get("loss").items():
                        self.writer.add_scalar(
                            f"train/{k}", v, log_dict["iter"])
                else:
                    self.writer.add_scalar(
                        "train/loss", log_dict["loss"], log_dict["iter"])
            if log_dict.get("metrics"):
                if isinstance(log_dict.get("metrics"), dict):
                    for k, v in log_dict.get("metrics").items():
                        self.writer.add_scalar(
                            f"train/{k}", v, log_dict["iter"])
                else:
                    self.writer.add_scalar(
                        "train/psnr", log_dict["psnr"], log_dict["iter"])
                    self.writer.add_scalar(
                        "train/ssim", log_dict["ssim"], log_dict["iter"])

        elif mode == "val":
            if isinstance(log_dict.get("metrics"), dict):
                for k, v in log_dict.get("metrics").items():
                    self.writer.add_scalar(f"val/{k}", v, log_dict["epoch"])
            else:
                self.writer.add_scalar(
                    "val/psnr", log_dict["psnr"], log_dict["epoch"])
                self.writer.add_scalar(
                    "val/ssim", log_dict["ssim"], log_dict["epoch"])

        else:
            raise NotImplementedError

        self._last_write = log_dict
