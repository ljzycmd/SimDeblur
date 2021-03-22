# CMD

import os
import logging
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from .metrics import calculate_psnr, calculate_ssim
from .dist_utils import master_only

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


class UniformMetricWtiter:
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.logger = logging.getLogger("simdeblur")
    
    def write(self, log_dict, mode="train"):
        pass


class SimpleMetricPrinter:
    def __init__(self, log_dir, log_name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(os.path.join(log_dir, log_name + ".json")))
        self._last_write = None

    def write(self,log_dict, mode="train"):
        if mode == "train":
            self.logger.info("e[{}/{}]--iter[{}/{}]--lr:{}--loss:{:.3f}--psnr:{:.2f}--ssim:{:.4f}--data_time:{:.4f}s--iter_time:{:.4f}s--eta:{}".format(
                log_dict["epoch"], log_dict["total_epochs"],
                log_dict["iter"], log_dict["iters_per_epoch"],
                log_dict["lr"],
                log_dict["loss"], 
                log_dict["psnr"], 
                log_dict["ssim"],
                log_dict["data_time"],
                log_dict["iter_time"],
                log_dict["eta_time"],
            ))
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

    def write(self,log_dict, mode="train"):
        if mode == "train":
            if "lr" in log_dict:
                self.writer.add_scalar("lr", log_dict["lr"], log_dict["epoch"])
            self.writer.add_scalar("train/loss", log_dict["loss"], log_dict["iter"])
            self.writer.add_scalar("train/psnr", log_dict["psnr"], log_dict["iter"])
            self.writer.add_scalar("train/ssim", log_dict["ssim"], log_dict["iter"])

        elif mode == "val":
            self.writer.add_scalar("val/psnr", log_dict["psnr"], log_dict["epoch"])
            self.writer.add_scalar("val/ssim", log_dict["ssim"], log_dict["epoch"])

        else:
            raise NotImplementedError

        self._last_write = log_dict


class JSONMetricWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def write(self, log_dict, mode="train"):
        self.logger.info("{} SimDeblur: epoch[{}/{}]--iter[{}/{}]--lr: {}--loss: {:.3f}--psnr: {:.2f}--ssim: {:.4f}--est_time: ".format(
            datetime.now(),
            log_dict["epoch"], log_dict["total_epochs"],
            log_dict["iter"], log_dict["iters_per_epoch"],
            log_dict["lr"],
            log_dict["loss"], 
            log_dict["psnr"], 
            log_dict["ssim"],
        ))
