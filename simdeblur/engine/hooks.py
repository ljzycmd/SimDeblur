import os
import time

import torch
import torch.distributed as dist

from datetime import datetime
from utils.logger import LogBuffer
from torchvision.utils import save_image
from utils.metrics import calculate_psnr, calculate_ssim


class HookBase:
    def before_train(self, trainer):
        pass

    def after_train(self, trainer):
        pass

    def before_epoch(self, trainer):
        pass

    def after_epoch(self, trainer):
        pass

    def before_iter(self, trainer):
        pass

    def after_iter(self, trainer):
        pass


class LRScheduler(HookBase):
    def __init__(self, lr_scheduler, optimizer):
        self._lr_scheduler = lr_scheduler
        self._optimizer = optimizer
    
    def before_train(self, trainer):
        self._optimizer = self._optimizer or trainer.optimizer
        self._lr_scheduler = self._lr_scheduler or trainer.lr_scheduler
    
    def after_epoch(self, trainer):
        self._lr_scheduler.step()


class CKPTSaver(HookBase):
    def __init__(self, save_path=None, period=5):
        self.save_path = save_path
        self.period = period
    
    def after_epoch(self, trainer):
        if trainer.cfg.args.local_rank == 0:
            if (trainer.epochs + 1) % self.period == 0: 
                trainer.save_ckpt()


class PeriodicWriter(HookBase):
    def __init__(self, writers, period=20):
        self._writers = writers
        self._period = period
        self.timer_dict = {
            "before_train" : time.time(),
            "after_train" : time.time(),
            "before_epoch" : time.time(),
            "after_epoch" : time.time(),
            "before_iter" : time.time(),
            "after_iter" : time.time(),
            "data_time" : 0,
            "iter_time" : 0,
            "total_time": 0,
        }
        self.log_buffer = LogBuffer()
    
    def before_train(self, trainer):
        self.timer_dict["before_train"] = time.time()
    
    def before_epoch(self, trainer):
        self.timer_dict["before_epoch"] = time.time()
    
    def after_epoch(self, trainer):
        self.timer_dict["after_epoch"] = time.time()

        # when evaluating...
        if not trainer.model.training:
            with torch.no_grad():
                psnr = torch.mean(torch.tensor(self.log_buffer.var_history["psnr"]).to(trainer.device))
                ssim = torch.mean(torch.tensor(self.log_buffer.var_history["ssim"]).to(trainer.device))
            self.log_buffer.clear_vars()
            
            if trainer.cfg.args.gpus > 1:
                dist.reduce(psnr, 0)
                dist.reduce(ssim, 0)
            
            # logging on the main GPU:0
            if trainer.cfg.args.local_rank == 0:
                log_dict = {
                    "epoch": trainer.epochs + 1,
                    "psnr": psnr.item() / trainer.cfg.args.gpus if trainer.cfg.args.gpus > 0 else trainer.loss.item(),
                    "ssim": ssim.item() / trainer.cfg.args.gpus if trainer.cfg.args.gpus > 0 else trainer.loss.item(),
                    }

                for writer in self._writers:
                    writer.write(log_dict, mode="val")

    
    def before_iter(self, trainer):
        self.timer_dict["before_iter"] = time.time()
    
    def after_iter(self, trainer):
        """
        different logging when the model in different mode: [trainer.model.training]
        """
        self.timer_dict["data_time"] += self.timer_dict["before_iter"] - self.timer_dict["after_iter"]
        self.timer_dict["iter_time"] += time.time() - self.timer_dict["after_iter"]

        if trainer.model.training:
            self.timer_dict["total_time"] += self.timer_dict["iter_time"]
            if (trainer.iters + 1) % self._period == 0:
                total_train_epochs = trainer.cfg.schedule.epochs - trainer.start_epoch
                epochs_left = trainer.cfg.schedule.epochs - trainer.epochs
                # iters_left = len(trainer.train_dataloader) * epochs_left - trainer.iters
                total_train_iters = len(trainer.train_dataloader) * total_train_epochs
                iters_left = len(trainer.train_dataloader) * trainer.cfg.schedule.epochs - trainer.iters
                eta_time = iters_left / self._period * self.timer_dict["iter_time"]

                # calculate metrics
                # flatten the gt_frames(b, n, c, h, w) to 4 dims tensor (b, n, c, h, w)
                psnr = calculate_psnr(trainer.batch_data["gt_frames"].flatten(0, 1).to(trainer.device), trainer.outputs.detach())
                ssim = calculate_ssim(trainer.batch_data["gt_frames"].flatten(0, 1).to(trainer.device), trainer.outputs.detach())
                
                # reduce the tensor from 
                if trainer.cfg.args.gpus > 1:
                    dist.reduce(psnr, 0)
                    dist.reduce(ssim, 0)
                    dist.reduce(trainer.loss, 0)
                
                if trainer.cfg.args.local_rank == 0:
                    log_dict = {
                        "epoch": trainer.epochs + 1,
                        "total_epochs": trainer.cfg.schedule.epochs,
                        "iter": trainer.iters + 1,
                        "iters_per_epoch": len(trainer.train_dataloader),
                        "lr": trainer.get_current_lr(),
                        "loss": trainer.loss.item() / trainer.cfg.args.gpus if trainer.cfg.args.gpus > 0 else trainer.loss.item(),
                        "psnr": psnr.item() / trainer.cfg.args.gpus if trainer.cfg.args.gpus > 0 else trainer.loss.item(),
                        "ssim": ssim.item() / trainer.cfg.args.gpus if trainer.cfg.args.gpus > 0 else trainer.loss.item(),
                        "data_time": self.timer_dict["data_time"],
                        "iter_time": self.timer_dict["iter_time"],
                        "eta_time" : self._format_time(eta_time),
                    }
                    for writer in self._writers:
                        writer.write(log_dict, mode="train")
                    
                self.timer_dict["data_time"] = 0
                self.timer_dict["iter_time"] = 0
        
        # evaluating ...
        else:
            # calculate metrics
            psnr = calculate_psnr(trainer.batch_data["gt_frames"].flatten(0, 1).to(trainer.device), trainer.outputs.detach())
            ssim = calculate_ssim(trainer.batch_data["gt_frames"].flatten(0, 1).to(trainer.device), trainer.outputs.detach())

            self.log_buffer.update({"psnr": psnr, "ssim": ssim}, count=1)
            
            # save the last validation results
            if trainer.epochs ==  trainer.cfg.schedule.epochs - 1:
                save_path = os.path.join(
                    os.path.abspath(trainer.current_work_dir),
                    "test",
                    trainer.batch_data["video_name"][0]
                )
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                save_image(
                    trainer.outputs.data.clamp(0, 1),
                    os.path.join(save_path, trainer.batch_data["gt_names"][0][0]+".png")
                )
        
        self.timer_dict["after_iter"] = time.time()
    
    def after_train(self, trainer):
        for writer in self._writers:
            writer.write(trainer)
    
    def _format_time(self, seconds=0):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        if d > 0:
            return f"{d}d{h:02}:{m:02}:{s:02}"

        return f"{h:02}:{m:02}:{s:02}"


class MetriCalculator(HookBase):
    def __init__(self):
        print("calculating the psnr and ssim metric.")
    
    def after_iter(self, trainer):
        # calculate metrics
        # flatten the gt_frames(b, n, c, h, w) to 4 dims tensor (b, n, c, h, w)
        psnr = calculate_psnr(trainer.batch_data["gt_frames"].flatten(0, 1).to(trainer.device), trainer.outputs.detach())
        ssim = calculate_ssim(trainer.batch_data["gt_frames"].flatten(0, 1).to(trainer.device), trainer.outputs.detach())
        
        ret = {}
        ret["psnr"] = psnr.item()
        ret["ssim"] = ssim.item()

        return ret