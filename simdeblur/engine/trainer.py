""" ************************************************
* fileName: trainer.py
* desc: The trainer class of SimDeblur framework, which builds the training loop automatically.
* author: mingdeng_cao
* lsat revised: 2021.4.7
************************************************ """

import os
import sys
import copy
import logging

import torch
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime


from simdeblur.dataset import build_dataset
from simdeblur.scheduler import build_optimizer, build_lr_scheduler
from simdeblur.model import build_backbone, build_meta_arch, build_loss
from simdeblur.utils.logger import LogBuffer, SimpleMetricPrinter, TensorboardWriter
from simdeblur.utils.metrics import calculate_psnr, calculate_ssim
from simdeblur.utils import dist_utils

from simdeblur.engine import hooks


logging.basicConfig(format='%(asctime)s - %(levelname)s - SimDeblur: %(message)s',level=logging.INFO)
logging.info("******* A simple deblurring framework ********")


class Trainer:
    def __init__(self, cfg):
        """
        Args
            cfg(edict): the config file, which contains arguments form comand line
        """
        self.cfg = copy.deepcopy(cfg)
        # initialize the distributed training
        if cfg.args.gpus > 1:
            dist_utils.init_distributed(cfg)

        # create the working dirs
        self.current_work_dir = os.path.join(cfg.work_dir, cfg.name)
        if not os.path.exists(self.current_work_dir):
            os.makedirs(self.current_work_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        # default logger
        logger = logging.getLogger("simdeblur")
        logger.setLevel(logging.INFO)
        logger.addHandler(
            logging.FileHandler(
                os.path.join(
                    self.current_work_dir, self.cfg.name.split("_")[0] + ".json"))
        )
        
        # construct the modules
        self.model = self.build_model(cfg).to(self.device)
        self.criterion = build_loss(cfg.loss).to(self.device)
        self.train_dataloader, self.train_sampler = self.build_dataloder(cfg, mode="train")
        self.val_datalocaer, _ = self.build_dataloder(cfg, mode="val")
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        
        # trainer hooks
        self._hooks = self.build_hooks()

        # some induces when training
        self.epochs = 0
        self.iters = 0
        self.batch_idx = 0 

        self.start_epoch = 0
        self.start_iter = 0
        self.total_train_epochs = self.cfg.schedule.epochs
        self.total_train_iters = self.total_train_epochs * len(self.train_dataloader)

        # resume or load the ckpt as init-weights
        if self.cfg.resume_from != "None":
            self.resume_or_load_ckpt(ckpt_path=self.cfg.resume_from)

        # log bufffer(dict to save) 
        self.log_buffer = LogBuffer()
    
    def preprocess(self, batch_data):
        """
        prepare for input
        """
        return batch_data["input_frames"].to(self.device)

    def postprocess(self):
        """
        post process for model outputs
        """
        # When the outputs is a img tensor
        if isinstance(self.outputs, torch.Tensor) and self.outputs.dim() == 5:
            self.outputs = self.outputs.flatten(0, 1)

    def calculate_loss(self, batch_data, model_outputs):
        """
        calculate the loss
        """
        gt_frames = batch_data["gt_frames"].to(self.device).flatten(0, 1)
        if model_outputs.dim() == 5:
                model_outputs = model_outputs.flatten(0, 1) # (b*n, c, h, w)
        return self.criterion(gt_frames, model_outputs)
    
    def update_params(self):
        """
        update params
        pipline: zero_grad, backward and update grad
        """
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def train(self, **kwargs):
        self.model.train()
        self.before_train()
        logger = logging.getLogger("simdeblur")
        logger.info("Starting training...")
        for self.epochs in range(self.start_epoch, self.cfg.schedule.epochs):
            # shuffle the dataloader when dist training: dist_data_loader.set_epoch(epoch)
            self.before_epoch()
            for self.batch_idx, self.batch_data in enumerate(self.train_dataloader):
                self.before_iter()

                input_frames = self.preprocess(self.batch_data)

                self.outputs = self.model(input_frames)
                self.postprocess()
                
                self.loss = self.calculate_loss(self.batch_data, self.outputs)

                self.update_params()

                self.iters += 1
                self.after_iter()
            
            if self.epochs % self.cfg.schedule.val_epochs == 0:
                self.val()

            self.after_epoch()
    
    def before_train(self):
        for h in self._hooks:
            h.before_train(self)

    def after_train(self):
        for h in self._hooks:
            h.after_train(self)
    
    def before_epoch(self):
        for h in self._hooks:
            h.before_epoch(self)
        # shuffle the data when dist training ...
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epochs)
    
    def after_epoch(self):
        for h in self._hooks:
            h.after_epoch(self)
        
        self.model.train()

    def before_iter(self):
        for h in self._hooks:
            h.before_iter(self)

    def after_iter(self):
        for h in self._hooks:
            h.after_iter(self)

    def run_iter(self, batch_data):
        pass

    @torch.no_grad()
    def val(self):
        self.model.eval()
        for self.batch_data in tqdm(self.val_datalocaer, desc="validation on gpu{}: ".format(self.cfg.args.local_rank)):
            self.before_iter()
            input_frames = self.preprocess(self.batch_data)
            self.outputs = self.model(input_frames)
            if isinstance(self.outputs, list):
                self.outputs = self.outputs[0]
            self.postprocess()

            self.after_iter()

    def build_writers(self):
        return [
            SimpleMetricPrinter(),
            ]

    def build_hooks(self):
        ret = [
            hooks.LRScheduler(self.lr_scheduler, self.optimizer),
            hooks.CKPTSaver(**self.cfg.ckpt),
            # logging on the main process
            hooks.PeriodicWriter([
                SimpleMetricPrinter(self.current_work_dir, self.cfg.name.split("_")[0]), 
                TensorboardWriter(os.path.join(self.current_work_dir, self.cfg.name.split("_")[0], str(datetime.now()))),
                ],
                **self.cfg.logging),
        ]
        
        return ret


    def resume_or_load_ckpt(self, ckpt=None, ckpt_path=None):
        if ckpt is not None:
            try:
                self.model.load_state_dict(ckpt)
            except:
                logging.warning("Connot load the ckpt from the input ckpt !!!")
        else:
            try:
                kwargs={'map_location':lambda storage, loc: storage.cuda(self.cfg.args.local_rank)}
                ckpt = torch.load(ckpt_path, **kwargs)
                
                meta_info = ckpt["mata"]
                model_ckpt = ckpt["model"]
                optimizer_ckpt = ckpt["optimizer"]
                lr_scheduler_ckpt = ckpt["lr_scheduler"]
                
                if self.cfg.args.gpus <= 1:
                    model_ckpt = {k[7:]:v for k, v in model_ckpt.items()} # for cpu or single gpu model, it doesn't have the .module property

                # initial mode
                if not self.cfg.get("init_mode"):
                    # strict=True if resume from exist .pth, 
                    self.model.load_state_dict(model_ckpt, strict=True)
                    # load optimizer and lr_scheduler
                    self.optimizer.load_state_dict(optimizer_ckpt)
                    self.lr_scheduler.load_state_dict(lr_scheduler_ckpt)
                    # generate the idx
                    self.start_epoch = self.epochs = meta_info["epochs"]
                    self.start_iter = self.iters = self.start_epoch * len(self.train_dataloader)
                else:
                    # strict=Fasle if fine-tune from exist .pth, 
                    self.model.load_state_dict(model_ckpt, strict=False)
                    
                logging.info("Inittial mode: %s, checkpoint loaded from %s."%(self.cfg.get("init_mode"), self.cfg.resume_from))
            except:
                logging.warning("Checkpoint loaded failed, cannot find ckpt file from %s."%(self.cfg.resume_from))

    
    def save_ckpt(self, out_dir=None, ckpt_name="epoch_{}.pth"):
        meta_info = {"epochs": self.epochs + 1, "iters": self.iters + 1}
        
        ckpt_name = ckpt_name.format(self.epochs + 1)
        if out_dir is None:
            out_dir = os.path.join(self.cfg.work_dir, self.cfg.name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, ckpt_name)

        ckpt = {
            # TODO change the key mata to meta...
            "mata": meta_info,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }

        with open(ckpt_path, "wb") as f:
            torch.save(ckpt, ckpt_path)
            f.flush()

    def get_current_lr(self):
        assert self.lr_scheduler.get_last_lr()[0] == self.optimizer.param_groups[0]["lr"]
        return self.optimizer.param_groups[0]["lr"]
    
    @classmethod
    def build_model(cls, cfg):
        """
        """
        # TODO change the build backbone to build model
        model = build_backbone(cfg.model)
        if cfg.args.gpus > 1:
            rank = cfg.args.local_rank
            model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.args.local_rank], output_device=cfg.args.local_rank)
        if cfg.args.local_rank == 0:
            logger = logging.getLogger(__name__)
            logger.info("Model:\n{}".format(model))
        return model
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        """
        return build_optimizer(cfg, model)
        

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        """
        return build_lr_scheduler(cfg, optimizer)
    
    @classmethod
    def build_dataloder(cls, cfg, mode="train"):
        if mode == "train":
            dataset_cfg = cfg.dataset.train
        elif mode == "val":
            dataset_cfg = cfg.dataset.val
        elif mode == "test":
            dataset_cfg = cfg.dataset.test
        else:
            raise NotImplementedError
        dataset = build_dataset(dataset_cfg)
        if cfg.args.gpus > 1:
            # TODO reimplement the dist dataloader partition without distsampler, 
            # that is because we must shuffe the dataloader by ourself before each epoch
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=cfg.args.gpus, rank=cfg.args.local_rank, shuffle=True)
            dataloder = torch.utils.data.DataLoader(dataset, **dataset_cfg.loader, sampler=sampler)
            return dataloder, sampler
        
        else:
            dataloder = torch.utils.data.DataLoader(dataset, **dataset_cfg.loader)
            return dataloder, None
    
    @classmethod
    def test(cls, cfg):
        """
        Args:
            cfg(edict): the config file for testing, which contains "model" and "test dataloader" configs etc.
        """
        logger = logging.getLogger(__name__)

        if cfg.args.gpus > 1:
            dist_utils.init_distributed(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        model = Trainer.build_model(cfg).to(device)
        test_datalocaer, _ = Trainer.build_dataloder(cfg, "val")
        
        try:
            kwargs={'map_location':lambda storage, loc: storage.cuda(cfg.args.local_rank)}
            ckpt = torch.load(cfg.args.ckpt_file, **kwargs)

            model_ckpt = ckpt["model"]
            print(model_ckpt.keys())
            if self.cfg.args.gpus <= 1:
                model_ckpt = {k[7:]:v for k, v in model_ckpt.items()} # for cpu or single gpu model, it doesn't have the .module property
            # strict=false if fine-tune from exist .pth, 
            model.load_state_dict(model_ckpt, strict=True)
            
            logging.info("Using checkpoint loaded from %s for testing."%(cfg.args.ckpt_file))
        except:
            logging.warning("Checkpoint loaded failed, cannot find ckpt file from %s."%(cfg.args.ckpt_file))
        # writers
        # SimpleMetricPrinter(cfg.current_work_dir, cfg.name.split("_")[0])
        # TensorboardWriter(os.path.join(cfg.current_work_dir, self.cfg.name.split("_")[0], str(datetime.now())))

        model.eval()
        psnr_dict = {}
        ssim_dict = {}
        for batch_data in tqdm(test_datalocaer, desc="validation on gpu{}: ".format(cfg.args.local_rank)):
            input_frames = batch_data["input_frames"].to(device)
            gt_frames = batch_data["gt_frames"].to(device)
            outputs = model(input_frames)
            
            print("video name: ", batch_data["video_name"])
            print("frame name: ", batch_data["gt_names"])
            break
            # calculate metrics
            b, n, c, h, w = gt_frames.shape
            # single image output
            if outputs.dim() == 4:
                outputs = outputs.detach().unsqueeze(1) # (b, 1, c, h, w)
            for b_idx in range(b):
                for n_idx in range(n):
                    frame_name = "{}_{}".format(batch_data["video_name"][b_idx], batch_data["gt_names"][n_idx][b_idx])
                    psnr_dict[frame_name] = calculate_psnr(gt_frames[b_idx, n_idx:n_idx+1], outputs[b_idx, n_idx:n_idx+1]).item()
                    ssim_dict[frame_name] = calculate_ssim(gt_frames[b_idx, n_idx:n_idx+1], outputs[b_idx, n_idx:n_idx+1]).item()
                    print(frame_name, "psnr: ", psnr_dict[frame_name], "ssim: ", ssim_dict[frame_name])
        print("mean psnr: ", psnr_dict.values())
        print("mean ssim: ", ssim_dict.values())