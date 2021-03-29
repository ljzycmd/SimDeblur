""" ************************************************
* fileName: default_config.py
* desc: The default configs for SimDeblur
* author: minton_cao
* last revised: None
************************************************ """


from easydict import EasyDict as edict

cfg = edict()

# default config name
cfg.name = "000_Dafault"

# dataset config
cfg.dataset = edict()
# training dataset
cfg.dataset.train = edict()
cfg.dataset.train.name = "DVD"
cfg.dataset.train.mode = "train"
cfg.dataset.train.sampling = "n_c"
cfg.dataset.train.overlapping = True
cfg.dataset.train.interval = 1
cfg.dataset.train.root_gt = "./datasets/DVD/quantitative_datasets"
cfg.dataset.train.data_type = "imgs"
cfg.dataset.train.num_frames = 5

cfg.dataset.train.augmentation = edict()
cfg.dataset.train.augmentation.RandomCrop = edict()
cfg.dataset.train.augmentation.RandomCrop.size = [256, 256]
cfg.dataset.train.augmentation.RandomHorizontalFlip = edict()
cfg.dataset.train.augmentation.RandomHorizontalFlip.p = 0.5
cfg.dataset.train.augmentation.RandomVerticalFlip = edict()
cfg.dataset.train.augmentation.RandomVerticalFlip.p = 0.5
cfg.dataset.train.augmentation.RandomRotation90 = edict()
cfg.dataset.train.augmentation.RandomRotation90.p = 0.5
cfg.dataset.train.augmentation.RandomReverse = edict()
cfg.dataset.train.augmentation.RandomReverse.p = 0.5
cfg.dataset.train.augmentation.Normalize = edict()
cfg.dataset.train.augmentation.Normalize.pixel_max = 255

cfg.dataset.train.loader = edict()
cfg.dataset.train.loader.batch_size = 1
cfg.dataset.train.loader.num_workers = 1

# test dataset
cfg.dataset.test = edict()
cfg.dataset.test.name = "DVD"
cfg.dataset.test.mode = "val"
cfg.dataset.test.sampling = "n_c"
cfg.dataset.test.overlapping = False
cfg.dataset.test.interval = 1
cfg.dataset.test.root_gt = "./datasets/DVD/quantitative_datasets"
cfg.dataset.test.data_type = "imgs"
cfg.dataset.test.num_frames = 5

cfg.dataset.test.loader = edict()
cfg.dataset.test.loader.batch_size = 1
cfg.dataset.test.loader.num_workers = 1


# model config
cfg.model = edict()
cfg.model.name = "DBN"
cfg.model.num_frames = 5
cfg.model.in_channels = 3
cfg.model.inner_channels = 64

# loss
cfg.loss = edict()
cfg.loss.name = "CharbonnierLoss"

# scheduler
cfg.schedule = edict()
cfg.schedule.epochs = 651
cfg.schedule.val_epochs = 5

cfg.schedule.optimizer = edict()
cfg.schedule.optimizer.name = "Adam"
cfg.schedule.optimizer.lr = 0.0002
cfg.schedule.optimizer.betas = [0.9, 0.99]
cfg.schedule.optimizer.weight_decay = 0.

cfg.schedule.lr_scheduler = edict()
cfg.schedule.lr_scheduler.name = "MultiStepLR"
cfg.schedule.lr_scheduler.milestones = [500, 600, 620]
cfg.schedule.lr_scheduler.gamma = 0.1


# logging config
cfg.logging = edict()
cfg.logging.period = 20


# checkpoint config
cfg.ckpt = edict()
cfg.ckpt.period = 10


# work directory
cfg.work_dir = "./workdir/dbn"

# resume training or init 
cfg.resume_from = None
cfg.init_mode = False