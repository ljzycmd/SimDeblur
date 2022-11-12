# <p align=center>`SimDeblur`</p>

SimDeblur (**Sim**ple **Deblur**ring) is an open-sourced unifying training and testing framework for image and video deblurring based on PyTorch. It supports most deep-learning based state-of-the-art deblurring algorithms, and provides easy way to implement your own image or video deblurring and restoration algorithms.

<div align=center>
    <img src=./docs/reds_020.gif>
</div>


### Major Features

- Modular Design

The toolbox decomposes the deblurring framework into different components and one can easily construct a customized restoration framework by combining different modules.

- State-of-the-art

The toolbox contains most deep-learning based state-of-the-art deblurring algorithms, including MSCNN, SRN, DeblurGAN, EDVR, *etc*.

- Efficient Training

SimDeblur supports distributed data-parallel training.

### New Features

[2022/11/12] SimDeblur supports [MIMOUnet](https://github.com/chosj95/MIMO-UNet) model.

[2022/3/8] We further provide a image deblurring-based inference code, please refer to **Usage** section for the using.

[2022/2/18] We add PVDNet model for video deblurring. Note that it requires the pretrained BIMNet for motion estimation. Thus please modify the CKPT path of BIMNet in the source codes.

[2022/1/21] We add Restormer model. Note that it can only works on PyTorch1.8+.

[2022/1/20] We transfer some checkpoints from the open-sourced repos into SimDeblur framework! You can find them [here](https://drive.google.com/drive/folders/1ukr-iLyxSMHVJJswHgGOF6QcQqOZwaLO?usp=sharing).

[2022/1/1] Support real-world video deblurring dataset: BSD.

[2021/3/31] Support DVD, GoPro and REDS video deblurring datasets. 

[2021/3/21] First release.

### Surpported Methods and Benchmarks

We will gradually release the checkpoints of each model in [checkpoints.md](./docs/checkpoints.md).
* Single Image Deblurring
    - [x] MSCNN [[Paper](https://arxiv.org/abs/1612.02177), [Project](https://github.com/SeungjunNah/DeepDeblur-PyTorch)]
    - [x] SRN [[Paper](https://arxiv.org/abs/1802.01770), [Project](https://github.com/jiangsutx/SRN-Deblur)]
    - [ ] DeblurGAN [[Paper](https://arxiv.org/abs/1711.07064), [Project](https://github.com/KupynOrest/DeblurGAN)]
    - [ ] DMPHN [[Paper](https://arxiv.org/abs/1904.03468), [Project](https://github.com/HongguangZhang/DMPHN-cvpr19-master)]
    - [ ] DeblurGAN_V2 [[Paper](https://arxiv.org/abs/1908.03826), [Project](https://github.com/VITA-Group/DeblurGANv2)]
    - [ ] SAPHN [[Paper](https://arxiv.org/abs/2004.05343)]
    - [x] MIMOUNet [[Paper](https://arxiv.org/abs/2108.05054), [Project](https://github.com/chosj95/MIMO-UNet)]
    - [x] Restormer [[Paper](https://arxiv.org/abs/2111.09881), [Project](https://github.com/swz30/Restormer)]

* Video Deblurring
    - [x] DBN [[Paper](https://arxiv.org/abs/1611.08387), [Project](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/), [GitHub](https://github.com/shuochsu/DeepVideoDeblurring)]
    - [x] STRCNN [[paper](https://arxiv.org/abs/1704.03285)]
    - [x] DBLRNet [[Paper](https://arxiv.org/abs/1804.00533)]
    - [x] EDVR [[Paper](https://arxiv.org/abs/1905.02716), [Project](https://github.com/xinntao/EDVR)]
    - [x] STFAN [[Paper](https://arxiv.org/abs/1904.12257), [Project](https://shangchenzhou.com/projects/stfan/)]
    - [x] IFIRNN [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Nah_Recurrent_Neural_Networks_With_Intra-Frame_Iterations_for_Video_Deblurring_CVPR_2019_paper.html)]
    - [x] CDVD-TSP [[Paper](https://arxiv.org/abs/2004.02501), [Project](https://github.com/csbhr/CDVD-TSP)]
    - [x] ESTRNN [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5116_ECCV_2020_paper.php), [Project](https://github.com/zzh-tech/ESTRNN)]
    - [x] PVDNet [[Paper](https://arxiv.org/abs/2108.09982),[Project](https://github.com/codeslake/PVDNet)]

* Benchmarks
    - [x] GoPro [[Paper](https://arxiv.org/abs/1612.02177), [Data](https://seungjunnah.github.io/Datasets/gopro)]
    - [x] DVD [[Paper](https://arxiv.org/abs/1611.08387), [Data](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)]
    - [x] REDS [[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_and_Super-Resolution_Dataset_and_CVPRW_2019_paper.html), [Data](https://seungjunnah.github.io/Datasets/reds)]
    - [x] BSD [[Paper](https://arxiv.org/abs/2106.16028), [Project](https://github.com/zzh-tech/ESTRNN), [Data](https://drive.google.com/file/d/19cel6QgofsWviRbA5IPMEv_hDbZ30vwH/view)]

### Dependencies and Installation
* Python 3 (Conda is recommended)
* Pytorch 1.5.1 (with GPU, note some methods require higher version)
* CUDA 10.1+ with NVCC (for code compilation in some models)

1. Clone the repositry or download the zip file
   ```
    git clone https://github.com/ljzycmd/SimDeblur.git
   ```
2. Install SimDeblur
   ```bash
   # create a pytorch env
   conda create -n simdeblur python=3.7
   conda activate simdeblur   
   # install the packages
   cd SimDeblur
   bash Install.sh  # some problems may occur due to wrong NVCC configurations for CUDA codes compiling
   ```

# Usage
You can open the [Colab Notebook](https://colab.research.google.com/drive/13dNBB38U_scDI46EHQ-V1GEgDMVtj5mx?usp=sharing) to learn about basic usage and see the deblurring performance.

The design of SimDeblur consists of **FOUR** main parts as follows:
| Dataset | Model  | Scheduler | Engine |
|:-------:|:------:|:---------:|:------:|
|Dataset-specific classes | The backbone, losses, and meta_archs | Opeimizer, LR scheduler | **Trainer**, and some hook functions during model training |

Note that the dataset, model and scheduler can be constructed with config (`EasyDict`) with corresponding `build_{dataset, backbone, meta_arch, scheduler, optimizer, etc.}` functions. The Trainer class automatically construct all reqiured elements for model training in a general way. This means that if you want to do some specific modeling training, you may modify the training logics.

## 0 Quick Inference

We provide a image deblurring inference code, and you can run it to deblur a blurry image as follows:

```bash
python inference_image.py CONFIG_PATH  CKPT_PATH  --img=BLUR_IMAGE_PATH  --save_path=DEBLURRED_OUT_PATH
```

the deblurred latent image will be stored at `./inference_resutls` in default.

## 1 Start with Trainer

You can construct a simple training process using the default Trainer as follows (refer to the `train.py` for more details):

```python
from easydict import EasyDict as edict
from simdeblur.config import build_config, merge_args
from simdeblur.engine.parse_arguments import parse_arguments
from simdeblur.engine.trainer import Trainer


args = parse_arguments()

cfg = build_config(args.config_file)
cfg = merge_args(cfg, args)
cfg.args = edict(vars(args))

trainer = Trainer(cfg)
trainer.train()
```

Start training with single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash ./tools/train.sh ./configs/dbn/dbn_dvd.yaml 1
```

or multiple GPUs training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/train.sh ./configs/dbn/dbn_dvd.yaml 4
```

For the testing, SimDeblur only supports **single GPU testing** and validation util now:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py ./configs/dbn/dbn_dvd.yaml PATH_TO_CKPT
```

## 2 Build specific module

SimDeblur also provides you to build some specific modules, including dataset, model, loss, *etc.*

**Build a dataset**:

```python
from easydict import EasyDict as edict
from simdeblur.dataset import build_dataset

# construct configs of target dataset.
# SimDeblur adopts EasyDict to store configs.
dataset_cfg = edict({
    "name": "DVD",
    "mode": "train",
    "sampling": "n_c",
    "overlapping": True,
    "interval": 1,
    "root_gt": "./dataset/DVD/quantitative_datasets",
    "num_frames": 5,
    "augmentation": {
        "RandomCrop": {
            "size": [256, 256] },
        "RandomHorizontalFlip": {
            "p": 0.5 },
        "RandomVerticalFlip": {
            "p": 0.5 },
        "RandomRotation90": {
            "p": 0.5 },
    }
})

dataset = build_dataset(dataset_cfg)

print(dataset[0])
```

**Build a model**:

```python
from easydict import EasyDict as edict
from simdeblur.model import build_backbone

model_cfg = edict({
    "name": "DBN",
    "num_frames": 5,
    "in_channels": 3,
    "inner_channels": 64
})

model = build_backbone(model_cfg)

x = torch.randn(1, 5, 3, 256, 256)
out = model(x)
```

**Build a loss**:

```python
from easydict import EasyDict as edict
from simdeblur.model import build_loss

criterion_cfg = {
    "name": "MSELoss",
}

criterion = build_loss()

x = torch.randn(2, 3, 256, 256)
y = torch.randn(2, 3, 256, 256)

print(criterion(x, y))
```

And the optimizer and lr_scheduler also can be created by the functions `build_optimizer` and `build_lr_scheduler` in the `simdeblur.scheduler`, *etc*.

### Dataset Description

SimDeblur supports the most popular image and video deblurring datasets, including GOPRO, DVD, REDS, BSD. We design different data reading strategies that can meet the input requirements of different image and video deblurring models.

You can click [here](./simdeblur/dataset/README.md) for more information about the design of the dataset.

**To start, note that you should change the path of the dataset in related config files.**

### Acknowledgment

The design spirit of SimDeblur comes most from Detectron2 [1], we highly thank for this amazing open-sourced toolbox. We also thank for the paper and code collections in Awesome-Deblurring repositry [2].

[1] facebookresearch. detectron2. [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

[2] subeeshvasu. Awesome-Deblurring. [https://github.com/subeeshvasu/Awesome-Deblurring](https://github.com/subeeshvasu/Awesome-Deblurring)

### Citations

If SimDeblur helps your research or work, please consider citing SimDeblur.

```bibtex
@misc{cao2021simdeblur,
  author       = {Mingdeng Cao},
  title        = {SimDeblur: A Simple Framwork for Image and Video Deblurring},
  howpublished = {\url{https://github.com/ljzycmd/SimDeblur}},
  year         = {2021}
}
```

Last, if you have any questions about SimDeblur, please feel free to [open an new issue](https://github.com/ljzycmd/SimDeblur/issues/new) or contact me at `mingdengcao [AT] gmail.com`, and I will try to solve your problem. Meanwhile, any contribution to this Repo is highly welcome. Let's make SimDeblur more powerful!
