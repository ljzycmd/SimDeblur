# <p align=center>`SimDeblur`</p>

SimDeblur (**Sim**ple **Deblur**ring) is an open source framework for image and video deblurring based on PyTorch, which contains most deep-learning based state-of-the-art deblurring algorithms. It is easy to implement your own image or video deblurring and restoration algorithms. To the best of our knowledge, this is the first general framework for image/video delburring. 

<div align=center> 
<img src=./docs/reds_020.gif>
</div>

### Major Features

- Modular Design

The toolbox decomposes the deblurring framework into different components and one can easily construct a customized restoration framework by combining different modules.

- State of the art

The toolbox contains most deep-learning based state-of-the-art deblurring algorithms, including MSCNN, SRN, DeblurGAN, EDVR, *etc*.

- Distributed Training


### New Features
[2021/4/7] Provide some project demos implemented under SimDeblur: SRN and MSCNN.

[2021/3/31] Support DVD, GoPro and REDS video deblurring datasets. 

[2021/3/21] First release.

### Surpported Methods and Benchmarks

* Single Image Deblurring 
    - [x] MSCNN [[Paper](https://arxiv.org/abs/1612.02177), [Project](https://github.com/SeungjunNah/DeepDeblur-PyTorch)]
    - [x] SRN [[Paper](https://arxiv.org/abs/1802.01770), [Project](https://github.com/jiangsutx/SRN-Deblur)]
    - [ ] DeblurGAN [[Paper](https://arxiv.org/abs/1711.07064), [Project](https://github.com/KupynOrest/DeblurGAN)]
    - [ ] DMPHN [[Paper](https://arxiv.org/abs/1904.03468), [Project](https://github.com/HongguangZhang/DMPHN-cvpr19-master)]
    - [ ] DeblurGAN_V2 [[Paper](https://arxiv.org/abs/1908.03826), [Project](https://github.com/VITA-Group/DeblurGANv2)]
    - [ ] SAPHN [[Paper](https://arxiv.org/abs/2004.05343)]

* Video Deblurring
    - [x] DBN [[Paper](https://arxiv.org/abs/1611.08387), [Project](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/), [GitHub](https://github.com/shuochsu/DeepVideoDeblurring)]
    - [x] STRCNN [[paper](https://arxiv.org/abs/1704.03285)]
    - [x] DBLRNet [[Paper](https://arxiv.org/abs/1804.00533)]
    - [x] EDVR [[Paper](https://arxiv.org/abs/1905.02716), [Project](https://github.com/xinntao/EDVR)]
    - [x] STFAN [[Paper](https://arxiv.org/abs/1904.12257), [Project](https://shangchenzhou.com/projects/stfan/)]
    - [x] IFIRNN [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Nah_Recurrent_Neural_Networks_With_Intra-Frame_Iterations_for_Video_Deblurring_CVPR_2019_paper.html)]
    - [ ] CDVD-TSP [[Paper](https://arxiv.org/abs/2004.02501), [Project](https://github.com/csbhr/CDVD-TSP)]
    - [x] ESTRNN [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5116_ECCV_2020_paper.php), [Project](https://github.com/zzh-tech/ESTRNN)]

* Benchmarks
    - [x] GoPro [[Paper](https://arxiv.org/abs/1612.02177), [Data](https://seungjunnah.github.io/Datasets/gopro)]
    - [x] DVD [[Paper](https://arxiv.org/abs/1611.08387), [Data](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)]
    - [x] REDS [[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_and_Super-Resolution_Dataset_and_CVPRW_2019_paper.html), [Data](https://seungjunnah.github.io/Datasets/reds)]

### Dependencies and Installation
* Python 3 (Conda is recommended)
* Pytorch 1.5.1 (with GPU)
* CUDA 10.2+ 
1. Clone the repositry or download the zip file
   ```git
    git clone https://github.com/ljzycmd/SimDeblur.git
   ```
2. Install SimDeblur
   ```bash
   # create a pytorch env
   conda create -n simdeblur python=3.7
   conda activate simdeblur   
   # install the packages
   cd SimDeblur
   bash Install.sh
   ```

# Usage
## 1 Start with trainer
You can construct a simple training process use the default trainer like following:
```python
from simdeblur.config import build_config, merge_args
from simdeblur.engine.parse_arguments import parse_arguments
from simdeblur.engine.trainer import Trainer


args = parse_arguments()

cfg = build_config(args.config_file)
cfg = merge_args(cfg, args)
cfg.args = args

trainer = Trainer(cfg)
trainer.train()
```
Then start training with single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 bash ./tools/train.sh ./config/dbn/dbn_dvd.yaml 1
```
multi GPU training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./tools/train.sh ./config/dbn/dbn_dvd.yaml 4
```

## 2 Build a module
The SimDeblur also provides you to build each module.

Build a dataset:
```python
from easydict import EasyDict as edict
from simdeblur.dataset import build_dataset

dataset = build_dataset(edict({
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
}))

print(dataset[0])
```

Build a model:
```python
from simdeblur.model import build_backbone

model = build_backbone({
    "name": "DBN",
    "num_frames": 5,
    "in_channels": 3,
    "inner_channels": 64
})

x = torch.randn(1, 5, 3, 256, 256)
out = model(x)
```
Build the loss:
```python 
from simdeblur.model import build_loss

criterion = build_loss({
    "name": "MSELoss",
})
x = torch.randn(2, 3, 256, 256)
y = torch.randn(2, 3, 256, 256)
print(criterion(x, y))
```
And the optimizer and lr_scheduler also can be created by "build_optimizer" and "build_lr_scheduler" etc. 

### Dataset Description

Click [here](./simdeblur/dataset/README.md) for more information. 

### Acknowledgment

[1] facebookresearch. detectron2. https://github.com/facebookresearch/detectron2

[2] subeeshvasu. Awesome-Deblurring. https://github.com/subeeshvasu/Awesome-Deblurring

### Citations

If SimDeblur helps your research or work, please consider citing SimDeblur.

```bibtex
@misc{cao2021simdeblur,
  author =       {Mingdeng Cao},
  title =        {SimDeblur},
  howpublished = {\url{https://github.com/ljzycmd/SimDeblur}},
  year =         {2021}
}
```
If you have any question, please contact me at `mingdengcao AT gmail.com`.