An implementation of MSCNN in cvpr2018 paper *Scale-recurrent Network for Deep Image Deblurring*. Please refer the original repo at [here](https://github.com/jiangsutx/SRN-Deblur).

Because of the multi-scale input and output are required by SRN, so we need to prepare this task-specific inputs before the training. We provide a simple way to implement this operation. If you want to generate some input formats corresponding to YOU, you can only override the *preprocess* function in simdeblur.engine.trainer.Trainer class. 

Please refer the srn_trainer.py and train.py for more details. 

**Warning:** remember that our workding directory is alway SimDeblur/, so when you train this project, you can start training like following:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10088 projects/srn/train.py configs/srn/srn_gopro.yaml --gpus=4
```
Or you can write a shell script like ./tools/train.sh to train the project.