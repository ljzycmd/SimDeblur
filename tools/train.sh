CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
# PORT=10086
# single gpu training
if [ GPUS == 1 ]
then
echo start single GPU training
python train.py $CONFIG --gpus=$GPUS
else
echo start distributed training
# distributed training
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        train.py $CONFIG --gpus=$GPUS
fi