"""
distributed utils
"""

import functools

import torch
import torch.distributed as dist
import pickle


def init_distributed(cfg, backend="nccl", init_method="evn://"):
    torch.distributed.init_process_group(backend=backend)
    assert torch.distributed.is_initialized(), "Torch distributed is not initialized!"
    torch.cuda.set_device(cfg.args.local_rank)


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


# function or class decorator
def master_only(func_or_class):
    @functools.wraps(func_or_class)
    def decorated(*args, **kwargs):
        rank = get_local_rank()
        if rank == 0:
            return func_or_class(*args, **kwargs)

    return decorated


def _object_to_tensor(obj):
    buffer = pickle.dumps(obj)
    byte_storage = torch.ByteStorage.from_buffer(buffer)  # type: ignore[attr-defined]
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def gather_objs(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if dist.get_world_size() == 1:
        return [data]
    rank = dist.get_rank()
    world_size = get_world_size()

    tensor, local_size = _object_to_tensor(data)
    size_list = [
        torch.zeros(1, dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.gather(local_size, size_list if rank == 0 else None)

    # receiving Tensor from all ranks
    if rank == dst:
        tensor_list = [
            torch.empty((size,), dtype=torch.uint8, device=tensor.device) for size in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, dst=dst)
        return []
