# CMD

import functools

import torch
import torch.distributed as dist 



def get_word_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized:
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

# def master_only(func_or_class):
#     rank = get_local_rank()
#     if rank == 0:
#         return func_or_class