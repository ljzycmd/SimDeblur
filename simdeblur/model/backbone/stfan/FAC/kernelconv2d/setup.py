import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75'
]

setup(
    version='1.0.0',
    name='kernelconv2d_cuda',
    ext_modules=[
        CUDAExtension('kernelconv2d_cuda', [
            'KernelConv2D_cuda.cpp',
            'KernelConv2D_kernel.cu'
        ], 
        include_dirs = ["/usr/local/cuda-10.2/include"],
        extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
