#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h> //for CUDA 10.2+
#include <stdio.h>

#include "KernelConv2D_kernel.h"


int KernelConv2D_forward_cuda(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& output
) {
    int success = KernelConv2D_forward_cuda_kernel(
        input,
        kernel,
        kernel_size,
        output,
        // at::cuda::getCurrentCUDAStream()
	//at::globalContext().getCurrentCUDAStream() //for torch 0.4.1
	c10::cuda::getCurrentCUDAStream() //for CUDA 10.2+

    );
    if (!success) {
    	AT_ERROR("CUDA call failed");
    }
	return 1;
}

int KernelConv2D_backward_cuda(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& grad_output,
	at::Tensor& grad_input,
	at::Tensor& grad_kernel
) {
    
    int success = KernelConv2D_backward_cuda_kernel(
        input,
        kernel,
        kernel_size,
        grad_output,
        grad_input,
        grad_kernel,
        // at::cuda::getCurrentCUDAStream()
	//at::globalContext().getCurrentCUDAStream() //for torch 0.4.1
	c10::cuda::getCurrentCUDAStream() //for CUDA 10.2+
    );
    if (!success) {
    	AT_ERROR("CUDA call failed");
    }
	return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &KernelConv2D_forward_cuda, "KernelConv2D forward (CUDA)");
    m.def("backward", &KernelConv2D_backward_cuda, "KernelConv2D backward (CUDA)");
}


