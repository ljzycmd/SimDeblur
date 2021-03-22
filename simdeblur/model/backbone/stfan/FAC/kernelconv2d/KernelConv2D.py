# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
import torch
from torch import nn
from torch.autograd import Function
import kernelconv2d_cuda
import random


class KernelConv2DFunction(Function):
    # def __init__(self, kernel_size=3):
    #     super(KernelConv2DFunction, self).__init__()
    #     self.kernel_size = kernel_size
    @staticmethod
    def forward(ctx, input, kernel, kernel_size):
        ctx.kernel_size = kernel_size
        assert (input.is_contiguous() == True)
        assert (kernel.is_contiguous() == True)
        ctx.save_for_backward(input, kernel)
        assert (ctx.kernel_size == int((kernel.size(1) / input.size(1)) ** 0.5))
        intKernelSize = ctx.kernel_size
        intBatches = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intOutputHeight = kernel.size(2)
        intOutputWidth = kernel.size(3)

        assert (intInputHeight - intKernelSize == intOutputHeight - 1)
        assert (intInputWidth - intKernelSize == intOutputWidth - 1)

        with torch.cuda.device_of(input):
            output = input.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()
            if input.is_cuda == True:
                kernelconv2d_cuda.forward(input, kernel, intKernelSize, output)
            elif input.is_cuda == False:
                raise NotImplementedError()  # CPU VERSION NOT IMPLEMENTED
                print(5)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, kernel = ctx.saved_tensors
        intKernelSize = ctx.kernel_size
        grad_output = grad_output.contiguous()
        with torch.cuda.device_of(input):
            grad_input = input.new().resize_(input.size()).zero_()
            grad_kernel = kernel.new().resize_(kernel.size()).zero_()
            if grad_output.is_cuda == True:
                kernelconv2d_cuda.backward(input, kernel, intKernelSize, grad_output, grad_input, grad_kernel)

            elif grad_output.is_cuda == False:
                raise NotImplementedError()  # CPU VERSION NOT IMPLEMENTED

        return grad_input, grad_kernel, None


def gradient_check():
    kernel_size_list = [1, 3]
    len_list = [8, 10]
    for i in range(10):
        B = random.randint(1, 4)
        C = i + 1
        K = random.choice(kernel_size_list)
        H = random.choice(len_list)
        W = random.choice(len_list)
        input = torch.randn(B, C, H + K - 1, W + K - 1, requires_grad=True).cuda()
        kernel = torch.randn(B, C * K * K, H, W, requires_grad=True).cuda()
        # linear function, thus eps set to 1e-1
        print(torch.autograd.gradcheck(KernelConv2DFunction(K), (input, kernel), eps=1e-1, atol=1e-5, rtol=1e-3,
                                       raise_exception=True))


class KernelConv2D(nn.Module):
    def __init__(self, kernel_size):
        super(KernelConv2D, self).__init__()
        assert (kernel_size % 2 == 1)
        self.kernel_size = kernel_size
        self.pad = torch.nn.ReplicationPad2d(
            [(kernel_size - 1) // 2, (kernel_size - 1) // 2, (kernel_size - 1) // 2, (kernel_size - 1) // 2])

    def forward(self, input, kernel):
        input_pad = self.pad(input)
        return KernelConv2DFunction.apply(input_pad, kernel, self.kernel_size)
