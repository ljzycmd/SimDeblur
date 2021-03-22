//Update the implemention to support 2D kernel by Shangchen Zhou 
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

#include "stdio.h"

#define THREAD_PER_BLOCK 512

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])

#ifdef __cplusplus
	extern "C" {
#endif

//Define forward operations
__global__ void KernelConv2D_forward_function(
	const int n_output,
	const int kernel_size,
	const float* input, const long4 input_shape, const long4 input_stride,
	const float* kernel, const long4 kernel_shape, const long4 kernel_stride,
	      float* output, const long4 output_shape, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n_output) {
		return;
	}

	float output_i = 0.0;

	int intBatch = ( intIndex / VEC_3(output_shape) / VEC_2(output_shape) / VEC_1(output_shape)) % VEC_0(output_shape);
	int intDepth = ( intIndex / VEC_3(output_shape) / VEC_2(output_shape)                      ) % VEC_1(output_shape);
	int intY     = ( intIndex / VEC_3(output_shape)                                            ) % VEC_2(output_shape);
	int intX     = ( intIndex                                                                  ) % VEC_3(output_shape);

	for (int intKernelY = 0; intKernelY < kernel_size; intKernelY += 1) {
		for (int intKernelX = 0; intKernelX < kernel_size; intKernelX += 1) {
			int intKernelDepth = kernel_size * kernel_size * intDepth + kernel_size * intKernelY + intKernelX;
			output_i += IDX_4(input, intBatch, intDepth, intY + intKernelY, intX + intKernelX) * IDX_4(kernel, intBatch, intKernelDepth, intY, intX);
		}
	}

	output[intIndex] = output_i;
}


int KernelConv2D_forward_cuda_kernel(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& output,
	cudaStream_t stream
) {
	int n_output = 0;
	n_output = output.size(0) * output.size(1) * output.size(2) * output.size(3);
	KernelConv2D_forward_function<<< (n_output + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK, 0, stream >>>(
		n_output,
		kernel_size,
	    input.data<float>(),		
		make_long4(input.size(0), input.size(1), input.size(2), input.size(3)), 
		make_long4(input.stride(0), input.stride(1), input.stride(2), input.stride(3)),
	    kernel.data<float>(),		
		make_long4(kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3)), 
		make_long4(kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3)),
	    output.data<float>(),
		make_long4(output.size(0), output.size(1), output.size(2), output.size(3)), 
		make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3))
	);

	cudaError_t err = cudaGetLastError();
    // check for errors
    if (err != cudaSuccess) {
    	printf("error in forward_cuda_kernel: %s\n", cudaGetErrorString(err));
    	return 0;
    }

    return 1;
}


//Define input backward operations
__global__ void KernelConv2D_backward_function_input(
	const int n_grad_input,
	const int kernel_size,
	const float* kernel, const long4 kernel_shape, const long4 kernel_stride,
	const float* grad_output, const long4 grad_output_shape, const long4 grad_output_stride,
	      float* grad_input, const long4 grad_input_shape, const long4 grad_input_stride	
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n_grad_input) {
		return;
	}

	float grad_input_i = 0.0;

	int intBatch = ( intIndex / VEC_3(grad_input_shape) / VEC_2(grad_input_shape) / VEC_1(grad_input_shape)) % VEC_0(grad_input_shape);
	int intDepth = ( intIndex / VEC_3(grad_input_shape) / VEC_2(grad_input_shape)                          ) % VEC_1(grad_input_shape);
	int intY     = ( intIndex / VEC_3(grad_input_shape)                                                    ) % VEC_2(grad_input_shape);
	int intX     = ( intIndex                                                                              ) % VEC_3(grad_input_shape);
 
	int kernel_H = VEC_2(kernel_shape);
	int kernel_W = VEC_3(kernel_shape);

	for (int intKernelY = 0; intKernelY < kernel_size; intKernelY += 1) {
		for (int intKernelX = 0; intKernelX < kernel_size; intKernelX += 1){
			// grad_input: B,C,H+k-1,W+k-1
			if (intY - intKernelY >= 0 && intY - intKernelY <= kernel_H - 1 && intX - intKernelX >= 0 && intX - intKernelX <= kernel_W - 1){
				int intKernelDepth = kernel_size * kernel_size * intDepth + kernel_size * intKernelY + intKernelX;
				grad_input_i += IDX_4(kernel, intBatch, intKernelDepth, intY - intKernelY, intX - intKernelX) * IDX_4(grad_output, intBatch, intDepth, intY - intKernelY, intX - intKernelX);
			}
		}
	}

	grad_input[intIndex] = grad_input_i;
}

//Define kernel backward operations
__global__ void KernelConv2D_backward_function_kernel(
	const int n_grad_kernel,
	const int kernel_size,
	const float* input, const long4 input_shape, const long4 input_stride,
	const float* grad_output, const long4 grad_output_shape, const long4 grad_output_stride,
	      float* grad_kernel, const long4 grad_kernel_shape, const long4 grad_kernel_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n_grad_kernel) {
		return;
	}

	int intBatch   = ( intIndex / VEC_3(grad_kernel_shape) / VEC_2(grad_kernel_shape) / kernel_size / kernel_size / VEC_1(grad_kernel_shape)) % VEC_0(grad_kernel_shape);
	int intDepth   = ( intIndex / VEC_3(grad_kernel_shape) / VEC_2(grad_kernel_shape) / kernel_size / kernel_size                           ) % VEC_1(grad_kernel_shape);
	int intKernelY = ( intIndex / VEC_3(grad_kernel_shape) / VEC_2(grad_kernel_shape) / kernel_size                                         ) % kernel_size;
	int intKernelX = ( intIndex / VEC_3(grad_kernel_shape) / VEC_2(grad_kernel_shape)                                                       ) % kernel_size;
	int intY       = ( intIndex / VEC_3(grad_kernel_shape)                                                                                  ) % VEC_2(grad_kernel_shape);
	int intX       = ( intIndex                                                                                                             ) % VEC_3(grad_kernel_shape);
	
	// grad_input: B,C,K,K,H,W
	grad_kernel[intIndex] = IDX_4(input, intBatch, intDepth, intY + intKernelY, intX + intKernelX) * IDX_4(grad_output, intBatch, intDepth, intY, intX);
}



int KernelConv2D_backward_cuda_kernel(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& grad_output,
	at::Tensor& grad_input,
	at::Tensor& grad_kernel,
	cudaStream_t stream
) {
	int n_grad_input = 0;
	n_grad_input = grad_input.size(0) * grad_input.size(1) * grad_input.size(2) * grad_input.size(3);

	int n_grad_kernel = 0;
	n_grad_kernel = grad_kernel.size(0) * grad_kernel.size(1) * grad_kernel.size(2) * grad_kernel.size(3);

    KernelConv2D_backward_function_input<<< (n_grad_input + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK, 0, stream >>>(
	    n_grad_input,
	    kernel_size,
	    kernel.data<float>(),
	    make_long4(kernel.size(0), kernel.size(1), kernel.size(2), kernel.size(3)), 
	    make_long4(kernel.stride(0), kernel.stride(1), kernel.stride(2), kernel.stride(3)),
	    grad_output.data<float>(),
	    make_long4(grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3)), 
	    make_long4(grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3)),
	    grad_input.data<float>(),
	    make_long4(grad_input.size(0), grad_input.size(1), grad_input.size(2), grad_input.size(3)), 
	    make_long4(grad_input.stride(0), grad_input.stride(1), grad_input.stride(2), grad_input.stride(3))
    );

	KernelConv2D_backward_function_kernel<<< (n_grad_kernel + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK, THREAD_PER_BLOCK, 0,stream >>>(
	    n_grad_kernel,
	    kernel_size,
	    input.data<float>(),	    
	    make_long4(input.size(0), input.size(1), input.size(2), input.size(3)), 
	    make_long4(input.stride(0), input.stride(1), input.stride(2), input.stride(3)),
	    grad_output.data<float>(),	    
	    make_long4(grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3)), 
	    make_long4(grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3)),
	    grad_kernel.data<float>(),	    
	    make_long4(grad_kernel.size(0), grad_kernel.size(1), grad_kernel.size(2), grad_kernel.size(3)),
	    make_long4(grad_kernel.stride(0), grad_kernel.stride(1), grad_kernel.stride(2), grad_kernel.stride(3))
    );
    
	// check for errors
    cudaError_t err = cudaGetLastError();
  	if (err != cudaSuccess) {
    	printf("error in backward_cuda_kernel: %s\n", cudaGetErrorString(err));
    	return 0;
    }
  	return 1;
}


#ifdef __cplusplus
	}
#endif
