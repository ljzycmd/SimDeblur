#ifdef __cplusplus
	extern "C" {
#endif

int KernelConv2D_forward_cuda_kernel(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& output,
	cudaStream_t stream
);

int KernelConv2D_backward_cuda_kernel(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& grad_output,
	at::Tensor& grad_input,
	at::Tensor& grad_kernel,
    cudaStream_t stream
);


#ifdef __cplusplus
	}
#endif
