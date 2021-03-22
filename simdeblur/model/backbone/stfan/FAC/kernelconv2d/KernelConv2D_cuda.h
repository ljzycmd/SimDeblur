int KernelConv2D_forward_cuda(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& output
);

int KernelConv2D_backward_cuda(
	at::Tensor& input,
	at::Tensor& kernel,
	int kernel_size,
	at::Tensor& grad_output,
	at::Tensor& grad_input,
	at::Tensor& grad_kernel
);
