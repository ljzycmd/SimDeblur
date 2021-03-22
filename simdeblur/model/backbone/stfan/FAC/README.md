# Filter Adaptive Convolutional (FAC) Layer 
FAC layer applies generated spatially variant filters (element-wise) to the features. 

(Here we release the full code of FAC layer, including both the forwards and the backwards pass.)

## Prerequisites
- CUDA 8.0/9.0/10.0
- gcc 4.9+
- Pytorch 1.0+

Note that if your CUDA is 10.2+, you need to modify KernelConv2D_cuda.cpp:
```
1. Uncomment: #include <c10/cuda/CUDAStream.h>
2. Modify: at::cuda::getCurrentCUDAStream() -> c10::cuda::getCurrentCUDAStream()
```

## Install
```
bash install.sh

```
