#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gru_pointwise_kernel(
    const float* gi,   
    const float* gh,    
    const float* h,    
    float* h_new,       
    int batch,
    int hidden)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    if (idx >= batch * hidden) return;                 

    int b = idx / hidden;   
    int j = idx % hidden;  

}

torch::Tensor gru_pointwise(torch::Tensor gi, torch::Tensor gh, torch::Tensor h) {
    TORCH_CHECK(gi.is_cuda() && gh.is_cuda() && h.is_cuda(), "tensors on GPU");
    TORCH_CHECK(gi.is_contiguous() && gh.is_contiguous() && h.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(gi.scalar_type() == torch::kFloat32, "float32 only");

    int batch = h.size(0);
    int hidden = h.size(1);
    auto h_new = torch::empty_like(h);

    int total = batch * hidden;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;   
    gru_pointwise_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        gi.data_ptr<float>(), gh.data_ptr<float>(), h.data_ptr<float>(),
        h_new.data_ptr<float>(), batch, hidden);
    return h_new;
}
