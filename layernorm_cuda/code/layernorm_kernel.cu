#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#define WARP_SIZE 32

template<typename T>
__global__ void LayerNormKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int hidden_size,
    const float eps) {
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    
    const T* row_input = input + batch_idx * hidden_size;
    T* row_output = output + batch_idx * hidden_size;
    
    // Step 1: Calculate mean
    float sum = 0.0f;
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        sum += static_cast<float>(row_input[i]);
    }
    
    // Warp reduce
    #pragma unroll
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ float s_partial_sums[32];  // Maximum 32 warps
    if (lane_id == 0) {
        s_partial_sums[tid / WARP_SIZE] = sum;
    }
    __syncthreads();
    
    // First warp reduces partial sums
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        sum = s_partial_sums[tid];
    } else {
        sum = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        #pragma unroll
        for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum / hidden_size;
    }
    __syncthreads();
    
    // Step 2: Calculate variance
    float variance = 0.0f;
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = static_cast<float>(row_input[i]) - s_mean;
        variance += diff * diff;
    }
    
    // Warp reduce variance
    #pragma unroll
    for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        variance += __shfl_down_sync(0xffffffff, variance, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        s_partial_sums[tid / WARP_SIZE] = variance;
    }
    __syncthreads();
    
    // First warp reduces partial sums
    if (tid < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        variance = s_partial_sums[tid];
    } else {
        variance = 0.0f;
    }
    
    if (tid < WARP_SIZE) {
        #pragma unroll
        for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            variance += __shfl_down_sync(0xffffffff, variance, offset);
        }
    }
    
    __shared__ float s_variance;
    if (tid == 0) {
        s_variance = variance / hidden_size;
    }
    __syncthreads();
    
    // Step 3: Normalize and apply weight/bias
    const float inv_std = rsqrtf(s_variance + eps);
    for(int i = tid; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        val = (val - s_mean) * inv_std;
        if(weight != nullptr && bias != nullptr) {
            val = val * static_cast<float>(weight[i]) + static_cast<float>(bias[i]);
        }
        row_output[i] = static_cast<T>(val);
    }
}

torch::Tensor layer_norm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {
    
    const auto batch_size = input.size(0);
    const auto hidden_size = input.size(1);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const dim3 blocks(batch_size);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "layer_norm_cuda", ([&] {
        LayerNormKernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            hidden_size,
            eps
        );
    }));
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        throw std::runtime_error(cudaGetErrorString(error));
    }
    
    return output;
}
