#include <cuda_runtime.h>

__global__ void dropout_kernel(const float* input, const float* mask, float* output, float p, int N) {
    // Calculate the global index for the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check boundaries to avoid out-of-bounds memory access
    if (i < N) {
        // Calculate the inverted-dropout rescale factor
        float scale = 1.0f / (1.0f - p);
        
        // Apply the mask and the scaling factor
        output[i] = input[i] * mask[i] * scale;
    }
}

extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Launch the kernel
    dropout_kernel<<<blocks, threads>>>(input, mask, output, p, N);
    cudaDeviceSynchronize();
}