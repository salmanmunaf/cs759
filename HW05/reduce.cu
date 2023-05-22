#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned int tid = threadIdx.x;
    extern __shared__ float ShMem[];
    // each thread loads two elements from global to shared mem
    ShMem[tid] = g_idata[i];
    if (i+blockDim.x < n) {
        ShMem[tid] += g_idata[i+blockDim.x];
    }
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s=(blockDim.x+1)/2; s>0; s>>=1) {
        if (tid < s && (tid+s) < (blockDim.x * 2) && (i+s) < n) {
            ShMem[tid] += ShMem[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = ShMem[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block) {
    for (unsigned int i = N; i > 1; i=(i + (threads_per_block*2)-1) / (threads_per_block*2)) {
        unsigned int num_blocks=(i + threads_per_block-1) / threads_per_block;
        reduce_kernel<<<num_blocks, threads_per_block, threads_per_block*sizeof(float)>>>(*input, *output, i);
        cudaMemcpy(*input, *output, num_blocks * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}