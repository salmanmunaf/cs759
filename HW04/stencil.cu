#include "stencil.cuh"
#include <iostream>

__host__ void stencil(const float* image, const float* mask, float* output, 
                    unsigned int n, unsigned int R, unsigned int threads_per_block) {
    const unsigned int num_blocks = (n + threads_per_block-1) / threads_per_block;
    stencil_kernel<<<num_blocks, threads_per_block, (4*R + 1 + 2*threads_per_block)*sizeof(float)>>>(image, mask, output, n, R);
}

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) {
        return;
    }
    extern __shared__ float ShMem[];
    // split shared memory for image, mask and output
    float *shared_mask = ShMem;
    float *shared_output = (float*)&shared_mask[2*R+1];
    float *shared_image = (float*)&shared_output[blockDim.x];
    // threads with index < 2R+1 will allocate values for shared mask
    if (threadIdx.x < (2*R+1)) {
        shared_mask[threadIdx.x] = mask[threadIdx.x];
    }

    int lindex = threadIdx.x + R;
    shared_image[lindex] = image[i];
    // check bounds of the image and allocate 1 if out of bounds
    if (threadIdx.x < R) {
        shared_image[lindex - R] = ((int)(i - R) >= 0) ? image[i - R] : 1;
    } else {
        shared_image[lindex + R] = ((int)(i + R) < n) ? image[i + R] : 1;
    }

    __syncthreads();
    // perform 1d convolution and store result in shared memory
    for (int j = 0 ; j <= 2*R ; j++) {
        shared_output[threadIdx.x] += shared_image[lindex + j - R] * shared_mask[j];
    }
    output[i] = shared_output[threadIdx.x];
}