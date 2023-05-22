#include "scan.cuh"

__global__ void scan_kernel(const float *g_idata, float *g_odata, unsigned int n) {
    extern volatile __shared__ float temp[]; // allocated on invocation
    int thid = threadIdx.x;
    int pout = 0; 
    int pin = 1;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        temp[thid] = 0;
    } else {
        // load input into shared memory. 
        // **exclusive** scan: shift right by one element and set first output to 0
        temp[thid] = g_idata[i];
    }
    __syncthreads();
    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout; // swap double buffer indices
        pin  = 1 - pout;
        if (thid >= offset) {
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid] + temp[pin * blockDim.x + thid - offset];
        }
        else {
            temp[pout * blockDim.x + thid] = temp[pin * blockDim.x + thid];
        }
        __syncthreads(); // I need this here before I start next iteration 
    }

    if (i < n) {
        g_odata[i] = temp[pout * blockDim.x + thid]; // write output
    }
}

// copy and store block sum to a new array
__global__ void copy_kernel(float* input, float* output, unsigned int threads_per_block, unsigned int n) {
    unsigned int i = ((threadIdx.x + 1) * threads_per_block) - 1;
    if (i >= n) {
        i = n-1;
    }
    output[threadIdx.x] = input[i];
}

// add scanned block sum i to all values of scanned block i+1
__global__ void add_kernel(float* scanned_block_sum, float* scanned_blocks, float* output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (blockIdx.x > 0) {
        output[i] = scanned_blocks[i] + scanned_block_sum[blockIdx.x - 1];
    } else {
        output[i] = scanned_blocks[i];
    }
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    unsigned int num_blocks=(n + threads_per_block-1) / threads_per_block;
    float *scanned_blocks, *block_sum, *scanned_block_sum;
    cudaMallocManaged((void **)&scanned_blocks, n * sizeof(float));
    cudaMallocManaged((void **)&block_sum, num_blocks * sizeof(float));
    cudaMallocManaged((void **)&scanned_block_sum, num_blocks * sizeof(float));
    scan_kernel<<<num_blocks, threads_per_block, 2*threads_per_block*sizeof(float)>>>(input, scanned_blocks, n);
    copy_kernel<<<1, num_blocks>>>(scanned_blocks, block_sum, threads_per_block, n);
    scan_kernel<<<1, threads_per_block, 2*threads_per_block*sizeof(float)>>>(block_sum, scanned_block_sum, num_blocks);
    add_kernel<<<num_blocks, threads_per_block>>>(scanned_block_sum, scanned_blocks, output, n);
    cudaFree(scanned_blocks);
    cudaFree(block_sum);
    cudaFree(scanned_block_sum);
}