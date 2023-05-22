#include "matmul.cuh"
#include <iostream>

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    const unsigned int BLOCK_SIZE = blockDim.x;
    // Block index
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;
    // Thread index
    const unsigned int tx = threadIdx.x; 
    const unsigned int ty = threadIdx.y;
    // Calculate row and column
    const unsigned int Row = by * blockDim.y + ty;
    const unsigned int Col = bx * blockDim.x + tx;
    
    T Pvalue = 0;

    extern __shared__ char ShMem[];
    T *s_data = reinterpret_cast<T *>(ShMem);
    
    // Shared memory for the sub-matrices (tiles) of  A and B
    T *As = s_data;
    T *Bs = (T*)&As[BLOCK_SIZE*BLOCK_SIZE];

    // Loop over the A and B tiles required to compute the C element
    for (unsigned int p = 0; p < ((n-1)/BLOCK_SIZE)+1; ++p) {
        // Collaborative loading of A and B tiles into shared memory
        if(Row < n && (p * BLOCK_SIZE+tx) < n) {
            As[ty * BLOCK_SIZE + tx] = A[Row*n + p*BLOCK_SIZE+tx];
        } else {
            As[ty * BLOCK_SIZE + tx] = 0;
        }
        if ((p*BLOCK_SIZE+ty) < n && Col < n) {
            Bs[ty * BLOCK_SIZE + tx] = B[(p*BLOCK_SIZE+ty)*n + Col];
        } else {
            Bs[ty * BLOCK_SIZE + tx] = 0;
        }
        __syncthreads();
        if (Row < n && Col < n) {
            for (unsigned int i = 0; i < BLOCK_SIZE; ++i) {
                Pvalue += As[ty * BLOCK_SIZE + i] * Bs[i * BLOCK_SIZE + tx];
            }
        }
        __syncthreads();
    }
    if (Row < n && Col < n) {
        C[Row*n+Col] = Pvalue;
    }
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim) {
    // Compute the execution configuration
    const unsigned int num_blocks = (n + block_dim - 1) / block_dim;
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(num_blocks, num_blocks);
    const unsigned int shared_mem = (block_dim * block_dim) * sizeof(int) + (block_dim * block_dim) * sizeof(int);
    // Launch the device computation
    matmul_kernel<int><<<dimGrid, dimBlock, shared_mem>>>(A, B, C, n);
}
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim) {
    // Compute the execution configuration
    const unsigned int num_blocks = (n + block_dim - 1) / block_dim;
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(num_blocks, num_blocks);
    const unsigned int shared_mem = (block_dim * block_dim) * sizeof(float) + (block_dim * block_dim) * sizeof(float);
    // Launch the device computation
    matmul_kernel<float><<<dimGrid, dimBlock, shared_mem>>>(A, B, C, n);
    return;
}
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim) {
    // Compute the execution configuration
    const unsigned int num_blocks = (n + block_dim - 1) / block_dim;
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid(num_blocks, num_blocks);
    const unsigned int shared_mem = (block_dim * block_dim) * sizeof(double) + (block_dim * block_dim) * sizeof(double);
    // Launch the device computation
    matmul_kernel<double><<<dimGrid, dimBlock, shared_mem>>>(A, B, C, n);
}