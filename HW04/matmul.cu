#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n*n) {
        // calculate row and column for matrix multiplication
        unsigned int i = index / n;
        unsigned int j = index % n;
        // Pvalue will store the result of per row by column multiplication
        float Pvalue = 0;
        for (int k = 0; k < n; k++) {
            float Melement = A[i * n + k];
            float Nelement = B[k * n + j];
            Pvalue += Melement * Nelement;
        }
        C[index] = Pvalue;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    const unsigned int NUM_BLOCKS = ((n*n) + threads_per_block-1) / threads_per_block;
    matmul_kernel<<<NUM_BLOCKS,threads_per_block>>>(A, B, C, n);
}