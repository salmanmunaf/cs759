#include "mmul.h"
#include <iostream>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t stat;
    //Call cuBLAS library function with appropriate parameters and print message if a call fails
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::printf("CUBLAS matrix multiplication failed\n");
        return;
    }
}