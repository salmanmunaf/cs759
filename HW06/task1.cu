#include <cuda.h>
#include <random>
#include <iostream>
#include "mmul.h"
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

const size_t NUM_PARAM = 3;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    const unsigned int N = atoi(argv[1]);
    const unsigned int n_tests = atoi(argv[2]);

    //Initialize random number generator for array A and B
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1., 1.);

    // Initialize and allocate device memory for input and output
    float *A, *B, *C;
    cudaMallocManaged((void **)&A, N * N * sizeof(float));
    cudaMallocManaged((void **)&B, N * N * sizeof(float));
    cudaMallocManaged((void **)&C, N * N * sizeof(float));

    //Generating random numbers for matrix A and B
    for (unsigned int j = 0; j < N; j++) {
        for (unsigned int i = 0; i < N; i++) {
            A[IDX2C(i,j,N)] = dist(generator);
            B[IDX2C(i,j,N)] = dist(generator);
            C[IDX2C(i,j,N)] = 0;
        }
    }

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    //Setup timers and record the time it takes for reduce function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //Call the mmul function n_test times
    for (unsigned int i = 0; i < n_tests; i++) {
        mmul(handle, A, B, C, N);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << milliseconds/n_tests << std::endl;

    // Cleanup after kernel execution
    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}