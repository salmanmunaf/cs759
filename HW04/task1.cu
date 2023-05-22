#include <cuda.h>
#include <random>
#include <iostream>
#include "matmul.cuh"

const size_t NUM_PARAM = 3;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    const unsigned int N = atoi(argv[1]);
    const unsigned int THREADS_PER_BLOCK = atoi(argv[2]);

    //Generating a random numbers for array a and b
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    
    std::uniform_real_distribution<float> dist(-1., 1.);

    float* a = new float[N*N];
    float* b = new float[N*N];
    float* c = new float[N*N];
    for (unsigned int i = 0; i < N*N; i++) {
        a[i] = dist(generator);
        b[i] = dist(generator);
        c[i] = 0;
    }

    float *d_a, *d_b, *d_c;
    // Allocate device memory for d_a, d_b and d_c
    cudaMalloc((void**)&d_a, sizeof(float) * N * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N * N);
    cudaMalloc((void**)&d_c, sizeof(float) * N * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    //Setup timers and record the time it takes for matmul function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul(d_a, d_b, d_c, N, THREADS_PER_BLOCK);
    cudaEventRecord(stop);

    //bring the result back from the GPU into the hostArray
    cudaMemcpy(c, d_c, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << c[N*N-1] << std::endl;
    std::cout << milliseconds << std::endl;

    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete [] a;
    delete [] b;
    delete [] c;

    return 0;
}