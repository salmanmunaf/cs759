#include <cuda.h>
#include <random>
#include <iostream>
#include "vscale.cuh"

const size_t NUM_PARAM = 2;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    const unsigned int N = atoi(argv[1]);
    const unsigned int THREADS_PER_BLOCK = 512;

    //Generating a random numbers for array a and b
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    
    std::uniform_real_distribution<float> dist_a(-10., 10.);
    std::uniform_real_distribution<float> dist_b(0., 1.);

    float* a = new float[N];
    float* b = new float[N];
    for (unsigned int i = 0; i < N; i++) {
        a[i] = dist_a(generator);
        b[i] = dist_b(generator);
    }

    float *d_a, *d_b;
    // Allocate device memory for d_a and d_b
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    const unsigned int NUM_BLOCKS = (N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    //Setup timers and record the time it takes for vscale kernel
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vscale<<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(d_a, d_b, N);
    cudaEventRecord(stop);

    //bring the result back from the GPU into the hostArray
    cudaMemcpy(b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << milliseconds << std::endl;
    std::cout << b[0] << std::endl;
    std::cout << b[N-1] << std::endl;

    // Cleanup after kernel execution
    cudaFree(d_a);
    cudaFree(d_b);

    delete [] a;
    delete [] b;

    return 0;
}