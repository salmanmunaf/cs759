#include <cuda.h>
#include <random>
#include <iostream>
#include "scan.cuh"

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

    // Initialize and allocate device memory for input and output
    float *input, *output;
    cudaMallocManaged((void **)&input, N * sizeof(float));
    cudaMallocManaged((void **)&output, N * sizeof(float));

    for (unsigned int i = 0; i < N; i++) {
        input[i] = dist(generator);
        output[i] = 0;
    }

    //Setup timers and record the time it takes for reduce function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    scan(input, output, N, THREADS_PER_BLOCK);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << output[N-1] << std::endl;
    std::cout << milliseconds << std::endl;

    // Cleanup after kernel execution
    cudaFree(input);
    cudaFree(output);

    return 0;
}