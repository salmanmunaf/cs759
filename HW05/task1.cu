#include <cuda.h>
#include <random>
#include <iostream>
#include "reduce.cuh"

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

    float* input = new float[N];
    float output;
    for (unsigned int i = 0; i < N; i++) {
        input[i] = dist(generator);
    }

    float *d_input, *d_output;
    // Allocate device memory for input and output
    cudaMalloc((void**)&d_input, sizeof(float) * N);
    cudaMalloc((void**)&d_output, sizeof(float) * ((N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK));

    // Transfer data from host to device memory
    cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);

    //Setup timers and record the time it takes for reduce function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce(&d_input, &d_output, N, THREADS_PER_BLOCK);
    cudaEventRecord(stop);
    
    //bring the result back from the GPU into the hostArray
    cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << output << std::endl;
    std::cout << milliseconds<< std::endl;

    // Cleanup after kernel execution
    cudaFree(d_input);
    cudaFree(d_output);

    delete [] input;

    return 0;
}