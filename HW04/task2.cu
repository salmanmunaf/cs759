#include <cuda.h>
#include <random>
#include <iostream>
#include "stencil.cuh"

const size_t NUM_PARAM = 4;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    const unsigned int N = atoi(argv[1]);
    const unsigned int R = atoi(argv[2]);
    const unsigned int THREADS_PER_BLOCK = atoi(argv[3]);

    //Generating a random numbers for array a and b
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    
    std::uniform_real_distribution<float> dist(-1., 1.);

    float* image = new float[N];
    float* output = new float[N];
    float* mask = new float[2*R+1];
    for (unsigned int i = 0; i < N; i++) {
        image[i] = dist(generator);
        output[i] = 0.;
    }

    for (unsigned int i = 0; i < 2*R+1; i++) {
        mask[i] = dist(generator);
    }

    float *d_image, *d_output, *d_mask;
    // // Allocate device memory for image, mask and output
    cudaMalloc((void**)&d_image, sizeof(float) * N);
    cudaMalloc((void**)&d_output, sizeof(float) * N);
    cudaMalloc((void**)&d_mask, sizeof(float) * (2*R+1));

    // Transfer data from host to device memory
    cudaMemcpy(d_image, image, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, sizeof(float) * (2*R+1), cudaMemcpyHostToDevice);

    //Setup timers and record the time it takes for stencil function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, N, R, THREADS_PER_BLOCK);
    cudaEventRecord(stop);
    
    //bring the result back from the GPU into the hostArray
    cudaMemcpy(output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << output[N-1] << std::endl;
    std::cout << milliseconds << std::endl;

    // Cleanup after kernel execution
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    delete [] image;
    delete [] mask;
    delete [] output;

    return 0;
}