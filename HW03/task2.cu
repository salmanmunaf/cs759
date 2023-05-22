#include <cuda.h>
#include <random>
#include <iostream>

const size_t NUM_PARAM = 1;

__global__ void add(int* data, int a, int N) {
    //this adds a value to a variable stored in global memory
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        data[index] = a * threadIdx.x + blockIdx.x;
    }
}

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    int N = 16;
    int THREADS_PER_BLOCK = 8;
    int *dA;
    int hA[N];
    // Allocate device memory for d_a
    cudaMalloc((void**)&dA, sizeof(int) * N);
    //Initialize values to 0
    cudaMemset(dA, 0, sizeof(int) * N);

    //Generating a random number between 1 and 20 for variable: a
    std::random_device entropy_source;
	std::mt19937 generator(entropy_source()); 

	std::uniform_real_distribution<float> dist(1., 20.);
    int a = int(dist(generator));

    add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(dA, a, N);

    //bring the result back from the GPU into the hostArray
    cudaMemcpy(&hA, dA, sizeof(int) * N, cudaMemcpyDeviceToHost);
    //print out the result to confirm that things are looking good 
    for(int i = 0; i < N; i++) {
        std::cout<< hA[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup after kernel execution
    cudaFree(dA);

    return 0;
}