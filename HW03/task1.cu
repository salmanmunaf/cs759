#include <cuda.h>
#include <iostream>

const size_t NUM_PARAM = 1;

__global__ void factorial(int* data, int N) {
    int index = threadIdx.x;
    if (index < N) {
        data[index] = index+1;
        for (int i = index; i > 0; i--) {
            data[index] *= i;
        }
        std::printf("%d!=%d\n", index+1, data[index]);
    }
}

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    int N = 8;
    int *d_a;
    // Allocate device memory for d_a
    cudaMalloc((void**)&d_a, sizeof(int) * N);
    //Initialize values to 0
    cudaMemset(d_a, 0, sizeof(int) * N);

    factorial<<<1,N>>>(d_a, N);
    //Wait for kernel to finish
    cudaDeviceSynchronize();
    // Cleanup after kernel execution
    cudaFree(d_a);

    return 0;
}