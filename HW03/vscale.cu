#include "vscale.cuh"
#include <iostream>

__global__ void vscale(const float *a, float *b, unsigned int n) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        b[index] = a[index] * b[index];
    }
}