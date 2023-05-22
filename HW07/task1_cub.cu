#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <random>
#include <cmath>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

const size_t NUM_PARAM = 2;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    const size_t num_items = atoi(argv[1]);

    //Initialize random number generator for array h_in
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1., 1.);

    // Set up host arrays
    float* h_in = new float[num_items];
    for (unsigned int i = 0; i < num_items; i++) {
        h_in[i] = dist(generator);
    }

    // Set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * num_items));
    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    // Setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));
    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // Do the actual reduce operation
    //Setup timers and record the time it takes for reduce function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items));
    DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, num_items);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float) * 1, cudaMemcpyDeviceToHost));
    // Print out results
    std::cout << gpu_sum << std::endl;
    std::cout << milliseconds << std::endl;

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}