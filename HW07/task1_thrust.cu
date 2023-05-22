#include <iostream>
#include <random>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>

const size_t NUM_PARAM = 2;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    const unsigned int N = atoi(argv[1]);

    //Initialize random number generator for array A and B
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1., 1.);

    // generate N random numbers on the host
    thrust::host_vector<float> h_vec(N);
    for (unsigned int i = 0; i < N; i++) {
        h_vec[i] = dist(generator);
    }
    // transfer data to the device
    thrust::device_vector<float> d_vec = h_vec;
    //Setup timers and record the time it takes for reduce function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // reduce data on the device
    float sum = thrust::reduce(thrust::cuda::par, d_vec.begin(), d_vec.end(), 0.0, thrust::plus<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //print out the result to confirm that things are looking good
    std::cout << sum << std::endl;
    std::cout << milliseconds << std::endl;
    return 0;
}