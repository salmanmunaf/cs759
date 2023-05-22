#include <iostream>
#include <random>
#include <cmath>
#include "count.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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
    std::uniform_real_distribution<float> dist(0., 501.);

    // generate N random numbers on the host
    thrust::host_vector<int> h_in(N);
    for (unsigned int i = 0; i < N; i++) {
        h_in[i] = int(dist(generator));
    }
    // transfer data to the device
    const thrust::device_vector<int> d_in = h_in;
    thrust::device_vector<int> values(N);
    thrust::device_vector<int> counts(N);
    //Setup timers and record the time it takes for reduce function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // call count function
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //print out the result to confirm that things are looking good
    std::cout << values.back() << std::endl;
    std::cout << counts.back() << std::endl;
    std::cout << milliseconds << std::endl;
    return 0;
}