#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include "cluster.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

const size_t NUM_PARAM = 3;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    const std::size_t N = std::atoi(argv[1]);
    const unsigned int t = std::atoi(argv[2]);

    if (t < 1 || t > 10) {
        std::cerr << "Invalid value of t" << std::endl;
        return 1;
    }

    omp_set_num_threads(t);

    //Initialize random number generator for array arr
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(0., (float) N);

    float* arr = new float[N];
    // generate N random numbers
    for (std::size_t i = 0; i < N; i++) {
        arr[i] = dist(generator);
    }
    
    //sort arr
    std::sort(arr, arr + N);

    float* centers = new float[t];
    float* dists = new float[t];
    for (std::size_t i = 0; i < t; i++) {
        centers[i] = ((2*(i+1) - 1) * N) / (2 * t);
        dists[i] = 0;
    }
    //Setup timers and record the time it takes for cluster function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    
    start = high_resolution_clock::now();
    cluster(N, t, arr, centers, dists);
    end = high_resolution_clock::now();
    // Get the elapsed time in milliseconds
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Get the max distance
    float max = *std::max_element(dists,dists+t);
    // Get the index for the max value
    int index = std::find(dists, dists+t, max) - dists;
    std::cout << max << std::endl;
    std::cout << index << std::endl;
    std::cout << duration_msec.count() << std::endl;

    delete [] arr;
    delete [] centers;
    delete [] dists;
    return 0;
}