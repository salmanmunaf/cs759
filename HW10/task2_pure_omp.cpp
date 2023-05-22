#include <iostream>
#include <random>
#include <chrono>
#include "reduce.h"

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

    if (t < 1 || t > 20) {
        std::cerr << "Invalid value of t" << std::endl;
        return 1;
    }

    omp_set_num_threads(t);

    //Initialize random number generator for array arr
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1., 1.);

    float* arr = new float[N];
    // generate N random numbers
    for (std::size_t i = 0; i < N; i++) {
        arr[i] = dist(generator);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    start = high_resolution_clock::now();
    float res = reduce(arr, 0, N);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //Print results
    std::cout << res << std::endl;
    std::cout << duration_msec.count() << std::endl;

    delete [] arr;
    return 0;
}