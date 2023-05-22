#include <iostream>
#include <random>
#include <chrono>
#include "montecarlo.h"

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
    const float radius = 1.0;

    if (t < 1 || t > 10) {
        std::cerr << "Invalid value of t" << std::endl;
        return 1;
    }

    omp_set_num_threads(t);

    //Initialize random number generator for array arr
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-radius, radius);

    float* x = new float[N];
    float* y = new float[N];
    // generate N random numbers
    for (std::size_t i = 0; i < N; i++) {
        x[i] = dist(generator);
        y[i] = dist(generator);
    }

    //Setup timers and record the time it takes for montecarlo function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    
    start = high_resolution_clock::now();
    int count = montecarlo(N, x, y, radius);
    end = high_resolution_clock::now();
    // Get the elapsed time in milliseconds
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Calculate pi
    float pi = ((float)(4 * count)) / ((float)N);
    std::cout << pi << std::endl;
    std::cout << duration_msec.count() << std::endl;

    delete [] x;
    delete [] y;
    return 0;
}