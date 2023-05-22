#include <iostream>
#include <random>
#include <chrono>
#include "convolution.h"

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

    omp_set_num_threads(t);

    //Initialize random number generator for array A and B
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist_mask(-1., 1.);
    std::uniform_real_distribution<float> dist(-10., 10.);

    float* image = new float[N*N];
    float* output = new float[N*N];
    float* mask = new float[3*3];
    for (std::size_t i = 0; i < N*N; i++) {
        image[i] = dist_mask(generator);
        output[i] = 0;
    }
    for (std::size_t i = 0; i < 3*3; i++) {
        mask[i] = dist(generator);
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    
    start = high_resolution_clock::now();
    convolve(image, output, N, mask, 3);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << output[0] << std::endl;
    std::cout << output[N*N-1] << std::endl;
    std::cout << duration_msec.count() << std::endl;

    delete [] image;
    delete [] output;
    delete [] mask;
    return 0;
}