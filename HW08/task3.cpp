#include <iostream>
#include <random>
#include <chrono>
#include "msort.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

const size_t NUM_PARAM = 4;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    const std::size_t N = std::atoi(argv[1]);
    const unsigned int t = std::atoi(argv[2]);
    const unsigned int threshold = std::atoi(argv[3]);

    omp_set_num_threads(t);

    //Initialize random number generator for array A and B
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1001., 1001.);

    int* input = new int[N];
    for (std::size_t i = 0; i < N; i++) {
        input[i] = int(dist(generator));
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    
    start = high_resolution_clock::now();
    msort(input, N, threshold);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << input[0] << std::endl;
    std::cout << input[N-1] << std::endl;
    std::cout << duration_msec.count() << std::endl;

    delete [] input;
    return 0;
}