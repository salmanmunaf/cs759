#include <iostream>
#include <random>
#include <chrono>
#include "matmul.h"

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
    std::uniform_real_distribution<float> dist(-1., 1.);

    float* A = new float[N*N];
    float* B = new float[N*N];
    float* C = new float[N*N];

    for (std::size_t i = 0; i < N*N; i++) {
        A[i] = dist(generator);
        B[i] = dist(generator);
        C[i] = 0;
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    
    start = high_resolution_clock::now();
    mmul(A, B, C, N);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << C[0] << std::endl;
    std::cout << C[N*N-1] << std::endl;
    std::cout << duration_msec.count() << std::endl;

    delete [] A;
    delete [] B;
    delete [] C;
    return 0;
}