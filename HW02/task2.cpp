#include <chrono>
#include <iostream>
#include "convolution.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

const size_t NUM_PARAM = 3;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    float* image = new float[N*N];
    float* output = new float[N*N];
    float* mask = new float[M*M];
    for (int i = 0; i < N*N; i++) {
        image[i] = (rand() / float(RAND_MAX)) * 20.f - 10.f;
        output[i] = 0;
    }
    for (int i = 0; i < M*M; i++) {
        mask[i] = (rand() / float(RAND_MAX)) * 2.f - 1.f;
    }
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    start = high_resolution_clock::now();
    convolve(image, output, N, mask, M);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_msec.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[N*N-1] << std::endl;
    delete [] image;
    delete [] output;
    delete [] mask;
    return 0;
}