#include <chrono>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include "scan.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

const size_t NUM_PARAM = 2;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    srand(time(NULL));
    int N = atoi(argv[1]);
    float* input = new float[N];
    float* output = new float[N];
    for (int i = 0; i < N; i++) {
        input[i] = (rand() / float(RAND_MAX)) * 2.f - 1.f;
        output[i] = 0;
    }
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    start = high_resolution_clock::now();
    scan(input, output, N);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_msec.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[N-1] << std::endl;
    delete [] input;
    delete [] output;
    return 0;
}