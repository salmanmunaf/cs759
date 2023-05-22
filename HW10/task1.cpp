#include <iostream>
#include <chrono>
#include "optimize.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

const size_t NUM_PARAM = 2;
const unsigned int NUM_ITERATIONS = 10;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    const std::size_t N = std::atoi(argv[1]);

    vec arr = vec(N);
    data_t* data = new data_t[N];

    // generate N random numbers
    for (std::size_t i = 0; i < N; i++) {
        data[i] = i;
    }

    arr.data = data;

    data_t dest;

    //Setup timers and record the time it takes for cluster function
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    
    start = high_resolution_clock::now();
    for (unsigned int i = 0 ; i < NUM_ITERATIONS; i++) {
        optimize1(&arr, &dest);
    }
    end = high_resolution_clock::now();
    // Get the elapsed time in milliseconds
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Print results
    std::cout << dest << std::endl;
    std::cout << duration_msec.count()/NUM_ITERATIONS << std::endl;

    start = high_resolution_clock::now();
    for (unsigned int i = 0 ; i < NUM_ITERATIONS; i++) {
        optimize2(&arr, &dest);
    }
    end = high_resolution_clock::now();
    // Get the elapsed time in milliseconds
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Print results
    std::cout << dest << std::endl;
    std::cout << duration_msec.count()/NUM_ITERATIONS << std::endl;

    start = high_resolution_clock::now();
    for (unsigned int i = 0 ; i < NUM_ITERATIONS; i++) {
        optimize3(&arr, &dest);
    }
    end = high_resolution_clock::now();
    // Get the elapsed time in milliseconds
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Print results
    std::cout << dest << std::endl;
    std::cout << duration_msec.count()/NUM_ITERATIONS << std::endl;

    start = high_resolution_clock::now();
    for (unsigned int i = 0 ; i < NUM_ITERATIONS; i++) {
        optimize4(&arr, &dest);
    }
    end = high_resolution_clock::now();
    // Get the elapsed time in milliseconds
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Print results
    std::cout << dest << std::endl;
    std::cout << duration_msec.count()/NUM_ITERATIONS << std::endl;

    start = high_resolution_clock::now();
    for (unsigned int i = 0 ; i < NUM_ITERATIONS; i++) {
        optimize5(&arr, &dest);
    }
    end = high_resolution_clock::now();
    // Get the elapsed time in milliseconds
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    // Print results
    std::cout << dest << std::endl;
    std::cout << duration_msec.count()/NUM_ITERATIONS << std::endl;

    delete [] data;
    return 0;
}