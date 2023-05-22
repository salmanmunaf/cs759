#include <omp.h>
#include <iostream>

const size_t NUM_PARAM = 1;

#define NUM_THREADS 4

int factorial(int n) {
    if (n <= 1) {
        return n;
    }

    return n * factorial(n-1);
}

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    omp_set_num_threads(NUM_THREADS);
    std::printf("Number of threads: %d\n", NUM_THREADS);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::printf("I am thread No. %d\n", tid);        
    }

    #pragma omp parallel for
    for (int i = 1; i <= 8; i++) {
        std::printf("%d!=%d\n", i, factorial(i));
    }
    
    return 0;
}