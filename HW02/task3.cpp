#include <chrono>
#include <iostream>
#include "matmul.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

const size_t NUM_PARAM = 1;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    int N = (rand() % 1000) + 1000;
    std::cout << N << std::endl;
    double* A = new double[N*N];
    double* B = new double[N*N];
    double* C = new double[N*N];
    std::vector<double> A_vec;
    std::vector<double> B_vec;

    for (int i = 0; i < N*N; i++) {
        A[i] = (rand() / float(RAND_MAX));
        B[i] = (rand() / float(RAND_MAX));
        A_vec.push_back(A[i]);
        B_vec.push_back(B[i]);
        C[i] = 0;
    }

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    
    start = high_resolution_clock::now();
    mmul1(A, B, C, N);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_msec.count() << std::endl;
    std::cout << C[N*N-1] << std::endl;

    for (int i = 0; i < N*N; i++) {
        C[i] = 0;
    }
    start = high_resolution_clock::now();
    mmul2(A, B, C, N);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_msec.count() << std::endl;
    std::cout << C[N*N-1] << std::endl;
    
    for (int i = 0; i < N*N; i++) {
        C[i] = 0;
    }
    start = high_resolution_clock::now();
    mmul3(A, B, C, N);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_msec.count() << std::endl;
    std::cout << C[N*N-1] << std::endl;
    
    for (int i = 0; i < N*N; i++) {
        C[i] = 0;
    }
    start = high_resolution_clock::now();
    mmul4(A_vec, B_vec, C, N);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    std::cout << duration_msec.count() << std::endl;
    std::cout << C[N*N-1] << std::endl;
    delete [] A;
    delete [] B;
    delete [] C;
    return 0;
}