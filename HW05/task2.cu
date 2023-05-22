#include <cuda.h>
#include <random>
#include <iostream>
#include "matmul.cuh"

const size_t NUM_PARAM = 3;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    const unsigned int N = atoi(argv[1]);
    const unsigned int BLOCK_DIM = atoi(argv[2]);

    //Generating a random numbers for array a and b
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    
    std::uniform_real_distribution<float> dist(-10., 10.);

    int *A, *B, *C;
    cudaMallocManaged((void **)&A, N * N * sizeof(int));
    cudaMallocManaged((void **)&B, N * N * sizeof(int));
    cudaMallocManaged((void **)&C, N * N * sizeof(int));

    for (unsigned int i = 0; i < N*N; i++) {
        A[i] = int(dist(generator));
        B[i] = int(dist(generator));
        C[i] = 0;
    }

    //Setup timers and record the time it takes for reduce function
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_1(A, B, C, N, BLOCK_DIM);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << C[0] << std::endl;
    std::cout << C[N*N-1] << std::endl;
    std::cout << milliseconds<< std::endl;

    // Cleanup after kernel execution
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    std::uniform_real_distribution<float> distf(-1., 1.);

    float *Af, *Bf, *Cf;
    cudaMallocManaged((void **)&Af, N * N * sizeof(float));
    cudaMallocManaged((void **)&Bf, N * N * sizeof(float));
    cudaMallocManaged((void **)&Cf, N * N * sizeof(float));

    for (unsigned int i = 0; i < N*N; i++) {
        Af[i] = distf(generator);
        Bf[i] = distf(generator);
        Cf[i] = 0;
    }

    //Setup timers and record the time it takes for reduce function
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_2(Af, Bf, Cf, N, BLOCK_DIM);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << Cf[0] << std::endl;
    std::cout << Cf[N*N-1] << std::endl;
    std::cout << milliseconds<< std::endl;

    // Cleanup after kernel execution
    cudaFree(Af);
    cudaFree(Bf);
    cudaFree(Cf);

    std::uniform_real_distribution<double> distd(-1., 1.);

    double *Ad, *Bd, *Cd;
    cudaMallocManaged((void **)&Ad, N * N * sizeof(double));
    cudaMallocManaged((void **)&Bd, N * N * sizeof(double));
    cudaMallocManaged((void **)&Cd, N * N * sizeof(double));

    for (unsigned int i = 0; i < N*N; i++) {
        Ad[i] = distd(generator);
        Bd[i] = distd(generator);
        Cd[i] = 0;
    }

    //Setup timers and record the time it takes for reduce function
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul_3(Ad, Bd, Cd, N, BLOCK_DIM);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    // Get the elapsed time in milliseconds
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //print out the result to confirm that things are looking good
    std::cout << Cd[0] << std::endl;
    std::cout << Cd[N*N-1] << std::endl;
    std::cout << milliseconds<< std::endl;

    // Cleanup after kernel execution
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    return 0;
}