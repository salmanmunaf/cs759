#include <iostream>
#include <random>
#include <chrono>
#include "mpi.h"
#include "reduce.h"

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

    if (t < 1 || t > 20) {
        std::cerr << "Invalid value of t" << std::endl;
        return 1;
    }

    omp_set_num_threads(t);

    //Initialize random number generator for array arr
    std::random_device entropy_source;
    std::mt19937 generator(entropy_source()); 
    std::uniform_real_distribution<float> dist(-1., 1.);

    float* arr = new float[N];
    // generate N random numbers
    for (std::size_t i = 0; i < N; i++) {
        arr[i] = dist(generator);
    }

    int my_rank;       /* rank of process      */
    // int p;             /* number of processes  */
    MPI_Init(&argc, &argv); // Start up MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Find out process rank
    // MPI_Comm_size(MPI_COMM_WORLD, &p); // Find out number of processes
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;
    float global_res;
    MPI_Barrier(MPI_COMM_WORLD);
    start = high_resolution_clock::now();
    float res = reduce(arr, 0, N);
    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    end = high_resolution_clock::now();
    duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    //Print results
    if (my_rank == 0) {
        std::cout << global_res << std::endl;
        std::cout << duration_msec.count() << std::endl;
    }

    delete [] arr;
    return 0;
}