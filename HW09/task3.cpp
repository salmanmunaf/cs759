#include "mpi.h"
#include <iostream>
#include <random>
#include <cassert>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration;

const size_t NUM_PARAM = 2;

int main(int argc, char* argv[]) {
    if (argc != NUM_PARAM) {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    const std::size_t N = std::atoi(argv[1]);

    //Initialize random number generator for message buffers
    // std::random_device entropy_source;
    // std::mt19937 generator(entropy_source()); 
    // std::uniform_real_distribution<float> dist(0., 1.);

    float* msg1 = new float[N];
    float* msg2 = new float[N];
    // generate N random numbers
    for (std::size_t i = 0; i < N; i++) {
        msg1[i] = 1;
        msg2[i] = 2;
    }

    int my_rank;       /* rank of process      */
    int p;             /* number of processes  */
    duration<double, std::milli> t0;
    duration<double, std::milli> t1;
    MPI_Status status;        /* return status for receive  */
    MPI_Init(&argc, &argv); // Start up MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // Find out process rank
    MPI_Comm_size(MPI_COMM_WORLD, &p); // Find out number of processes
    if(my_rank == 0) {
        //Setup timers and record the time it takes for send and recv function
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;
    
        start = high_resolution_clock::now();
        MPI_Send(msg1, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(msg2, N, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
        end = high_resolution_clock::now();
        t0 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        // std::cout << "rank0 - t0: " << t0.count() << std::endl;
        MPI_Send(&t0, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
    } else { 
        /* my_rank== 1 */
        //Setup timers and record the time it takes for send and recv function
        high_resolution_clock::time_point start;
        high_resolution_clock::time_point end;
    
        start = high_resolution_clock::now();
        MPI_Recv(msg1, N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Send(msg2, N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        end = high_resolution_clock::now();
        t1 = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        for (std::size_t i = 0; i < N; i++) {
            assert(msg1[i] == 1);
            assert(msg2[i] == 2);
        }
        MPI_Recv(&t0, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
        // std::cout << "rank1 - t0: " << t0.count() << std::endl;
        // std::cout << "rank1 - t1: " << t1.count() << std::endl;
        std::cout << (t0.count() + t1.count()) << std::endl;
    }
    MPI_Finalize(); // Shut down MPI
    return 0;
}