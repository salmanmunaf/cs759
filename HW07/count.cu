#include "count.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts) {
    thrust::device_vector<int> d_values(d_in.size());
    thrust::fill(d_values.begin(), d_values.end(), 1);
    thrust::device_vector<int> d_in_sorted = d_in;
    thrust::sort(d_in_sorted.begin(), d_in_sorted.end());
    auto new_end = thrust::reduce_by_key(thrust::cuda::par, d_in_sorted.begin(), d_in_sorted.end(), 
    d_values.begin(), values.begin(), counts.begin());
    values.resize(new_end.first - values.begin());
    counts.resize(new_end.second - counts.begin());
}