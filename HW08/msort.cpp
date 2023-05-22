#include <algorithm>
#include "msort.h"

void bubble_sort(int* arr, const std::size_t start, const std::size_t end) {
    for (std::size_t i = start; i < end; i++) {
        for (std::size_t j = i+1; j < end; j++) {
            if (arr[j] < arr[i]) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
    }
}

void msort_helper(int* arr, const std::size_t start, const std::size_t end, const std::size_t threshold) {
    if (end - start < 2) {
        return;
    }
    
    if (end - start <= threshold) {
        bubble_sort(arr, start, end);
        return;
    }
    int mid = start + (end - start) / 2;
    #pragma omp task 
    {
        msort_helper(arr, start, mid, threshold);
    }
    
    #pragma omp task 
    {
        msort_helper(arr, mid, end, threshold);
    }
    #pragma omp taskwait

    std::inplace_merge(arr+start, arr+mid, arr+end);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            msort_helper(arr, 0, n, threshold);
        }
    }
}