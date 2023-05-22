#include "montecarlo.h"
#include <random>

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
    int count = 0;
    #pragma omp parallel for simd reduction(+: count)
    // count number of points that lie inside circle
    for (std::size_t i = 0; i < n; i++) {
        count += ((x[i] * x[i] + y[i] * y[i]) <= (radius * radius));
    }
    return count;
}