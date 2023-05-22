#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    #pragma omp parallel for collapse(2)
    for (size_t x = 0; x < n; x++) {
        for (size_t y = 0; y < n; y++) {
            float result = 0;
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < m; j++) {
                    size_t image_i = x + i - (m-1)/2;
                    size_t image_j = y + j - (m-1)/2;
                    float f = 0;
                    if (image_i >= 0 && image_i < n && image_j >= 0 && image_j < n) {
                        f = image[image_i * n + image_j];
                    }
                    else if ((image_i >= 0 && image_i < n) || (image_j >= 0 && image_j < n)) {
                        f = 1;
                    }
                    else if (image_i < 0 || image_i >= n || image_j < 0 || image_j >= n) {
                        f = 0;
                    }
                    result += f * mask[i * m + j];
                }
            }
            output[x * n + y] = result;
        }
    }
}