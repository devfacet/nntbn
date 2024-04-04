#include "nn_debug.h"
#include <stddef.h>

// nn_dot_product_neon calculates the dot product of two vectors.
float nn_dot_product_generic(const float *a, const float *b, size_t length) {
    NN_DEBUG_PRINT(5, "function %s called with length = %zu\n", __func__, length);

    // Initialize vector sum to 0
    float result = 0.0f;

    // Iterate over the elements of the vectors
    for (size_t i = 0; i < length; ++i) {
        result += a[i] * b[i];
    }

    return result;
}
