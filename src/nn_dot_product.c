#include "nn_dot_product.h"
#include "nn_debug.h"
#include <stddef.h>

// nn_dot_product_neon calculates the dot product of two vectors.
float nn_dot_product(const float a[NN_DOT_PRODUCT_MAX_VECTOR_SIZE], const float b[NN_DOT_PRODUCT_MAX_VECTOR_SIZE], size_t vector_size) {
    NN_DEBUG_PRINT(5, "function %s called with vector_size = %zu\n", __func__, vector_size);

    // Initialize vector sum to 0
    float result = 0.0f;

    // Iterate over the elements of the vectors
    for (size_t i = 0; i < vector_size; ++i) {
        result += a[i] * b[i];
    }

    return result;
}
