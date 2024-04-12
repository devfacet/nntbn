#include "arch/arm/cmsis-dsp/nn_dot_product.h"
#include "arm_math.h"
#include "nn_debug.h"
#include <stddef.h>

// nn_dot_product_neon calculates the dot product of two vectors.
float nn_dot_product_cmsis(const float a[NN_DOT_PRODUCT_MAX_VECTOR_SIZE], const float b[NN_DOT_PRODUCT_MAX_VECTOR_SIZE], size_t vector_size) {
    NN_DEBUG_PRINT(5, "function %s called with vector_size = %zu\n", __func__, vector_size);

    float result = 0.0f;

    // CMSIS-DSP provides arm_dot_prod_f32 for Cortex-M cores with FPU
    arm_dot_prod_f32(a, b, vector_size, &result);

    return result;
}
