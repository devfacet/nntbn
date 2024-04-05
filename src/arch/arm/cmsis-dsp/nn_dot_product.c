#include "arch/arm/cmsis-dsp/nn_dot_product.h"
#include "arm_math.h"
#include "nn_debug.h"
#include <stddef.h>

// nn_dot_product_neon calculates the dot product of two vectors.
float nn_dot_product_cmsis(const float *a, const float *b, size_t length) {
    NN_DEBUG_PRINT(5, "function %s called with length = %zu\n", __func__, length);

    float result = 0.0f;

    // CMSIS-DSP provides arm_dot_prod_f32 for Cortex-M cores with FPU
    arm_dot_prod_f32(a, b, length, &result);

    return result;
}
