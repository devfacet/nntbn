#include "arch/arm/neon/nn_dot_product.h"
#include "nn_debug.h"
#include <arm_neon.h>
#include <stddef.h>

// nn_dot_product_neon calculates the dot product of two vectors.
float nn_dot_product_neon(const float *a, const float *b, size_t length) {
    NN_DEBUG_PRINT(5, "function %s called with length = %zu\n", __func__, length);

    // Initialize vector sum to 0
    float32x4_t sumVec = vdupq_n_f32(0.0);

    // Process 4 elements at a time using NEON SIMD instructions
    // NEON SIMD registers are 128 bits wide, which can hold 4 float32 values
    size_t i;
    for (i = 0; i + 3 < length; i += 4) {
        float32x4_t aVec = vld1q_f32(a + i);         // load 4 elements from a
        float32x4_t bVec = vld1q_f32(b + i);         // load 4 elements from b
        float32x4_t prodVec = vmulq_f32(aVec, bVec); // multiply elements
        sumVec = vaddq_f32(sumVec, prodVec);         // add to sum
    }

    // Reduce the sum vector to a single sum value
    float result = vgetq_lane_f32(sumVec, 0) + vgetq_lane_f32(sumVec, 1) + vgetq_lane_f32(sumVec, 2) + vgetq_lane_f32(sumVec, 3);

    // Handle remaining elements
    for (; i < length; ++i) {
        result += a[i] * b[i];
    }

    return result;
}
