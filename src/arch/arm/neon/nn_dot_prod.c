#include "arch/arm/neon/nn_dot_prod.h"
#include "nn_debug.h"
#include "nn_error.h"
#include "nn_tensor.h"
#include <arm_neon.h>
#include <math.h>
#include <stddef.h>

NNTensorUnit nn_dot_prod_neon(const NNTensor *a, const NNTensor *b, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with a.dims=%zu b.dims=%zu\n", __func__, a->dims, b->dims);

    if (a == NULL || b == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "a or b is NULL");
        return NAN;
    } else if (a->dims != 1 || b->dims != 1 || a->sizes[0] != b->sizes[0]) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 1-dimensional tensors of the same size are allowed");
        return false;
    }

    // TODO: Implement other data types based in the tensor data type
    // NEON SIMD registers are 128 bits wide, which can hold 4 float32 values
    float32x4_t sumVec = vdupq_n_f32(0.0);
    size_t i;
    size_t size = a->sizes[0];
    for (i = 0; i + 3 < size; i += 4) {
        float32x4_t aVec = vld1q_f32(a->data + i);   // load 4 elements from a
        float32x4_t bVec = vld1q_f32(b->data + i);   // load 4 elements from b
        float32x4_t prodVec = vmulq_f32(aVec, bVec); // multiply elements
        sumVec = vaddq_f32(sumVec, prodVec);         // add to sum
    }

    // Reduce the sum vector to a single sum value
    NNTensorUnit result = vgetq_lane_f32(sumVec, 0) + vgetq_lane_f32(sumVec, 1) + vgetq_lane_f32(sumVec, 2) + vgetq_lane_f32(sumVec, 3);

    // Handle remaining elements
    for (; i < size; i++) {
        result += a->data[i] * b->data[i];
    }

    return result;
}
