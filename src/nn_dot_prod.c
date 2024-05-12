#include "nn_dot_prod.h"
#include "nn_debug.h"
#include "nn_error.h"
#include "nn_tensor.h"
#include <math.h>
#include <stddef.h>

NNTensorUnit nn_dot_prod(const NNTensor *a, const NNTensor *b, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with a.dims=%zu b.dims=%zu\n", __func__, a->dims, b->dims);

    if (a == NULL || b == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "a or b is NULL");
        return NAN;
    } else if (a->dims != 1 || b->dims != 1 || a->sizes[0] != b->sizes[0]) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 1-dimensional tensors of the same size are allowed");
        return false;
    }

    NNTensorUnit result = 0;

    // Calculate the dot product
    size_t size = a->sizes[0];
    for (size_t i = 0; i < size; i++) {
        result += a->data[i] * b->data[i];
    }

    return result;
}
