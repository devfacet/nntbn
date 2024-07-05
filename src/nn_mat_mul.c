#include "nn_mat_mul.h"
#include "nn_debug.h"
#include "nn_error.h"
#include "nn_tensor.h"
#include <stddef.h>

bool nn_mat_mul(const NNTensor *a, const NNTensor *b, NNTensor *result, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with a.dims=%zu b.dims=%zu\n", __func__, a->dims, b->dims);

    if (!(a->flags & NN_TENSOR_FLAG_INIT) || !(b->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor a or b is not initialized");
        return NULL;
    } else if (a->dims != 2 || b->dims != 2 || a->sizes[1] != b->sizes[0]) {
        // Number of columns in the first tensor must be equal to the number of rows in the second tensor
        nn_error_setf(error, NN_ERROR_INVALID_ARGUMENT, "only 2-dimensional tensors with matching inner dimensions are allowed, a.dims=%zu b.dims=%zu a.sizes[1]=%zu b.sizes[0]=%zu", a->dims, b->dims, a->sizes[1], b->sizes[0]);
        return NULL;
    }

    size_t m = a->sizes[0]; // number of rows in a
    size_t n = a->sizes[1]; // number of columns in a, rows in b (b->sizes[0])
    size_t p = b->sizes[1]; // number of columns in b

    // Perform matrix multiplication
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            NNTensorUnit sum = 0;
            for (size_t k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }
            result->data[i * p + j] = sum;
        }
    }

    return result;
}
