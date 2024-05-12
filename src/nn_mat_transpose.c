#include "nn_mat_transpose.h"
#include "nn_debug.h"
#include "nn_error.h"
#include "nn_tensor.h"
#include <stddef.h>

bool nn_mat_transpose(const NNTensor *input, NNTensor *output, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with input.dims=%zu output.dims=%zu\n", __func__, input->dims, output->dims);

    if (!(input->flags & NN_TENSOR_FLAG_INIT) || !(output->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor input or output is not initialized");
        return false;
    } else if (input->dims != 2 || output->dims != 2) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 2-dimensional tensors are allowed");
        return false;
    }

    // Check if input and output tensors have the same shape
    bool inPlace = (input == output);
    NNTensorUnit *dataSrc = input->data;
    if (inPlace) {
        NN_DEBUG_PRINT(5, "function %s using in-place transpose\n", __func__);

        dataSrc = (NNTensorUnit *)malloc(input->sizes[0] * input->sizes[1] * sizeof(NNTensorUnit));
        if (!dataSrc) {
            nn_error_set(error, NN_ERROR_MEMORY_ALLOCATION, "could not allocate memory for in-place transpose");
            return false;
        }
        memcpy(dataSrc, input->data, input->sizes[0] * input->sizes[1] * sizeof(NNTensorUnit));
    }

    // Perform matrix transpose
    size_t m = input->sizes[0]; // number of rows in input
    size_t n = input->sizes[1]; // number of columns in input
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            output->data[j * m + i] = dataSrc[i * n + j];
        }
    }
    output->sizes[0] = n;
    output->sizes[1] = m;

    if (inPlace) {
        free(dataSrc);
    }

    return true;
}
