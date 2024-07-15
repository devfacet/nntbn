#include "nn_argmax.h"
#include "nn_debug.h"
#include "nn_error.h"

size_t nn_argmax(const NNTensor *input, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with input.dims=%zu\n", __func__, input->dims);

    if (!(input->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor input is not initialized");
        return 0;
    } else if (input->dims != 1) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 1-dimensional tensors are allowed");
        return 0;
    }

    // Find the index of the maximum value in the input tensor
    size_t max_index = 0;
    NNTensorUnit max_value = input->data[0];
    for (size_t i = 1; i < input->sizes[0]; ++i) {
        if (input->data[i] > max_value) {
            max_value = input->data[i];
            max_index = i;
        }
    }

    return max_index;
}

bool nn_argmax_tensor_batch(const NNTensor *input, NNTensor *output, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with input.dims=%zu output.dims=%zu\n", __func__, input->dims, output->dims);

    if (!(input->flags & NN_TENSOR_FLAG_INIT) || !(output->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor input or output is not initialized");
        return false;
    } else if (input->dims != 2 || output->dims != 1 || input->sizes[0] != output->sizes[0]) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 2-dimensional input tensor and 1-dimensional output tensor with the same batch size are allowed");
        return false;
    }

    // Find the index of the maximum value in each row of the input tensor batch
    size_t batch_size = input->sizes[0];
    size_t num_classes = input->sizes[1];
    for (size_t i = 0; i < batch_size; ++i) {
        size_t max_index = 0;
        NNTensorUnit max_value = input->data[i * num_classes]; // first element in the row
        for (size_t j = 1; j < num_classes; ++j) {
            if (input->data[i * num_classes + j] > max_value) {
                max_value = input->data[i * num_classes + j];
                max_index = j;
            }
        }
        output->data[i] = max_index;
    }

    return true;
}
