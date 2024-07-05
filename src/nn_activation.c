#include "nn_activation.h"
#include "nn_debug.h"
#include "nn_error.h"
#include "nn_tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

// TODO: Add tests

NNActFunc nn_act_func_init(NNActFuncType type, void *func) {
    NN_DEBUG_PRINT(5, "function %s called with type=%d\n", __func__, type);

    NNActFunc act_func;
    act_func.type = type;
    if (type == NN_ACT_FUNC_SCALAR) {
        act_func.scalar_func = (NNActFuncScalar)func;
    } else if (type == NN_ACT_FUNC_TENSOR) {
        act_func.tensor_func = (NNActFuncTensor)func;
    }
    return act_func;
}

bool nn_act_func(NNActFunc act_func, const NNTensor *input, NNTensor *output, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with input.dims=%zu output.dims=%zu\n", __func__, input->dims, output->dims);

    switch (act_func.type) {
    case NN_ACT_FUNC_SCALAR:
        return nn_act_func_scalar_batch(act_func.scalar_func, input, output, error);
    case NN_ACT_FUNC_TENSOR:
        return nn_act_func_tensor_batch(act_func.tensor_func, input, output, error);
    default:
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "invalid activation function type");
        return false;
    }
}

NNTensorUnit nn_act_func_identity(NNTensorUnit n) {
    NN_DEBUG_PRINT(5, "function %s called with n=%f\n", __func__, n);

    return n;
}

NNTensorUnit nn_act_func_sigmoid(NNTensorUnit n) {
    NN_DEBUG_PRINT(5, "function %s called with n=%f\n", __func__, n);

    // TODO: Implement exp macro for handling different types
    return 1 / (1 + expf(-n));
}

NNTensorUnit nn_act_func_relu(NNTensorUnit n) {
    NN_DEBUG_PRINT(5, "function %s called with n=%f\n", __func__, n);

    return n > 0 ? n : 0;
}

bool nn_act_func_softmax(const NNTensor *input, NNTensor *output, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with input.dims=%zu output.dims=%zu\n", __func__, input->dims, output->dims);

    if (!(input->flags & NN_TENSOR_FLAG_INIT) || !(input->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor input or output is not initialized");
        return false;
    } else if (input->dims != 1 || output->dims != 1 || input->sizes[0] != output->sizes[0]) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 1-dimensional tensors of the same size are allowed");
        return false;
    }

    // Find the maximum input value
    size_t input_size = input->sizes[0];
    NNTensorUnit max_input = 0;
    for (size_t i = 0; i < input_size; i++) {
        if (input->data[i] > max_input) {
            max_input = input->data[i];
        }
    }

    // Compute exp(input[i] - max_input) to prevent overflow
    NNTensorUnit sum = 0;
    for (size_t i = 0; i < input_size; i++) {
        // TODO: Implement exp macro for handling different types
        output->data[i] = expf(input->data[i] - max_input);
        sum += output->data[i];
    }

    if (sum == 0) {
        nn_error_set(error, NN_ERROR_INVALID_VALUE, "sum is zero");
        return false;
    }

    // Normalize to form a probability distribution
    for (size_t i = 0; i < input_size; i++) {
        output->data[i] /= sum;
    }

    return true;
}

bool nn_act_func_scalar_batch(const NNActFuncScalar act_func, const NNTensor *input, NNTensor *output, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with input.dims=%zu output.dims=%zu\n", __func__, input->dims, output->dims);

    if (!(input->flags & NN_TENSOR_FLAG_INIT) || !(output->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor input or output is not initialized");
        return false;
    } else if (input->dims != 2 || output->dims != 2 || input->sizes[0] != output->sizes[0]) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 2-dimensional tensors of the same size are allowed");
        return false;
    }

    // Apply the activation function to each element in the input tensor
    size_t batch_size = input->sizes[0];
    size_t sample_size = input->sizes[1];
    size_t sizes[1] = {sample_size};
    NNTensor input_slice;
    for (size_t i = 0; i < batch_size; i++) {
        nn_tensor_slice(input, i * sample_size, sizes, &input_slice);
        for (size_t j = 0; j < sample_size; ++j) {
            output->data[i * sample_size + j] = act_func(input_slice.data[j]);
        }
    }

    return true;
}

bool nn_act_func_tensor_batch(const NNActFuncTensor act_func, const NNTensor *input, NNTensor *output, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with input.dims=%zu output.dims=%zu\n", __func__, input->dims, output->dims);

    if (!(input->flags & NN_TENSOR_FLAG_INIT) || !(output->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor input or output is not initialized");
        return false;
    } else if (input->dims != 2 || output->dims != 2 || input->sizes[0] != output->sizes[0] || input->sizes[1] != output->sizes[1]) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 2-dimensional tensors of the same size are allowed");
        return false;
    }

    // Apply the activation function to each sample in the batch
    size_t batch_size = input->sizes[0];
    size_t sample_size = input->sizes[1];
    size_t sizes[1] = {sample_size};
    NNTensor input_slice;
    NNTensor output_slice;

    for (size_t i = 0; i < batch_size; i++) {
        nn_tensor_slice(input, i * sample_size, sizes, &input_slice);
        nn_tensor_slice(output, i * sample_size, sizes, &output_slice);

        if (!act_func(&input_slice, &output_slice, error)) {
            return false;
        }
    }

    return true;
}
