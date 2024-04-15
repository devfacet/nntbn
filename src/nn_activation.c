#include "nn_activation.h"
#include "nn_error.h"
#include <math.h>
#include <stdbool.h>

// TODO: Add tests

// nn_act_func_forward_scalar computes the given activation function with the given input and stores the result in output.
bool nn_act_func_forward_scalar(NNActFuncScalar act_func, const float input[NN_AF_FORWARD_MAX_SIZE], float output[NN_AF_FORWARD_MAX_SIZE], size_t input_size, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (act_func == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_FUNCTION, "act_func is NULL");
        return false;
    } else if (input_size == 0 || input_size > NN_AF_FORWARD_MAX_SIZE) {
        nn_error_set(error, NN_ERROR_INVALID_SIZE, "invalid input size");
        return false;
    }

    for (size_t i = 0; i < input_size; ++i) {
        output[i] = act_func(input[i]);
    }

    return true;
}

// nn_act_func_forward_vector computes the given activation function with the given input and stores the result in output.
bool nn_act_func_forward_vector(NNActFuncVector act_func, const float input[NN_AF_FORWARD_MAX_SIZE], float output[NN_AF_FORWARD_MAX_SIZE], size_t input_size, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (act_func == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_FUNCTION, "act_func is NULL");
        return false;
    } else if (input_size == 0 || input_size > NN_AF_FORWARD_MAX_SIZE) {
        nn_error_set(error, NN_ERROR_INVALID_SIZE, "invalid input size");
        return false;
    }

    return act_func(input, output, input_size, error);
}

// nn_act_func_identity returns x.
float nn_act_func_identity(float x) {
    return x;
}

// nn_act_func_sigmoid returns the sigmoid of x.
float nn_act_func_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// nn_act_func_relu returns the ReLU of x.
float nn_act_func_relu(float x) {
    return fmaxf(0, x);
}

// nn_act_func_softmax calculates the softmax of the input and stores the result in the output.
bool nn_act_func_softmax(const float input[NN_AF_VECTOR_MAX_SIZE], float output[NN_AF_VECTOR_MAX_SIZE], size_t input_size, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (input_size == 0 || input_size > NN_AF_VECTOR_MAX_SIZE) {
        nn_error_set(error, NN_ERROR_INVALID_SIZE, "invalid input size");
        return false;
    }

    // Find the maximum input value
    float max_input = input[0];
    for (size_t i = 1; i < input_size; ++i) {
        if (input[i] > max_input) {
            max_input = input[i];
        }
    }

    // Compute exp(input[i] - max_input) to prevent overflow
    float sum = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        output[i] = expf(input[i] - max_input);
        sum += output[i];
    }

    if (sum == 0.0f) {
        nn_error_set(error, NN_ERROR_INVALID_VALUE, "sum is zero");
        return false;
    }

    // Normalize to form a probability distribution
    for (size_t i = 0; i < input_size; ++i) {
        output[i] /= sum;
    }

    return true;
}
