#include "nn_activation.h"
#include "nn_error.h"
#include <math.h>
#include <stdbool.h>

// TODO: Add tests

// nn_activation_func_identity returns x.
float nn_activation_func_identity(float x) {
    return x;
}

// nn_activation_func_sigmoid returns the sigmoid of x.
float nn_activation_func_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// nn_activation_func_relu returns the ReLU of x.
float nn_activation_func_relu(float x) {
    return fmaxf(0, x);
}

// nn_activation_func_softmax calculates the softmax of the input and stores the result in the output.
bool nn_activation_func_softmax(const float input[NN_SOFTMAX_MAX_SIZE], float output[NN_SOFTMAX_MAX_SIZE], size_t input_size, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (input == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "input is NULL");
        return false;
    } else if (output == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "output is NULL");
        return false;
    } else if (input_size == 0 || input_size > NN_SOFTMAX_MAX_SIZE) {
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
