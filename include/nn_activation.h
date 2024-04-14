#ifndef NN_ACTIVATION_FUNCTION_H
#define NN_ACTIVATION_FUNCTION_H

#include "nn_error.h"
#include <stdbool.h>
#include <stddef.h>

#ifndef NN_SOFTMAX_MAX_SIZE
#define NN_SOFTMAX_MAX_SIZE 64
#endif

// NNActivationFunction represents an activation function.
typedef float (*NNActivationFunctionScalar)(float);
typedef bool (*NNActivationFunctionVector)(const float input[NN_SOFTMAX_MAX_SIZE], float output[NN_SOFTMAX_MAX_SIZE], size_t input_size, NNError *error);
typedef union {
    NNActivationFunctionScalar scalar;
    NNActivationFunctionVector vector;
} NNActivationFunction;

// nn_activation_func_identity returns x.
float nn_activation_func_identity(float x);

// nn_activation_func_sigmoid returns the sigmoid of x.
float nn_activation_func_sigmoid(float x);

// nn_activation_func_relu returns the ReLU of x.
float nn_activation_func_relu(float x);

// nn_activation_func_softmax calculates the softmax of the input and stores the result in the output.
bool nn_activation_func_softmax(const float input[NN_SOFTMAX_MAX_SIZE], float output[NN_SOFTMAX_MAX_SIZE], size_t input_size, NNError *error);

#endif // NN_ACTIVATION_FUNCTION_H
