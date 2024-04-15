#ifndef NN_ACTIVATION_FUNCTION_H
#define NN_ACTIVATION_FUNCTION_H

#include "nn_error.h"
#include <stdbool.h>
#include <stddef.h>

#ifndef NN_AF_FORWARD_MAX_SIZE
#define NN_AF_FORWARD_MAX_SIZE 64
#endif

#ifndef NN_AF_VECTOR_MAX_SIZE
#define NN_AF_VECTOR_MAX_SIZE 64
#endif

// NNActFuncScalar represents a scalar activation function.
typedef float (*NNActFuncScalar)(float);

// NNActFuncVector represents a vector activation function.
typedef bool (*NNActFuncVector)(const float input[NN_AF_VECTOR_MAX_SIZE], float output[NN_AF_VECTOR_MAX_SIZE], size_t input_size, NNError *error);

// nn_act_func_forward_scalar computes the given activation function with the given input and stores the result in output.
bool nn_act_func_forward_scalar(NNActFuncScalar act_func, const float input[NN_AF_FORWARD_MAX_SIZE], float output[NN_AF_FORWARD_MAX_SIZE], size_t input_size, NNError *error);

// nn_act_func_forward_vector computes the given activation function with the given input and stores the result in output.
bool nn_act_func_forward_vector(NNActFuncVector act_func, const float input[NN_AF_FORWARD_MAX_SIZE], float output[NN_AF_FORWARD_MAX_SIZE], size_t input_size, NNError *error);

// nn_act_func_identity returns x.
float nn_act_func_identity(float x);

// nn_act_func_sigmoid returns the sigmoid of x.
float nn_act_func_sigmoid(float x);

// nn_act_func_relu returns the ReLU of x.
float nn_act_func_relu(float x);

// nn_act_func_softmax calculates the softmax of the input and stores the result in the output.
bool nn_act_func_softmax(const float input[NN_AF_VECTOR_MAX_SIZE], float output[NN_AF_VECTOR_MAX_SIZE], size_t input_size, NNError *error);

#endif // NN_ACTIVATION_FUNCTION_H
