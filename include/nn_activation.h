#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "nn_error.h"
#include "nn_tensor.h"
#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Represents a scalar activation function.
 *
 * @param n The input value.
 *
 * @return The output value.
 */
typedef NNTensorUnit (*NNActFuncScalar)(NNTensorUnit n);

/**
 * @brief Represents a tensor activation function.
 *
 * @param input The input tensor.
 * @param output The output tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return True if the operation was successful, false otherwise.
 */
typedef bool (*NNActFuncTensor)(const NNTensor *input, NNTensor *output, NNError *error);

/**
 * @brief Represents the type of activation function.
 */
typedef enum {
    NN_ACT_FUNC_SCALAR,
    NN_ACT_FUNC_TENSOR
} NNActFuncType;

/**
 * @brief Represents the activation function.
 */
typedef struct {
    NNActFuncType type;
    union {
        NNActFuncScalar scalar_func;
        NNActFuncTensor tensor_func;
    };
} NNActFunc;

/**
 * @brief Initializes a new activation function.
 *
 * @param type The type of activation function (scalar or tensor).
 * @param func The activation function.
 */
NNActFunc nn_act_func_init(NNActFuncType type, void *func);

/**
 * @brief Computes an activation function on the input tensor and stores the result in the output tensor.
 *
 * @param act_func The activation function.
 * @param input The input tensor.
 * @param output The output tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false.
 */
bool nn_act_func(NNActFunc act_func, const NNTensor *input, NNTensor *output, NNError *error);

/**
 * @brief Returns the identity activation function result.
 *
 * @param n The input value.
 *
 * @return The input value.
 *
 * @note Implemented for testing and theoretical purposes. Not recommended for practical use.
 */
NNTensorUnit nn_act_func_identity(NNTensorUnit n);

/**
 * @brief Returns the sigmoid activation function result.
 *
 * @param n The input value.
 *
 * @return Sigmoid of the input value.
 */
NNTensorUnit nn_act_func_sigmoid(NNTensorUnit n);

/**
 * @brief Returns the ReLU activation function result.
 *
 * @param n The input value.
 *
 * @return ReLU of the input value.
 */
NNTensorUnit nn_act_func_relu(NNTensorUnit n);

/**
 * @brief Calculates the softmax of the input tensor and stores the result in the output tensor.
 *
 * @param input The input tensor.
 * @param output The output tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false.
 */
bool nn_act_func_softmax(const NNTensor *input, NNTensor *output, NNError *error);

/**
 * @brief Calculates the scalar activation function result for each element in the input tensor and stores the result in the output tensor.
 *
 * @param act_func The scalar activation function.
 * @param input The input tensor.
 * @param output The output tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false.
 */
bool nn_act_func_scalar_batch(const NNActFuncScalar act_func, const NNTensor *input, NNTensor *output, NNError *error);

/**
 * @brief Calculates the tensor activation function result for the input tensor and stores the result in the output tensor.
 *
 * @param act_func The tensor activation function.
 * @param input The input tensor.
 * @param output The output tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false.
 */
bool nn_act_func_tensor_batch(const NNActFuncTensor act_func, const NNTensor *input, NNTensor *output, NNError *error);

#endif // NN_ACTIVATION_H
