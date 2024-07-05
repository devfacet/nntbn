#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "nn_activation.h"
#include "nn_dot_prod.h"
#include "nn_error.h"
#include "nn_mat_mul.h"
#include "nn_mat_transpose.h"
#include "nn_tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Represents flags that can be set on a neural network layer.
 */
typedef enum {
    NN_LAYER_FLAG_NONE = 0x00,                   // no flags set
    NN_LAYER_FLAG_INIT = 0x01,                   // initialized
    NN_LAYER_FLAG_WEIGHTS_SET = 0x02,            // weights set
    NN_LAYER_FLAG_BIASES_SET = 0x04,             // biases set
    NN_LAYER_FLAG_FORWARD_READY = 0x08,          // ready for forward pass
    NN_LAYER_FLAG_MAT_MUL_FUNC_SET = 0x10,       // matrix multiplication function set
    NN_LAYER_FLAG_MAT_TRANSPOSE_FUNC_SET = 0x20, // matrix transpose function set
    NN_LAYER_FLAG_ACT_FUNC_SET = 0x40,           // activation function set
    NN_LAYER_FLAG_ACT_FUNC_SCALAR = 0x80,        // scalar activation function set
    NN_LAYER_FLAG_ACT_FUNC_TENSOR = 0x100        // tensor activation function set
} NNLayerFlags;

/**
 * @brief Represents a neural network layer.
 *
 * @param flags The flags set on the layer.
 * @param input_size The number of inputs to the layer.
 * @param output_size The number of outputs from the layer.
 * @param mat_mul_func The matrix multiplication function.
 * @param mat_transpose_func The matrix transpose function.
 * @param act_func The activation function.
 * @param weights The weights of the layer.
 * @param biases The biases of the layer.
 *
 * @note Use the `nn_layer_init` function to create and `nn_layer_destroy` to destroy.
 */
typedef struct {
    NNLayerFlags flags;
    size_t input_size;
    size_t output_size;
    NNMatMulFunc mat_mul_func;
    NNMatTransposeFunc mat_transpose_func;
    NNActFunc act_func;
    NNTensor *weights;
    NNTensor *biases;
} NNLayer;

/**
 * @brief Initializes a new neural network layer.
 *
 * @param input_size The number of inputs to the layer.
 * @param output_size The number of outputs from the layer.
 * @param error The error instance to set if an error occurs.
 *
 * @return The pointer to the newly created layer instance or NULL if an error occurs.
 */
NNLayer *nn_layer_init(size_t input_size, size_t output_size, NNError *error);

/**
 * @brief Sets the weights for the specified layer.
 *
 * @param layer The layer instance.
 * @param weights The weights to set.
 * @param error The error instance to set if an error occurs.
 */
bool nn_layer_set_weights(NNLayer *layer, const NNTensor *weights, NNError *error);

/**
 * @brief Sets the biases for the specified layer.
 *
 * @param layer The layer instance.
 * @param biases The biases to set.
 * @param error The error instance to set if an error occurs.
 */
bool nn_layer_set_biases(NNLayer *layer, const NNTensor *biases, NNError *error);

/**
 * @brief Sets the matrix multiplication function for the specified layer.
 *
 * @param layer The layer instance.
 * @param mat_mul_func The matrix multiplication function to set.
 * @param error The error instance to set if an error occurs.
 */
bool nn_layer_set_mat_mul_func(NNLayer *layer, NNMatMulFunc mat_mul_func, NNError *error);

/**
 * @brief Sets the matrix transpose function for the specified layer.
 *
 * @param layer The layer instance.
 * @param mat_transpose_func The matrix transpose function to set.
 * @param error The error instance to set if an error occurs.
 */
bool nn_layer_set_mat_transpose_func(NNLayer *layer, NNMatTransposeFunc mat_transpose_func, NNError *error);

/**
 * @brief Sets the activation function for the specified layer.
 *
 * @param layer The layer instance.
 * @param act_func_type The type of activation function.
 * @param act_func The activation function to set.
 * @param error The error instance to set if an error occurs.
 */
bool nn_layer_set_act_func(NNLayer *layer, NNActFuncType act_func_type, NNActFunc act_func, NNError *error);

/**
 * @brief Computes the forward pass of the given layer with the given inputs.
 *
 * @param layer The layer instance.
 * @param inputs The inputs to the layer.
 * @param outputs The tensor instance to set the outputs to.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false
 */
bool nn_layer_forward(const NNLayer *layer, const NNTensor *inputs, NNTensor *outputs, NNError *error);

/**
 * @brief Destroys the specified layer.
 *
 * @param layer The layer instance to destroy.
 */
void nn_layer_destroy(NNLayer *layer);

#endif // NN_LAYER_H
