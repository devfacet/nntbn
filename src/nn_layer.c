#include "nn_layer.h"
#include "nn_error.h"
#include "nn_mat_mul.h"
#include "nn_mat_transpose.h"
#include "nn_tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Checks if the layer is ready for forward pass.
 *
 * @param layer The layer instance to check.
 *
 * @return True or false.
 */
static bool nn_layer_check_forward_ready(NNLayer *layer) {
    if (layer == NULL) {
        return false;
    } else if (layer->flags & NN_LAYER_FLAG_FORWARD_READY) {
        return true;
    }

    int required_flags = NN_LAYER_FLAG_INIT;

    if ((layer->flags & required_flags) == required_flags) {
        layer->flags |= NN_LAYER_FLAG_FORWARD_READY;
        return true;
    }

    return false;
}

NNLayer *nn_layer_init(size_t input_size, size_t output_size, NNError *error) {
    NNLayer *layer = (NNLayer *)malloc(sizeof(NNLayer));
    if (!layer) {
        nn_error_set(error, NN_ERROR_MEMORY_ALLOCATION, "could not allocate memory for the new layer");
        return NULL;
    }

    // Init the layer
    layer->flags = NN_LAYER_FLAG_NONE;
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Init the weights
    layer->weights = nn_tensor_init_NNTensor(2, (const size_t[]){output_size, input_size}, true, NULL, error); // initialized to 0
    if (!layer->weights) {
        nn_layer_destroy(layer);
        nn_error_setf(error, NN_ERROR_MEMORY_ALLOCATION, "could create weights tensor: %s", error->message);
        return NULL;
    }

    // Init the biases
    layer->biases = nn_tensor_init_NNTensor(1, (const size_t[]){output_size}, true, NULL, error); // initialized to 0
    if (!layer->biases) {
        nn_layer_destroy(layer);
        nn_error_setf(error, NN_ERROR_MEMORY_ALLOCATION, "could create biases tensor: %s", error->message);
        return NULL;
    }
    layer->flags = NN_LAYER_FLAG_INIT;
    nn_layer_check_forward_ready(layer); // update flags

    return layer;
}

bool nn_layer_set_weights(NNLayer *layer, const NNTensor *weights, NNError *error) {
    if (layer == NULL || !(layer->flags & NN_LAYER_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "layer is not initialized");
        return false;
    } else if (weights == NULL || weights->data == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "given weights are NULL");
        return false;
    } else if (!nn_tensor_set_NNTensor(layer->weights, NULL, 0, weights, error)) {
        return false;
    }

    // TODO: Check if there is a "better" way to set the dims and sizes
    layer->weights->dims = weights->dims;
    layer->weights->sizes = weights->sizes;

    layer->flags |= NN_LAYER_FLAG_WEIGHTS_SET;
    nn_layer_check_forward_ready(layer);

    return true;
}

bool nn_layer_set_biases(NNLayer *layer, const NNTensor *biases, NNError *error) {
    if (layer == NULL || !(layer->flags & NN_LAYER_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "layer is not initialized");
        return false;
    } else if (biases == NULL || biases->data == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "given biases are NULL");
        return false;
    } else if (!nn_tensor_set_NNTensor(layer->biases, NULL, 0, biases, error)) {
        return false;
    }
    layer->flags |= NN_LAYER_FLAG_BIASES_SET;
    nn_layer_check_forward_ready(layer);

    return true;
}

bool nn_layer_set_mat_mul_func(NNLayer *layer, NNMatMulFunc mat_mul_func, NNError *error) {
    if (layer == NULL || !(layer->flags & NN_LAYER_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "layer is not initialized");
        return false;
    } else if (mat_mul_func == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "matrix multiplication function is NULL");
        return false;
    }
    layer->mat_mul_func = mat_mul_func;
    layer->flags |= NN_LAYER_FLAG_MAT_MUL_FUNC_SET;
    nn_layer_check_forward_ready(layer);

    return true;
}

bool nn_layer_set_mat_transpose_func(NNLayer *layer, NNMatTransposeFunc mat_transpose_func, NNError *error) {
    if (layer == NULL || !(layer->flags & NN_LAYER_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "layer is not initialized");
        return false;
    } else if (mat_transpose_func == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "matrix transpose function is NULL");
        return false;
    }
    layer->mat_transpose_func = mat_transpose_func;
    layer->flags |= NN_LAYER_FLAG_MAT_TRANSPOSE_FUNC_SET;
    nn_layer_check_forward_ready(layer);

    return true;
}

bool nn_layer_set_act_func(NNLayer *layer, NNActFuncType act_func_type, NNActFunc act_func, NNError *error) {
    if (layer == NULL || !(layer->flags & NN_LAYER_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "layer is not initialized");
        return false;
    }
    layer->act_func = act_func;
    layer->flags |= NN_LAYER_FLAG_ACT_FUNC_SET;
    if (act_func_type == NN_ACT_FUNC_SCALAR) {
        layer->flags |= NN_LAYER_FLAG_ACT_FUNC_SCALAR;
    } else if (act_func_type == NN_ACT_FUNC_TENSOR) {
        layer->flags |= NN_LAYER_FLAG_ACT_FUNC_TENSOR;
    }
    nn_layer_check_forward_ready(layer);

    return true;
}

bool nn_layer_forward(const NNLayer *layer, const NNTensor *inputs, NNTensor *outputs, NNError *error) {
    if (layer == NULL || !(layer->flags & NN_LAYER_FLAG_FORWARD_READY)) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is not ready for forward pass");
        return false;
    } else if (inputs == NULL || outputs == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "inputs or outputs are NULL");
        return false;
    }

    // Check if weights are set
    if (layer->flags & NN_LAYER_FLAG_WEIGHTS_SET) {
        // Check if matrix multiplication function is set
        if (layer->flags & NN_LAYER_FLAG_MAT_MUL_FUNC_SET) {
            // Check if matrix transpose function is set
            if (layer->flags & NN_LAYER_FLAG_MAT_TRANSPOSE_FUNC_SET) {
                // Transpose weights
                NNTensor *weights = nn_tensor_init_NNTensor(layer->weights->dims, layer->weights->sizes, true, NULL, error);
                if (!weights) {
                    return false;
                }
                // TODO: Should we overwrite the weights tensor so that we don't have to allocate a new tensor every time?
                if (!nn_mat_transpose(layer->weights, weights, error)) {
                    return false;
                }
                // Perform matrix multiplication
                if (!layer->mat_mul_func(inputs, weights, outputs, error)) {
                    return false;
                }
            } else {
                // Perform matrix multiplication
                if (!layer->mat_mul_func(inputs, layer->weights, outputs, error)) {
                    return false;
                }
            }
        } else {
            nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "matrix multiplication function is not set");
            return false;
        }
    } else {
        // If weights are not set, just copy the inputs to the outputs
        if (!nn_tensor_set_NNTensor(outputs, NULL, 0, inputs, error)) {
            return false;
        }
    }

    // Add biases
    if (layer->flags & NN_LAYER_FLAG_BIASES_SET) {
        size_t batch_size = outputs->sizes[0];
        size_t sample_size = outputs->sizes[1];
        for (size_t i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < sample_size; ++j) {
                outputs->data[i * sample_size + j] += layer->biases->data[j];
            }
        }
    }

    // Apply activation function
    if (layer->flags & NN_LAYER_FLAG_ACT_FUNC_SET) {
        if (layer->flags & NN_LAYER_FLAG_ACT_FUNC_SCALAR) {
            if (!nn_act_func_scalar_batch(layer->act_func.scalar_func, outputs, outputs, error)) {
                return false;
            }
        } else if (layer->flags & NN_LAYER_FLAG_ACT_FUNC_TENSOR) {
            if (!nn_act_func_tensor_batch(layer->act_func.tensor_func, outputs, outputs, error)) {
                return false;
            }
        }
    }

    return true;
}

void nn_layer_destroy(NNLayer *layer) {
    if (layer) {
        nn_tensor_destroy_NNTensor(layer->weights);
        nn_tensor_destroy_NNTensor(layer->biases);
        free(layer);
    }
}
