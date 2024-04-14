#include "nn_layer.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// M_PI is not defined in some compilers.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// nn_layer_init initializes a layer with the given arguments.
bool nn_layer_init(NNLayer *layer, size_t input_size, size_t output_size, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }
    if (input_size == 0) {
        nn_error_set(error, NN_ERROR_INVALID_SIZE, "invalid input size");
        return false;
    }
    if (output_size == 0) {
        nn_error_set(error, NN_ERROR_INVALID_SIZE, "invalid output size");
        return false;
    }
    layer->input_size = input_size;
    layer->output_size = output_size;

    return true;
}

// nn_layer_init_weights_gaussian initializes the weights of the layer with a Gaussian distribution.
bool nn_layer_init_weights_gaussian(NNLayer *layer, float scale, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }

    // Initialize weights with Gaussian random values scaled by 'scale'
    for (size_t i = 0; i < layer->output_size; ++i) {
        for (size_t j = 0; j < layer->input_size; ++j) {
            float u1 = (float)rand() / (float)RAND_MAX;
            float u2 = (float)rand() / (float)RAND_MAX;
            float rand_std_normal = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); // Box-Muller transform
            layer->weights[i][j] = scale * rand_std_normal;
        }
    }

    return true;
}

// nn_layer_init_biases_zeros initializes the biases of the layer to zero.
bool nn_layer_init_biases_zeros(NNLayer *layer, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }

    // Initialize biases to zero
    for (size_t i = 0; i < layer->output_size; ++i) {
        layer->biases[i] = 0.0;
    }

    return true;
}

// nn_layer_set_weights sets the weights of the given layer.
bool nn_layer_set_weights(NNLayer *layer, const float weights[NN_LAYER_MAX_OUTPUT_SIZE][NN_LAYER_MAX_INPUT_SIZE], NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }
    for (size_t i = 0; i < layer->output_size; ++i) {
        for (size_t j = 0; j < layer->input_size; ++j) {
            layer->weights[i][j] = weights[i][j];
        }
    }

    return true;
}

// nn_layer_set_biases sets the biases of the given layer.
bool nn_layer_set_biases(NNLayer *layer, const float biases[NN_LAYER_MAX_BIASES], NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }
    for (size_t i = 0; i < layer->output_size; ++i) {
        layer->biases[i] = biases[i];
    }

    return true;
}

// nn_layer_set_dot_product_func sets the dot product function of the given layer.
bool nn_layer_set_dot_product_func(NNLayer *layer, NNDotProductFunction dot_product_func, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }
    layer->dot_product_func = dot_product_func;

    return true;
}

// nn_layer_set_activation_func sets the activation function of the given layer.
bool nn_layer_set_activation_func(NNLayer *layer, NNActivationFunction act_func, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }
    layer->act_func = act_func;

    return true;
}

// nn_layer_forward computes the given layer with the given inputs and stores the result in outputs.
bool nn_layer_forward(const NNLayer *layer, const float inputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_INPUT_SIZE], float outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE], size_t batch_size, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    } else if (batch_size == 0) {
        nn_error_set(error, NN_ERROR_INVALID_SIZE, "invalid batch size");
        return false;
    }

    // Iterate over each input in the batch
    for (size_t i = 0; i < batch_size; ++i) {
        // Iterate over each output in the layer
        for (size_t j = 0; j < layer->output_size; ++j) {
            outputs[i][j] = layer->biases[j];
            if (layer->dot_product_func != NULL) {
                outputs[i][j] += layer->dot_product_func(inputs[i], layer->weights[j], layer->input_size);
            }
            if (layer->act_func.scalar != NULL) {
                outputs[i][j] = layer->act_func.scalar(outputs[i][j]);
            }
        }
    }

    return true;
}
