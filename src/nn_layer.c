#include "nn_layer.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

// nn_layer_init initializes a layer with the given arguments.
bool nn_layer_init(NNLayer *layer, size_t input_size, size_t output_size, NNActivationFunction act_func, NNDotProductFunction dot_product_func, NNError *error) {
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
    if (act_func) {
        layer->act_func = act_func;
    }
    if (dot_product_func) {
        layer->dot_product_func = dot_product_func;
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

// nn_layer_compute computes the given layer with the given inputs and stores the result in outputs.
bool nn_layer_compute(const NNLayer *layer, const float inputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_INPUT_SIZE], float outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE], size_t batch_size, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (layer == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "layer is NULL");
        return false;
    }
    if (batch_size == 0) {
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
            if (layer->act_func != NULL) {
                outputs[i][j] = layer->act_func(outputs[i][j]);
            }
        }
    }
    return true;
}
