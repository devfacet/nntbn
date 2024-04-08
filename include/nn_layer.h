#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_error.h"
#include <stdbool.h>
#include <stddef.h>

// LAYER_MAX_INPUT_SIZE defines the maximum input size a layer can have.
#ifndef LAYER_MAX_INPUT_SIZE
#define LAYER_MAX_INPUT_SIZE 64
#endif

// LAYER_MAX_OUTPUT_SIZE defines the maximum output size a layer can have.
#ifndef LAYER_MAX_OUTPUT_SIZE
#define LAYER_MAX_OUTPUT_SIZE 64
#endif

// LAYER_MAX_BIASES defines the maximum number of biases a layer can have.
#ifndef LAYER_MAX_BIASES
#define LAYER_MAX_BIASES 64
#endif

// LAYER_MAX_BATCH_SIZE defines the maximum batch size a layer can have.
#ifndef LAYER_MAX_BATCH_SIZE
#define LAYER_MAX_BATCH_SIZE 32
#endif

// NNLayer represents a single layer in a neural network.
typedef struct {
    size_t input_size;
    size_t output_size;
    float weights[LAYER_MAX_OUTPUT_SIZE][LAYER_MAX_INPUT_SIZE];
    float biases[LAYER_MAX_BIASES];
    NNActivationFunction act_func;
    NNDotProductFunction dot_product_func;
} NNLayer;

// nn_layer_init initializes a layer with the given arguments.
bool nn_layer_init(NNLayer *layer, size_t input_size, size_t output_size, NNActivationFunction act_func, NNDotProductFunction dot_product_func, NNError *error);

// nn_layer_set_weights sets the weights of the given layer.
bool nn_layer_set_weights(NNLayer *layer, const float weights[LAYER_MAX_OUTPUT_SIZE][LAYER_MAX_INPUT_SIZE], NNError *error);

// nn_layer_set_biases sets the biases of the given layer.
bool nn_layer_set_biases(NNLayer *layer, const float biases[LAYER_MAX_BIASES], NNError *error);

// nn_layer_compute computes the given layer with the given inputs and stores the result in outputs.
bool nn_layer_compute(const NNLayer *layer, const float inputs[LAYER_MAX_BATCH_SIZE][LAYER_MAX_INPUT_SIZE], float outputs[LAYER_MAX_BATCH_SIZE][LAYER_MAX_OUTPUT_SIZE], size_t batch_size, NNError *error);

#endif // NN_LAYER_H
