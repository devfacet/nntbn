#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_error.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

// M_PI is not defined in some compilers.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// NN_LAYER_MAX_INPUT_SIZE defines the maximum input size a layer can have.
#ifndef NN_LAYER_MAX_INPUT_SIZE
#define NN_LAYER_MAX_INPUT_SIZE 64
#endif

// NN_LAYER_MAX_OUTPUT_SIZE defines the maximum output size a layer can have.
#ifndef NN_LAYER_MAX_OUTPUT_SIZE
#define NN_LAYER_MAX_OUTPUT_SIZE 64
#endif

// NN_LAYER_MAX_BIASES defines the maximum number of biases a layer can have.
#ifndef NN_LAYER_MAX_BIASES
#define NN_LAYER_MAX_BIASES 64
#endif

// NN_LAYER_MAX_BATCH_SIZE defines the maximum batch size a layer can have.
#ifndef NN_LAYER_MAX_BATCH_SIZE
#define NN_LAYER_MAX_BATCH_SIZE 32
#endif

// NNLayer represents a single layer in a neural network.
typedef struct {
    size_t input_size;
    size_t output_size;
    float weights[NN_LAYER_MAX_OUTPUT_SIZE][NN_LAYER_MAX_INPUT_SIZE];
    float biases[NN_LAYER_MAX_BIASES];
    NNDotProdFunc dot_prod_func;
} NNLayer;

// nn_layer_init initializes a layer with the given arguments.
bool nn_layer_init(NNLayer *layer, size_t input_size, size_t output_size, NNError *error);

// nn_layer_init_weights_gaussian initializes the weights of the layer with a Gaussian distribution.
bool nn_layer_init_weights_gaussian(NNLayer *layer, float scale, NNError *error);

// nn_layer_init_biases_zeros initializes the biases of the layer to zero.
bool nn_layer_init_biases_zeros(NNLayer *layer, NNError *error);

// nn_layer_set_weights sets the weights of the given layer.
bool nn_layer_set_weights(NNLayer *layer, const float weights[NN_LAYER_MAX_OUTPUT_SIZE][NN_LAYER_MAX_INPUT_SIZE], NNError *error);

// nn_layer_set_biases sets the biases of the given layer.
bool nn_layer_set_biases(NNLayer *layer, const float biases[NN_LAYER_MAX_BIASES], NNError *error);

// nn_layer_set_dot_prod_func sets the dot product function of the given layer.
bool nn_layer_set_dot_prod_func(NNLayer *layer, NNDotProdFunc dot_prod_func, NNError *error);

// nn_layer_forward computes the given layer with the given inputs and stores the result in outputs.
bool nn_layer_forward(const NNLayer *layer, const float inputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_INPUT_SIZE], float outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE], size_t batch_size, NNError *error);

#endif // NN_LAYER_H
