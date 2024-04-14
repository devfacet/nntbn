#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand((unsigned int)time(NULL));

    // Init vars
    NNLayer layer;
    NNError error;
    const int input_size = 4;
    const int output_size = 3;
    const int batch_size = 2;

    // Initialize a layer with the given input and output sizes, ReLU activation function, and dot product function
    if (!nn_layer_init(&layer, input_size, output_size, &error)) {
        fprintf(stderr, "error: %s\n", error.message);
        return 1;
    }

    // Initialize the weights of the layer with Gaussian random values scaled by 0.01
    if (!nn_layer_init_weights_gaussian(&layer, 0.01f, &error)) {
        fprintf(stderr, "error: %s\n", error.message);
        return 1;
    }

    // Initialize the biases of the layer to zero
    if (!nn_layer_init_biases_zeros(&layer, &error)) {
        fprintf(stderr, "error: %s\n", error.message);
        return 1;
    }

    // Set the dot product function of the layer
    if (!nn_layer_set_dot_product_func(&layer, nn_dot_product, &error)) {
        fprintf(stderr, "error: %s\n", error.message);
        return 1;
    }

    // Set the activation function of the layer
    NNActivationFunction act_func = {.scalar = nn_activation_func_relu};
    if (!nn_layer_set_activation_func(&layer, act_func, &error)) {
        fprintf(stderr, "error: %s\n", error.message);
        return 1;
    }

    // Generate random inputs
    float inputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_INPUT_SIZE];
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            inputs[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }

    // Compute the layer with the given inputs
    float outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE];
    if (!nn_layer_forward(&layer, inputs, outputs, 2, &error)) {
        fprintf(stderr, "error: %s\n", error.message);
        return 1;
    }

    // Print the inputs
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            printf("inputs[%zu][%zu] = %f\n", i, j, inputs[i][j]);
        }
    }

    // Print the outputs
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_size; ++j) {
            printf("outputs[%zu][%zu] = %f\n", i, j, outputs[i][j]);
        }
    }

    return 0;
}
