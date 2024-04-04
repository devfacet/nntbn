#include "nn_neuron.h"
#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_error.h"
#include <math.h>
#include <stdio.h>

// nn_init_neuron initializes a neuron with the given arguments.
void nn_init_neuron(Neuron *neuron, const float *weights, size_t n_inputs, float bias, NNActivationFunction act_func, NNDotProductFunction dot_product_func) {
    if (weights == NULL) {
        return;
    }
    for (size_t i = 0; i < n_inputs; ++i) {
        neuron->weights[i] = weights[i];
    }
    neuron->n_inputs = n_inputs;
    neuron->bias = bias;
    if (act_func != NULL) {
        neuron->act_func = act_func;
    }
    if (dot_product_func != NULL) {
        neuron->dot_product_func = dot_product_func;
    }
}

// nn_compute_neuron computes the given neuron and returns the output.
float nn_compute_neuron(const Neuron *neuron, const float *inputs, size_t input_size, NNError *error) {
    if (error) {
        error->code = NN_ERROR_NONE;
        error->message = NULL;
    }

    // Check if the number of inputs matches the expected number of inputs for the neuron
    if (neuron->n_inputs != input_size) {
        if (error) {
            error->code = NN_ERROR_INVALID_INPUT_SIZE;
            error->message = "invalid input size";
        }
        return NAN;
    }

    // Compute the output of the neuron:
    // 1. Sum the weighted inputs (dot product)
    // 2. Add the bias
    // 3. Apply the activation function
    float result = 0.0f;
    if (neuron->dot_product_func != NULL) {
        // Sum the weighted inputs
        result = neuron->dot_product_func(neuron->weights, inputs, neuron->n_inputs);
    }
    // Add the bias
    result += neuron->bias;
    if (neuron->act_func != NULL) {
        // Apply the activation function
        result = neuron->act_func(result);
    }
    return result;
}
