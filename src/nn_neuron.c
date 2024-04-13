#include "nn_neuron.h"
#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_error.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// nn_neuron_init initializes a neuron with the given arguments.
bool nn_neuron_init(NNNeuron *neuron, const float weights[NN_NEURON_MAX_WEIGHTS], size_t input_size, float bias, NNActivationFunction act_func, NNDotProductFunction dot_product_func, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (neuron == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "neuron is NULL");
        return false;
    }
    if (weights == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "weights is NULL");
        return false;
    }
    neuron->input_size = input_size;
    for (size_t i = 0; i < neuron->input_size; ++i) {
        neuron->weights[i] = weights[i];
    }
    neuron->bias = bias;
    if (act_func != NULL) {
        neuron->act_func = act_func;
    }
    if (dot_product_func != NULL) {
        neuron->dot_product_func = dot_product_func;
    }
    return true;
}

// nn_neuron_set_weights sets the weights of the given neuron.
bool nn_neuron_set_weights(NNNeuron *neuron, const float weights[NN_NEURON_MAX_WEIGHTS], NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (neuron == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "neuron is NULL");
        return false;
    }
    for (size_t i = 0; i < neuron->input_size; ++i) {
        neuron->weights[i] = weights[i];
    }
    return true;
}

// nn_neuron_set_bias sets the bias of the given neuron.
bool nn_neuron_set_bias(NNNeuron *neuron, float bias, NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (neuron == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "neuron is NULL");
        return false;
    }
    neuron->bias = bias;
    return true;
}

// nn_neuron_compute computes the given neuron and returns the output.
float nn_neuron_compute(const NNNeuron *neuron, const float inputs[NN_NEURON_MAX_WEIGHTS], NNError *error) {
    nn_error_set(error, NN_ERROR_NONE, NULL);
    if (neuron == NULL) {
        nn_error_set(error, NN_ERROR_INVALID_INSTANCE, "neuron is NULL");
        return NAN;
    }

    // Initialize the result
    float result = 0.0f;

    // Compute the output of the neuron:
    // 1. Sum the weighted inputs (dot product)
    // 2. Add the bias
    // 3. Apply the activation function

    // Compute the dot product
    if (neuron->dot_product_func != NULL) {
        // Sum the weighted inputs
        result = neuron->dot_product_func(neuron->weights, inputs, neuron->input_size);
    }
    // Add the bias
    result += neuron->bias;
    // Apply the activation function
    if (neuron->act_func != NULL) {
        result = neuron->act_func(result);
    }
    return result;
}
