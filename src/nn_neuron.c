#include "nn_neuron.h"
#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_error.h"
#include <math.h>
#include <stdio.h>

// nn_neuron_init initializes a neuron with the given arguments.
void nn_neuron_init(NNNeuron *neuron, const float *weights, size_t n_weights, float bias, NNActivationFunction act_func, NNDotProductFunction dot_product_func) {
    if (weights == NULL) {
        return;
    }
    neuron->n_weights = n_weights;
    for (size_t i = 0; i < neuron->n_weights; ++i) {
        neuron->weights[i] = weights[i];
    }
    neuron->bias = bias;
    if (act_func != NULL) {
        neuron->act_func = act_func;
    }
    if (dot_product_func != NULL) {
        neuron->dot_product_func = dot_product_func;
    }
}

// nn_neuron_set_weights sets the weights of the given neuron.
void nn_neuron_set_weights(NNNeuron *neuron, const float *weights) {
    if (weights == NULL) {
        return;
    }
    for (size_t i = 0; i < neuron->n_weights; ++i) {
        neuron->weights[i] = weights[i];
    }
}

// nn_neuron_set_bias sets the bias of the given neuron.
void nn_neuron_set_bias(NNNeuron *neuron, float bias) {
    neuron->bias = bias;
}

// nn_neuron_compute computes the given neuron and returns the output.
float nn_neuron_compute(const NNNeuron *neuron, const float *inputs) {
    // Compute the output of the neuron:
    // 1. Sum the weighted inputs (dot product)
    // 2. Add the bias
    // 3. Apply the activation function
    float result = 0.0f;
    if (neuron->dot_product_func != NULL) {
        // Sum the weighted inputs
        result = neuron->dot_product_func(neuron->weights, inputs, neuron->n_weights);
    }
    // Add the bias
    result += neuron->bias;
    if (neuron->act_func != NULL) {
        // Apply the activation function
        result = neuron->act_func(result);
    }
    return result;
}
