#ifndef NN_NEURON_H
#define NN_NEURON_H

#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_error.h"
#include <stdbool.h>
#include <stddef.h>

// NEURON_MAX_WEIGHTS defines the maximum number of weights a neuron can have.
#ifndef NEURON_MAX_WEIGHTS
#define NEURON_MAX_WEIGHTS 64
#endif

// Neuron represents a single neuron in a neural network.
typedef struct {
    NNActivationFunction act_func;
    NNDotProductFunction dot_product_func;
    float bias;
    float weights[NEURON_MAX_WEIGHTS];
    size_t n_inputs;
} Neuron;

// nn_init_neuron initializes a neuron with the given arguments.
void nn_init_neuron(Neuron *neuron, const float *weights, size_t n_inputs, float bias, NNActivationFunction act_func, NNDotProductFunction dot_product_func);

// nn_compute_neuron computes the given neuron and returns the output.
float nn_compute_neuron(const Neuron *neuron, const float *inputs, size_t input_size, NNError *error);

#endif // NN_NEURON_H
