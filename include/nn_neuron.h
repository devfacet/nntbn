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

// NNNeuron represents a single neuron in a neural network.
// It is intended for experimentation and prototyping, and not recommended for
// real-world applications since it's not optimized for performance.
typedef struct {
    float weights[NEURON_MAX_WEIGHTS];
    float bias;
    size_t n_inputs;
    NNActivationFunction act_func;
    NNDotProductFunction dot_product_func;
} NNNeuron;

// nn_neuron_init initializes a neuron with the given arguments.
void nn_neuron_init(NNNeuron *neuron, const float *weights, size_t n_inputs, float bias, NNActivationFunction act_func, NNDotProductFunction dot_product_func);

// nn_neuron_compute computes the given neuron and returns the output.
float nn_neuron_compute(const NNNeuron *neuron, const float *inputs);

#endif // NN_NEURON_H
