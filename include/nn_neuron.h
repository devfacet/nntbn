#ifndef NN_NEURON_H
#define NN_NEURON_H

#include "nn_activation.h"
#include "nn_dot_product.h"
#include "nn_error.h"
#include <stdbool.h>
#include <stddef.h>

// NN_NEURON_MAX_WEIGHTS defines the maximum number of weights a neuron can have.
#ifndef NN_NEURON_MAX_WEIGHTS
#define NN_NEURON_MAX_WEIGHTS 64
#endif

// NNNeuron represents a single neuron in a neural network.
// It is intended for experimentation and prototyping, and not recommended for
// real-world applications since it's not optimized for performance.
typedef struct {
    float weights[NN_NEURON_MAX_WEIGHTS];
    size_t input_size;
    float bias;
    NNActivationFunction act_func;
    NNDotProductFunction dot_product_func;
} NNNeuron;

// nn_neuron_init initializes a neuron with the given arguments.
bool nn_neuron_init(NNNeuron *neuron, const float weights[NN_NEURON_MAX_WEIGHTS], size_t input_size, float bias, NNActivationFunction act_func, NNDotProductFunction dot_product_func, NNError *error);

// nn_neuron_set_weights sets the weights of the given neuron.
bool nn_neuron_set_weights(NNNeuron *neuron, const float weights[NN_NEURON_MAX_WEIGHTS], NNError *error);

// nn_neuron_set_bias sets the bias of the given neuron.
bool nn_neuron_set_bias(NNNeuron *neuron, float bias, NNError *error);

// nn_neuron_compute computes the given neuron and returns the output.
float nn_neuron_compute(const NNNeuron *neuron, const float inputs[NN_NEURON_MAX_WEIGHTS], NNError *error);

#endif // NN_NEURON_H
