#include "nn_activation.h"
#include "nn_app.h"
#include "nn_config.h"
#include "nn_neuron.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);

    // Init vars
    NNNeuron neuron;
    int n_inputs = 3;
    float inputs[NEURON_MAX_WEIGHTS] = {1, 2, 3};
    float weights[NEURON_MAX_WEIGHTS] = {0.2f, 0.8f, -0.5f};
    float bias = 2.0f;

    // Compute the output
    nn_neuron_init(&neuron, weights, n_inputs, bias, nn_activation_func_identity, nn_dot_product);
    printf("output (generic): %f\n", nn_neuron_compute(&neuron, inputs));

    return 0;
}
