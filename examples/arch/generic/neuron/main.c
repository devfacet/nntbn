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
    NNError error;
    float inputs[NEURON_MAX_WEIGHTS] = {1, 2, 3};
    float weights[NEURON_MAX_WEIGHTS] = {0.2f, 0.8f, -0.5f};
    int n_weights = 3;
    float bias = 2.0f;

    // Compute the output
    if (!nn_neuron_init(&neuron, weights, n_weights, bias, nn_activation_func_identity, nn_dot_product, &error)) {
        printf("error: %s\n", error.message);
        return 1;
    }
    const float output = nn_neuron_compute(&neuron, inputs, &error);
    if (error.code != NN_ERROR_NONE) {
        printf("error: %s\n", error.message);
        return 1;
    }
    printf("output (generic): %f\n", output);

    return 0;
}
