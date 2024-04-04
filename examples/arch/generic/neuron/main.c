#include "nn_activation.h"
#include "nn_app.h"
#include "nn_config.h"
#include "nn_neuron.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);

    // Init vars
    Neuron neuron;
    NNError error;
    int n_inputs = 3;
    float inputs[NEURON_MAX_WEIGHTS] = {1, 2, 3};
    float weights[NEURON_MAX_WEIGHTS] = {0.2f, 0.8f, -0.5f};
    float bias = 2.0f;

    // Compute the output
    nn_init_neuron(&neuron, weights, n_inputs, bias, nn_activation_func_identity, nn_dot_product_generic);
    const float output = nn_compute_neuron(&neuron, inputs, n_inputs, &error);
    if (isnan(output) || error.code != NN_ERROR_NONE) {
        printf("error (%d): %s\n", error.code, error.message);
        return 1;
    }
    printf("output (generic): %f\n", output);

    return 0;
}
