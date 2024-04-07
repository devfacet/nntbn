#include "arch/arm/neon/nn_dot_product.h"
#include "nn_activation.h"
#include "nn_app.h"
#include "nn_config.h"
#include "nn_neuron.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);

    if (!nn_neon_available()) {
        printf("dot_product (NEON): ARM NEON not available\n");
    }

    // Init vars
    NNNeuron neuron;
    float inputs[NEURON_MAX_WEIGHTS] = {1, 2, 3};
    float weights[NEURON_MAX_WEIGHTS] = {0.2f, 0.8f, -0.5f};
    int n_weights = 3;
    float bias = 2.0f;

    // Compute the output
    nn_neuron_init(&neuron, weights, n_weights, bias, nn_activation_func_identity, nn_dot_product_neon);
    printf("output (NEON): %f\n", nn_neuron_compute(&neuron, inputs));

    return 0;
}
