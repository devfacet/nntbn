#include "arch/arm/cmsis-dsp/nn_dot_product.h"
#include "nn_activation.h"
#include "nn_app.h"
#include "nn_config.h"
#include "nn_neuron.h"
#include "nn_test.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);

    if (!nn_cmsis_dsp_available()) {
        printf("ARM CMSIS-DSP not available\n");
    }

    // Init vars
    NNNeuron neuron;
    NNError error;
    size_t input_size = 3;
    float weights[NEURON_MAX_WEIGHTS] = {0.2f, 0.8f, -0.5f};
    float bias = 2.0f;
    const int n_runs = 1000;
    const int n_inputs = n_runs * input_size;
    long long total_time = 0;
    struct timespec start, end;
    float *inputs = malloc(n_inputs * sizeof(float));
    for (int i = 0; i < n_inputs; ++i) {
        inputs[i] = (float)rand() / (float)RAND_MAX;
    }

    if (!nn_neuron_init(&neuron, weights, input_size, bias, nn_activation_func_identity, nn_dot_product_cmsis, &error)) {
        printf("error: %s\n", error.message);
        return 1;
    }

    // Benchmark
    for (int i = 0; i < n_runs; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        nn_neuron_compute(&neuron, inputs + i * input_size, &error);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += nn_timespec_diff_ns(&start, &end);
    }
    if (error.code != NN_ERROR_NONE) {
        printf("error: %s\n", error.message);
        return 1;
    }
    printf("avg_time_ns=%lld total_time_ms=%lld info=nn_neuron_compute\n", total_time / n_runs, total_time / 1000000);

    return 0;
}
