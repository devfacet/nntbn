#include "arch/arm/neon/nn_dot_product.h"
#include "nn_config.h"
#include "nn_neuron.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
const float DEFAULT_OUTPUT_TOLERANCE = 0.0001f;

// TestCase defines a single test case.
typedef struct {
    float inputs[NEURON_MAX_WEIGHTS];
    float weights[NEURON_MAX_WEIGHTS];
    int n_inputs;
    float bias;
    NNDotProductFunction dot_product_func;
    float output_tolerance;
    float expected_output;
} TestCase;

// run_test_cases runs the test cases.
void run_test_cases(TestCase *test_cases, int n_cases, char *info, NNDotProductFunction dot_product_func) {
    for (int i = 0; i < n_cases; ++i) {
        TestCase tc = test_cases[i];
        Neuron neuron;
        NNError error;

        nn_init_neuron(&neuron, tc.weights, tc.n_inputs, tc.bias, nn_activation_func_identity, dot_product_func);
        const float output = nn_compute_neuron(&neuron, tc.inputs, tc.n_inputs, &error);
        assert(isnan(output) == false);
        assert(error.code == NN_ERROR_NONE);
        assert(fabs(output - tc.expected_output) < tc.output_tolerance);
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main() {
    const int n_test_cases = 10;
    TestCase test_cases[n_test_cases] = {
        {
            .inputs = {0.5f, 1.2f, -0.8f},
            .weights = {0.2f, 0.3f, -0.1f},
            .n_inputs = 3,
            .bias = 0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 1.04f,
        },

        {
            .inputs = {-0.6f, -1.1f, 0.9f},
            .weights = {-0.2f, 0.5f, 0.3f},
            .n_inputs = 3,
            .bias = -0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -0.66f,
        },

        {
            .inputs = {1.5f, 2.0f, -1.0f},
            .weights = {0.4f, 0.4f, -0.2f},
            .n_inputs = 3,
            .bias = 2.0f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 3.6f,
        },

        {
            .inputs = {0.1f, -0.2f, 0.3f},
            .weights = {0.3f, -0.2f, 0.1f},
            .n_inputs = 3,
            .bias = 0.05f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.15f,
        },

        {
            .inputs = {-2.5f, 3.0f, -1.5f},
            .weights = {0.5f, -0.5f, 0.75f},
            .n_inputs = 3,
            .bias = 1.0f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -2.875f,
        },

        {
            .inputs = {0.0f, 0.0f, 0.0f},
            .weights = {0.25f, -0.75f, 0.5f},
            .n_inputs = 3,
            .bias = 0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.5f,
        },

        {
            .inputs = {1.2f, -1.2f, 0.8f},
            .weights = {0.0f, 0.0f, 0.0f},
            .n_inputs = 3,
            .bias = 0.25f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.25f,
        },

        {
            .inputs = {1.0f, -1.0f, 1.0f},
            .weights = {-1.0f, 1.0f, -1.0f},
            .n_inputs = 3,
            .bias = -0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -3.5f,
        },

        {
            .inputs = {0.123f, 0.456f, -0.789f},
            .weights = {0.321f, -0.654f, 0.987f},
            .n_inputs = 3,
            .bias = 0.1f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -0.937484,
        },

        {
            .inputs = {0.001f, -0.002f, 0.003f},
            .weights = {0.004f, 0.005f, -0.006f},
            .n_inputs = 3,
            .bias = 0.0f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.000012f,
        },
    };
    run_test_cases(test_cases, n_test_cases, "nn_dot_product_neon", nn_dot_product_neon);
    return 0;
}
