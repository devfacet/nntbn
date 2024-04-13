#include "arch/arm/neon/nn_dot_product.h"
#include "nn_config.h"
#include "nn_neuron.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// N_TEST_CASES defines the number of test cases.
#define N_TEST_CASES 10
// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
#define DEFAULT_OUTPUT_TOLERANCE 0.0001f

// TestCase defines a single test case.
typedef struct {
    float inputs[NEURON_MAX_WEIGHTS];
    float weights[NEURON_MAX_WEIGHTS];
    size_t input_size;
    float bias;
    NNDotProductFunction dot_product_func;
    float output_tolerance;
    float expected_output;
} TestCase;

// run_test_cases runs the test cases.
void run_test_cases(TestCase *test_cases, int n_cases, char *info, NNDotProductFunction dot_product_func) {
    for (int i = 0; i < n_cases; ++i) {
        TestCase tc = test_cases[i];
        NNNeuron neuron;
        NNError error;

        nn_neuron_init(&neuron, tc.weights, tc.input_size, tc.bias, nn_activation_func_identity, dot_product_func, &error);
        assert(error.code == NN_ERROR_NONE);
        const float output = nn_neuron_compute(&neuron, tc.inputs, &error);
        assert(error.code == NN_ERROR_NONE);
        assert(isnan(output) == false);
        assert(fabs(output - tc.expected_output) < tc.output_tolerance);
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main() {
    TestCase test_cases[N_TEST_CASES] = {
        {
            .inputs = {0.5f, 1.2f, -0.8f},
            .weights = {0.2f, 0.3f, -0.1f},
            .input_size = 3,
            .bias = 0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 1.04f,
        },

        {
            .inputs = {-0.6f, -1.1f, 0.9f},
            .weights = {-0.2f, 0.5f, 0.3f},
            .input_size = 3,
            .bias = -0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -0.66f,
        },

        {
            .inputs = {1.5f, 2.0f, -1.0f},
            .weights = {0.4f, 0.4f, -0.2f},
            .input_size = 3,
            .bias = 2.0f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 3.6f,
        },

        {
            .inputs = {0.1f, -0.2f, 0.3f},
            .weights = {0.3f, -0.2f, 0.1f},
            .input_size = 3,
            .bias = 0.05f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.15f,
        },

        {
            .inputs = {-2.5f, 3.0f, -1.5f},
            .weights = {0.5f, -0.5f, 0.75f},
            .input_size = 3,
            .bias = 1.0f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -2.875f,
        },

        {
            .inputs = {0.0f, 0.0f, 0.0f},
            .weights = {0.25f, -0.75f, 0.5f},
            .input_size = 3,
            .bias = 0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.5f,
        },

        {
            .inputs = {1.2f, -1.2f, 0.8f},
            .weights = {0.0f, 0.0f, 0.0f},
            .input_size = 3,
            .bias = 0.25f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.25f,
        },

        {
            .inputs = {1.0f, -1.0f, 1.0f},
            .weights = {-1.0f, 1.0f, -1.0f},
            .input_size = 3,
            .bias = -0.5f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -3.5f,
        },

        {
            .inputs = {0.123f, 0.456f, -0.789f},
            .weights = {0.321f, -0.654f, 0.987f},
            .input_size = 3,
            .bias = 0.1f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -0.937484,
        },

        {
            .inputs = {0.001f, -0.002f, 0.003f},
            .weights = {0.004f, 0.005f, -0.006f},
            .input_size = 3,
            .bias = 0.0f,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.000012f,
        },
    };
    run_test_cases(test_cases, N_TEST_CASES, "nn_neuron", nn_dot_product_neon);
    return 0;
}
