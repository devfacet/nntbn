#include "nn_activation.h"
#include "nn_config.h"
#include "nn_dot_product.h"
#include "nn_layer.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// N_TEST_CASES defines the number of test cases.
#define N_TEST_CASES 3
// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
#define DEFAULT_OUTPUT_TOLERANCE 0.0001f

// TestCase defines a single test case.
typedef struct {
    size_t input_size;
    size_t output_size;
    float weights[NN_LAYER_MAX_OUTPUT_SIZE][NN_LAYER_MAX_INPUT_SIZE];
    float biases[NN_LAYER_MAX_BIASES];
    float weights2[NN_LAYER_MAX_OUTPUT_SIZE][NN_LAYER_MAX_INPUT_SIZE];
    float biases2[NN_LAYER_MAX_BIASES];
    NNActivationFunction act_func;
    NNDotProductFunction dot_product_func;
    size_t batch_size;
    float inputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_INPUT_SIZE];
    float output_tolerance;
    float expected_outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE];
} TestCase;

// run_test_cases runs the test cases.
void run_test_cases(TestCase *test_cases, int n_cases, char *info) {
    for (int i = 0; i < n_cases; ++i) {
        TestCase tc = test_cases[i];
        NNLayer layer;
        NNError error;

        nn_layer_init(&layer, tc.input_size, tc.output_size, tc.act_func, tc.dot_product_func, &error);
        assert(error.code == NN_ERROR_NONE);
        nn_layer_set_weights(&layer, tc.weights, &error);
        assert(error.code == NN_ERROR_NONE);
        nn_layer_set_biases(&layer, tc.biases, &error);
        assert(error.code == NN_ERROR_NONE);
        float intermediate_outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE];
        const bool first_layer_success = nn_layer_compute(&layer, tc.inputs, intermediate_outputs, tc.batch_size, &error);
        assert(first_layer_success == true);
        assert(error.code == NN_ERROR_NONE);
        nn_layer_set_weights(&layer, tc.weights2, &error);
        assert(error.code == NN_ERROR_NONE);
        nn_layer_set_biases(&layer, tc.biases2, &error);
        assert(error.code == NN_ERROR_NONE);
        float final_outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE];
        const bool second_layer_success = nn_layer_compute(&layer, intermediate_outputs, final_outputs, tc.batch_size, &error);
        assert(second_layer_success == true);
        assert(error.code == NN_ERROR_NONE);
        for (size_t i = 0; i < tc.batch_size; ++i) {
            for (size_t j = 0; j < tc.output_size; ++j) {
                assert(fabs(final_outputs[i][j] - tc.expected_outputs[i][j]) <= tc.output_tolerance);
            }
        }
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main() {
    TestCase test_cases[N_TEST_CASES] = {
        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {0.34f, -0.78f, 0.59f, 1.25f},
                {0.45f, 0.12f, -0.33f, 0.1f},
                {0.14f, 0.76f, -0.48f, -0.81f},
            },
            .biases = {0.1f, -0.2f, 0.4f},
            .weights2 = {
                {0.25f, -0.15f, 0.2f},
                {0.3f, 0.45f, -0.25f},
                {0.5f, -0.9f, 0.1f},
            },
            .biases2 = {0.5f, 1.5f, -0.2f},
            .act_func = nn_activation_func_identity,
            .dot_product_func = nn_dot_product,
            .batch_size = 3,
            .inputs = {
                {0.9f, -0.3f, 2.2f, 1.9f},
                {1.4f, 0.6f, -1.3f, 2.7f},
                {0.6f, -0.5f, 1.8f, -0.9f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {1.1739, 3.203, 2.0571},
                {0.89665, 2.983, 0.026},
                {0.75265, 1.39375, 0.719},
            },
        },
        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {-0.45f, 0.88f, -0.14f, 0.23f},
                {0.52f, 0.21f, -0.88f, 0.45f},
                {-0.33f, 0.44f, 0.62f, -0.67f},
            },
            .biases = {1.0f, -1.2f, 0.3f},
            .weights2 = {
                {0.39f, 0.17f, -0.41f},
                {-0.29f, 0.36f, 0.27f},
                {0.13f, -0.31f, 0.11f},
            },
            .biases2 = {-0.1f, 1.0f, 0.2f},
            .act_func = nn_activation_func_identity,
            .dot_product_func = nn_dot_product,
            .batch_size = 3,
            .inputs = {
                {-0.5f, 2.1f, 1.9f, -1.3f},
                {1.2f, 0.5f, -0.7f, 2.2f},
                {0.3f, 1.1f, -1.5f, 1.8f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {-1.08838, 0.02158, 1.91978},
                {1.41095, 0.49076, -0.15257},
                {1.67703, 0.36982, -0.04847},
            },
        },
        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {0.62f, -0.32f, 0.71f, 0.14f},
                {0.39f, 0.24f, -0.56f, -0.21f},
                {-0.29f, -0.51f, 0.28f, 0.67f},
            },
            .biases = {0.25f, 0.75f, -0.15f},
            .weights2 = {
                {0.19f, -0.45f, 0.28f},
                {0.54f, -0.33f, 0.47f},
                {-0.35f, 0.62f, -0.2f},
            },
            .biases2 = {0.7f, -1.1f, 0.3f},
            .act_func = nn_activation_func_identity,
            .dot_product_func = nn_dot_product,
            .batch_size = 3,
            .inputs = {
                {0.2f, 2.8f, -1.5f, 1.6f},
                {1.1f, -0.8f, 2.3f, 0.5f},
                {-0.9f, 1.6f, 0.7f, -0.2f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {-0.73629, -2.95982, 2.21633},
                {1.68903, 1.02658, -1.14717},
                {0.25842, -1.73464, 0.81991},
            },
        },
    };

    run_test_cases(test_cases, N_TEST_CASES, "nn_layer.multi");
    return 0;
}
