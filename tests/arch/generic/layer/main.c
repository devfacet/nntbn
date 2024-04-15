#include "nn_activation.h"
#include "nn_config.h"
#include "nn_dot_product.h"
#include "nn_layer.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// N_TEST_CASES defines the number of test cases.
#define N_TEST_CASES 9
// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
#define DEFAULT_OUTPUT_TOLERANCE 0.0001f

// TestCase defines a single test case.
typedef struct {
    size_t input_size;
    size_t output_size;
    float weights[NN_LAYER_MAX_OUTPUT_SIZE][NN_LAYER_MAX_INPUT_SIZE];
    float biases[NN_LAYER_MAX_BIASES];
    NNDotProductFunction dot_product_func;
    NNActivationFunctionScalar act_func_scalar;
    NNActivationFunctionVector act_func_vector;
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

        nn_layer_init(&layer, tc.input_size, tc.output_size, &error);
        assert(error.code == NN_ERROR_NONE);
        nn_layer_set_dot_product_func(&layer, tc.dot_product_func, &error);
        assert(error.code == NN_ERROR_NONE);
        nn_layer_set_weights(&layer, tc.weights, &error);
        assert(error.code == NN_ERROR_NONE);
        nn_layer_set_biases(&layer, tc.biases, &error);
        assert(error.code == NN_ERROR_NONE);
        float outputs[NN_LAYER_MAX_BATCH_SIZE][NN_LAYER_MAX_OUTPUT_SIZE];
        const bool lfr = nn_layer_forward(&layer, tc.inputs, outputs, tc.batch_size, &error);
        assert(lfr == true);
        assert(error.code == NN_ERROR_NONE);
        for (size_t i = 0; i < tc.batch_size; ++i) {
            if (tc.act_func_scalar != NULL) {
                const bool laf = nn_activation_func_forward_scalar(tc.act_func_scalar, outputs[i], outputs[i], tc.output_size, &error);
                assert(laf == true);
                assert(error.code == NN_ERROR_NONE);
            } else if (tc.act_func_vector != NULL) {
                const bool laf = nn_activation_func_forward_vector(tc.act_func_vector, outputs[i], outputs[i], tc.output_size, &error);
                assert(laf == true);
                assert(error.code == NN_ERROR_NONE);
            }
        }
        for (size_t i = 0; i < tc.batch_size; ++i) {
            for (size_t j = 0; j < tc.output_size; ++j) {
                assert(fabs(outputs[i][j] - tc.expected_outputs[i][j]) <= tc.output_tolerance);
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
                {0.1f, 0.2f, -0.1f, 0.5f},
                {0.3f, -0.2f, 0.4f, 0.1f},
                {-0.3f, 0.4f, 0.2f, -0.5f},
            },
            .biases = {0.5f, -0.1f, 0.2f},
            .dot_product_func = nn_dot_product,
            .act_func_scalar = nn_activation_func_identity,
            .batch_size = 2,
            .inputs = {
                {1.5f, -2.0f, 1.0f, -1.5f},
                {-1.0f, 2.0f, -0.5f, 1.0f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {-0.6f, 1.0f, -0.1f},
                {1.35f, -0.9f, 0.7f},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {0.1f, 0.2f, -0.1f, 0.5f},
                {0.3f, -0.2f, 0.4f, 0.1f},
                {-0.3f, 0.4f, 0.2f, -0.5f},
            },
            .biases = {0.5f, -0.1f, 0.2f},
            .dot_product_func = nn_dot_product,
            .act_func_scalar = nn_activation_func_relu,
            .batch_size = 2,
            .inputs = {
                {1.5f, -2.0f, 1.0f, -1.5f},
                {-1.0f, 2.0f, -0.5f, 1.0f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {0.0f, 1.0f, 0.0f},
                {1.35f, 0.0f, 0.7f},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {0.1f, 0.2f, -0.1f, 0.5f},
                {0.3f, -0.2f, 0.4f, 0.1f},
                {-0.3f, 0.4f, 0.2f, -0.5f},
            },
            .biases = {0.5f, -0.1f, 0.2f},
            .dot_product_func = nn_dot_product,
            .act_func_vector = nn_activation_func_softmax,
            .batch_size = 2,
            .inputs = {
                {1.5f, -2.0f, 1.0f, -1.5f},
                {-1.0f, 2.0f, -0.5f, 1.0f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {0.13154859, 0.65156444, 0.21688696},
                {0.61446009, 0.06476362, 0.32077629},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {-0.5f, 0.8f, -0.2f, 0.4f},
                {0.2f, -0.3f, 0.5f, -0.1f},
                {0.4f, 0.1f, -0.4f, 0.6f},
            },
            .biases = {1.0f, 0.5f, -0.2f},
            .dot_product_func = nn_dot_product,
            .act_func_scalar = nn_activation_func_identity,
            .batch_size = 2,
            .inputs = {
                {0.5f, 0.1f, -0.2f, 0.4f},
                {1.2f, -1.2f, 0.5f, -0.3f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {1.03f, 0.43f, 0.33f},
                {-0.78f, 1.38f, -0.22f},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {-0.5f, 0.8f, -0.2f, 0.4f},
                {0.2f, -0.3f, 0.5f, -0.1f},
                {0.4f, 0.1f, -0.4f, 0.6f},
            },
            .biases = {1.0f, 0.5f, -0.2f},
            .dot_product_func = nn_dot_product,
            .act_func_scalar = nn_activation_func_relu,
            .batch_size = 2,
            .inputs = {
                {0.5f, 0.1f, -0.2f, 0.4f},
                {1.2f, -1.2f, 0.5f, -0.3f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {1.03f, 0.43f, 0.33f},
                {0.0f, 1.38f, 0.0f},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {-0.5f, 0.8f, -0.2f, 0.4f},
                {0.2f, -0.3f, 0.5f, -0.1f},
                {0.4f, 0.1f, -0.4f, 0.6f},
            },
            .biases = {1.0f, 0.5f, -0.2f},
            .dot_product_func = nn_dot_product,
            .act_func_vector = nn_activation_func_softmax,
            .batch_size = 2,
            .inputs = {
                {0.5f, 0.1f, -0.2f, 0.4f},
                {1.2f, -1.2f, 0.5f, -0.3f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {0.48890266, 0.26831547, 0.24278187},
                {0.0875518, 0.75917368, 0.15327452},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {0.6f, -0.1f, 0.2f, 0.3f},
                {-0.4f, 0.2f, -0.5f, 0.1f},
                {0.1f, 0.4f, 0.2f, -0.2f},
            },
            .biases = {0.2f, -0.3f, 0.4f},
            .dot_product_func = nn_dot_product,
            .act_func_scalar = nn_activation_func_identity,
            .batch_size = 3,
            .inputs = {
                {2.0f, -1.5f, 0.5f, 0.6f},
                {-1.2f, 1.3f, -0.4f, 0.5f},
                {0.5f, 0.6f, -1.0f, 0.2f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {1.83f, -1.59f, -0.02f},
                {-0.58f, 0.69f, 0.62f},
                {0.3f, 0.14f, 0.45f},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {0.6f, -0.1f, 0.2f, 0.3f},
                {-0.4f, 0.2f, -0.5f, 0.1f},
                {0.1f, 0.4f, 0.2f, -0.2f},
            },
            .biases = {0.2f, -0.3f, 0.4f},
            .dot_product_func = nn_dot_product,
            .act_func_vector = nn_activation_func_softmax,
            .batch_size = 3,
            .inputs = {
                {2.0f, -1.5f, 0.5f, 0.6f},
                {-1.2f, 1.3f, -0.4f, 0.5f},
                {0.5f, 0.6f, -1.0f, 0.2f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {0.84037173, 0.02749061, 0.13213767},
                {0.12688794, 0.45182925, 0.4212828},
                {0.33178742, 0.28273059, 0.38548199},
            },
        },

        {
            .input_size = 4,
            .output_size = 3,
            .weights = {
                {0.6f, -0.1f, 0.2f, 0.3f},
                {-0.4f, 0.2f, -0.5f, 0.1f},
                {0.1f, 0.4f, 0.2f, -0.2f},
            },
            .biases = {0.2f, -0.3f, 0.4f},
            .dot_product_func = nn_dot_product,
            .act_func_scalar = nn_activation_func_relu,
            .batch_size = 3,
            .inputs = {
                {2.0f, -1.5f, 0.5f, 0.6f},
                {-1.2f, 1.3f, -0.4f, 0.5f},
                {0.5f, 0.6f, -1.0f, 0.2f},
            },
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_outputs = {
                {1.83f, 0.0f, 0.0f},
                {0.0f, 0.69f, 0.62f},
                {0.3f, 0.14f, 0.45f},
            },
        },
    };

    run_test_cases(test_cases, N_TEST_CASES, "NNLayer");
    return 0;
}
