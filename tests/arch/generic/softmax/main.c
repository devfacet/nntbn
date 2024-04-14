#include "nn_activation.h"
#include "nn_config.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// N_TEST_CASES defines the number of test cases.
#define N_TEST_CASES 4
// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
#define DEFAULT_OUTPUT_TOLERANCE 0.000001f

// TestCase defines a single test case.
typedef struct {
    float input[NN_SOFTMAX_MAX_SIZE];
    size_t input_size;
    NNActivationFunctionVector activation_func;
    float output_tolerance;
    float expected_output[NN_SOFTMAX_MAX_SIZE];
} TestCase;

// run_test_cases runs the test cases.
void run_test_cases(TestCase *test_cases, int n_cases, char *info, NNActivationFunctionVector activation_func) {
    for (int i = 0; i < n_cases; ++i) {
        TestCase tc = test_cases[i];
        NNError error;

        float output[NN_SOFTMAX_MAX_SIZE];
        const bool result = activation_func(tc.input, output, tc.input_size, &error);
        assert(result == true);
        assert(error.code == NN_ERROR_NONE);
        float sum = 0;
        for (size_t i = 0; i < tc.input_size; ++i) {
            assert(fabs(output[i] - tc.expected_output[i]) < tc.output_tolerance);
            sum += output[i];
        }
        assert(sum == 1.0f);
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main() {
    TestCase test_cases[N_TEST_CASES] = {
        {
            .input = {1.0, 2.0, 3.0},
            .input_size = 3,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {0.09003057317038046, 0.24472847105479767, 0.6652409557748219},
        },

        {
            .input = {-1.0, -2.0, -3.0},
            .input_size = 3,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {0.6652409557748219, 0.24472847105479764, 0.09003057317038046},
        },

        {
            .input = {3.12, 0.845, -0.917},
            .input_size = 3,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {0.89250074, 0.09174632, 0.01575295},
        },

        {
            .input = {1.8, -3.21, 2.44},
            .input_size = 3,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {0.34445323, 0.00229781, 0.65324896},
        },

    };
    run_test_cases(test_cases, N_TEST_CASES, "nn_activation_func_softmax", nn_activation_func_softmax);

    return 0;
}
