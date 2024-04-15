#include "nn_config.h"
#include "nn_neuron.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// N_TEST_CASES defines the number of test cases.
#define N_TEST_CASES 4
// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
#define DEFAULT_OUTPUT_TOLERANCE 0.0001f

// TestCase defines a single test case.
typedef struct {
    float a[4];
    float b[4];
    size_t vector_size;
    float bias;
    NNDotProdFunc dot_product_func;
    float output_tolerance;
    float expected_output;
} TestCase;

// run_test_cases runs the test cases.
void run_test_cases(TestCase *test_cases, int n_cases, char *info, NNDotProdFunc dot_product_func) {
    for (int i = 0; i < n_cases; ++i) {
        TestCase tc = test_cases[i];

        const float output = dot_product_func(tc.a, tc.b, tc.vector_size);
        assert(isnan(output) == false);
        assert(fabs(output - tc.expected_output) < tc.output_tolerance);
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main() {
    TestCase test_cases[N_TEST_CASES] = {
        {
            .a = {0.5f},
            .b = {0.2f},
            .vector_size = 1,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.1f,
        },

        {
            .a = {-0.6f, -1.1f},
            .b = {-0.2f, 0.5f},
            .vector_size = 2,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = -0.43f,
        },

        {
            .a = {1.5f, 2.0f, -1.0f},
            .b = {0.4f, 0.4f, -0.2f},
            .vector_size = 3,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 1.6f,
        },

        {
            .a = {0.36f, 0.17f, 0.96f, 0.12f},
            .b = {0.77f, 0.09f, 0.12f, 0.81f},
            .vector_size = 4,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = 0.5048,
        },

    };
    run_test_cases(test_cases, N_TEST_CASES, "nn_dot_prod", nn_dot_prod);
    return 0;
}
