#include "nn_app.h"
#include "nn_config.h"
#include "nn_dot_prod.h"
#include "nn_error.h"
#include "nn_tensor.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

typedef struct {
    NNTensor *vec_a;
    NNTensor *vec_b;
    NNTensorUnit expected_output;
    NNTensorUnit output_tolerance;
} TestCase;

void run_test_cases(TestCase *test_cases, int n_cases, char *info) {
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        NNError error = {0};
        const NNTensorUnit output = nn_dot_prod(tc.vec_a, tc.vec_b, &error);
        assert(isnan(output) == false);
        assert(error.code == NN_ERROR_NONE);
        assert(fabs(output - tc.expected_output) < tc.output_tolerance);
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);
    // nn_set_debug_level(5); // for debugging

    const float default_output_tolerance = 0.000001f;
    const int test_cases_size = 4;
    TestCase test_cases[] = {
        {
            .vec_a = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, (const NNTensorUnit[]){0.5f}, NULL),
            .vec_b = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, (const NNTensorUnit[]){0.2f}, NULL),
            .expected_output = 0.1f,
            .output_tolerance = default_output_tolerance,
        },

        {
            .vec_a = nn_tensor_init_NNTensor(1, (const size_t[]){2}, false, (const NNTensorUnit[]){-0.6f, -1.1f}, NULL),
            .vec_b = nn_tensor_init_NNTensor(1, (const size_t[]){2}, false, (const NNTensorUnit[]){-0.2f, 0.5f}, NULL),
            .expected_output = -0.43f,
            .output_tolerance = default_output_tolerance,
        },

        {
            .vec_a = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){1.5f, 2.0f, -1.0f}, NULL),
            .vec_b = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){0.4f, 0.4f, -0.2f}, NULL),
            .expected_output = 1.6f,
            .output_tolerance = default_output_tolerance,
        },

        {
            .vec_a = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){0.36f, 0.17f, 0.96f, 0.12f}, NULL),
            .vec_b = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){0.77f, 0.09f, 0.12f, 0.81f}, NULL),
            .expected_output = 0.504899f,
            .output_tolerance = default_output_tolerance,
        },

    };
    run_test_cases(test_cases, test_cases_size, "nn_dot_prod");

    return 0;
}
