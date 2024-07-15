#include "nn_argmax.h"
#include "nn_tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    NNTensor *input;
    size_t expected_index;
} TestCase;

const NNTensorUnit default_expected_tolerance = 0.000001f;

void test_nn_argmax() {
    TestCase test_cases[] = {
        {
            .input = nn_tensor_init_NNTensor(1, (const size_t[]){5}, false, (const NNTensorUnit[]){1.0, 3.0, 2.0, 5.0, 4.0}, NULL),
            .expected_index = 3,
        },
        {
            .input = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){0.1, 0.5, 0.2}, NULL),
            .expected_index = 1,
        },
        {
            .input = nn_tensor_init_NNTensor(1, (const size_t[]){4}, false, (const NNTensorUnit[]){-1.0, -2.0, -0.5, -3.0}, NULL),
            .expected_index = 2,
        },
    };

    const int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        NNError error = {0};
        size_t index = nn_argmax(tc.input, &error);
        assert(error.code == NN_ERROR_NONE);
        assert(index == tc.expected_index);
        printf("passed: %s case=%d\n", __func__, i + 1);

        // Cleanup
        nn_tensor_destroy_NNTensor(tc.input);
    }
}
