#include "nn_loss.h"
#include "nn_tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    NNTensor *predictions;
    NNTensor *actual;
    NNTensorUnit expected_value;
    NNTensorUnit expected_tolerance;
} TestCase;

void test_nn_loss_cross_entropy() {
    const NNTensorUnit default_expected_tolerance = 0.000001f;

    // See scripts/test/gen/nn_loss_cross_entropy.py
    TestCase test_cases[] = {
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.1, 0.7, 0.2, 0.3, 0.4, 0.3, 0.8, 0.1, 0.1}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0, 1, 0, 1, 0, 0, 0, 1, 0}, NULL),
            .expected_value = 1.2877442804195713f,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.2, 0.5, 0.3, 0.4, 0.4, 0.2, 0.7, 0.2, 0.1}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){1, 0, 0, 0, 1, 0, 0, 0, 1}, NULL),
            .expected_value = 1.6094379124341003f,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.3, 0.4, 0.3, 0.6, 0.3, 0.1, 0.5, 0.2, 0.3}, NULL),
            .actual = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0, 0, 1, 0, 1, 0, 1, 0, 0}, NULL),
            .expected_value = 1.0336975964039392f,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.1, 0.7, 0.2, 0.3, 0.4, 0.3, 0.8, 0.1, 0.1}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){1, 0, 1}, NULL),
            .expected_value = 1.2877442804195713f,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.2, 0.5, 0.3, 0.4, 0.4, 0.2, 0.7, 0.2, 0.1}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){0, 1, 2}, NULL),
            .expected_value = 1.6094379124341003f,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){3, 3}, false, (const NNTensorUnit[]){0.3, 0.4, 0.3, 0.6, 0.3, 0.1, 0.5, 0.2, 0.3}, NULL),
            .actual = nn_tensor_init_NNTensor(1, (const size_t[]){3}, false, (const NNTensorUnit[]){2, 1, 0}, NULL),
            .expected_value = 1.0336975964039392f,
            .expected_tolerance = default_expected_tolerance,
        },
    };

    const int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        NNError error = {0};
        NNTensorUnit loss = nn_loss_cross_entropy(tc.predictions, tc.actual, &error);
        assert(error.code == NN_ERROR_NONE);
        for (size_t j = 0; j < tc.predictions->sizes[0]; j++) {
            assert(fabs(loss - tc.expected_value) < tc.expected_tolerance);
        }
        printf("passed: %s case=%d\n", __func__, i + 1);
    }
}
