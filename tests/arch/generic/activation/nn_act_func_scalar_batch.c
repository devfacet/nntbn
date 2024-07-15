#include "nn_activation.h"
#include "nn_tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    NNActFunc act_func;
    NNTensor *input;
    NNTensor *expected_value;
    NNTensorUnit expected_tolerance;
} TestCase;

void test_nn_act_func_scalar_batch() {
    const NNTensorUnit default_expected_tolerance = 0.000001f;

    // See scripts/test/gen/nn_act_func_scalar_batch.py
    TestCase test_cases[] = {
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_relu),
            .input = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.8, 0.2, -0.1, 0.0}, NULL),
            .expected_value = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.8, 0.2, 0.0, 0.0}, NULL),
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_relu),
            .input = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){-0.6, 0.0, 0.6, -1.0}, NULL),
            .expected_value = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.0, 0.0, 0.6, 0.0}, NULL),
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_relu),
            .input = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.3, -0.3, 1.0, 0.0}, NULL),
            .expected_value = nn_tensor_init_NNTensor(2, (const size_t[]){1, 4}, false, (const NNTensorUnit[]){0.3, 0.0, 1.0, 0.0}, NULL),
            .expected_tolerance = default_expected_tolerance,
        },
    };

    const int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        NNError error = {0};
        NNTensor *output = nn_tensor_init_NNTensor(tc.input->dims, tc.input->sizes, false, NULL, NULL);
        bool success = nn_act_func_scalar_batch(tc.act_func.scalar_func, tc.input, output, &error);
        assert(success);
        assert(error.code == NN_ERROR_NONE);
        for (size_t j = 0; j < tc.input->sizes[0]; j++) {
            assert(fabs(output->data[j] - tc.expected_value->data[j]) < tc.expected_tolerance);
        }
        printf("passed: %s case=%d\n", __func__, i + 1);

        // Cleanup
        nn_tensor_destroy_NNTensor(output);
    }
}
