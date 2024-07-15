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

void test_nn_act_func() {
    const NNTensorUnit default_expected_tolerance = 0.000001f;

    // See scripts/test/gen/nn_act_func.py
    TestCase test_cases[] = {
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_softmax),
            .input = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.8, 0.2, 0.1}, NULL),
            .expected_value = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.48890265771885366, 0.26831546747340185, 0.24278187480774444}, NULL),
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_softmax),
            .input = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){-0.6, 0.0, 0.6}, NULL),
            .expected_value = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.1628071674674988, 0.2966540006808555, 0.5405388318516458}, NULL),
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_softmax),
            .input = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.3, -0.3, 0.0}, NULL),
            .expected_value = nn_tensor_init_NNTensor(2, (const size_t[]){1, 3}, false, (const NNTensorUnit[]){0.4367518169107908, 0.23969447920584977, 0.32355370388335947}, NULL),
            .expected_tolerance = default_expected_tolerance,
        },
    };

    const int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        NNError error = {0};
        NNTensor *output = nn_tensor_init_NNTensor(tc.input->dims, tc.input->sizes, false, NULL, NULL);
        bool success = nn_act_func(tc.act_func, tc.input, output, &error);
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
