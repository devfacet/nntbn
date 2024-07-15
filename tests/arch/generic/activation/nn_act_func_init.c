#include "nn_activation.h"
#include <assert.h>
#include <stdio.h>

typedef struct {
    NNActFuncType type;
    void *func;
    NNActFunc expected_act_func;
} TestCase;

void test_nn_act_func_init() {
    TestCase test_cases[] = {
        {
            .type = NN_ACT_FUNC_SCALAR,
            .func = nn_act_func_identity,
            .expected_act_func = {
                .type = NN_ACT_FUNC_SCALAR,
                .scalar_func = nn_act_func_identity,
            },
        },
        {
            .type = NN_ACT_FUNC_TENSOR,
            .func = nn_act_func_softmax,
            .expected_act_func = {
                .type = NN_ACT_FUNC_TENSOR,
                .tensor_func = nn_act_func_softmax,
            },
        },
    };

    const int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        NNActFunc act_func = nn_act_func_init(tc.type, tc.func);
        assert(act_func.type == tc.expected_act_func.type);
        if (act_func.type == NN_ACT_FUNC_SCALAR) {
            assert(act_func.scalar_func == tc.expected_act_func.scalar_func);
        } else {
            assert(act_func.tensor_func == tc.expected_act_func.tensor_func);
        }
        printf("passed: %s case=%d\n", __func__, i + 1);
    }
}
