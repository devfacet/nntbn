#include "nn_activation.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    NNActFunc act_func;
    NNTensorUnit input;
    NNTensorUnit expected_value;
    NNTensorUnit expected_tolerance;
} TestCase;

void test_nn_act_func_sigmoid() {
    const NNTensorUnit default_expected_tolerance = 0.000001f;

    // See scripts/test/gen/nn_act_func_sigmoid.py
    TestCase test_cases[] = {
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_sigmoid),
            .input = -2.0,
            .expected_value = 0.11920292202211755,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_sigmoid),
            .input = -1.0,
            .expected_value = 0.2689414213699951,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_sigmoid),
            .input = 0.0,
            .expected_value = 0.5,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_sigmoid),
            .input = 1.0,
            .expected_value = 0.7310585786300049,
            .expected_tolerance = default_expected_tolerance,
        },
        {
            .act_func = nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_sigmoid),
            .input = 2.0,
            .expected_value = 0.8807970779778823,
            .expected_tolerance = default_expected_tolerance,
        },
    };

    const int n_cases = sizeof(test_cases) / sizeof(test_cases[0]);
    for (int i = 0; i < n_cases; i++) {
        TestCase tc = test_cases[i];

        const NNTensorUnit output = tc.act_func.scalar_func(tc.input);
        assert(isnan(output) == false);
        assert(fabs(output - tc.expected_value) < tc.expected_tolerance);
        printf("passed: %s case=%d\n", __func__, i + 1);
    }
}
