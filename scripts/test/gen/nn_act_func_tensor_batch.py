# This script generates test cases for nn_act_func_softmax function.

import numpy as np

# Returns the softmax activation function result.
def nn_act_func_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Generates a test case.
def generate_test_case(input):
    input_c = ", ".join(map(str, input.flatten()))
    expected_value = nn_act_func_softmax(input)
    expected_value_c = ", ".join(map(str, expected_value.flatten()))
    return f"""
    {{
        .act_func = nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_softmax),
        .input = nn_tensor_init_NNTensor(2, (const size_t[]){{1, {len(input)}}}, false, (const NNTensorUnit[]){{{input_c}}}, NULL),
        .expected_value = nn_tensor_init_NNTensor(2, (const size_t[]){{1, {len(input)}}}, false, (const NNTensorUnit[]){{{expected_value_c}}}, NULL),
        .expected_tolerance = default_expected_tolerance,
    }}"""

# Generate test cases
np.random.seed(2024)
test_cases = []
inputs = [
    np.array([-0.1, 0.2, 0.8, 0.0]),
    np.array([1.0, 0.5, -0.4, -1.0]),
    np.array([0.9, -0.3, 0.1, 0.0])
]
for input in inputs:
    test_cases.append(generate_test_case(input))

print(f"TestCase test_cases[] = {{{', '.join(test_cases)},\n}};")
