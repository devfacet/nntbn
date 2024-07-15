# This script generates test cases for nn_act_func_sigmoid function.

import numpy as np

# Returns the sigmoid activation function result.
def nn_act_func_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generates test cases.
def generate_test_case(input):
    expected_value = nn_act_func_sigmoid(input)
    return f"""
    {{
        .act_func = nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_sigmoid),
        .input = {input},
        .expected_value = {expected_value},
        .expected_tolerance = default_expected_tolerance,
    }}"""

# Generate test cases
np.random.seed(2024)
test_cases = []
inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
for input in inputs:
    test_cases.append(generate_test_case(input))

print(f"TestCase test_cases[] = {{{', '.join(test_cases)},\n}};")
