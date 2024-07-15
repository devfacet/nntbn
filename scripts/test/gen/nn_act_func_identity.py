# This script generates test cases for nn_act_func_identity function.

import numpy as np

# Generates a test case.
def generate_test_case(input):
    return f"""
    {{
        .act_func = nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_identity),
        .input = {input},
        .expected_value = {input},
        .expected_tolerance = default_expected_tolerance,
    }}"""

# Generate test cases
np.random.seed(2024)
test_cases = []
inputs = [-1.0, 0.0, 1.0]
for input in inputs:
    test_cases.append(generate_test_case(input))

print(f"TestCase test_cases[] = {{{', '.join(test_cases)},\n}};")
