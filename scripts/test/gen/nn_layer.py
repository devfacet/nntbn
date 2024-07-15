# This script generates test cases for NNLayer struct.

import numpy as np

# Returns the identity activation function result.
def nn_act_func_identity(x):
    return x

# Returns the sigmoid activation function result.
def nn_act_func_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Returns the ReLU activation function result.
def nn_act_func_relu(x):
    return np.maximum(0, x)

# Returns the softmax activation function result.
def nn_act_func_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Activation function map
act_func_map = {
    "identity": "nn_act_func_identity",
    "sigmoid": "nn_act_func_sigmoid",
    "relu": "nn_act_func_relu",
    "softmax": "nn_act_func_softmax"
}

# Returns a test case by the given parameters.
def generate_test_case(batch_size, inputs_size, output_size, act_func_name):
    # Init vars
    weights = np.random.uniform(-0.5, 0.5, (output_size, inputs_size))
    biases = np.random.uniform(-0.5, 0.5, output_size)
    inputs = np.random.uniform(-2.0, 2.0, (batch_size, inputs_size))

    # Determine activation function
    if act_func_name == "softmax":
        act_func_c = act_func_map[act_func_name]
        act_func_type = "NN_ACT_FUNC_TENSOR"
        act_func = nn_act_func_softmax
    else:
        act_func_c = act_func_map[act_func_name]
        act_func_type = "NN_ACT_FUNC_SCALAR"
        act_func = globals()[f"nn_act_func_{act_func_name}"]

    # Calculate expected outputs
    raw_outputs = np.dot(inputs, weights.T) + biases
    expected_outputs = act_func(raw_outputs)

    # Generate the partial C code
    weights_c = ", ".join(map(str, weights.flatten()))
    biases_c = ", ".join(map(str, biases))
    inputs_c = ", ".join(map(str, inputs.flatten()))
    expected_outputs_c = ", ".join(map(str, expected_outputs.flatten()))
    return f"""
    {{
        .batch_size = {batch_size},
        .inputs_size = {inputs_size},
        .output_size = {output_size},
        .mat_mul_func = nn_mat_mul,
        .mat_transpose_func = nn_mat_transpose,
        .act_func = nn_act_func_init({act_func_type}, {act_func_c}),
        .weights = nn_tensor_init_NNTensor(2, (const size_t[]){{{output_size}, {inputs_size}}}, false, (const NNTensorUnit[]){{{weights_c}}}, NULL),
        .biases = nn_tensor_init_NNTensor(1, (const size_t[]){{{output_size}}}, false, (const NNTensorUnit[]){{{biases_c}}}, NULL),
        .inputs = nn_tensor_init_NNTensor(2, (const size_t[]){{{batch_size}, {inputs_size}}}, false, (const NNTensorUnit[]){{{inputs_c}}}, NULL),
        .expected_outputs = nn_tensor_init_NNTensor(2, (const size_t[]){{{batch_size}, {output_size}}}, false, (const NNTensorUnit[]){{{expected_outputs_c}}}, NULL),
        .output_tolerance = default_output_tolerance,
    }}"""

# Returns test cases by the given parameters.
def generate_test_cases(batch_sizes, input_sizes, output_sizes, act_functions):
    test_cases = []
    for batch_size in batch_sizes:
        for inputs_size in input_sizes:
            for output_size in output_sizes:
                for act_func_name in act_functions:
                    test_case = generate_test_case(batch_size, inputs_size, output_size, act_func_name)
                    test_cases.append(test_case)

    return test_cases

# Generate test cases
np.random.seed(2024)
batch_sizes = [1, 2, 3, 4]
input_sizes = [1, 2, 3, 4]
output_sizes = [1, 2, 3, 4]
act_functions = ["identity", "sigmoid", "relu", "softmax"]
test_cases = generate_test_cases(batch_sizes, input_sizes, output_sizes, act_functions)

print(f"TestCase test_cases[] = {{{','.join(test_cases)},\n}};")
