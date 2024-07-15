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

# Layer represents a layer.
class Layer:
    def __init__(self, inputs_size, output_size, layer_idx, layer_type, act_func_name, weights=None, biases=None):
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.inputs_size = inputs_size
        self.output_size = output_size
        self.act_func_name = act_func_name
        self.act_func = globals()[f"nn_act_func_{act_func_name}"]
        self.weights = weights
        self.biases = biases

    def compute_outputs(self, inputs):
        if self.weights is None:
            return inputs
        z = np.dot(inputs, self.weights.T)
        if self.biases is not None:
            z += self.biases
        return self.act_func(z)

# Returns test cases by the given parameters.
def generate_test_cases(batch_sizes, input_sizes, output_sizes, hidden_layers):
    test_cases = []
    for batch_size in batch_sizes:
        for inputs_size in input_sizes:
            for output_size in output_sizes:
                for hidden_layer in hidden_layers:
                    # Init vars
                    layers = []
                    inputs = np.random.uniform(-1.0, 1.0, (batch_size, inputs_size))

                    # Input layer (no activation function, no weights, no biases)
                    layer = Layer(inputs_size, inputs_size, 0, 0, "identity")
                    layers.append(layer)
                    last_outputs = layer.compute_outputs(inputs)
                    last_outputs_size = inputs_size

                    # Hidden layers (only non-linear activation functions)
                    for i in range(hidden_layer):
                        hidden_layer_size = np.random.choice(output_sizes)
                        weights = np.random.uniform(-0.5, 0.5, (hidden_layer_size, last_outputs_size))
                        biases = np.random.uniform(-0.5, 0.5, hidden_layer_size)
                        layer = Layer(last_outputs_size, hidden_layer_size, i+1, 1, "relu", weights, biases)
                        layers.append(layer)
                        last_outputs = layer.compute_outputs(last_outputs)
                        last_outputs_size = hidden_layer_size

                    # Output layer (softmax or sigmoid activation function)
                    act_func_name = "softmax" if output_size > 1 else "sigmoid"
                    weights = np.random.uniform(-0.5, 0.5, (output_size, last_outputs_size))
                    biases = np.random.uniform(-0.5, 0.5, output_size)
                    layer = Layer(last_outputs_size, output_size, len(layers), 2, act_func_name, weights, biases)
                    layers.append(layer)
                    last_outputs = layer.compute_outputs(last_outputs)

                    # Generate the partial C code
                    expected_outputs_c = ", ".join(map(str, last_outputs.flatten()))
                    layer_configs_c = []
                    for layer in layers:
                        if layer.act_func_name == "softmax":
                            act_func_c = act_func_map[layer.act_func_name]
                            act_func_type = "NN_ACT_FUNC_TENSOR"
                        else:
                            act_func_c = act_func_map[layer.act_func_name]
                            act_func_type = "NN_ACT_FUNC_SCALAR"

                        weights_c = ", ".join(map(str, layer.weights.flatten())) if layer.weights is not None else None
                        biases_c = ", ".join(map(str, layer.biases.flatten())) if layer.biases is not None else None
                        inputs_c = ", ".join(map(str, inputs.flatten())) if layer.layer_idx == 0 else None
                        layer_configs_c.append(f"""
                        {{
                            .layer_idx = {layer.layer_idx},
                            .layer_type = {layer.layer_type},
                            .inputs_size = {layer.inputs_size},
                            .output_size = {layer.output_size},
                            .mat_mul_func = {'NULL' if layer.layer_idx == 0 else 'nn_mat_mul'},
                            .mat_transpose_func = {'NULL' if layer.layer_idx == 0 else 'nn_mat_transpose'},
                            .act_func = nn_act_func_init({act_func_type}, {act_func_c}),
                            .weights = {"nn_tensor_init_NNTensor(2, (const size_t[]){" + str(layer.output_size) + ", " + str(layer.inputs_size) + "}, false, (const NNTensorUnit[]){ " + weights_c + "}, NULL)" if weights_c is not None else 'NULL'},
                            .biases = {"nn_tensor_init_NNTensor(1, (const size_t[]){" + str(layer.output_size) + "}, false, (const NNTensorUnit[]){ " + biases_c + "}, NULL)" if biases_c is not None else 'NULL'},
                            .inputs = {"nn_tensor_init_NNTensor(2, (const size_t[]){" + str(batch_size) + ", " + str(layer.inputs_size) + "}, false, (const NNTensorUnit[]){ " + inputs_c + "}, NULL)" if inputs_c is not None else 'NULL'},
                        }},""")

                    test_case = f"""
                    {{
                        .batch_size = {batch_size},
                        .expected_outputs = nn_tensor_init_NNTensor(2, (const size_t[]){{{batch_size}, {output_size}}}, false, (const NNTensorUnit[]){{{expected_outputs_c}}}, NULL),
                        .output_tolerance = default_output_tolerance,
                        .n_layers = {len(layers)},
                        .layers = (TestCaseLayer[]){{ {''.join(layer_configs_c)}\n}},
                    }}"""
                    test_cases.append(test_case)

    return test_cases

# Generate test cases
np.random.seed(2024)
batch_sizes = [1, 2, 3, 4]
input_sizes = [1, 2, 3, 4]
output_sizes = [1, 2, 3, 4]
hidden_layers = [1, 2, 3, 4]
test_cases = generate_test_cases(batch_sizes, input_sizes, output_sizes, hidden_layers)

print(f"TestCase test_cases[] = {{{','.join(test_cases)},\n}};")
