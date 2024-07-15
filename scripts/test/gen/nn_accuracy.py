# This script generates test cases for nn_accuracy function.

import numpy as np

# Returns the softmax activation function result.
def nn_act_func_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Generates a one-hot encoded vector.
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

# Generates a test case.
def generate_test_case(batch_size, num_classes, one_hot):
    # Init vars
    predictions = np.random.uniform(0, 1, (batch_size, num_classes))
    predictions = nn_act_func_softmax(predictions)  # ensure the predictions are probabilities
    actual = np.random.randint(0, num_classes, batch_size)
    if one_hot:
        actual = one_hot_encode(actual, num_classes)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(actual, axis=1)) if one_hot else np.mean(np.argmax(predictions, axis=1) == actual)

    # Generate the partial C code
    predictions_c = ", ".join(map(str, predictions.flatten()))
    actual_c = ", ".join(map(str, actual.flatten()))
    return f"""
    {{
        .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){{{batch_size}, {num_classes}}}, false, (const NNTensorUnit[]){{{predictions_c}}}, NULL),
        .actual = nn_tensor_init_NNTensor({2 if one_hot else 1}, (const size_t[]){{{batch_size}{', ' + str(num_classes) if one_hot else ''}}}, false, (const NNTensorUnit[]){{{actual_c}}}, NULL),
        .expected_value = {accuracy},
        .expected_tolerance = default_expected_tolerance,
    }}"""

# Generates test cases.
def generate_test_cases(batch_sizes, num_classes_list, one_hot_encodings):
    test_cases = []
    for batch_size in batch_sizes:
        for num_classes in num_classes_list:
            for one_hot in one_hot_encodings:
                test_case = generate_test_case(batch_size, num_classes, one_hot)
                test_cases.append(test_case)

    return test_cases

# Generate test cases
np.random.seed(2024)
batch_sizes = [1, 2, 3, 4, 5]
num_classes_list = [2, 3, 4, 5, 6]
one_hot_encodings = [True, False]
test_cases = generate_test_cases(batch_sizes, num_classes_list, one_hot_encodings)

print(f"TestCase test_cases[] = {{{','.join(test_cases)},\n}};")
