import numpy as np

NN_EPSILON = 1e-7

# Returns the cross-entropy loss between the predictions and actual.
def nn_loss_cross_entropy(predictions, actual):
    batch_size = predictions.shape[0]
    predictions = np.clip(predictions, NN_EPSILON, 1 - NN_EPSILON)

    if len(actual.shape) == 2:
        # One-hot encoded
        correct_confidences = np.sum(predictions * actual, axis=1)
    else:
        # Categorical
        correct_confidences = predictions[np.arange(batch_size), actual.astype(int)]

    loss = -np.mean(np.log(correct_confidences))

    return loss


# Generates a test case.
def generate_test_case(predictions, actual):
    predictions_c = ", ".join(map(str, predictions.flatten()))
    actual_c = ", ".join(map(str, actual.flatten()))
    expected_value = nn_loss_cross_entropy(predictions, actual)
    actual_size_str = f"{len(actual)}, {len(actual[0])}" if len(actual.shape) > 1 else f"{len(actual)}"
    return f"""
    {{
        .predictions = nn_tensor_init_NNTensor(2, (const size_t[]){{{len(predictions)}, {len(predictions[0])}}}, false, (const NNTensorUnit[]){{{predictions_c}}}, NULL),
        .actual = nn_tensor_init_NNTensor({len(actual.shape)}, (const size_t[]){{{actual_size_str}}}, false, (const NNTensorUnit[]){{{actual_c}}}, NULL),
        .expected_value = {expected_value}f,
        .expected_tolerance = default_expected_tolerance,
    }}"""


# Generate test cases
np.random.seed(2024)
test_cases = []
inputs = [
    # One-hot encoded
    (np.array([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3], [0.8, 0.1, 0.1]]), np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])),
    (np.array([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
    (np.array([[0.3, 0.4, 0.3], [0.6, 0.3, 0.1], [0.5, 0.2, 0.3]]), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])),

    # Categorical labels
    (np.array([[0.1, 0.7, 0.2], [0.3, 0.4, 0.3], [0.8, 0.1, 0.1]]), np.array([1, 0, 1])),
    (np.array([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]]), np.array([0, 1, 2])),
    (np.array([[0.3, 0.4, 0.3], [0.6, 0.3, 0.1], [0.5, 0.2, 0.3]]), np.array([2, 1, 0])),
]
for predictions, actual in inputs:
    test_cases.append(generate_test_case(predictions, actual))

print(f"TestCase test_cases[] = {{{', '.join(test_cases)},\n}};")
