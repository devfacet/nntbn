#include "nn_accuracy.h"
#include "nn_argmax.h"
#include "nn_debug.h"
#include "nn_error.h"
#include "nn_tensor.h"

NNTensorUnit nn_accuracy(const NNTensor *predictions, const NNTensor *actual, NNError *error) {
    NN_DEBUG_PRINT(5, "function %s called with predictions.dims=%zu actual.dims=%zu\n", __func__, predictions->dims, actual->dims);

    if (!(predictions->flags & NN_TENSOR_FLAG_INIT) || !(actual->flags & NN_TENSOR_FLAG_INIT)) {
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor predictions or actual is not initialized");
        return 0;
    } else if (predictions->dims != 2 || actual->dims < 1 || actual->dims > 2 || predictions->sizes[0] != actual->sizes[0]) {
        // Only one-hot encoded or categorical tensors with the same batch size are allowed
        nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "only 2-dimensional predictions tensor and 1 or 2-dimensional actual tensor with the same batch size are allowed");
        return 0;
    }

    // Determine the batch size, the number of classes and if the actual tensor is one-hot encoded
    size_t batch_size = predictions->sizes[0];
    size_t num_classes = predictions->sizes[1];
    bool one_hot = (actual->dims == 2 && actual->sizes[1] == num_classes);

    // Find the index of the maximum value in the predictions tensor
    NNTensor *predictions_argmax = nn_tensor_init_NNTensor(1, (size_t[]){batch_size}, true, NULL, error);
    if (!predictions_argmax) {
        return 0;
    }
    if (!nn_argmax_tensor_batch(predictions, predictions_argmax, error)) {
        nn_tensor_destroy_NNTensor(predictions_argmax);
        return 0;
    }

    // Compute the accuracy
    NNTensorUnit accuracy = 0;
    if (one_hot) {
        // Find the index of the maximum value in the actual tensor
        NNTensor *actual_argmax = nn_tensor_init_NNTensor(1, (size_t[]){batch_size}, true, NULL, error);
        if (!actual_argmax) {
            nn_tensor_destroy_NNTensor(predictions_argmax);
            return 0;
        }
        if (!nn_argmax_tensor_batch(actual, actual_argmax, error)) {
            nn_tensor_destroy_NNTensor(predictions_argmax);
            nn_tensor_destroy_NNTensor(actual_argmax);
            return 0;
        }

        // Iterate over the batch
        for (size_t i = 0; i < batch_size; i++) {
            // If the predicted class is equal to the actual class
            if (predictions_argmax->data[i] == actual_argmax->data[i]) {
                accuracy += 1;
            }
        }
        nn_tensor_destroy_NNTensor(actual_argmax);
    } else {
        // Iterate over the batch
        for (size_t i = 0; i < batch_size; i++) {
            // Find the index of the maximum value in the actual tensor
            size_t actual_argmax = (size_t)actual->data[i];
            // If the predicted class is equal to the actual class
            if (predictions_argmax->data[i] == actual_argmax) {
                accuracy += 1;
            }
        }
    }
    nn_tensor_destroy_NNTensor(predictions_argmax);

    // Average the accuracy
    accuracy /= batch_size;

    return accuracy;
}
