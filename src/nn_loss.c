#include "nn_loss.h"
#include "nn_constants.h"
#include "nn_debug.h"
#include "nn_error.h"
#include "nn_tensor.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

// TODO: Implement macro for fminf and fmaxf
// TODO: Implement macro for logf

NNTensorUnit nn_loss_cross_entropy(const NNTensor *predictions, const NNTensor *actual, NNError *error) {
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

    // Compute the cross-entropy loss
    NNTensorUnit loss = 0;
    if (one_hot) {
        // Iterate over the batch
        for (size_t i = 0; i < batch_size; i++) {
            // Iterate over the classes
            for (size_t j = 0; j < num_classes; j++) {
                // Clip the predictions value to avoid log(0)
                NNTensorUnit predictions_val = fminf(fmaxf(predictions->data[i * num_classes + j], NN_EPSILON), 1 - NN_EPSILON);
                // If the actual value is greater than 0
                // if (actual->data[i * num_classes + j] > 0) {
                //     loss -= logf(predictions_val);
                // }
                loss -= actual->data[i * num_classes + j] * logf(predictions_val);
            }
        }
    } else {
        // Iterate over the batch
        for (size_t i = 0; i < batch_size; i++) {
            // Clip the predictions value to avoid log(0)
            size_t class_idx = (size_t)actual->data[i];
            NNTensorUnit predictions_val = fminf(fmaxf(predictions->data[i * num_classes + class_idx], NN_EPSILON), 1 - NN_EPSILON);
            loss -= logf(predictions_val);
        }
    }

    // Average the loss
    loss /= batch_size;

    return loss;
}
