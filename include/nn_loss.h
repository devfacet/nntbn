#ifndef NN_LOSS_H
#define NN_LOSS_H

#include "nn_error.h"
#include "nn_tensor.h"
#include <stddef.h>

/**
 * @brief Returns the cross-entropy loss between the predictions and actual tensors.
 *
 * @param predictions The predictions (output of the network) tensor.
 * @param actual The actual (ground truth) tensor (one-hot encoded or categorical).
 * @param error The error instance to set if an error occurs.
 *
 * @return The cross-entropy loss.
 */
NNTensorUnit nn_loss_cross_entropy(const NNTensor *predictions, const NNTensor *actual, NNError *error);

#endif // NN_LOSS_H
