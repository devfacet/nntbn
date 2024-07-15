#ifndef NN_ACCURACY_H
#define NN_ACCURACY_H

#include "nn_tensor.h"
#include "nn_error.h"

/**
 * @brief Returns the accuracy between the predictions and actual tensors.
 *
 * @param predictions The predictions (output of the network) tensor.
 * @param actual The actual (ground truth) tensor (one-hot encoded or categorical).
 * @param error The error instance to set if an error occurs.
 *
 * @return The accuracy.
 */
NNTensorUnit nn_accuracy(const NNTensor *predictions, const NNTensor *actual, NNError *error);

#endif // NN_ACCURACY_H
