#ifndef NN_ARGMAX_H
#define NN_ARGMAX_H

#include "nn_tensor.h"
#include "nn_error.h"
#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Finds the index of the maximum value in a 1-dimensional tensor.
 *
 * @param input The input tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return The index of the maximum value.
 */
size_t nn_argmax(const NNTensor *input, NNError *error);

/**
 * @brief Finds the indices of the maximum values in each row of a 2-dimensional tensor batch.
 *
 * @param input The input tensor.
 * @param output The output tensor to store the indices.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false.
 */
bool nn_argmax_tensor_batch(const NNTensor *input, NNTensor *output, NNError *error);

#endif // NN_ARGMAX_H
