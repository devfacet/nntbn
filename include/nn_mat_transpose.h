#ifndef NN_MAT_TRANSPOSE_H
#define NN_MAT_TRANSPOSE_H

#include "nn_error.h"
#include "nn_tensor.h"

/**
 * @brief Represents a matrix transpose function.
 *
 * @param input The tensor to transpose.
 * @param output The tensor instance to store the transposed tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false
 */
typedef bool (*NNMatTransposeFunc)(const NNTensor *input, NNTensor *output, NNError *error);

/**
 * @brief Computes a transpose of a 2-dimensional tensor.
 *
 * @param input The tensor to transpose.
 * @param output The tensor instance to store the transposed tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false
 */
bool nn_mat_transpose(const NNTensor *input, NNTensor *output, NNError *error);

#endif // NN_MAT_TRANSPOSE_H
