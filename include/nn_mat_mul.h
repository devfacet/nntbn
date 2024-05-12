#ifndef NN_MAT_MUL_H
#define NN_MAT_MUL_H

#include "nn_error.h"
#include "nn_tensor.h"

/**
 * @brief Represents a matrix multiplication function.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param result The tensor instance to store the result.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false
 */
typedef bool (*NNMatMulFunc)(const NNTensor *a, const NNTensor *b, NNTensor *result, NNError *error);

/**
 * @brief Computes a matrix multiplication of two 2-dimensional tensors.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param result The tensor instance to store the result.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false
 */
bool nn_mat_mul(const NNTensor *a, const NNTensor *b, NNTensor *result, NNError *error);

#endif // NN_MAT_MUL_H
