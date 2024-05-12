#ifndef NN_DOT_PROD_NEON_H
#define NN_DOT_PROD_NEON_H

#include "nn_error.h"
#include "nn_tensor.h"

/**
 * @brief Returns the dot product of two 1-dimensional tensors.
 *
 * @param a The first tensor.
 * @param b The second tensor.
 * @param error The error instance to set if an error occurs.
 *
 * @return The result of the dot product or NAN if an error occurs.
 */
NNTensorUnit nn_dot_prod_neon(const NNTensor *a, const NNTensor *b, NNError *error);

#endif // NN_DOT_PROD_NEON_H
