#ifndef NN_DOT_PRODUCT_H
#define NN_DOT_PRODUCT_H

#include "nn_error.h"
#include <stddef.h>

// NNDotProductFunction represents a function that calculates the dot product of two vectors.
typedef float (*NNDotProductFunction)(const float *a, const float *b, size_t length);

// nn_dot_product_generic calculates the dot product of two vectors.
float nn_dot_product_generic(const float *a, const float *b, size_t length);

#endif // NN_DOT_PRODUCT_H
