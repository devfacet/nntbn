#ifndef NN_DOT_PRODUCT_NEON_H
#define NN_DOT_PRODUCT_NEON_H

#include <stddef.h>

// nn_dot_product_neon calculates the dot product of two vectors.
float nn_dot_product_neon(const float *a, const float *b, size_t length);

#endif // NN_DOT_PRODUCT_NEON_H
