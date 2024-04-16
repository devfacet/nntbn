#ifndef NN_DOT_PRODUCT_MATRIX_H
#define NN_DOT_PRODUCT_MATRIX_H

#include "nn_error.h"
#include <stddef.h>

// NN_MATRIX_MAX_ROWS defines the maximum number of rows in a matrix.
#ifndef NN_MATRIX_MAX_ROWS
#define NN_MATRIX_MAX_ROWS 3
#endif

// NN_MATRIX_MAX_COLS defines the maximum number of columns in a matrix.
#ifndef NN_MATRIX_MAX_COLS
#define NN_MATRIX_MAX_COLS 3
#endif

// NNDotProdMatrixFunc represents a function that calculates
// the dot product of two matrices.
typedef void (*NNDotProdMatrixFunc)(const float a[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS], const float b[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS], float output[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS]);

// nn_dot_product_matrix calculates the dot product of two square
// matrices.
//
// The dimensions of the input matrices and the resultant matrix are
// implicitly the same.
void nn_dot_product_matrix(const float a[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS], const float b[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS], float output[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS]);

#endif // NN_DOT_PRODUCT_MATRIX_H
