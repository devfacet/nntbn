#ifndef NN_DOT_PRODUCT_MATRIX_H
#define NN_DOT_PRODUCT_MATRIX_H

#include "nn_error.h"
#include <stddef.h>

// MATRIX_MAX_ROWS defines the maximum number of rows in a matrix.
#ifndef MATRIX_MAX_ROWS
#define MATRIX_MAX_ROWS 3
#endif

// MATRIX_MAX_COLS defines the maximum number of columns in a matrix.
#ifndef MATRIX_MAX_COLS
#define MATRIX_MAX_COLS 3
#endif

// NNDotProductMatrixFunction represents a function that calculates
// the dot product of two matrices.
typedef void (*NNDotProductMatrixFunction)(float result[MATRIX_MAX_ROWS][MATRIX_MAX_COLS], const float a[MATRIX_MAX_ROWS][MATRIX_MAX_COLS], const float b[MATRIX_MAX_ROWS][MATRIX_MAX_COLS]);

// nn_dot_product_matrix calculates the dot product of two square
// matrices.
//
// The dimensions of the input matrices and the resultant matrix are
// implicitly the same.
void nn_dot_product_matrix(float result[MATRIX_MAX_ROWS][MATRIX_MAX_COLS], const float a[MATRIX_MAX_ROWS][MATRIX_MAX_COLS], const float b[MATRIX_MAX_ROWS][MATRIX_MAX_COLS]);

#endif // NN_DOT_PRODUCT_MATRIX_H
