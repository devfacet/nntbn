#ifndef NN_DOT_PRODUCT_MATRIX_H
#define NN_DOT_PRODUCT_MATRIX_H

// NNDotProdMatrixFunc represents a function that calculates
// the dot product of two matrices.
typedef void (*NNDotProdMatrixFunc)(const float a[], int a_rows, int a_cols, const float b[], int b_cols, float output[]);

// nn_dot_product_matrix calculates the dot product of two square
// matrices.
//
// The dimensions of the input matrices and the resultant matrix are
// implicitly the same.
void nn_dot_product_matrix(const float a[], int a_rows, int a_cols, const float b[], int b_cols, float output[]);

#endif // NN_DOT_PRODUCT_MATRIX_H
