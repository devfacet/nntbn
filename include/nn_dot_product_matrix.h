#ifndef NN_DOT_PRODUCT_MATRIX_H
#define NN_DOT_PRODUCT_MATRIX_H

typedef struct {
    int rows;
    int cols;
    float *data;
} NNMatrix;

// NNDotProdMatrixFunc represents a function that calculates
// the dot product of two matrices.
typedef void (*NNDotProdMatrixFunc)(const NNMatrix *a, const float b[], int b_cols, float output[]);

// nn_dot_product_matrix calculates the dot product of two matrices.
//
// PRECONDITIONS:
// The dimensions of the input matrices and the resultant matrix
// must be compatible.
void nn_dot_product_matrix(const NNMatrix *a, const float b[], int b_cols, float output[]);

#endif // NN_DOT_PRODUCT_MATRIX_H
