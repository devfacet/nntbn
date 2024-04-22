#include "nn_dot_product_matrix.h"
#include "nn_debug.h"
#include <stddef.h>
#include <string.h>

// nn_dot_product_matrix calculates the dot product of two matrices.
void nn_dot_product_matrix(const NNMatrix *a, const float b[], int b_cols, float output[]) {
    NN_DEBUG_PRINT(5, "function %s called with matrix a sized %dX%d and matrix b sized %dX%d\n", __func__, a->rows, a->cols, a->cols, b_cols);

    int output_rows = a->cols;
    int output_cols = b_cols;

    // Initialize the result matrix.
    for (int i = 0; i < output_rows * output_cols; i++) {
            output[i] = 0.0f;
    }

    // Multiply two square matrices.
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < output_cols; j++) {
            for (int k = 0; k < output_rows; k++) {
                output[i * output_cols + j] = output[i * output_cols + j] + a->data[i * a->cols + k] * b[k * b_cols + j];
            }
        }
    }
}
