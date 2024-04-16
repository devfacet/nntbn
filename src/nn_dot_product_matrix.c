#include "nn_dot_product_matrix.h"
#include "nn_debug.h"
#include <stddef.h>
#include <string.h>

// nn_dot_product_matrix calculates the dot product of two square
// matrices.
void nn_dot_product_matrix(const float a[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS], const float b[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS], float output[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS]) {
    NN_DEBUG_PRINT(5, "function %s called\n", __func__);

    // Initialize the result matrix.
    for (int i = 0; i < NN_MATRIX_MAX_ROWS; i++) {
        memset(&output[i], 0, NN_MATRIX_MAX_COLS * sizeof(float));
    }

    // Multiply two square matrices.
    for (int i = 0; i < NN_MATRIX_MAX_ROWS; i++) {
        for (int j = 0; j < NN_MATRIX_MAX_COLS; j++) {
            for (int k = 0; k < NN_MATRIX_MAX_ROWS; k++) {
                output[i][j] = output[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}
