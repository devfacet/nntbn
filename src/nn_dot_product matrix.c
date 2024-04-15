#define NN_DOT_PRODUCT_MATRIX_C
#include "nn_dot_product_matrix.h"
#include "nn_debug.h"
#include <stddef.h>
#include <string.h>

// nn_dot_product_matrix calculates the dot product of two square
// matrices.
void nn_dot_product_matrix(float result[MATRIX_MAX_ROWS][MATRIX_MAX_COLS], const float a[MATRIX_MAX_ROWS][MATRIX_MAX_COLS], const float b[MATRIX_MAX_ROWS][MATRIX_MAX_COLS]) {
    NN_DEBUG_PRINT(5, "function %s called\n", __func__);

    // Initialize the result matrix.
    for (int i = 0; i < MATRIX_MAX_ROWS; i++) {
        memset(&result[i], 0, MATRIX_MAX_COLS * sizeof(float));
    }

    // Multiply two square matrices.
    for (int i = 0; i < MATRIX_MAX_ROWS; i++) {
        for (int j = 0; j < MATRIX_MAX_COLS; j++) {
            for (int k = 0; k < MATRIX_MAX_ROWS; k++) {
                result[i][j] = result[i][j] + a[i][k] * b[k][j];
            }
        }
    }
}
