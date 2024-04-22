#include "nn_debug.h"
#include "nn_dot_product_matrix.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// N_TEST_CASES defines the number of test cases.
#define N_TEST_CASES 5
// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
#define DEFAULT_OUTPUT_TOLERANCE 0.0001f

// TestCase defines a single test case.
typedef struct {
    float *b;
    int b_rows;
    int b_cols;
    NNDotProdMatrixFunc dot_product_matrix_func;
    float output_tolerance;
    float bias;
    float *expected_output;
    int expected_output_rows;
    int expected_output_cols;
    NNMatrix matrix_a;
} TestCase;

// run_test_cases runs the test cases.
void run_test_cases(TestCase *test_cases, int n_cases, char *info, NNDotProdMatrixFunc dot_product_matrix_func) {
    for (int i = 0; i < n_cases; ++i) {
        TestCase tc = test_cases[i];

        float output[tc.matrix_a.rows * tc.b_cols];

        NN_DEBUG_PRINT(5, "A:\n");
        for (int i = 0; i < tc.matrix_a.rows; i++) {
            for (int j = 0; j < tc.matrix_a.cols; j++) {
                NN_DEBUG_PRINT(5, " %f", tc.matrix_a.data[i * tc.matrix_a.cols + j]);
            }
            NN_DEBUG_PRINT(5, "\n");
        }

        NN_DEBUG_PRINT(5, "B:\n");
        for (int i = 0; i < tc.b_rows; i++) {
            for (int j = 0; j < tc.b_cols; j++) {
                NN_DEBUG_PRINT(5, " %f", tc.b[i * tc.b_cols + j]);
            }
            NN_DEBUG_PRINT(5, "\n");
        }

        dot_product_matrix_func(&tc.matrix_a, tc.b, tc.b_cols, output);
            
        NN_DEBUG_PRINT(5, "Output:\n");
        for (int i = 0; i < tc.matrix_a.rows; i++) {
            for (int j = 0; j < tc.b_cols; j++) {
                NN_DEBUG_PRINT(5, " %f", output[i * tc.b_cols + j]);
            }
            NN_DEBUG_PRINT(5, "\n");
        }

        NN_DEBUG_PRINT(5, "Expected:\n");
        for (int i = 0; i < tc.expected_output_rows; i++) {
            for (int j = 0; j < tc.expected_output_cols; j++) {
                NN_DEBUG_PRINT(5, " %f", tc.expected_output[i * tc.expected_output_cols + j]);
            }
            NN_DEBUG_PRINT(5, "\n");
        }

        for (int i = 0; i < tc.matrix_a.rows; i++) {
            for (int j = 0; j < tc.b_cols; j++) {
                assert(isnan(output[i * tc.b_cols + j]) == false);
                assert(fabs(output[i * tc.b_cols + j] - tc.expected_output[i * tc.b_cols + j]) < tc.output_tolerance);
            }
        }
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main() {
    // nn_set_debug_level(10);
    
    TestCase test_cases[N_TEST_CASES] = {
        // a is a 1x1 square matrix and b is a square 1x1 matrix
        {
            .matrix_a = {
                .rows = 1,
                .cols = 1,
                .data = (float[]){ 1 },
            },
            .b = (float[]){ 1 },
            .b_rows = 1,
            .b_cols = 1,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = (float[]){ 1 },
            .expected_output_rows = 1,
            .expected_output_cols = 1,
        },

        // a is a 3x3 square matrix and b is a square 3x3 matrix; both consist of all zeros
        {
            .matrix_a = {
                .rows = 3,
                .cols = 3,
                .data = (float[]){
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                },
            },
            .b = (float[]){
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            },
            .b_rows = 3,
            .b_cols = 3,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = (float[]){
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
            },
            .expected_output_rows = 3,
            .expected_output_cols = 3,
        },

        // a is a 3x3 square matrix and b is a square 3x3 matrix
        {
            .matrix_a = {
                .rows = 3,
                .cols = 3,
                .data = (float[]){
                     1, 5, 2,
                    -1, 0, 1,
                     3, 2, 4,
                },
            },
            .b = (float[]){
                 6, 1, 3,
                -1, 1, 2,
                 4, 1, 3,
            },
            .b_rows = 3,
            .b_cols = 3,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = (float[]){
                9, 8, 19,
                -2, 0,  0,
                32, 9, 25,
            },
            .expected_output_rows = 3,
            .expected_output_cols = 3,
        },

        // a is a 3x2 matrix and b is a 2x2 square matrix
        {
            .matrix_a = {
                .rows = 3,
                .cols = 2,
                .data = (float[]){
                     3, 0,
                    -1, 2,
                     1, 1,
                },
            },
            .b = (float[]){
                 4, -1,
                 0,  2,
            },
            .b_rows = 2,
            .b_cols = 2,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = (float[]){
                12, -3,
                -4,  5,
                 4,  1,
            },
            .expected_output_rows = 3,
            .expected_output_cols = 2,
        },

        // a is a 2x2 matrix and b is a 2x1 square matrix
        {
            .matrix_a = {
                .rows = 2,
                .cols = 2,
                .data = (float[]){
                    1, 2,
                    3, 4,
                },
            },
            .b = (float[]){
                 5,
                 6,
            },
            .b_rows = 2,
            .b_cols = 1,
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = (float[]){
                17,
                39,
            },
            .expected_output_rows = 2,
            .expected_output_cols = 1,
        },

    };
    run_test_cases(test_cases, N_TEST_CASES, "nn_dot_product_matrix", nn_dot_product_matrix);
    return 0;
}
