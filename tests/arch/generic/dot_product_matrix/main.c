#include "nn_config.h"
#include "nn_debug.h"
#include "nn_dot_product_matrix.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

// N_TEST_CASES defines the number of test cases.
#define N_TEST_CASES 4
// DEFAULT_OUTPUT_TOLERANCE defines the default tolerance for comparing output values.
#define DEFAULT_OUTPUT_TOLERANCE 0.0001f

// TestCase defines a single test case.
typedef struct {
    float a[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS];
    float b[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS];
    float bias;
    NNDotProdMatrixFunc dot_product_matrix_func;
    float output_tolerance;
    float expected_output[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS];
} TestCase;

// run_test_cases runs the test cases.
void run_test_cases(TestCase *test_cases, int n_cases, char *info, NNDotProdMatrixFunc dot_product_matrix_func) {
    for (int i = 0; i < n_cases; ++i) {
        TestCase tc = test_cases[i];

        float output[NN_MATRIX_MAX_ROWS][NN_MATRIX_MAX_COLS];

        NN_DEBUG_PRINT(5, "A:\n");
        for (int i = 0; i < NN_MATRIX_MAX_ROWS; i++) {
            for (int j = 0; j < NN_MATRIX_MAX_COLS; j++) {
                NN_DEBUG_PRINT(5, " %f", tc.a[i][j]);
            }
            NN_DEBUG_PRINT(5, "\n");
        }

        NN_DEBUG_PRINT(5, "B:\n");
        for (int i = 0; i < NN_MATRIX_MAX_ROWS; i++) {
            for (int j = 0; j < NN_MATRIX_MAX_COLS; j++) {
                NN_DEBUG_PRINT(5, " %f", tc.b[i][j]);
            }
            NN_DEBUG_PRINT(5, "\n");
        }

        dot_product_matrix_func(tc.a, tc.b, output);
            
        NN_DEBUG_PRINT(5, "C:\n");
        for (int i = 0; i < NN_MATRIX_MAX_ROWS; i++) {
            for (int j = 0; j < NN_MATRIX_MAX_COLS; j++) {
                NN_DEBUG_PRINT(5, " %f", tc.expected_output[i][j]);
            }
            NN_DEBUG_PRINT(5, "\n");
        }

        for (int m = 0; m < NN_MATRIX_MAX_ROWS; m++) {
            for (int n = 0; n < NN_MATRIX_MAX_COLS; n++) {
                assert(isnan(output[m][n]) == false);
                assert(fabs(output[m][n] - tc.expected_output[m][n]) < tc.output_tolerance);
            }
        }
        printf("passed: %s case=%d info=%s\n", __func__, i + 1, info);
    }
}

int main() {
    // nn_set_debug_level(10);
    
    TestCase test_cases[N_TEST_CASES] = {
        {
            .a = {{0}},
            .b = {{0}},
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {{0}},
        },

        {
            .a = {{ 3, 0},
                  {-1, 2},
                  { 1, 1}},
            .b = {{4, -1},
                  {0,  2}},
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {{ 12, -3},
                                {-4,   5},
                                { 4,   1}},
        },

        {
            .a = {{ 1, 5, 2},
                  {-1, 0, 1},
                  { 3, 2, 4}},
            .b = {{ 6, 1, 3},
                  {-1, 1, 2},
                  { 4, 1, 3}},
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {{ 9, 8, 19},
                                {-2, 0,  0},
                                {32, 9, 25}},
        },

        {
            .a = {{1, 2},
                  {3, 4}},
            .b = {{5},
                  {6}},
            .output_tolerance = DEFAULT_OUTPUT_TOLERANCE,
            .expected_output = {{17},
                                {39}},
        },

    };
    run_test_cases(test_cases, N_TEST_CASES, "nn_dot_product_matrix", nn_dot_product_matrix);
    return 0;
}
