#include "nn_accuracy.h"
#include "nn_test.h"
#include <stdio.h>
#include <time.h>

void test_nn_accuracy() {
    struct timespec start, end;
    long long total_time = 0;
    const int batch_size = 32;
    const int vector_size = 10;
    const int n_tensors = 100000;
    NNTensor *predictions[n_tensors];
    NNTensor *actual[n_tensors];
    for (int i = 0; i < n_tensors; i++) {
        predictions[i] = nn_tensor_init_NNTensor(2, (const size_t[]){batch_size, vector_size}, false, NULL, NULL);
        actual[i] = nn_tensor_init_NNTensor(1, (const size_t[]){batch_size}, false, NULL, NULL);
        for (int j = 0; j < batch_size; ++j) {
            for (int k = 0; k < vector_size; ++k) {
                predictions[i]->data[j * vector_size + k] = (NNTensorUnit)rand() / (NNTensorUnit)RAND_MAX;
            }
            actual[i]->data[j] = rand() % vector_size;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < n_tensors; i++) {
        nn_accuracy(predictions[i], actual[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = nn_timespec_diff_ns(&start, &end);
    printf("perf: %s avg_time_ns=%lld total_time_ms=%lld\n", __func__, total_time / n_tensors, total_time / 1000000);
}
