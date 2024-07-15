#include "nn_argmax.h"
#include "nn_tensor.h"
#include "nn_test.h"
#include <stdio.h>
#include <time.h>

void test_nn_argmax() {
    struct timespec start, end;
    long long total_time = 0;
    const int iterations = 100000;
    NNTensor *inputs[iterations];
    for (int i = 0; i < iterations; i++) {
        inputs[i] = nn_tensor_init_NNTensor(1, (const size_t[]){1}, false, NULL, NULL);
        inputs[i]->data[0] = (NNTensorUnit)rand() / (NNTensorUnit)RAND_MAX;
    }
    NNError error = {0};
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        nn_argmax(inputs[i], &error);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = nn_timespec_diff_ns(&start, &end);
    printf("avg_time_ns=%lld total_time_ms=%lld info=nn_argmax\n", total_time / iterations, total_time / 1000000);
}
