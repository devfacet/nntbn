#include "nn_activation.h"
#include "nn_test.h"
#include <stdio.h>
#include <time.h>

void test_nn_act_func_sigmoid() {
    struct timespec start, end;
    long long total_time = 0;
    const int iterations = 100000;
    NNTensorUnit inputs[iterations];
    for (int i = 0; i < iterations; i++) {
        inputs[i] = (NNTensorUnit)rand() / (NNTensorUnit)RAND_MAX;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        nn_act_func_sigmoid(inputs[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = nn_timespec_diff_ns(&start, &end);
    printf("avg_time_ns=%lld total_time_ms=%lld info=nn_act_func_sigmoid\n", total_time / iterations, total_time / 1000000);
}
