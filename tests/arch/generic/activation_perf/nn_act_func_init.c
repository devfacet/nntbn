#define _POSIX_C_SOURCE 199309L

#include "nn_activation.h"
#include "nn_test.h"
#include <stdio.h>
#include <time.h>

void test_nn_act_func_init() {
    struct timespec start, end;
    long long total_time = 0;
    const int iterations = 100000;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        nn_act_func_init(NN_ACT_FUNC_SCALAR, nn_act_func_identity);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = nn_timespec_diff_ns(&start, &end);
    printf("avg_time_ns=%lld total_time_ms=%lld info=nn_act_func_init details=NN_ACT_FUNC_SCALAR\n", total_time / iterations, total_time / 1000000);

    total_time = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < iterations; i++) {
        nn_act_func_init(NN_ACT_FUNC_TENSOR, nn_act_func_sigmoid);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = nn_timespec_diff_ns(&start, &end);
    printf("avg_time_ns=%lld total_time_ms=%lld info=nn_act_func_init details=NN_ACT_FUNC_TENSOR\n", total_time / iterations, total_time / 1000000);
}
