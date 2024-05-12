#define _POSIX_C_SOURCE 199309L

#include "arch/arm/cmsis-dsp/nn_dot_prod.h"
#include "nn_app.h"
#include "nn_config.h"
#include "nn_dot_prod.h"
#include "nn_tensor.h"
#include "nn_test.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);
    // nn_set_debug_level(5); // for debugging

    srand((unsigned int)time(NULL));

    // Init vars
    struct timespec start, end;
    long long total_time = 0;
    const int batch_size = 1024;
    const int n_vectors = 4096;
    NNTensor *vec_a[batch_size];
    NNTensor *vec_b[batch_size];
    for (int i = 0; i < batch_size; i++) {
        vec_a[i] = nn_tensor_init_NNTensor(1, (const size_t[]){4096}, false, NULL, NULL);
        vec_b[i] = nn_tensor_init_NNTensor(1, (const size_t[]){4096}, false, NULL, NULL);
        for (int j = 0; j < n_vectors; ++j) {
            vec_a[i]->data[j] = (NNTensorUnit)rand() / (NNTensorUnit)RAND_MAX;
            vec_b[i]->data[j] = (NNTensorUnit)rand() / (NNTensorUnit)RAND_MAX;
        }
    }

    // Benchmark
    for (int i = 0; i < batch_size; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        nn_dot_prod_cmsis_dsp(vec_a[i], vec_b[i], NULL);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += nn_timespec_diff_ns(&start, &end);
    }
    printf("avg_time_ns=%lld total_time_ms=%lld info=nn_dot_prod_cmsis_dsp\n", total_time / batch_size, total_time / 1000000);

    return 0;
}
