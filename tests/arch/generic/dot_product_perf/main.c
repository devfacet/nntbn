#include "nn_app.h"
#include "nn_config.h"
#include "nn_dot_product.h"
#include "nn_test.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);

    // Init vars
    const int n_runs = 1000;
    const int n_vectors = 100000;
    long long total_time = 0;
    struct timespec start, end;
    float *a = malloc(n_vectors * sizeof(float));
    float *b = malloc(n_vectors * sizeof(float));
    for (int i = 0; i < n_vectors; ++i) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    // Benchmark
    for (int i = 0; i < n_runs; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        nn_dot_product(a, b, n_vectors);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time += nn_timespec_diff_ns(&start, &end);
    }
    printf("avg_time_ns=%lld total_time_ms=%lld info=nn_dot_product\n", total_time / n_runs, total_time / 1000000);

    return 0;
}
