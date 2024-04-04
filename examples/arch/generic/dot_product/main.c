#include "nn_app.h"
#include "nn_config.h"
#include "nn_dot_product.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// timespec_diff_ns returns the difference between two timespec structs in nanoseconds.
static long long timespec_diff_ns(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000000000LL + (end->tv_nsec - start->tv_nsec);
}

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);

    // Init vars
    const int n_runs = 1000;
    const int n_vectors = 100000;
    struct timespec start, end;
    float *a = malloc(n_vectors * sizeof(float));
    float *b = malloc(n_vectors * sizeof(float));
    for (int i = 0; i < n_vectors; ++i) {
        a[i] = (float)rand() / (float)RAND_MAX;
        b[i] = (float)rand() / (float)RAND_MAX;
    }

    // Benchmark generic implementation
    long long total_time_generic = 0;
    for (int i = 0; i < n_runs; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        nn_dot_product(a, b, n_vectors);
        clock_gettime(CLOCK_MONOTONIC, &end);
        total_time_generic += timespec_diff_ns(&start, &end);
    }
    printf("dot_product (generic): n_vectors=%d n_runs=%d total_time_ns=%lld avg_time_ns=%lld\n", n_vectors, n_runs, total_time_generic, total_time_generic / n_runs / n_vectors);

    return 0;
}
