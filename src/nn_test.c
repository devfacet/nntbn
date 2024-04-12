#define _POSIX_C_SOURCE 199309L
#include "nn_test.h"
#include <time.h>

// nn_timespec_diff_ns returns the difference between two timespec structs in nanoseconds.
long long nn_timespec_diff_ns(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000000000LL + (end->tv_nsec - start->tv_nsec);
}
