#ifndef NN_TEST_H
#define NN_TEST_H

#include <time.h>

// nn_timespec_diff_ns returns the difference between two timespec structs in nanoseconds.
long long nn_timespec_diff_ns(struct timespec *start, struct timespec *end);

#endif // NN_TEST_H
