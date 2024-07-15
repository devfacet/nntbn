#ifndef NN_TEST_H
#define NN_TEST_H

#include <time.h>

/**
 * @brief Returns the difference between two timespec in nanoseconds.
 *
 * @param start The start timespec.
 * @param end The end timespec.
 *
 * @return The difference between the two timespec structures in nanoseconds.
 */
long long nn_timespec_diff_ns(struct timespec *start, struct timespec *end);

#endif // NN_TEST_H
