#ifndef NN_DEBUG_H
#define NN_DEBUG_H

#include "nn_config.h"
#include <stdio.h>

/**
 * @brief Prints a debug message.
 *
 * @param level The debug level.
 * @param fmt The format string.
 * @param ... The arguments.
 */
#define NN_DEBUG_PRINT(level, fmt, ...)          \
    do {                                         \
        if (nn_get_debug_level() >= level)       \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
    } while (0)

#endif // NN_DEBUG_H
