#ifndef NN_DEBUG_H
#define NN_DEBUG_H

#include "nn_config.h"
#include <stdio.h>

// NN_DEBUG_PRINT defines a macro for printing debug messages.
#define NN_DEBUG_PRINT(level, fmt, ...)          \
    do {                                         \
        if (nn_get_debug_level() >= level)       \
            fprintf(stderr, fmt, ##__VA_ARGS__); \
    } while (0)

#endif // NN_DEBUG_H
