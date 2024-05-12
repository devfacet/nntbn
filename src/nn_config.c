#include "nn_config.h"
#include <stdbool.h>

// TODO: Add tests

/**
 * @brief debug_level holds the debug level.
 */
static int debug_level = 0;

int nn_get_debug_level() {
    return debug_level;
}

bool nn_set_debug_level(int level) {
    debug_level = level;
    return true;
}

bool nn_neon_available() {
#if NN_NEON_AVAILABLE
    return true;
#else
    return false;
#endif
}

bool nn_cmsis_dsp_available() {
#if NN_CMSIS_DSP_AVAILABLE
    return true;
#else
    return false;
#endif
}
