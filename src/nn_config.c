#include "nn_config.h"
#include "nn_error.h"
#include <stdbool.h>
#include <stdio.h>

// TODO: Add tests

// debug_level holds the debug level.
static int debug_level = 0;

// use_neon holds the flag for using ARM NEON instructions.
static bool use_neon = false;

// use_cmsis holds the flag for using ARM CMSIS-DSP functions.
static bool use_cmsis = false;

// neon_available holds whether ARM NEON is available.
#if NN_NEON_AVAILABLE
static bool neon_available = true;
#else
static bool neon_available = false;
#endif

// cmsis_dsp_available holds whether ARM CMSIS-DSP is available.
#if NN_CMSIS_DSP_AVAILABLE
static bool cmsis_dsp_available = true;
#else
static bool cmsis_dsp_available = false;
#endif

// nn_get_debug_level returns the debug level.
int nn_get_debug_level() {
    return debug_level;
}

// nn_set_debug_level sets the debug level.
bool nn_set_debug_level(int level) {
    debug_level = level;
    return true;
}

// nn_neon_available returns whether ARM NEON is available.
bool nn_neon_available() {
    return neon_available;
}

// nn_get_use_neon returns the ARM NEON use flag.
bool nn_get_use_neon() {
    return use_neon;
}

// nn_set_use_neon sets the ARM NEON use flag.
bool nn_set_use_neon(bool flag, NNError *error) {
    if (neon_available) {
        use_neon = flag;
        return true;
    }
    if (error) {
        error->code = NN_ERROR_NEON_NOT_AVAILABLE;
        error->message = "ARM NEON not available";
    }
    return false;
}

// nn_cmsis_dsp_available returns whether ARM CMSIS-DSP is available.
bool nn_cmsis_dsp_available() {
    return cmsis_dsp_available;
}

// nn_get_use_cmsis returns the ARM CMSIS-DSP flag.
bool nn_get_use_cmsis() {
    return use_cmsis;
}

// nn_set_use_cmsis_dsp sets the ARM CMSIS-DSP flag.
bool nn_set_use_cmsis_dsp(bool flag, NNError *error) {
    if (cmsis_dsp_available) {
        use_cmsis = flag;
        return true;
    }
    if (error) {
        error->code = NN_ERROR_CMSIS_DSP_NOT_AVAILABLE;
        error->message = "ARM CMSIS-DSP not available";
    }
    return false;
}
