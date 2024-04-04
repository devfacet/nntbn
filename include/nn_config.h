#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#include "nn_error.h"
#include <stdbool.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NN_NEON_AVAILABLE 1
#else
#define NN_NEON_AVAILABLE 0
#endif

#if defined(__ARM_ARCH)
#define NN_CMSIS_DSP_AVAILABLE 1
#else
#define NN_CMSIS_DSP_AVAILABLE 0
#endif

// nn_get_debug_level returns the debug level.
int nn_get_debug_level();

// nn_set_debug_level sets the debug level.
bool nn_set_debug_level(int level);

// nn_neon_available returns whether ARM NEON is available.
bool nn_neon_available();

// nn_get_use_neon returns the ARM NEON use flag.
bool nn_get_use_neon();

// nn_set_use_neon sets the ARM NEON use flag.
bool nn_set_use_neon(bool flag, NNError *error);

// nn_cmsis_dsp_available returns whether ARM CMSIS-DSP is available.
bool nn_cmsis_dsp_available();

// nn_get_use_cmsis returns the ARM CMSIS-DSP flag.
bool nn_get_use_cmsis();

// nn_set_use_cmsis_dsp sets the ARM CMSIS-DSP flag.
bool nn_set_use_cmsis_dsp(bool flag, NNError *error);

#endif // NN_CONFIG_H
