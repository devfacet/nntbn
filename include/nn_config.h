#ifndef NN_CONFIG_H
#define NN_CONFIG_H

#include "nn_error.h"
#include <stdbool.h>

/**
 * @brief Defines whether ARM NEON is available.
 *
 * @note Available for certain ARM architectures.
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define NN_NEON_AVAILABLE 1
#else
#define NN_NEON_AVAILABLE 0
#endif

/**
 * @brief Defines whether ARM CMSIS-DSP is available.
 *
 * @note Available for all ARM architectures.
 */
#if defined(__ARM_ARCH)
#define NN_CMSIS_DSP_AVAILABLE 1
#else
#define NN_CMSIS_DSP_AVAILABLE 0
#endif

/**
 * @brief Returns the debug level.
 *
 * @return The debug level.
 */
int nn_get_debug_level();

/**
 * @brief Sets the debug level.
 *
 * @param level The debug level.
 *
 * @return True or false.
 */
bool nn_set_debug_level(int level);

/**
 * @brief Returns whether ARM NEON is available.
 *
 * @return True or false.
 */
bool nn_neon_available();

/**
 * @brief Returns whether ARM CMSIS-DSP is available.
 *
 * @return True or false.
 */
bool nn_cmsis_dsp_available();

#endif // NN_CONFIG_H
