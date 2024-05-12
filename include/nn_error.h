#ifndef NN_ERROR_H
#define NN_ERROR_H

#include <stddef.h>

/**
 * @brief Represents an error code.
 */
typedef enum {
    NN_ERROR_NONE = 0,                    // no error
    NN_ERROR_INIT,                        // initialization error
    NN_ERROR_INVALID_ARGUMENT,            // invalid argument
    NN_ERROR_INVALID_FUNCTION,            // invalid function
    NN_ERROR_INVALID_INSTANCE,            // invalid instance
    NN_ERROR_INVALID_SIZE,                // invalid size
    NN_ERROR_INVALID_TYPE,                // invalid type
    NN_ERROR_INVALID_VALUE,               // invalid value
    NN_ERROR_MEMORY_ALLOCATION,           // memory allocation error
    NN_ERROR_NOT_IMPLEMENTED,             // not implemented
    NN_ERROR_ARM_CMSIS_DSP_NOT_AVAILABLE, // ARM CMSIS-DSP functions not available
    NN_ERROR_ARM_NEON_NOT_AVAILABLE,      // ARM NEON instructions not available
} NNErrorCode;

/**
 * @brief Represents an error.
 *
 * @param code The error code.
 * @param message_size The size of the error message.
 * @param message The error message.
 */
typedef struct {
    NNErrorCode code;
    size_t message_size;
    const char *message;
} NNError;

/**
 * @brief Sets the error with the given code and message.
 *
 * @param error The error to set.
 * @param code The error code.
 * @param message The error message.
 */
void nn_error_set(NNError *error, NNErrorCode code, const char *message);

/**
 * @brief Sets the error with the given code and formatted message.
 *
 * @param error The error to set.
 * @param code The error code.
 * @param format The formatted error message.
 */
void nn_error_setf(NNError *error, NNErrorCode code, const char *format, ...);

#endif // NN_ERROR_H
