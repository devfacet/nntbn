#ifndef NN_ERROR_H
#define NN_ERROR_H

#include <stddef.h>

// NNErrorCode defines the error codes.
typedef enum {
    NN_ERROR_NONE = 0,                // no error
    NN_ERROR_NOT_IMPLEMENTED,         // not implemented
    NN_ERROR_INVALID_INSTANCE,        // invalid instance
    NN_ERROR_INVALID_SIZE,            // invalid size
    NN_ERROR_INVALID_VALUE,           // invalid value
    NN_ERROR_INVALID_TYPE,            // invalid type
    NN_ERROR_NEON_NOT_AVAILABLE,      // NEON instructions not available
    NN_ERROR_CMSIS_DSP_NOT_AVAILABLE, // CMSIS-DSP functions not available
} NNErrorCode;

// NNError represents an error.
typedef struct {
    NNErrorCode code;    // error code
    const char *message; // error message
} NNError;

// nn_error_set sets the error code and message.
void nn_error_set(NNError *error, NNErrorCode code, const char *message);

#endif // NN_ERROR_H
