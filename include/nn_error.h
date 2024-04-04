#ifndef NN_ERROR_H
#define NN_ERROR_H

#include <stddef.h>

// NNErrorCode defines the error codes.
typedef enum {
    NN_ERROR_NONE = 0,                // no error
    NN_ERROR_INVALID_INPUT_SIZE,      // invalid input size
    NN_ERROR_NEON_NOT_AVAILABLE,      // NEON instructions not available
    NN_ERROR_CMSIS_DSP_NOT_AVAILABLE, // CMSIS-DSP functions not available
} NNErrorCode;

// NNError represents an error.
typedef struct {
    NNErrorCode code;    // error code
    const char *message; // error message
} NNError;

#endif // NN_ERROR_H
