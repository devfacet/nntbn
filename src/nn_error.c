#include "nn_error.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO: Add tests
// TODO: Optimize theses functions

void nn_error_set(NNError *error, NNErrorCode code, const char *message) {
    if (!error) {
        error = malloc(sizeof(NNError));
    }

    error->code = code;
    if (error->message) {
        free((void *)error->message);
    }
    error->message_size = strlen(message) + 1; // include the null terminator
    error->message = malloc(error->message_size);
    if (error->message) {
        strncpy((char *)error->message, message, error->message_size);
    }
}

void nn_error_setf(NNError *error, NNErrorCode code, const char *format, ...) {
    if (!error) {
        error = malloc(sizeof(NNError));
    }

    va_list args;
    va_start(args, format);
    int size = vsnprintf(NULL, 0, format, args);
    va_end(args);
    if (size < 0) {
        return;
    }
    char *message = malloc(size + 1); // include the null terminator
    if (!message) {
        return;
    }
    va_start(args, format);
    vsnprintf(message, size + 1, format, args);
    va_end(args);

    nn_error_set(error, code, message);
    free(message);
}
