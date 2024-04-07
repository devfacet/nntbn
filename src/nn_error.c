#include "nn_error.h"

// nn_error_set sets the error code and message.
void nn_error_set(NNError *error, NNErrorCode code, const char *message) {
    if (error) {
        error->code = code;
        error->message = message;
    }
}
