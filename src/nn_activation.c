#include "nn_activation.h"
#include <math.h>

// TODO: Add tests
// TODO: Add softmax activation function.

// nn_activation_func_identity returns x.
float nn_activation_func_identity(float x) {
    return x;
}

// nn_activation_func_sigmoid returns the sigmoid of x.
float nn_activation_func_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// nn_activation_func_relu returns the ReLU of x.
float nn_activation_func_relu(float x) {
    return fmaxf(0, x);
}
