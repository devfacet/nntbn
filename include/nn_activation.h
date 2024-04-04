#ifndef NN_ACTIVATION_FUNCTIONS_H
#define NN_ACTIVATION_FUNCTIONS_H

// NNActivationFunction represents an activation function.
typedef float (*NNActivationFunction)(float);

// nn_activation_func_identity returns x.
float nn_activation_func_identity(float x);

// nn_activation_func_sigmoid returns the sigmoid of x.
float nn_activation_func_sigmoid(float x);

// nn_activation_func_relu returns the ReLU of x.
float nn_activation_func_relu(float x);

#endif // NN_ACTIVATION_FUNCTIONS_H
