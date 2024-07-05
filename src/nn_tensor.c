#include "nn_tensor.h"

void nn_tensor_slice(const NNTensor *input, const size_t offset, const size_t *sizes, NNTensor *output) {
    output->flags = NN_TENSOR_FLAG_INIT;
    output->dims = 1;
    output->sizes = (size_t *)sizes;
    output->data = &input->data[offset];
}
