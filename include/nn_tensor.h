#ifndef NN_TENSOR_H
#define NN_TENSOR_H

#include "nn_error.h"
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Represents the tensor flags.
 */
typedef enum {
    NN_TENSOR_FLAG_NONE = 0x00, // no flags set
    NN_TENSOR_FLAG_INIT = 0x01, // initialized
} NNTensorFlags;

/**
 * @brief Represents the unit type of a tensor.
 *
 * @note Different hardware platforms may have different unit types.
 *       Also, on certain platforms `int` is faster than `float`.
 */
#define NN_TENSOR_DEFINE_UNIT_TYPE(type, name) \
    typedef type name##Unit;

/**
 * @brief Represents a multi dimensional array.
 *
 * @param flags The flags set on the tensor.
 * @param dims The number of dimensions.
 * @param sizes The array of sizes for each dimension.
 * @param data The array of values in row-major order.
 *
 * @note Use the `nn_tensor_init` and `nn_tensor_destroy` functions to create and destroy tensors.
 */
#define NN_TENSOR_DEFINE_TYPE(type, name) \
    typedef struct {                      \
        NNTensorFlags flags;              \
        size_t dims;                      \
        size_t *sizes;                    \
        name##Unit *data;                 \
    } name;

/**
 * @brief Returns a new tensor with the specified dimensions, sizes, and values.
 *
 * @param dims The number of dimensions.
 * @param sizes The array of sizes for each dimension.
 * @param zero Whether to initialize the data array to zero using calloc.
 * @param values The optional values to set (row-major order).
 * @param error The error instance to set if an error occurs.
 *
 * @return The pointer to the newly created tensor instance or NULL if an error occurs.
 *
 * @note It makes separate memory allocations for, struct, sizes array, and data array.
 *       If zero is true and values is NULL then it initializes the data array to zero.
 *       If zero is false and values is NULL then it initializes the data array to an uninitialized state.
 */
#define NN_TENSOR_DEFINE_INIT(type, name)                                                                                        \
    static inline name *nn_tensor_init_##name(size_t dims, const size_t *sizes, bool zero, const type *values, NNError *error) { \
        name *tensor = (name *)malloc(sizeof(name));                                                                             \
        if (!tensor) {                                                                                                           \
            nn_error_set(error, NN_ERROR_MEMORY_ALLOCATION, "could not allocate memory for the new tensor");                     \
            return NULL;                                                                                                         \
        }                                                                                                                        \
                                                                                                                                 \
        tensor->flags = NN_TENSOR_FLAG_NONE;                                                                                     \
        tensor->dims = dims;                                                                                                     \
        tensor->sizes = (size_t *)malloc(dims * sizeof(size_t));                                                                 \
        if (!tensor->sizes) {                                                                                                    \
            free(tensor);                                                                                                        \
            return NULL;                                                                                                         \
        }                                                                                                                        \
        size_t total_elements = 1;                                                                                               \
        for (size_t i = 0; i < dims; i++) {                                                                                      \
            tensor->sizes[i] = sizes[i];                                                                                         \
            total_elements *= sizes[i];                                                                                          \
        }                                                                                                                        \
                                                                                                                                 \
        if (values) {                                                                                                            \
            tensor->data = (type *)malloc(total_elements * sizeof(type));                                                        \
            if (!tensor->data) {                                                                                                 \
                free(tensor->sizes);                                                                                             \
                free(tensor);                                                                                                    \
                return NULL;                                                                                                     \
            }                                                                                                                    \
            memcpy(tensor->data, values, total_elements * sizeof(type));                                                         \
        } else if (zero) {                                                                                                       \
            tensor->data = (type *)calloc(total_elements, sizeof(type));                                                         \
            if (!tensor->data) {                                                                                                 \
                free(tensor->sizes);                                                                                             \
                free(tensor);                                                                                                    \
                return NULL;                                                                                                     \
            }                                                                                                                    \
        } else {                                                                                                                 \
            tensor->data = (type *)malloc(total_elements * sizeof(type));                                                        \
            if (!tensor->data) {                                                                                                 \
                free(tensor->sizes);                                                                                             \
                free(tensor);                                                                                                    \
                return NULL;                                                                                                     \
            }                                                                                                                    \
        }                                                                                                                        \
        tensor->flags = NN_TENSOR_FLAG_INIT;                                                                                     \
                                                                                                                                 \
        return tensor;                                                                                                           \
    }

/**
 * @brief Destroys the given tensor instance and frees the allocated memory.
 *
 * @param tensor The tensor instance to destroy.
 */
#define NN_TENSOR_DEFINE_DESTROY(name)                          \
    static inline void nn_tensor_destroy_##name(name *tensor) { \
        if (tensor) {                                           \
            free(tensor->data);                                 \
            free(tensor->sizes);                                \
            free(tensor);                                       \
        }                                                       \
    }

/**
 * @brief Get dimension values of the given tensor instance.
 *
 * @param tensor The tensor instance to get values from.
 * @param indices The indices specifying the dimension to get.
 * @param indices_size The number of indices.
 * @param values The tensor instance to set the values to.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false
 *
 * @note: If indices is NULL then it gets the entire tensor values.
 *        Some of the checks are omitted for performance reasons.
 */
#define NN_TENSOR_DEFINE_GET(type, name)                                                                                        \
    static inline bool nn_tensor_get_##name(name *tensor, size_t *indices, size_t indices_size, name *values, NNError *error) { \
        if (tensor == NULL) {                                                                                                   \
            nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor is NULL");                                                   \
            return false;                                                                                                       \
        } else if (values == NULL) {                                                                                            \
            nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "values are NULL");                                                  \
            return false;                                                                                                       \
        }                                                                                                                       \
                                                                                                                                \
        if (indices == NULL) {                                                                                                  \
            size_t total_elements = 1;                                                                                          \
            for (size_t i = 0; i < tensor->dims; i++) {                                                                         \
                total_elements *= tensor->sizes[i];                                                                             \
            }                                                                                                                   \
            memcpy(values->data, tensor->data, total_elements * sizeof(type));                                                  \
            return true;                                                                                                        \
        }                                                                                                                       \
                                                                                                                                \
        size_t block_size = 1;                                                                                                  \
        size_t flat_index = 0;                                                                                                  \
        size_t stride = 1;                                                                                                      \
        for (size_t i = tensor->dims - 1; i != (size_t) - 1; i--) {                                                             \
            if (i >= tensor->dims - indices_size) {                                                                             \
                flat_index += indices[tensor->dims - 1 - i] * stride;                                                           \
            } else {                                                                                                            \
                block_size *= tensor->sizes[i];                                                                                 \
            }                                                                                                                   \
            stride *= tensor->sizes[i];                                                                                         \
        }                                                                                                                       \
        memcpy(values->data, &tensor->data[flat_index], block_size * sizeof(type));                                             \
                                                                                                                                \
        return true;                                                                                                            \
    }

/**
 * @brief Set dimension values of the given tensor instance.
 *
 * @param tensor The tensor instance to set values to.
 * @param indices The indices specifying the dimension to set.
 * @param indices_size The number of indices.
 * @param values The tensor instance to set the values from.
 * @param error The error instance to set if an error occurs.
 *
 * @return True or false
 *
 * @note: If indices is NULL then it sets the entire tensor values.
 *        Some of the checks are omitted for performance reasons.
 */
#define NN_TENSOR_DEFINE_SET(type, name)                                                                                              \
    static inline bool nn_tensor_set_##name(name *tensor, size_t *indices, size_t indices_size, const name *values, NNError *error) { \
        if (tensor == NULL) {                                                                                                         \
            nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "tensor is NULL");                                                         \
            return false;                                                                                                             \
        } else if (values == NULL) {                                                                                                  \
            nn_error_set(error, NN_ERROR_INVALID_ARGUMENT, "values are NULL");                                                        \
            return false;                                                                                                             \
        }                                                                                                                             \
                                                                                                                                      \
        if (indices == NULL) {                                                                                                        \
            size_t total_elements = 1;                                                                                                \
            for (size_t i = 0; i < tensor->dims; i++) {                                                                               \
                total_elements *= tensor->sizes[i];                                                                                   \
            }                                                                                                                         \
            memcpy(tensor->data, values->data, total_elements * sizeof(type));                                                        \
            return true;                                                                                                              \
        }                                                                                                                             \
                                                                                                                                      \
        size_t block_size = 1;                                                                                                        \
        size_t flat_index = 0;                                                                                                        \
        size_t stride = 1;                                                                                                            \
        for (size_t i = tensor->dims - 1; i != (size_t) - 1; i--) {                                                                   \
            if (i >= tensor->dims - indices_size) {                                                                                   \
                flat_index += indices[tensor->dims - 1 - i] * stride;                                                                 \
            } else {                                                                                                                  \
                block_size *= tensor->sizes[i];                                                                                       \
            }                                                                                                                         \
            stride *= tensor->sizes[i];                                                                                               \
        }                                                                                                                             \
        memcpy(&tensor->data[flat_index], values->data, block_size * sizeof(type));                                                   \
                                                                                                                                      \
        return true;                                                                                                                  \
    }

// Define the default tensor type if not defined
#ifndef NN_TENSOR_DEFINED
#define NN_TENSOR_DEFINED
#if defined(__ARM_ARCH) && !defined(__ARM_FP)
NN_TENSOR_DEFINE_UNIT_TYPE(int, NNTensor);
NN_TENSOR_DEFINE_TYPE(int, NNTensor);
NN_TENSOR_DEFINE_INIT(int, NNTensor);
NN_TENSOR_DEFINE_GET(int, NNTensor);
NN_TENSOR_DEFINE_SET(int, NNTensor);
NN_TENSOR_DEFINE_DESTROY(NNTensor);
#else
NN_TENSOR_DEFINE_UNIT_TYPE(float, NNTensor);
NN_TENSOR_DEFINE_TYPE(float, NNTensor);
NN_TENSOR_DEFINE_INIT(float, NNTensor);
NN_TENSOR_DEFINE_GET(float, NNTensor);
NN_TENSOR_DEFINE_SET(float, NNTensor);
NN_TENSOR_DEFINE_DESTROY(NNTensor);
#endif
#endif // NN_TENSOR_DEFINED

#endif // NN_TENSOR_H
