// include/cuda/cuda_tensor.cuh
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda {

// ============= Error Checking Macro =============
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                   cudaGetErrorString(error));               \
        }                                                                     \
    } while(0)

// ============= Function Declarations =============

// Allocate GPU memory
float* cuda_malloc(size_t size);

// Free GPU memory
void cuda_free(float* d_ptr);

// Copy from host to device
void cuda_copy_to_device(float* d_dst, const float* h_src, size_t size);

// Copy from device to host
void cuda_copy_to_host(float* h_dst, const float* d_src, size_t size);

// Copy from device to device
void cuda_copy_device_to_device(float* d_dst, const float* d_src, size_t size);

// Set device memory to zero
void cuda_zero(float* d_ptr, size_t size);

// Device info
void print_cuda_info();
void print_memory_info();
bool is_available();
void set_device(int device_id = 0);
int get_device();
void synchronize();

} // namespace cuda