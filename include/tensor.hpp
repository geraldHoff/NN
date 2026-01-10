// include/tensor.hpp
#pragma once
#include <vector>
#include <memory>

enum class Device {
    CPU,
    CUDA
};

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;      // CPU data
    float* d_data = nullptr;      // GPU data pointer
    Device device = Device::CPU;
    
    // Move tensor to GPU
    void to_cuda();
    
    // Move tensor to CPU
    void to_cpu();
    
    // Free GPU memory
    void free_cuda();
    
    ~Tensor() {
        if (d_data != nullptr) {
            free_cuda();
        }
    }
};