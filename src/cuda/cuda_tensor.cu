// src/cuda/cuda_tensor.cu
#include "cuda/cuda_tensor.cuh"
#include <iostream>

namespace cuda {

float* cuda_malloc(size_t size) {
    float* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(float)));
    return d_ptr;
}

void cuda_free(float* d_ptr) {
    if (d_ptr != nullptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

void cuda_copy_to_device(float* d_dst, const float* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size * sizeof(float), 
                          cudaMemcpyHostToDevice));
}

void cuda_copy_to_host(float* h_dst, const float* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size * sizeof(float), 
                          cudaMemcpyDeviceToHost));
}

void cuda_copy_device_to_device(float* d_dst, const float* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size * sizeof(float), 
                          cudaMemcpyDeviceToDevice));
}

void cuda_zero(float* d_ptr, size_t size) {
    CUDA_CHECK(cudaMemset(d_ptr, 0, size * sizeof(float)));
}

void print_cuda_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!\n";
        return;
    }
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        std::cout << "Device " << dev << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    }
}

void print_memory_info() {
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    std::cout << "GPU Memory:\n";
    std::cout << "  Free: " << free_bytes / (1024*1024) << " MB\n";
    std::cout << "  Total: " << total_bytes / (1024*1024) << " MB\n";
    std::cout << "  Used: " << (total_bytes - free_bytes) / (1024*1024) << " MB\n";
}

bool is_available() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

int get_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace cuda