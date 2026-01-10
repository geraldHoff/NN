// src/cuda/layers_gpu.cu
#include "cuda/layers_gpu.cuh"
#include "cuda/kernels.cuh"
#include "cuda/cuda_tensor.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

// ============= DenseGPU Implementation =============

DenseGPU::DenseGPU(size_t in, size_t out) 
    : in_features(in), out_features(out), cached_batch_size(0) {
    
    // Allocate GPU memory
    d_W = cuda::cuda_malloc(in * out);
    d_b = cuda::cuda_malloc(out);
    d_dW = cuda::cuda_malloc(in * out);
    d_db = cuda::cuda_malloc(out);
    d_x_cache = nullptr;
    
    // Initialize weights on CPU
    std::vector<float> h_W(in * out);
    std::vector<float> h_b(out, 0.0f);
    
    float scale = std::sqrt(1.0f / in);
    for (auto& val : h_W) {
        val = scale * (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }
    
    // Copy to GPU
    cuda::cuda_copy_to_device(d_W, h_W.data(), in * out);
    cuda::cuda_copy_to_device(d_b, h_b.data(), out);
    cuda::cuda_zero(d_dW, in * out);
    cuda::cuda_zero(d_db, out);
}

DenseGPU::~DenseGPU() {
    cuda::cuda_free(d_W);
    cuda::cuda_free(d_b);
    cuda::cuda_free(d_dW);
    cuda::cuda_free(d_db);
    if (d_x_cache != nullptr) {
        cuda::cuda_free(d_x_cache);
    }
}

Tensor DenseGPU::forward(const Tensor& x) {
    size_t batch = x.shape[0];
    size_t in = x.shape[1];
    
    // Cache input
    if (d_x_cache == nullptr || cached_batch_size != batch) {
        if (d_x_cache != nullptr) cuda::cuda_free(d_x_cache);
        d_x_cache = cuda::cuda_malloc(batch * in);
        cached_batch_size = batch;
    }
    cuda::cuda_copy_to_device(d_x_cache, x.data.data(), batch * in);
    
    // Allocate output
    Tensor result;
    result.shape = {batch, out_features};
    result.data.resize(batch * out_features);
    float* d_result = cuda::cuda_malloc(batch * out_features);
    
    // Matrix multiply: result = x * W
    cuda::matmul(d_x_cache, d_W, d_result, batch, in_features, out_features);
    
    // Add bias
    cuda::add_bias(d_result, d_b, batch, out_features);
    
    // Copy back to CPU
    cuda::cuda_copy_to_host(result.data.data(), d_result, batch * out_features);
    cuda::cuda_free(d_result);
    
    return result;
}

Tensor DenseGPU::backward(const Tensor& grad) {
    // TODO: Implement full backward pass
    Tensor dx;
    dx.shape = {cached_batch_size, in_features};
    dx.data.resize(cached_batch_size * in_features, 0.0f);
    return dx;
}

void DenseGPU::step(float lr) {
    cuda::sgd_step(d_W, d_dW, lr, in_features * out_features);
    cuda::sgd_step(d_b, d_db, lr, out_features);
}

// ============= ReLUGPU Implementation =============

ReLUGPU::ReLUGPU() : d_x_cache(nullptr), cached_size(0) {}

ReLUGPU::~ReLUGPU() {
    if (d_x_cache != nullptr) {
        cuda::cuda_free(d_x_cache);
    }
}

Tensor ReLUGPU::forward(const Tensor& x) {
    size_t size = x.data.size();
    
    // Cache input
    if (d_x_cache == nullptr || cached_size != size) {
        if (d_x_cache != nullptr) cuda::cuda_free(d_x_cache);
        d_x_cache = cuda::cuda_malloc(size);
        cached_size = size;
    }
    cuda::cuda_copy_to_device(d_x_cache, x.data.data(), size);
    
    // Apply ReLU
    Tensor result;
    result.shape = x.shape;
    result.data.resize(size);
    float* d_result = cuda::cuda_malloc(size);
    
    cuda::relu_forward(d_x_cache, d_result, size);
    
    // Copy back
    cuda::cuda_copy_to_host(result.data.data(), d_result, size);
    cuda::cuda_free(d_result);
    
    return result;
}

Tensor ReLUGPU::backward(const Tensor& grad) {
    size_t size = grad.data.size();
    
    Tensor result;
    result.shape = grad.shape;
    result.data.resize(size);
    
    float* d_grad = cuda::cuda_malloc(size);
    float* d_result = cuda::cuda_malloc(size);
    
    cuda::cuda_copy_to_device(d_grad, grad.data.data(), size);
    cuda::relu_backward(d_grad, d_x_cache, d_result, size);
    cuda::cuda_copy_to_host(result.data.data(), d_result, size);
    
    cuda::cuda_free(d_grad);
    cuda::cuda_free(d_result);
    
    return result;
}

// ============= MSELossGPU Implementation =============

MSELossGPU::MSELossGPU() : d_y_pred_cache(nullptr), d_y_true_cache(nullptr), cached_size(0) {}

MSELossGPU::~MSELossGPU() {
    if (d_y_pred_cache != nullptr) cuda::cuda_free(d_y_pred_cache);
    if (d_y_true_cache != nullptr) cuda::cuda_free(d_y_true_cache);
}

float MSELossGPU::forward(const Tensor& y_pred, const Tensor& y_true) {
    size_t size = y_pred.data.size();
    
    // Cache predictions and targets
    if (d_y_pred_cache == nullptr || cached_size != size) {
        if (d_y_pred_cache != nullptr) cuda::cuda_free(d_y_pred_cache);
        if (d_y_true_cache != nullptr) cuda::cuda_free(d_y_true_cache);
        d_y_pred_cache = cuda::cuda_malloc(size);
        d_y_true_cache = cuda::cuda_malloc(size);
        cached_size = size;
    }
    
    cuda::cuda_copy_to_device(d_y_pred_cache, y_pred.data.data(), size);
    cuda::cuda_copy_to_device(d_y_true_cache, y_true.data.data(), size);
    
    return cuda::mse_forward(d_y_pred_cache, d_y_true_cache, size);
}

Tensor MSELossGPU::backward() {
    Tensor grad;
    grad.shape = {1, cached_size};
    grad.data.resize(cached_size);
    
    float* d_grad = cuda::cuda_malloc(cached_size);
    cuda::mse_backward(d_y_pred_cache, d_y_true_cache, d_grad, cached_size);
    cuda::cuda_copy_to_host(grad.data.data(), d_grad, cached_size);
    
    cuda::cuda_free(d_grad);
    return grad;
}