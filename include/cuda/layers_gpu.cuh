// include/cuda/layers_gpu.cuh
#pragma once

#include "nn.hpp"
#include <memory>

// GPU-accelerated Dense layer
class DenseGPU : public Layer {
private:
    float* d_W;           // Weights on GPU
    float* d_b;           // Biases on GPU
    float* d_dW;          // Weight gradients on GPU
    float* d_db;          // Bias gradients on GPU
    float* d_x_cache;     // Cached input on GPU
    
    size_t in_features;
    size_t out_features;
    size_t cached_batch_size;
    
public:
    DenseGPU(size_t in, size_t out);
    ~DenseGPU();
    
    DenseGPU(const DenseGPU&) = delete;
    DenseGPU& operator=(const DenseGPU&) = delete;
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    void step(float lr) override;
};

// GPU-accelerated ReLU layer
class ReLUGPU : public Layer {
private:
    float* d_x_cache;
    size_t cached_size;
    
public:
    ReLUGPU();
    ~ReLUGPU();
    
    ReLUGPU(const ReLUGPU&) = delete;
    ReLUGPU& operator=(const ReLUGPU&) = delete;
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
};

// GPU-accelerated MSE Loss
class MSELossGPU {
private:
    float* d_y_pred_cache;
    float* d_y_true_cache;
    size_t cached_size;
    
public:
    MSELossGPU();
    ~MSELossGPU();
    
    MSELossGPU(const MSELossGPU&) = delete;
    MSELossGPU& operator=(const MSELossGPU&) = delete;
    
    float forward(const Tensor& y_pred, const Tensor& y_true);
    Tensor backward();
};