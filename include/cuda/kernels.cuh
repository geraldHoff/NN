// include/cuda/kernels.cuh
#pragma once

namespace cuda {

// Matrix multiplication: C = A * B
// A: [m, n], B: [n, p], C: [m, p]
void matmul(const float* d_A, const float* d_B, float* d_C,
            int m, int n, int p);

// Element-wise ReLU: out[i] = max(0, in[i])
void relu_forward(const float* d_in, float* d_out, int size);

// ReLU backward: out[i] = grad[i] if in[i] > 0 else 0
void relu_backward(const float* d_grad, const float* d_input,
                   float* d_out, int size);

// Add bias to each row: out[i,j] = in[i,j] + bias[j]
void add_bias(float* d_data, const float* d_bias,
              int batch_size, int features);

// MSE loss
float mse_forward(const float* d_pred, const float* d_true, int size);

// MSE gradient
void mse_backward(const float* d_pred, const float* d_true,
                  float* d_grad, int size);

// SGD update: params[i] -= lr * grad[i]
void sgd_step(float* d_params, const float* d_grad,
              float lr, int size);

} // namespace cuda