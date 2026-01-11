// src/cuda/kernels.cu
#include "cuda/kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

namespace cuda {

// ============= Matrix Multiplication (Tiled) =============
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                               int m, int n, int p) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile into shared memory
        if (row < m && t * TILE_SIZE + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < p && t * TILE_SIZE + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * p + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

void matmul(const float* d_A, const float* d_B, float* d_C,
            int m, int n, int p) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((p + TILE_SIZE - 1) / TILE_SIZE,
                   (m + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);
    cudaDeviceSynchronize();
}

// ============= ReLU Forward =============
__global__ void relu_forward_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

void relu_forward(const float* d_in, float* d_out, int size) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_forward_kernel<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, size);
    cudaDeviceSynchronize();
}

// ============= ReLU Backward =============
__global__ void relu_backward_kernel(const float* grad, const float* input,
                                     float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = (input[idx] > 0.0f) ? grad[idx] : 0.0f;
    }
}

void relu_backward(const float* d_grad, const float* d_input,
                   float* d_out, int size) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    relu_backward_kernel<<<numBlocks, BLOCK_SIZE>>>(d_grad, d_input, d_out, size);
    cudaDeviceSynchronize();
}

// ============= Add Bias =============
__global__ void add_bias_kernel(float* data, const float* bias,
                                int batch_size, int features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < features) {
        data[row * features + col] += bias[col];
    }
}

void add_bias(float* d_data, const float* d_bias,
              int batch_size, int features) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((features + 15) / 16, (batch_size + 15) / 16);
    
    add_bias_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_bias,
                                                     batch_size, features);
    cudaDeviceSynchronize();
}

// ============= MSE Forward =============
__global__ void mse_forward_kernel(const float* pred, const float* true_val,
                                   float* partial_loss, int size) {
    __shared__ float shared_loss[BLOCK_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Compute squared error
    float loss = 0.0f;
    if (idx < size) {
        float diff = pred[idx] - true_val[idx];
        loss = diff * diff;
    }
    shared_loss[tid] = loss;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_loss[tid] += shared_loss[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_loss[blockIdx.x] = shared_loss[0];
    }
}

float mse_forward(const float* d_pred, const float* d_true, int size) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    float* d_partial;
    cudaMalloc(&d_partial, numBlocks * sizeof(float));
    
    mse_forward_kernel<<<numBlocks, BLOCK_SIZE>>>(d_pred, d_true, d_partial, size);
    
    // Sum partial results on CPU (small array)
    std::vector<float> h_partial(numBlocks);
    cudaMemcpy(h_partial.data(), d_partial, numBlocks * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    float total = 0.0f;
    for (float val : h_partial) {
        total += val;
    }
    
    cudaFree(d_partial);
    return total / size;
}

// ============= MSE Backward =============
__global__ void mse_backward_kernel(const float* pred, const float* true_val,
                                    float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0f * (pred[idx] - true_val[idx]) / size;
    }
}

void mse_backward(const float* d_pred, const float* d_true,
                  float* d_grad, int size) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mse_backward_kernel<<<numBlocks, BLOCK_SIZE>>>(d_pred, d_true, d_grad, size);
    cudaDeviceSynchronize();
}

// ============= SGD Step =============
__global__ void sgd_step_kernel(float* params, const float* grad,
                                float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= lr * grad[idx];
    }
}

void sgd_step(float* d_params, const float* d_grad,
              float lr, int size) {
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sgd_step_kernel<<<numBlocks, BLOCK_SIZE>>>(d_params, d_grad, lr, size);
    cudaDeviceSynchronize();
}

} // namespace cuda