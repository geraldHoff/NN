#include "nn.hpp"
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <random>

static Tensor matmul(const Tensor& A, const Tensor& B) {
    
    assert(A.shape.size() == 2 && "Matrix A must be 2D");
    assert(B.shape.size() == 2 && "Matrix B must be 2D");
    assert(A.shape[1] == B.shape[0] && "Inner dimensions must match");
    
    size_t m = A.shape[0];
    size_t n = A.shape[1];
    size_t p = B.shape[1];
    
    Tensor result;
    result.shape = {m, p};
    result.data.resize(m * p, 0.0f);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                sum += A.data[i * n + k] * B.data[k * p + j];
            }
            result.data[i * p + j] = sum;
        }
    }
    
    return result;
}

Dense::Dense(size_t in, size_t out) {
    W.shape = {in, out};
    W.data.resize(in * out);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / (in + out));  // initialization
    std::uniform_real_distribution<float> dis(-limit, limit);
    
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] = dis(gen);
    }
    
    // Initialize biases to zero
    b.shape = {1, out};
    b.data.resize(out, 0.0f);
    
    // Initialize gradient tensors
    dW.shape = {in, out};
    dW.data.resize(in * out, 0.0f);
    
    db.shape = {1, out};
    db.data.resize(out, 0.0f);
}

Tensor Dense::forward(const Tensor& x) {
    x_cache = x;
    
    // Compute: output = x · W + b
    Tensor result = matmul(x, W);
    
    // Add bias to each row
    assert(result.shape.size() == 2 && "Result must be 2D");
    assert(result.shape[1] == b.shape[1] && "Bias dimension mismatch");
    
    size_t batch_size = result.shape[0];
    size_t out_features = result.shape[1];
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            result.data[i * out_features + j] += b.data[j];
        }
    }
    
    return result;
}

Tensor Dense::backward(const Tensor& grad) {
   
    assert(grad.shape.size() == 2 && "Gradient must be 2D");
    assert(x_cache.shape.size() == 2 && "Cached input must be 2D");
    
    size_t batch_size = grad.shape[0];
    size_t out_features = grad.shape[1];
    size_t in_features = x_cache.shape[1];
    
    // Compute gradient w.r.t. weights: dW = x_cache^T · grad
    Tensor x_T;
    x_T.shape = {in_features, batch_size};
    x_T.data.resize(in_features * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < in_features; ++j) {
            x_T.data[j * batch_size + i] = x_cache.data[i * in_features + j];
        }
    }
    
    dW = matmul(x_T, grad);
    
    // Compute gradient w.r.t. bias: db = sum(grad, axis=0)
    db.data.assign(out_features, 0.0f);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            db.data[j] += grad.data[i * out_features + j];
        }
    }
    
    // Compute gradient w.r.t. input: dx = grad · W^T
    Tensor W_T;
    W_T.shape = {out_features, in_features};
    W_T.data.resize(out_features * in_features);
    for (size_t i = 0; i < in_features; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            W_T.data[j * in_features + i] = W.data[i * out_features + j];
        }
    }
    
    Tensor dx = matmul(grad, W_T);
    
    return dx;
}

void Dense::step(float lr) {
    // Update weights: W = W - lr * dW
    assert(W.data.size() == dW.data.size() && "Weight and gradient size mismatch");
    for (size_t i = 0; i < W.data.size(); ++i) {
        W.data[i] -= lr * dW.data[i];
    }
    
    // Update biases: b = b - lr * db
    assert(b.data.size() == db.data.size() && "Bias and gradient size mismatch");
    for (size_t i = 0; i < b.data.size(); ++i) {
        b.data[i] -= lr * db.data[i];
    }
}

// ReLU
Tensor ReLU::forward(const Tensor& x) {
    // Cache input for backward pass
    x_cache = x;
    
    // Apply ReLU: f(x) = max(0, x) element-wise
    Tensor result;
    result.shape = x.shape;
    result.data.resize(x.data.size());
    
    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = std::max(0.0f, x.data[i]);
    }
    
    return result;
}

Tensor ReLU::backward(const Tensor& grad) {
    // Gradient of ReLU: 
    // d(ReLU)/dx = 1 if x > 0, else 0
    
    assert(grad.shape == x_cache.shape && "Gradient shape mismatch");
    
    Tensor result;
    result.shape = grad.shape;
    result.data.resize(grad.data.size());
    
    for (size_t i = 0; i < grad.data.size(); ++i) {
        // Pass gradient through if input was positive, block if negative
        result.data[i] = (x_cache.data[i] > 0.0f) ? grad.data[i] : 0.0f;
    }
    
    return result;
}

// Loss
float MSELoss::forward(const Tensor& y_pred, const Tensor& y_true) {
    // Cache predictions and targets for backward pass
    y_pred_cache = y_pred;
    y_true_cache = y_true;
    
    // MSE = (1/n) * sum((y_pred - y_true)^2)
    assert(y_pred.shape == y_true.shape && "Prediction and target shapes must match");
    assert(y_pred.data.size() == y_true.data.size() && "Data size mismatch");
    
    float sum_squared_error = 0.0f;
    size_t n = y_pred.data.size();
    
    for (size_t i = 0; i < n; ++i) {
        float diff = y_pred.data[i] - y_true.data[i];
        sum_squared_error += diff * diff;
    }
    
    float mse = sum_squared_error / n;
    
    return mse;
}

Tensor MSELoss::backward() {
    // Derivative of MSE with respect to y_pred:
    // d(MSE)/d(y_pred) = (2/n) * (y_pred - y_true)
    
    assert(y_pred_cache.shape == y_true_cache.shape && "Cached shapes mismatch");
    
    Tensor grad;
    grad.shape = y_pred_cache.shape;
    grad.data.resize(y_pred_cache.data.size());
    
    size_t n = y_pred_cache.data.size();
    float scale = 2.0f / n;
    
    for (size_t i = 0; i < n; ++i) {
        grad.data[i] = scale * (y_pred_cache.data[i] - y_true_cache.data[i]);
    }
    
    return grad;
}

void Model::add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

Tensor Model::forward(const Tensor& x) {
    Tensor output = x;
    
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    
    return output;
}

void Model::backward(const Tensor& grad) {
    Tensor current_grad = grad;
    
    for (int i = layers.size() - 1; i >= 0; --i) {
        current_grad = layers[i]->backward(current_grad);
    }
}

void Model::step(float lr) {
    // Update parameters for all layers that have them
    for (auto& layer : layers) {
        layer->step(lr);
    }
}