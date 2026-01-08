#pragma once
#include <vector>
#include <memory>

// ---- Tensor ----
struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;
};

// ---- Layer base class ----
class Layer {
public:
    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad) = 0;
    virtual void step(float lr) {}
    virtual ~Layer() = default;
};

// ---- Concrete layers ----
class Dense : public Layer {
private:
    Tensor W, b;           // weights and biases
    Tensor dW, db;         // gradients
    Tensor x_cache;        // cached input for backward pass
public:
    Dense(size_t in, size_t out);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    void step(float lr) override;
};

class ReLU : public Layer {
private:
    Tensor x_cache;        // cached input for backward pass
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
};

// ---- Loss ----
class MSELoss {
private:
    Tensor y_pred_cache, y_true_cache;  // cached for backward pass
public:
    float forward(const Tensor& y_pred, const Tensor& y_true);
    Tensor backward();
};

// ---- Model ----
class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    void add(std::unique_ptr<Layer> layer);
    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad);
    void step(float lr);
};
