#include "nn.hpp"
#include "cuda/layers_gpu.cuh"
#include <iostream>

int main() {
    std::cout << "Creating GPU model...\n";
    
    Model model;
    model.add(std::make_unique<DenseGPU>(784, 128));
    model.add(std::make_unique<ReLUGPU>());
    model.add(std::make_unique<DenseGPU>(128, 10));
    
    MSELossGPU loss;
    
    Tensor x;
    x.shape = {1, 784};
    x.data.resize(784, 0.5f);
    
    Tensor y;
    y.shape = {1, 10};
    y.data.resize(10, 0.0f);
    y.data[5] = 1.0f;
    
    std::cout << "Training...\n";
    for (int epoch = 0; epoch < 10; ++epoch) {
        Tensor y_pred = model.forward(x);
        float L = loss.forward(y_pred, y);
        Tensor grad = loss.backward();
        model.backward(grad);
        model.step(0.01f);
        
        if (epoch % 1 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << L << "\n";
        }
    }
    
    std::cout << "Training complete!\n";
    return 0;
}