#include "nn.hpp"

int main() {
    Model model;
    model.add(std::make_unique<Dense>(2, 16));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(16, 1));

    MSELoss loss;

    for (int epoch = 0; epoch < 1000; ++epoch) {
        //Tensor y_pred = model.forward(x);
        //float L = loss.forward(y_pred, y);

        //Tensor grad = loss.backward();
        //model.backward(grad);
        //model.step(0.01f);
    }
}
