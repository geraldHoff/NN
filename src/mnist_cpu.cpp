#include "../include/nn.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>

// Helper to reverse bytes (MNIST uses big-endian)
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Read MNIST images
std::vector<Tensor> read_mnist_images(const std::string& filename, int max_images = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char*)&num_images, sizeof(num_images));
    num_images = reverseInt(num_images);

    file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = reverseInt(num_rows);

    file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = reverseInt(num_cols);

    if (max_images > 0 && max_images < num_images) {
        num_images = max_images;
    }

    std::cout << "Reading " << num_images << " images (" << num_rows << "x" << num_cols << ")...\n";

    std::vector<Tensor> images;
    int image_size = num_rows * num_cols;

    for (int i = 0; i < num_images; ++i) {
        Tensor img;
        img.shape = {1, (size_t)image_size};
        img.data.resize(image_size);

        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            img.data[j] = pixel / 255.0f;
        }

        images.push_back(img);
    }

    file.close();
    return images;
}

// Read MNIST labels
std::vector<Tensor> read_mnist_labels(const std::string& filename, int max_labels = -1) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    int magic_number = 0, num_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = reverseInt(num_labels);

    if (max_labels > 0 && max_labels < num_labels) {
        num_labels = max_labels;
    }

    std::cout << "Reading " << num_labels << " labels...\n";

    std::vector<Tensor> labels;

    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));

        // One-hot encode: 10 outputs for digits 0-9
        Tensor lbl;
        lbl.shape = {1, 10};
        lbl.data.resize(10, 0.0f);
        lbl.data[label] = 1.0f;

        labels.push_back(lbl);
    }

    file.close();
    return labels;
}

int main() {
    std::cout << "=== MNIST Digit Classification ===\n\n";

    // Load dataset (use subset for faster training)
    std::cout << "Loading MNIST dataset...\n";
    auto train_images = read_mnist_images("train-images-idx3-ubyte", 5000);
    auto train_labels = read_mnist_labels("train-labels-idx1-ubyte", 5000);
    auto test_images = read_mnist_images("t10k-images-idx3-ubyte", 1000);
    auto test_labels = read_mnist_labels("t10k-labels-idx1-ubyte", 1000);

    if (train_images.empty() || train_labels.empty()) {
        std::cerr << "Failed to load training data!\n";
        return 1;
    }

    std::cout << "\nDataset loaded successfully!\n";
    std::cout << "Training samples: " << train_images.size() << "\n";
    std::cout << "Test samples: " << test_images.size() << "\n\n";

    // Create neural network
    // Architecture: 784 inputs -> 128 hidden -> 64 hidden -> 10 outputs
    std::cout << "Creating neural network (784 -> 128 -> 64 -> 10)...\n\n";
    Model model;
    model.add(std::make_unique<Dense>(784, 128));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(128, 64));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(64, 10));

    MSELoss loss_fn;
    float learning_rate = 0.01;
    int epochs = 10;

    std::cout << "Training for " << epochs << " epochs...\n";
    std::cout << "Learning rate: " << learning_rate << "\n\n";

    // Training
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < train_images.size(); ++i) {
            // Forward
            Tensor pred = model.forward(train_images[i]);
            float loss = loss_fn.forward(pred, train_labels[i]);
            total_loss += loss;

            // Backward
            Tensor grad = loss_fn.backward();
            model.backward(grad);

            // Update
            model.step(learning_rate);
        }

        float avg_loss = total_loss / train_images.size();

        // Evaluate
        int correct = 0;
        for (size_t i = 0; i < test_images.size(); ++i) {
            Tensor pred = model.forward(test_images[i]);

            int pred_digit = std::max_element(pred.data.begin(), pred.data.end()) - pred.data.begin();
            int true_digit = std::max_element(test_labels[i].data.begin(), test_labels[i].data.end()) - test_labels[i].data.begin();

            if (pred_digit == true_digit) {
                correct++;
            }
        }

        float accuracy = 100.0f * correct / test_images.size();

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - Loss: " << avg_loss
                  << " - Test Accuracy: " << accuracy << "% ("
                  << correct << "/" << test_images.size() << ")\n";
    }

    std::cout << "\n=== Training Complete ===\n";

    // Show some example predictions
    std::cout << "\nSample predictions (first 10 test images):\n";
    for (int i = 0; i < 10 && i < (int)test_images.size(); ++i) {
        Tensor pred = model.forward(test_images[i]);
        int pred_digit = std::max_element(pred.data.begin(), pred.data.end()) - pred.data.begin();
        int true_digit = std::max_element(test_labels[i].data.begin(), test_labels[i].data.end()) - test_labels[i].data.begin();

        std::cout << "Image " << i << ": Predicted=" << pred_digit
                  << ", True=" << true_digit
                  << " " << (pred_digit == true_digit ? "✓" : "✗") << "\n";
    }

    return 0;
}