cmake_minimum_required(VERSION 3.18)
project(NeuralNetwork LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(include)

# CPU-only library
add_library(nn_cpu
    src/tensor.cpp
    src/layers.cpp
    src/nn.cpp
)

# GPU library
add_library(nn_gpu
    src/cuda/cuda_tensor.cu
    src/cuda/kernels.cu
    src/cuda/layers_gpu.cu
)

set_target_properties(nn_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75  # Change based on your GPU
)

# Executables
add_executable(mnist_cpu examples/mnist_cpu.cpp)
target_link_libraries(mnist_cpu nn_cpu)

add_executable(mnist_gpu examples/mnist_gpu.cpp)
target_link_libraries(mnist_gpu nn_cpu nn_gpu)