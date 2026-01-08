cmake_minimum_required(VERSION 3.16)

project(neuralnetwork
    VERSION 0.1
    LANGUAGES CXX
)

# Use modern C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create the executable
add_executable(neuralnetwork
    src/main.cpp
    src/nn.cpp
)

# Tell the compiler where headers live
target_include_directories(neuralnetwork
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
