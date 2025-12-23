# NeuralNetwork
🧠 NeuralNetwork-CPP
A High-Performance, From-Scratch C++ Neural Network Engine

Built with a focus on Systems Architecture, memory efficiency, and hardware-level optimization. This project implements a multi-layer perceptron (MLP) without the use of external machine learning libraries, targeting the XOR problem and MNIST handwritten digit recognition.

🚀 Key Features
Hardware-Up Design: Built using standard C++ vectors with a focus on cache locality and spatial data alignment.

Mini-Batch Gradient Descent: Implements efficient data batching to stabilize training and optimize CPU throughput.

OpenMP Parallelization: Utilizes multi-threading for neuron-level calculations in large layers.

FMA Optimized: Mathematical kernels designed to trigger Fused Multiply-Add (FMA) and AVX2 instructions during the forward pass.

Custom File Manager: Proprietary .nnet file format for saving and loading trained weights and biases.

🛠️ The Architecture
The engine is structured to separate the mathematical kernels from the network management:

NNET Core: Handles the Feedforward and Backpropagation logic.

EMath: A dedicated math namespace for Sigmoid activations, Matrix-Vector multiplication (IKJ ordered), and MSE calculation.

FileManager: Handles model serialization and binary data I/O.

📊 Current Status: XOR Validation
The network is currently undergoing validation on the XOR logic gate problem.

Topology: 2 Input Neurons → 3 Hidden Neurons → 1 Output Neuron.

Activation: Sigmoid.

Optimization: Stochastic Gradient Descent (SGD) with Mini-Batching.

⚙️ Build & Run
Ensure you have a C++17 compatible compiler and OpenMP installed.

Bash

# Clone the repository
git clone https://github.com/Low-Zi-Hong/NeuralNetwork.git

# Navigate to the directory
cd NeuralNetwork

# Compile (example using GCC with OpenMP and AVX2)
g++ -O3 -fopenmp -mavx2 main.cpp scr/*.cpp -o NeuralNetwork.exe

# Run the engine
./NeuralNetwork.exe
🎯 Project Goals
Phase 1: Solve XOR logic gate (Current).

Phase 2: Implement MNIST data loader and achieve >90% accuracy on handwritten digits.

Phase 3: Refactor to flattened 1D contiguous memory buffers for maximum cache performance.