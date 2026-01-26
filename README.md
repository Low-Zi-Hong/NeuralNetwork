🧠 NnetLZH: High-Performance Neural Engine
A hybrid deep learning framework featuring a hand-optimized C++ calculation backend and a flexible Python interface.

📖 Overview
NnetLZH is a custom-built Neural Network engine designed to bridge the gap between low-level performance and high-level usability.

Core Backend: Written in C++ with OpenMP parallelization, handling matrix operations, backpropagation, and memory management.

Python Interface: Exposed via pybind11, allowing users to script models, load datasets, and visualize results using Python's ecosystem while enjoying C++ speeds.

✨ Key Features
🐍 Python Bindings: Seamlessly import the C++ engine as a native Python module (import NnetLZH).

⚡ Parallel Compute: Uses OpenMP to distribute neuron activation and gradient calculations across P-Cores & E-Cores.

🔢 Native MNIST Loader: High-speed C++ parser for the MNIST dataset, exposed directly to Python.

💾 Model Persistence: Save/Load trained weights to disk via binary serialization.

🛠 Hybrid Workflow: Train in C++ for speed, or prototype in Python for flexibility.

⚙️ Build Instructions
Option 1: Standalone C++ App
Build the NeuralNetwork.exe for maximum raw performance without Python dependencies.

Define: Comment out #define PythonLib in main.cpp.

Compiler: Enable /openmp.

Option 2: Python Extension (.pyd)
Compile the engine as a shared library callable from Python.

Dependencies:

Python 3.x Development Headers

pybind11

Visual Studio Configuration:

Preprocessor: Add #define PythonLib.

Include Directories: Add paths to pybind11/include and Python/include.

Library Directories: Add path to Python/libs.

Configuration Type: Dynamic Library (.dll), rename output to NnetLZH.pyd.

Build: Run via Release mode.

🐍 Python Usage Example
Once built, place NnetLZH.pyd in your project folder.

Python
import NnetLZH
import numpy as np

# 1. Initialize Network (784 Input, 128 Hidden, 10 Output)
topology = [784, 128, 10]
model = NnetLZH.NeuralNet(topology)

# 2. Randomize Weights
NnetLZH.RandomInitialise(model)

# 3. Load Data (Using C++ High-Speed Loader)
print("[*] Loading MNIST via C++ backend...")
NnetLZH.LoadImg("train-images.idx3-ubyte")
NnetLZH.LoadLabel("train-labels.idx1-ubyte")
NnetLZH.ProcessData() 

# 4. Training Loop (Python controls logic, C++ does math)
print("[*] Starting Training...")
for epoch in range(5):
    # Note: In a real loop, you'd iterate through data here.
    # This example assumes the C++ backend manages the internal data cursor 
    # or you pass data vectors (depending on implementation).
    
    NnetLZH.FeedPropagation(model)
    cost = NnetLZH.CalculateError(model, target_labels)
    print(f"Epoch {epoch}: Cost = {cost}")

# 5. Save Model
NnetLZH.SaveModel(model, "my_model.bin")
🧠 Architecture
Code snippet
graph TD
    A[Python Script] -->|Calls| B(pybind11 Interface)
    B -->|Invokes| C{C++ Core Engine}
    C -->|#pragma omp| D[CPU Core 1]
    C -->|#pragma omp| E[CPU Core 2]
    C -->|#pragma omp| F[CPU Core N]
    C -->|Direct IO| G[Binary File / MNIST]
