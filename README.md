# LZH Neural Engine (C++) 🧠

A high-performance Neural Network engine built from scratch in C++, designed for modularity and bit-perfect persistence. This project is part of my 2026 graduation milestones at Kolej Matrikulasi Perak.

## 🚀 Current Milestone: XOR Mastery
The engine has successfully converged on the XOR non-linear manifold with **100% Accuracy** and a stabilized loss of **0.0001**.

### Key Features
* **Dynamic Topology**: Flexible `reinit` system allowing the model to "shape-shift" architectures (e.g., from {2,4,1} for XOR to {784,128,10} for MNIST).
* **Binary Persistence (FMANAGER)**: Custom `.nnet` serialization protocol utilizing bit-perfect binary I/O for model saving/loading.
* **Stochastic Mini-Batching**: Efficient training loop using accumulated gradients and real-time telemetry.

## 🛠️ Installation & Usage
1. Clone the repository.
2. Open the Visual Studio Solution (`.sln`).
3. Build and Run.
4. Use the console UI to create a new model or load the pre-trained `model_v1.nnet`.

## 📈 Roadmap
- [x] XOR Logical Convergence (100% Acc)
- [ ] MNIST Data Bridge Implementation
- [ ] MNIST Model Training (Targeting 95%+ Accuracy)