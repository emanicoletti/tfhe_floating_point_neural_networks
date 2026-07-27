# Training TFHE-Based Neural Networks with Approximated Floating-Point Arithmetic

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![DOI](https://img.shields.io/badge/Zenodo-Dataset-blue.svg)](https://zenodo.org/records/YOUR_RESERVED_REC_ID)

This repository contains the official Rust implementation companion for our paper: **"Training TFHE-Based Neural Networks with Approximated Floating-Point Arithmetic"** (Accepted at **PoPETS 2026**).

---

## 📝 Abstract & Project Summary

Training neural networks under Torus Fully Homomorphic Encryption (TFHE) has historically been severely bottlenecked by the scheme's native restriction to boolean and integer arithmetic. This framework breaks that barrier, enabling **approximate floating-point ($FP32$) training within TFHE** by reinterpreting IEEE 754 representations as encrypted integers and executing them via hardware-accelerated primitives.

### 🌟 Core Breakthroughs
* **Hybrid Arithmetic Suite:** Formulates exact encrypted operations (Addition, Subtraction, Square Root) alongside high-speed approximate primitives (Multiplication via adapted *L-mul*, and Division) implemented entirely over integer arithmetic.
* **Massive Performance Leaps:** Drastically reduces both evaluation wall-clock times and memory demand compared to state-of-the-art exact TFHE floating-point arithmetic.
* **Verified Plaintext Emulation:** Bypasses FHE hardware limitations through a verified plaintext emulator that guarantees strict output alignment (identical network parameters), enabling scaling projections up to deeper models like VGG and ResNet-20.

---

## 🚀 Features

* **Interactive CLI Tool:** Reproduce specific paper benchmarks with a single command via a guided arrow-key menu interface.
* **On-Demand Data Sourcing:** Local repository remains lightweight. Large preprocessed machine learning datasets are fetched automatically and securely from our verified Zenodo repository upon selection.
* **Deterministic Hyperparameter Enforcement:** Core execution layers dynamically configure underlying learning-rate schedules and architectures internally, guaranteeing faithful replica execution without manual source configuration.
* **CUDA Accelerated Backend:** Deep integration with GPU acceleration via `tfhe-cuda-backend` for cryptographic processing.

---

## 🛠️ Prerequisites & Setup

Ensure your local environment satisfies the following software and hardware toolchains:

* **Rust Toolchain:** Stable Rust (Edition 2021)
* **NVIDIA CUDA Toolkit:** CUDA `12.0` (or matching version supported by your active GPU drivers)
* **Host C++ Compiler:** `GCC 12` / `G++ 12` (*CUDA 12.0 strictly requires host compiler verification $\le$ GCC 12*)

### Quickstart: Build & Reproduce

Run the following commands sequentially to clone the repository, configure your host compiler, build the project, and launch the interactive experiment menu:

```bash
# 1. Clone the repository and navigate into it
git clone https://github.com/emanicoletti/tfhe_floating_point_neural_networks.git
cd tfhe_floating_point_neural_networks

# 2. Force the cryptographic build script to prioritize the supported host compiler
export CC=gcc-12
export CXX=g++-12

# 3. Build the production profile
cargo build --release

# 4. Launch the interactive UI to reproduce paper experiments
cargo run --release
```

## ⚙️ Advanced Execution Flags

### 1. Arithmetic Engine Variants (Plain Execution)
When conducting plain executions, you can toggle between standard exact evaluation or our proposed approximate arithmetic using Rust's compilation features:

* **For Exact Plain Execution:**
  ```bash
  cargo run --release --features exact
  ```
* **For Approximate Plain Execution [default]:** 
  ```bash
  cargo run --release --features lmul
  ```


## 📜 Citation

```bibtex
@article{nicoletti2026training,
  title={Training TFHE-Based Neural Networks with Approximated Floating-Point Arithmetic},
  author={Nicoletti, Emanuele and Pittorino, Fabrizio and Falcetta, Alessandro and Colombo, Luca and Roveri, Manuel},
  journal={Proceedings on Privacy Enhancing Technologies},
  volume={2026},
  number={4},
  pages={376--395},
  year={2026},
  doi={10.56553/popets-2026-0126}
}
```

## 🤝 Acknowledgments

This framework is built upon the foundational cryptographic engineering of the team at **[Zama](https://zama.ai/)**. It heavily utilizes the **[TFHE-rs](https://github.com/zama-ai/tfhe-rs)** library to implement our custom accelerated floating-point primitives over Torus Fully Homomorphic Encryption.

## 📄 License
This project is licensed under the MIT License, see the LICENSE file for details.
