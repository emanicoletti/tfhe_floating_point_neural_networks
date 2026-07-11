# Artifact Appendix

Paper title: **Training TFHE-Based Neural Networks with Approximated Floating-Point Arithmetic**

Requested Badge(s):
- [X] **Available**
- [X] **Functional**
- [X] **Reproduced**

## Description

This artifact contains the official Rust implementation accompanying our paper accepted at PoPETS 2026.

Our framework introduces an optimized implementation of fully homomorphic encryption (FHE) neural network training, utilizing tfhe-rs to accelerate floating-point operations directly over encrypted data. It includes our custom approximate floating-point suite (FP32 arithmetic via adapted L-mul operations), a command-line interface for running benchmarks, an on-demand dataset management module integrated with Zenodo, an encrypted pipeline for training tiny encrypted networks, and a verified plaintext emulation pipeline used to evaluate deep networks (LeNet-5, VGG, and ResNet-20).

### Security/Privacy Issues and Ethical Concerns

Evaluating or reusing this artifact presents no known security or privacy risks to the host machine. The code runs entirely within user space and does not disable any OS-level security configurations (such as ASLR or firewalls). It relies on standard cryptographic and parallel computing libraries (tfhe-rs and NVIDIA CUDA). No human subject data or tracking mechanisms are involved.

## Basic Requirements

### Hardware Requirements

- **Minimal Configuration**: The framework can be built and evaluated in "plain" mode, or run microbenchmarks on a standard laptop with at least 16 GB of RAM on common CPUs. To execute fully encrypted benchmarks and accelerated training paths efficiently, an NVIDIA GPU compatible with the CUDA 12.0 runtime is required.

- **Paper Reproduction Specifications**: The baseline benchmarks and wall-clock execution figures reported in the paper were evaluated on a workstation with an Intel Xeon Gold 5318S CPU (24 cores, 48 threads), 384 GB RAM, and an NVIDIA A40 GPU, running Ubuntu 24.04 LTS.

### Software Requirements

- **Operating System**: Ubuntu 22.04 LTS or newer (highly recommended for seamless CUDA toolchain compatibility).
- **Rust Toolchain**: Stable Rust compiler framework (Edition 2021).
- **NVIDIA CUDA Toolkit**: CUDA version 12.0 or newer. (Please refer to official NVIDIA documentation to install the correct toolkit for your specific OS architecture).
- **Host dependencies**: `cmake`, `pkg-config`, `libssl-dev`, `git-lfs`, as well as `gcc-12` and `g++-12` (strictly required for CUDA compatibility).
- **Dependencies**: All external crates (including tfhe-rs) are specified in `Cargo.toml` and are resolved automatically by Cargo during compilation.
- **Datasets & Models**: The benchmark suite includes an integrated, self-healing dataset downloader. Preprocessed dataset assets are pulled directly from our verified Zenodo repository on demand during runtime execution; no manual data ingestion is required.

### Estimated Time and Storage Consumption

- **Human Time**: 10–15 minutes for installation and environment configuration.
- **Compute Time**: Individual encrypted primitive operations complete within a few seconds each. Encrypted training can require anywhere from several hours to a few days. Plaintext training can require anywhere from a few minutes to 2–3 days.
- **Storage Consumption**: The base repository takes less than 50 MB. Depending on which benchmarks are triggered, the automated Zenodo download manager fetches asset arrays consuming up to 2–5 GB of local disk space within the `res/` directory.

## Environment

### Accessibility

The verified source code repository is persistently hosted on GitHub and can be accessed at: github.com/emanicoletti/tfhe_floating_point_neural_networks

### Set up the environment

Before cloning the repository, ensure your host machine has the required build tools and libraries installed:

```bash
sudo apt update
sudo apt install -y cmake pkg-config libssl-dev git-lfs build-essential
git lfs install
```

Run the following commands in your terminal to clone the project, target the compliant host compiler version, and build the production-ready release profile:

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

### Testing the Environment

To verify that the Rust toolchain, CUDA libraries, and compilation configurations are functioning correctly, execute the interactive routing menu without flags:

```bash
cargo run --release
```

**Expected Output:** The terminal should launch a visually navigable CLI menu (arrow-key navigation) containing a selection list of the paper's 10 core benchmarks alongside an Encrypted Ops Playground. If this panel renders and allows navigation, your environment setup is successful.

When running plain executions, you can toggle between standard exact evaluation and our proposed approximate arithmetic using Rust's compilation features:

- **For Exact Plain Execution:**
  ```bash
  cargo run --release --features exact
  ```
- **For Approximate Plain Execution (default):**
  ```bash
  cargo run --release --features lmul
  ```

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Performance and Memory Optimization

Our approximate encrypted floating-point arithmetic primitives achieve significant performance improvements on CPU backends compared to state-of-the-art exact TFHE floating-point baselines. Specifically, multiplication runs up to 2.1× faster, division runs up to 29.3× faster, and peak memory usage is reduced by up to 57×. This claim is supported by running evaluations in the Encrypted Ops Playground.

#### Main Result 2: Small-Scale Encrypted MLP + CNN Training

The framework successfully executes full, end-to-end encrypted training cycles under TFHE parameters for smaller network architectures. Specifically, users can run fully encrypted training for both a small-scale Multi-Layer Perceptron (MLP) and a small-scale Convolutional Neural Network (CNN). These claims are directly reproducible using the interactive CLI configurations.

#### Main Result 3: Emulation Alignment & Deep Network Scale

By exploiting a mathematically verified plaintext emulation layer, the framework enables evaluation of large architectures (LeNet-5, VGG, ResNet-20) over six distinct imaging datasets, achieving convergence alignment within 1% of the exact-arithmetic baselines. This claim is verified via our emulation feature tracking suites.

### Experiments

#### Experiment 1: Encrypted Ops Playground (Microbenchmarks)

- **Time**: 1 human minute + 2 compute minutes
- **Storage**: Negligible (<10 MB)

This experiment evaluates the processing latency of individual primitives to support the claims in Main Result 1.

Run the command:

```bash
cargo run --release
```

Navigate with the CLI to **11. Encrypted Ops Playground (Test Encrypted Operations with Custom Configurations)** and select the operations you want to run, the number of chained operations, the format, and the hardware backend (CPU/GPU).
To empirically validate the equivalence of encrypted and emulated computation claimed in Section 5.3: Select Miscellaneous inside the Playground and insert 50000 as the number of chained operations.

#### Experiment 2: Encrypted Emulation Check (Encrypted Training)

- **Time**: 1 human minute + 36 compute hours
- **Storage**: <100 MB (automatically managed via Zenodo)

This experiment reproduces Main Result 2: Small-Scale Encrypted MLP or CNN Training.

Run the command:

```bash
cargo run --release
```

Navigate with the CLI to **1. MLP[87] on Ternary-MNIST (Plain or Encrypted)** or **2. CNN[109] on Ternary-MNIST (Plain or Encrypted)**. Select **Run Both versions sequentially**. The dataset will be downloaded automatically. The plain version will run first, outputting loss values and printed weights; the encrypted version will follow (taking several hours).

#### Experiment 3: Plaintext Emulation Check (Deep Architectures)

- **Time**: 1 human minute + 48 compute hours
- **Storage**: ~2 GB (automatically managed via Zenodo)

This experiment verifies structural convergence over deep network architectures to support the claims in Main Result 3, running the most complex architecture. 

Run the command:

```bash
cargo run --release --features lmul
```

Navigate with the CLI to **7. ResNet-20[273k] on CIFAR-10 (Plain)**. The dataset will be downloaded automatically. Computation will begin and you will see epochs and batches being processed. Alternatively, users can evaluate any of the pre-configured benchmarks provided in the suite.
For any experiment, to reproduce the exact arithmetic accuracy metrics found in Table 5, you must run the plain execution using the exact feature flag: 

```bash
cargo run --release --features exact
```

## Limitations

Fully encrypted training of large networks such as ResNet-20 and VGG-style models is severely constrained by current hardware processing limits. To work around these systemic overhead restrictions, this artifact evaluates deep architectures via a mathematically verified plaintext emulation pipeline. For fully encrypted execution of these deeper architectures, the repository produces wall-clock mathematical projections based on structural primitive timings rather than live, multi-day model iterations. Note that even the emulation pipeline requires several hours of CPU processing time and significant RAM.

Memory consumption for each single operation (Figure 3) is not visible using the CLI. It was measured using external profiling tools. It is omitted from the artifact because a direct, fair comparison requires integrating external SoTA source code.

## Notes on Reusability

The approximate arithmetic engine developed for this framework is designed as a standalone module within the Rust codebase. Researchers can decouple the custom FP32 L-mul and L-div primitives from the neural network routing structures and reuse them as a general-purpose approximate floating-point library for other homomorphic encryption research tasks utilizing tfhe-rs. The same applies to the exact operation primitives.
