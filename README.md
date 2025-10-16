# TFHE-Based Floating-Point Neural Network Training
![Banner](./assets/banner.png)

This is the official repository for .... paper. In this study we make feasible, for the first time, the usage of Floating-Point arithmetic using TFHE encryption scheme for training Deep Neural Networks. 
The entire framework is implemented in Rust, and based on [TFHE-rs](https://docs.zama.ai/tfhe-rs) library. 

## Setup
To use it, just clone it:
```
git clone https://github.com/emanicoletti/tfhe_floating_point_neural_networks.git
cd tfhe_floating_point_neural_networks
```
and run it with Cargo:
```
cargo run --release
```

## Training
You can define and train your own encrypted model by specifying its architecture, number of epochs, batch size, learning rate, etc.
```
let mut encrypted_model = EncryptedNeuralNetworkU16GPU::create(Some(1));

// Define the architecture
encrypted_model.add_max_pooling(vec![16, 16], 4, 4);
encrypted_model.add_dense(16, 4);
encrypted_model.add_tanh_activation(4);
encrypted_model.add_dense(4, 2);
encrypted_model.add_tanh_activation(2);
encrypted_model.add_dense(2, 3);
encrypted_model.add_tanh_activation(3);

// Train the model 
encrypted_model.train(
    50,                     // epochs
    64,                     // batch size
    0.1,                    // learning rate
    &train_inputs,          
    &train_labels,
    vec![6400, 1, 16, 16],  // input shape
    vec![6400, 1, 1, 3],    // output shape
);
```
To emulate the learning process on plaintext data (useful for testing arithmetic or debugging), simply use the same architecture but replace encrypted constructs with plain ones:
```
plain_model.add_max_pooling(vec![16, 16], 4, 4);
plain_model.add_dense(16, 4);
plain_model.add_tanh_activation(4);
plain_model.add_dense(4, 2);
plain_model.add_tanh_activation(2);
plain_model.add_dense(2, 3);
plain_model.add_tanh_activation(3);

// Train the model
plain_model.train(
    50,                     // epochs
    64,                     // batch size
    0.1,                    // learning rate
    &train_inputs,
    &train_labels,
    vec![6400, 1, 16, 16],  // input shape
    vec![6400, 1, 1, 3],    // output shape
);
```
You can evaluate in plain using three different settings: exact, lmul (default), or pam. You can select the mode at runtime with Cargo features, e.g.:
```
cargo run --release --features pam
```
Encrypted training currently supports only the lmul mode.

## Replicate Experiments
To reproduce the experiments from the paper, call the corresponding function in your main():
```
experiment_x_fpy(...)
```
where x corresponds to one of the three experiments (1, 2, 3), and y the bit_width (16 or 32).

## Test arithmetic
You can test individual encrypted arithmetic operations by calling the corresponding test function in your main():
```
test_encrypted_ops("lmul", 32, false, 1, -5.0, 5.0)?;
```
where the parameters corresponds to the operation the list is below, then the size, the true or false for gpu acceleration, number of operations, and range.
| Parameter                 | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `op_name`                 | Operation name (e.g., `"lmul"`, `"add"`, `"pam_div"`) |
| `bit_width`               | Floating-point bit width (8, 16, 32, 64)              |
| `use_gpu`                 | `true` to enable GPU acceleration, `false` otherwise  |
| `num_ops`                 | Number of operations to perform                       |
| `range_min` / `range_max` | Value range for random input generation               |

Available operations:
| Operation         | Description              | Type        | FP Formats               | Speed             |
| ----------------- | ------------------------ | ----------- | ------------------------ | ----------------- |
| **add**           | Addition                 | Exact       | FP8 / FP16 / FP32 / FP64 | ⚙️ Medium          |
| **same_sign_add** | Same-sign addition       | Exact       | FP8 / FP16 / FP32 / FP64 | ⚙️ Medium          |
| **sub**           | Subtraction              | Exact       | FP8 / FP16 / FP32 / FP64 | ⚙️ Medium         |
| **lmul**          | Lmul multiplication      | Approximate | FP8 / FP16 / FP32 / FP64 | ⚡ Fast            |
| **ldiv**          | Lmul division            | Approximate | FP8 / FP16 / FP32 / FP64 | ⚡ Fast            |
| **lmul_tanh**     | PLA Tanh (based on Lmul) | Approximate | FP16 / FP32              | 🐢 Very Slow      |
| **pam_mul**       | PAM multiplication       | Approximate | FP16 / FP32              | ⚡ Fast            |
| **pam_div**       | PAM division             | Approximate | FP16 / FP32              | ⚡ Fast            |
| **pam_tanh**      | PLA Tanh (based on PAM)  | Approximate | FP16 / FP32              | 🐢 Very Slow      |
| **relu**          | ReLU activation          | Exact       | FP8 / FP16 / FP32 / FP64 | 🚀 Very Fast      |
| **sqrt**          | Square root              | Exact       | FP16 / FP32              | 🐢 Very Slow      |
| **log2**          | Base-2 logarithm         | Exact       | FP16 / FP32              | 🐌 Extremely Slow |


## Citation
