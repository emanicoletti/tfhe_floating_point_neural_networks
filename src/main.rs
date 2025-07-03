use half::f16;

use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use tfhe::{prelude::*, set_server_key, generate_keys};
use tfhe::{ ConfigBuilder, FheUint16, ClientKey, CompressedServerKey, CudaServerKey};

use rand::distributions::{Uniform};
use rand::{thread_rng, Rng};

use std::time::Instant;

mod encrypted_layers;
mod plain_layers;
mod encrypted_ops;
use crate::encrypted_layers::EncryptedDenseLayer;
use crate::plain_layers::PlainDenseLayer;


// Function to generate dataset
fn generate_dataset() -> Vec<(Vec<f32>, Vec<f32>)> {
    vec![
        (vec![1.0, 2.0, 3.0, 4.0], vec![22.0, 37.5, 42.2]),
        (vec![6.0, 2.0, 4.0, 2.0], vec![47.0, 45.0, 48.2]),
        (vec![1.5, 0.8, 3.7, 0.0], vec![18.3, 28.05, 34.76]),
        (vec![1.0, 1.9, 1.3, 2.0], vec![16.3, 21.1, 22.48]),
        (vec![0.3, 5.0, 2.1, 3.0], vec![24.07, 34.05, 35.04]),
        (vec![1.5, 4.0, 0.8, 1.0], vec![23.1, 20.05, 18.4]),
    ]
}

fn main() {
    
    // 1. Generate keys
    let config = ConfigBuilder::with_custom_parameters(
        PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
    )
    .build();

    let client_key = ClientKey::generate(config);
    let compressed_server_key = CompressedServerKey::new(&client_key);
    let cuda_server_key = compressed_server_key.decompress_to_gpu();


    set_server_key(cuda_server_key.clone());

    // Create encrypted zero and learning rate
    let encrypted_zero = FheUint16::encrypt(0u16, &client_key);
    let encrypted_1023 = FheUint16::encrypt(1023u16, &client_key);
    let learning_rate = FheUint16::encrypt(8479u16, &client_key); // ~0.01 in f16

    let limit = (6.0f32).sqrt() / ((4 + 3) as f32).sqrt(); // ≈ 0.92
    let dist = Uniform::new(-limit, limit);
    let mut rng = thread_rng();

    // 2. Initialize weights and biases
    let float_weights: Vec<Vec<f32>> = (0..3)
    .map(|_| (0..4).map(|_| (rng.sample(dist))).collect())
    .collect();
    let float_biases: Vec<f32> = vec![0.0, 0.0, 0.0];

    let plain_weights: Vec<Vec<u16>> = float_weights
        .iter()
        .map(|row| row.iter().map(|&f| f16::from_f32(f).to_bits()).collect())
        .collect();
    let encrypted_weights: Vec<Vec<FheUint16>> = plain_weights
        .iter()
        .map(|row| row.iter().map(|&w| FheUint16::encrypt(w, &client_key)).collect())
        .collect();

    let plain_biases: Vec<u16> = float_biases.iter().map(|&f| f16::from_f32(f).to_bits()).collect();
    let encrypted_biases: Vec<FheUint16> = plain_biases
        .iter()
        .map(|&b| FheUint16::encrypt(b, &client_key))
        .collect();

    let mut dense_layer = EncryptedDenseLayer::new(encrypted_weights, encrypted_biases);

    let float_weights_f16: Vec<Vec<f16>> = float_weights
    .iter()
    .map(|row| row.iter().map(|&w| f16::from_f32(w as f32)).collect())
    .collect();

    // Convert biases: Vec<f32> or Vec<f64> → Vec<f16>
    let float_biases_f16: Vec<f16> = float_biases
        .iter()
        .map(|&b| f16::from_f32(b as f32))
        .collect();

    // Now create the plain dense layer with f16 values
    let mut plain_dense_layer = PlainDenseLayer::new(float_weights_f16, float_biases_f16);

    let plain_learning_rate: f16 = f16::from_f32(0.01);
    let dataset = generate_dataset();
    let num_epochs = 10;

    for epoch in 0..num_epochs {
        println!("Epoch {}", epoch + 1);

        for (float_input, float_target) in &dataset[..dataset.len()] {
            
            // Convert f64 → f32 → f16
            let float_input_f16: Vec<f16> = float_input.iter()
                .map(|&f| f16::from_f32(f as f32))
                .collect();
        
            let float_target_f16: Vec<f16> = float_target.iter()
                .map(|&f| f16::from_f32(f as f32))
                .collect();

            
            // --- Plain Forward Pass ---
            let output = plain_dense_layer.forward(&float_input_f16);
        
            println!("Epoch {} Plain", epoch + 1);
            for (i, y) in output.iter().enumerate() {
                println!("  Output[{}] = {:.4}", i, f32::from(*y));
            }
        
            // --- Plain Backward Pass ---
            plain_dense_layer.backward(&float_input_f16, &float_target_f16, &output, plain_learning_rate);
            plain_dense_layer.print_learned_parameters();
            println!("------------------------");
            
            // --- Encrypt input and target ---
            
            let encrypted_input: Vec<FheUint16> = float_input_f16.iter()
                .map(|&f| FheUint16::encrypt(f.to_bits(), &client_key))
                .collect();
        
            let encrypted_target: Vec<FheUint16> = float_target_f16.iter()
                .map(|&f| FheUint16::encrypt(f.to_bits(), &client_key))
                .collect();
        
            // --- Encrypted Forward Pass ---
            let encrypted_output = dense_layer.forward(
                &encrypted_input,
                &encrypted_zero,
                &encrypted_1023,
                &cuda_server_key,
            );
        
            // --- Encrypted Backward Pass ---
            dense_layer.backward(
                &encrypted_input,
                &encrypted_target,
                &encrypted_output,
                &learning_rate,
                &encrypted_zero,
                &encrypted_1023,
                &cuda_server_key,
                &client_key
            );

            let plain_test_input = dataset.last().map(|(inputs, outputs)| {
                let inputs_f16_bits: Vec<u16> = inputs.iter()
                    .map(|&v| f16::from_f32(v).to_bits())
                    .collect();
                let outputs_f16_bits: Vec<u16> = outputs.iter()
                    .map(|&v| f16::from_f32(v).to_bits())
                    .collect();
                (inputs_f16_bits, outputs_f16_bits)
            }).unwrap();

            let encrypted_test_input: Vec<FheUint16> = plain_test_input.0.iter()
            .map(|&v| FheUint16::encrypt(v, &client_key))
            .collect();

            let encrypted_test_output = dense_layer.forward(&encrypted_test_input, &encrypted_zero, &encrypted_1023, &cuda_server_key);

            println!("  → Output after epoch {}:", epoch + 1);
            for (i, y_enc) in encrypted_test_output.iter().enumerate() {
                let y_plain: u16 = y_enc.decrypt(&client_key);
                let float_res = f16::from_bits(y_plain).to_f32();
                println!("    Output[{}] = {}", i, float_res);
            }

            dense_layer.print_learned_parameters(&client_key.clone());
            println!("------------------------");
        
        }
    }   
}