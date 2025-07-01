use crate::CudaServerKey; // your server key type
use tfhe::{set_server_key, FheUint16, ClientKey};

use crate::add::fhe_add;
use crate::add::fhe_negate;
use crate::mul::fhe_lmul16_parallel;

use tfhe::prelude::*;

use rayon::prelude::*;

use std::time::Instant;

use half::f16;


pub struct DenseLayer {
    weights: Vec<Vec<FheUint16>>, // [output_size][input_size] - already encrypted
    biases: Vec<FheUint16>,       // [output_size] - already encrypted
}

impl DenseLayer {
    /// Create a new DenseLayer with encrypted weights and biases
    pub fn new(weights: Vec<Vec<FheUint16>>, biases: Vec<FheUint16>) -> Self {
        assert_eq!(weights.len(), biases.len(), "Weights and biases size mismatch");
        Self { weights, biases }
    }

    pub fn forward(
        &self,
        input: &[FheUint16],           // encrypted input vector [input_size]
        encrypted_zero: &FheUint16,    // zero ciphertext for reuse
        encrypted_1023: &FheUint16,
        server_key: &CudaServerKey,    // your FHE server key for homomorphic ops
    ) -> Vec<FheUint16> {
        self.weights
            .par_iter()
            .zip(self.biases.par_iter())
            .map(|(weight_row, bias)| {
                // Start timing total neuron computation
                let neuron_timer = Instant::now();
    
                let mut acc = encrypted_zero.clone();
    
                let mut mul_time = std::time::Duration::ZERO;
                let mut add_time = std::time::Duration::ZERO;
    
                for (inp, w) in input.iter().zip(weight_row.iter()) {
                    let mul_start = Instant::now();
                    let prod = fhe_lmul16_parallel(
                        inp.clone(),
                        w.clone(),
                        encrypted_zero.clone(),
                        server_key.clone(),
                    );
                    mul_time += mul_start.elapsed();
    
                    let add_start = Instant::now();
                    acc = fhe_add(acc, prod, encrypted_zero.clone(), encrypted_1023.clone(), server_key.clone());
                    add_time += add_start.elapsed();
                }
    
                // Add bias after sum
                fhe_add(acc, bias.clone(), encrypted_zero.clone(), encrypted_1023.clone(), server_key.clone())
            })
            .collect()
    }

    pub fn backward(
        &mut self,
        input: &[FheUint16],
        target: &[FheUint16],
        output: &[FheUint16],
        learning_rate: &FheUint16,
        encrypted_zero: &FheUint16,
        encrypted_1023: &FheUint16,
        server_key: &CudaServerKey,
        client_key: &ClientKey, 
    ) {
        self.weights
            .iter_mut()
            .zip(self.biases.iter_mut())
            .zip(output.iter().zip(target.iter()))
            .for_each(|((weight_row, bias), (y_hat, mut y_true))| {
                let binding_y_true = fhe_negate(y_true.clone(), server_key.clone());
                y_true = &binding_y_true;
    
                let error = fhe_add(
                    y_hat.clone(),
                    y_true.clone(),
                    encrypted_zero.clone(),
                    encrypted_1023.clone(),
                    server_key.clone(),
                );
    
                // 🔍 Debug: Decrypt error
                let error_plain: u16 = error.decrypt(client_key);
                let error_f32 = f16::from_bits(error_plain).to_f32();
                println!("Error: {}", error_f32);
    
                for (w_i, x_i) in weight_row.iter_mut().zip(input.iter()) {
                    let grad = fhe_lmul16_parallel(
                        error.clone(),
                        x_i.clone(),
                        encrypted_zero.clone(),
                        server_key.clone(),
                    );
    
                    let mut update = fhe_lmul16_parallel(
                        grad,
                        learning_rate.clone(),
                        encrypted_zero.clone(),
                        server_key.clone(),
                    );
    
                    let binding_update = fhe_negate(update.clone(), server_key.clone());
                    update = binding_update;
    
                    // 🔍 Debug: Decrypt weight update
                    let update_plain: u16 = update.decrypt(client_key);
                    let update_f32 = f16::from_bits(update_plain).to_f32();
                    println!("Weight update: {}", update_f32);
    
                    *w_i = fhe_add(
                        w_i.clone(),
                        update,
                        encrypted_zero.clone(),
                        encrypted_1023.clone(),
                        server_key.clone(),
                    );
                }
    
                let mut bias_update = fhe_lmul16_parallel(
                    error,
                    learning_rate.clone(),
                    encrypted_zero.clone(),
                    server_key.clone(),
                );
    
                let binding_bias_update = fhe_negate(bias_update.clone(), server_key.clone());
                bias_update = binding_bias_update;
    
                // 🔍 Debug: Decrypt bias update
                let bias_update_plain: u16 = bias_update.decrypt(client_key);
                let bias_update_f32 = f16::from_bits(bias_update_plain).to_f32();
                println!("Bias update: {}", bias_update_f32);
    
                *bias = fhe_add(
                    bias.clone(),
                    bias_update,
                    encrypted_zero.clone(),
                    encrypted_1023.clone(),
                    server_key.clone(),
                );
            });

            self.print_learned_parameters(client_key);
    }

    pub fn print_learned_parameters(&self, client_key: &ClientKey) {
        println!("\nLearned Weights:");
        for (i, row) in self.weights.iter().enumerate() {
            let decrypted_row: Vec<f32> = row.iter()
                .map(|enc| {
                    let plain: u16 = enc.decrypt(client_key);
                    f16::from_bits(plain).to_f32()
                })
                .collect();
            println!("Neuron {}: {:?}", i, decrypted_row);
        }

        println!("\nLearned Biases:");
        let decrypted_biases: Vec<f32> = self.biases.iter()
            .map(|enc| {
                let plain: u16 = enc.decrypt(client_key);
                f16::from_bits(plain).to_f32()
            })
            .collect();

        println!("{:?}", decrypted_biases);
    }

}

