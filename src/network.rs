use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::{self, EncryptedContext};
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::encrypted_losses::loss_function::LossFunction;
use crate::encrypted_losses::loss_function::MseLoss;
use crate::encrypted_ops::*;
use crate::activations::tanh_activation::*;
use crate::encrypted_nn::EncryptedNeuralNetworkImpl;

use rayon::iter::Inspect;
use tfhe::prelude::FheTryEncrypt;
use tfhe::prelude::FheDecrypt;

use tfhe::shortint::parameters::v1_2::*;
use tfhe::{generate_keys, set_server_key, ClientKey, CompressedServerKey, ConfigBuilder, CudaServerKey, FheUint16, FheUint32, FheUint64, FheUint8, ServerKey};

use rand::thread_rng;
use rand_distr::{Normal, Distribution};

use half::f16;

use std::time::Instant;

pub trait EncryptedNeuralNetwork{
    fn create() -> Self;
    fn add_dense(&mut self, input_size: usize, output_size: usize);
    fn add_tanh_activation(&mut self, size: usize);
    fn train(
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        train_inputs: &[Vec<f32>],   
        train_labels: &[Vec<f32>],  
        val_inputs: &[Vec<f32>],
        val_labels: &[Vec<f32>],
    );
    fn print_plain_weights(&self, id: String);
    fn print_plain_biases(&self, id:String);
    fn print_plain_grad_weights(&self, id: String);
    fn print_plain_grad_biases(&self, id:String);
}

pub struct EncryptedNeuralNetworkU8CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint8>,
}

pub struct EncryptedNeuralNetworkU8GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint8>,
}

pub struct EncryptedNeuralNetworkU16CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint16>,
}

pub struct EncryptedNeuralNetworkU16GPU {
    pub inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint16>,
}

pub struct EncryptedNeuralNetworkU32CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint32>,
}

pub struct EncryptedNeuralNetworkU32GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint32>,
}

pub struct EncryptedNeuralNetworkU64CPU {
    inner: EncryptedNeuralNetworkImpl<ServerKey, FheUint64>,
}

pub struct EncryptedNeuralNetworkU64GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint64>,
}

/// Format: (min_input, max_input, a, b, derivative)
pub static TANH16_PLA_RANGES: &[(u16, u16, u16, u16, u16)] = &[
    (50176u16, 65535u16, 48128u16, 0u16,     0u16),     // [-inf, -4], output ~ -1, derivative ≈ 0
    (49664u16, 50175u16, 9699u16,  15454u16, 9699u16),  // [-4.0, -3.0], slope ≈ 0.023, intercept ≈ -1.092
    (49152u16, 49663u16, 11960u16, 47802u16, 11960u16), // [-3.0, -2.0], slope ≈ 0.105, intercept ≈ -0.841
    (48128u16, 49151u16, 13476u16, 46883u16, 13476u16), // [-2.0, -1.0], slope ≈ 0.290, intercept ≈ -0.446
    (32768u16, 48127u16, 14412u16, 0u16,     14412u16), // [-1.0, -0.0],  slope ≈ 0.537, intercept ≈ 0.0
    (0u16,     15360u16, 14412u16, 0u16,     14412u16), // [0.0, 1.0],   slope ≈ 0.537, intercept ≈ 0.0
    (15361u16, 16384u16, 13476u16, 14115u16, 13476u16), // [1.0, 2.0],   slope ≈ 0.290, intercept ≈ 0.446
    (16385u16, 16896u16, 11960u16, 15034u16, 11960u16), // [2.0, 3.0],   slope ≈ 0.105, intercept ≈ 0.841
    (16897u16, 17408u16, 9699u16,  15454u16, 9699u16),  // [3.0, 4.0],   slope ≈ 0.023, intercept ≈ 1.092
    (17409u16, 32767u16, 15360u16, 0u16,     0u16),     // [4.0, +inf], output ~ 1, derivative ≈ 0
];

pub static TANH32_PLA_RANGES: &[(u32, u32, u32, u32, u32)] = &[
    (3229614080u32, 4294967295u32, 3212836864u32, 0u32, 0u32,), // [-inf, -4], output ~ -1, derivative ≈ 0
    (3225419776u32, 3229614079u32, 999046974u32, 3212538397u32, 999046974u32), // [-4.0, -3.0], slope ≈ 0.00428, intercept ≈ -0.98221
    (3221225472u32, 3225419775u32, 1023286696u32, 3211192529u32, 1023286696u32), // [-3.0, -2.0], slope ≈ 0.03102, intercept ≈ -0.90199
    (3212836864u32, 3221225471u32, 1045384302u32, 3205440628u32, 1045384302u32), // [-2.0, -1.0], slope ≈ 0.20244, intercept ≈ -0.55915
    (3204448256u32, 3212836863u32, 1058624781u32, 3190197018u32, 1058624781u32), // [-1.0, -0.5], slope ≈ 0.598954, intercept ≈ −0.162640
    (2147483648u32, 3204448255u32, 1064082073u32, 0u32, 1064082073u32), // [-0.5, -0.0],  slope ≈ 0.924234, intercept ≈ 0.0
    (0u32, 1056964608u32, 1064082073u32, 0u32, 1064082073u32), // [0.0, 0.5],   slope ≈ 0.924234, intercept ≈ 0.0
    (1056964609u32, 1065353216u32, 1058624781u32, 1042713370u32, 1058624781u32), // // [0.5, 1], slope ≈ 0.598954, intercept ≈ 0.162640
    (1065353217u32, 1073741824u32, 1045384302u32, 1057956980u32, 1045384302u32), // [1.0, 2.0],   slope ≈ 0.20244, intercept ≈ 0.55915
    (1073741825u32, 1077936128u32, 1023286696u32, 1063708881u32, 1023286696u32), // [2.0, 3.0],   slope ≈ 0.03102, intercept ≈ 0.90199
    (1077936129u32, 1082130432u32, 999046974u32, 1065054749u32, 999046974u32), // [3.0, 4.0],   slope ≈ 0.00428, intercept ≈ 0.98221
    (1082130433u32, 2147483647u32, 1065353217u32, 0u32, 0u32), // [4.0, +inf], output ~ 1, derivative ≈ 0
];
/*
impl EncryptedNeuralNetwork for EncryptedNeuralNetworkU16GPU {
    fn create() -> Self{
        let config =
        ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let server_key = compressed_server_key.decompress_to_gpu();

        let encrypted_zero = FheUint16::try_encrypt(0u16, &client_key).unwrap();
        let encrypted_mask = FheUint16::try_encrypt(1023u16, &client_key).unwrap();
        
        let context = EncryptedContext {
            encrypted_zero,
            encrypted_mask,
            client_key,
            server_key,
        };
        let layers: Vec<Box<dyn EncryptedLayer<CudaServerKey, FheUint16>>> = vec![];

        // Step 4: Create the loss function
        let loss: Box<dyn LossFunction<CudaServerKey, FheUint16>> = Box::new(MseLoss{});

        // Step 5: Build the inner EncryptedNeuralNetworkImpl
        let inner = EncryptedNeuralNetworkImpl {
            layers,
            loss,
            context,
        };

        // Step 6: Wrap in the public struct
        EncryptedNeuralNetworkU16GPU { inner }
    }

    fn add_dense(&mut self, input_size: usize, output_size: usize) {
        let encrypted_weights = self.init_weights(output_size, input_size);
        let encrypted_biases = self.init_biases(output_size);
        let encrypted_grad_weights = self.init_gradients(&[output_size, input_size]);
        let encrypted_grad_biases = self.init_gradients(&[output_size]);
        self.inner.add_dense(encrypted_weights, encrypted_biases, encrypted_grad_weights, encrypted_grad_biases);
    }

    fn train(&mut self, epochs: usize, batch_size: usize, learning_rate: f32, train_inputs: &[Vec<f32>], train_labels: &[Vec<f32>], val_inputs: &[Vec<f32>], val_labels: &[Vec<f32>],) {
        set_server_key(self.inner.context.server_key.clone());
        let enc_learning_rate = FheUint16::try_encrypt(f16::from_f32(learning_rate).to_bits(), &self.inner.context.client_key).unwrap();
        let enc_train_inputs = self.encrypt_dataset(batch_size, train_inputs);
        let enc_train_labels = self.encrypt_dataset(batch_size, train_labels);
        let enc_val_inputs = self.encrypt_dataset(batch_size, val_inputs);
        let enc_val_labels = self.encrypt_dataset(batch_size, val_labels);
        let time = Instant::now();
        self.inner.train(epochs, batch_size, enc_learning_rate.clone(), enc_train_inputs.clone(), enc_train_labels.clone(), enc_val_inputs.clone(), enc_val_labels.clone());
        println!("Training completed in {:?}", time.elapsed());
    }

    fn print_plain_weights(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let weights = layer.get_weights();
                let shape = &weights.shape;
    
                if shape.len() != 2 {
                    println!("Expected 2D shape for weights, got {:?}", shape);
                    return;
                }
    
                let rows = shape[0];
                let cols = shape[1];
                let flat = weights.data;
    
                if flat.len() != rows * cols {
                    println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                    return;
                }
    
                println!("\nDecrypted Weights for Layer \"{}\":", id);
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        //let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                        let decrypted: u16 = EncryptableValueType::decrypt(&flat[index], &self.inner.context.client_key);
                        print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                    }
                    print!("]\n");
                }
                return;
            }
        }
    
        println!("No layer found with id: {}", id);
    }

    fn print_plain_biases(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_biases();
                let shape = &biases.shape;

                let columns = shape[1];
                let flat = biases.data;

                println!("\nDecrypted Biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    //let decrypted: u16 = flat[i].decrypt(&self.inner.context.client_key);
                    let decrypted: u16 = EncryptableValueType::decrypt(&flat[i], &self.inner.context.client_key);
                    print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                }
                print!("]\n");
                return;
            }
        }
    }

    fn print_plain_grad_weights(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let weights = layer.get_grad_weights();
                let shape = &weights.shape;
    
                if shape.len() != 2 {
                    println!("Expected 2D shape for grad_weights, got {:?}", shape);
                    return;
                }
    
                let rows = shape[0];
                let cols = shape[1];
                let flat = weights.data;
    
                if flat.len() != rows * cols {
                    println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                    return;
                }
    
                println!("\nDecrypted grad_weights for Layer \"{}\":", id);
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        //let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                        let decrypted: u16 = EncryptableValueType::decrypt(&flat[index], &self.inner.context.client_key);
                        print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                    }
                    print!("]\n");
                }
                return;
            }
        }
    
        println!("No layer found with id: {}", id);
    }

    fn print_plain_grad_biases(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_grad_biases();
                let shape = &biases.shape;

                let columns = shape[1];
                let flat = biases.data;

                println!("\nDecrypted grad_biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    //let decrypted: u16 = flat[i].decrypt(&self.inner.context.client_key);
                    let decrypted: u16 = EncryptableValueType::decrypt(&flat[i], &self.inner.context.client_key);
                    print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                }
                print!("]\n");
                return;
            }
        }
    }

}

impl EncryptedNeuralNetworkU16GPU {
    fn init_weights(&mut self, input_size: usize, output_size: usize) -> EncryptedTensor<FheUint16>{
        
        
        // Xavier Initialization
        let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
        let normal = Normal::new(0.0, std_dev).unwrap();

        let mut rng = thread_rng();
        
        let mut encrypted_weights = Vec::with_capacity(input_size * output_size);

        for i in 0..(input_size * output_size) {
            let sample = f16::from_f32(normal.sample(&mut rng) as f32);
            let u_sample = sample.to_bits();
            let encrypted_sample = FheUint16::try_encrypt(u_sample, &self.inner.context.client_key).expect("Weight initialization failed");;
            encrypted_weights.push(encrypted_sample);
        }
        
        EncryptedTensor { data: (encrypted_weights), shape: (vec![input_size, output_size]) }
    }

    fn init_biases(&mut self, output_size:usize) -> EncryptedTensor<FheUint16> {
        let zero_enc = &self.inner.context.encrypted_zero;
        let biases = vec![zero_enc.clone(); output_size];
        EncryptedTensor::new(biases, vec![1, output_size])
    }

    fn init_gradients(&self, shape: &[usize]) -> EncryptedTensor<FheUint16> {
        let zero_enc = &self.inner.context.encrypted_zero;
        let size = shape.iter().product();
        let zeros = vec![zero_enc.clone(); size];
        let shapes: Vec<usize>;
        if(shape.to_vec().len() == 1){
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        EncryptedTensor::new(zeros, shapes)
    }

    fn encrypt_dataset(&mut self, batch_size: usize, clear_data: &[Vec<f32>]) -> EncryptedTensor<FheUint16> {
        let mut encrypted_dataset = Vec::new();
        let feature_size = clear_data[0].len();

        for vec in clear_data{
            for sample in vec{
                let u16_sample = f16::from_f32(*sample).to_bits();
                let enc_sample = FheUint16::try_encrypt(u16_sample, &self.inner.context.client_key).unwrap();
                encrypted_dataset.push(enc_sample);
            }
        }

        EncryptedTensor { data: encrypted_dataset, shape: vec![batch_size, feature_size] }

    }

}
*/
impl EncryptedNeuralNetwork for EncryptedNeuralNetworkU32GPU {
    fn create() -> Self{
        let config =
        ConfigBuilder::with_custom_parameters(V1_2_PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let server_key = compressed_server_key.decompress_to_gpu();

        let encrypted_zero = FheUint32::try_encrypt(0u32, &client_key).unwrap();
        //let encrypted_mask = FheUint32::try_encrypt(1023u32, &client_key).unwrap();
        let encrypted_mask = FheUint32::try_encrypt(8388607u32, &client_key).unwrap();

        let ranges = TANH32_PLA_RANGES
        .iter()
        .map(|&(a, b, c, d, e)| {
            (
                FheUint32::try_encrypt(a, &client_key).unwrap(),
                FheUint32::try_encrypt(b, &client_key).unwrap(),
                FheUint32::try_encrypt(c, &client_key).unwrap(),
                FheUint32::try_encrypt(d, &client_key).unwrap(),
                FheUint32::try_encrypt(e, &client_key).unwrap(),
            )
        })
        .collect();
        
        let context = EncryptedContext {
            encrypted_zero,
            encrypted_mask,
            client_key,
            server_key,
            ranges,
        };
        let layers: Vec<Box<dyn EncryptedLayer<CudaServerKey, FheUint32>>> = vec![];

        // Step 4: Create the loss function
        let loss: Box<dyn LossFunction<CudaServerKey, FheUint32>> = Box::new(MseLoss{});

        // Step 5: Build the inner EncryptedNeuralNetworkImpl
        let inner = EncryptedNeuralNetworkImpl {
            layers,
            loss,
            context,
        };

        // Step 6: Wrap in the public struct
        EncryptedNeuralNetworkU32GPU { inner }
    }

    fn add_dense(&mut self, input_size: usize, output_size: usize) {
        let encrypted_weights = self.init_weights(output_size, input_size);
        let encrypted_biases = self.init_biases(output_size);
        let encrypted_grad_weights = self.init_gradients(&[output_size, input_size]);
        let encrypted_grad_biases = self.init_gradients(&[output_size]);
        self.inner.add_dense(encrypted_weights, encrypted_biases, encrypted_grad_weights, encrypted_grad_biases);
    }

    fn add_tanh_activation(&mut self, size: usize) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_tanh_activation(derivatives, self.inner.context.ranges.clone());
    }

    fn train(&mut self, epochs: usize, batch_size: usize, learning_rate: f32, train_inputs: &[Vec<f32>], train_labels: &[Vec<f32>], val_inputs: &[Vec<f32>], val_labels: &[Vec<f32>],) {
        let enc_learning_rate = FheUint32::try_encrypt(learning_rate.to_bits(), &self.inner.context.client_key).unwrap();
        let enc_train_inputs = self.encrypt_dataset(batch_size, train_inputs);
        let enc_train_labels = self.encrypt_dataset(batch_size, train_labels);
        let enc_val_inputs = self.encrypt_dataset(batch_size, val_inputs);
        let enc_val_labels = self.encrypt_dataset(batch_size, val_labels);
        let time = Instant::now();
        self.inner.train(epochs, batch_size, enc_learning_rate.clone(), enc_train_inputs.clone(), enc_train_labels.clone(), enc_val_inputs.clone(), enc_val_labels.clone());
        println!("Training completed in {:?}", time.elapsed());
    }

    fn print_plain_weights(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let weights = layer.get_weights();
                let shape = &weights.shape;
    
                if shape.len() != 2 {
                    println!("Expected 2D shape for weights, got {:?}", shape);
                    return;
                }
    
                let rows = shape[0];
                let cols = shape[1];
                let flat = weights.data;
    
                if flat.len() != rows * cols {
                    println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                    return;
                }
    
                println!("\nDecrypted Weights for Layer \"{}\":", id);
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        //let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                        let decrypted: u32 = EncryptableValueType::decrypt(&flat[index], &self.inner.context.client_key);
                        print!("{:<6} ", f32::from_bits(decrypted));
                    }
                    print!("]\n");
                }
                return;
            }
        }
    
        println!("No layer found with id: {}", id);
    }

    fn print_plain_biases(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_biases();
                let shape = &biases.shape;

                let columns = shape[1];
                let flat = biases.data;

                println!("\nDecrypted Biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    //let decrypted: u16 = flat[i].decrypt(&self.inner.context.client_key);
                    let decrypted: u32 = EncryptableValueType::decrypt(&flat[i], &self.inner.context.client_key);
                    print!("{:<6} ", f32::from_bits(decrypted));
                }
                print!("]\n");
                return;
            }
        }
    }

    fn print_plain_grad_weights(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let weights = layer.get_grad_weights();
                let shape = &weights.shape;
    
                if shape.len() != 2 {
                    println!("Expected 2D shape for grad_weights, got {:?}", shape);
                    return;
                }
    
                let rows = shape[0];
                let cols = shape[1];
                let flat = weights.data;
    
                if flat.len() != rows * cols {
                    println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                    return;
                }
    
                println!("\nDecrypted grad_weights for Layer \"{}\":", id);
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        //let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                        let decrypted: u32 = EncryptableValueType::decrypt(&flat[index], &self.inner.context.client_key);
                        print!("{:<6} ", f32::from_bits(decrypted));
                    }
                    print!("]\n");
                }
                return;
            }
        }
    
        println!("No layer found with id: {}", id);
    }

    fn print_plain_grad_biases(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_grad_biases();
                let shape = &biases.shape;

                let columns = shape[1];
                let flat = biases.data;

                println!("\nDecrypted grad_biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    //let decrypted: u16 = flat[i].decrypt(&self.inner.context.client_key);
                    let decrypted: u32 = EncryptableValueType::decrypt(&flat[i], &self.inner.context.client_key);
                    print!("{:<6} ", f32::from_bits(decrypted));
                }
                print!("]\n");
                return;
            }
        }
    }

}

impl EncryptedNeuralNetworkU32GPU {
    fn init_weights(&mut self, input_size: usize, output_size: usize) -> EncryptedTensor<FheUint32>{
        
        
        // Xavier Initialization
        let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
        let normal = Normal::new(0.0, std_dev).unwrap();

        let mut rng = thread_rng();
        
        let mut encrypted_weights = Vec::with_capacity(input_size * output_size);

        for i in 0..(input_size * output_size) {
            let sample = normal.sample(&mut rng) as f32;
            let u_sample = sample.to_bits();
            let encrypted_sample = FheUint32::try_encrypt(u_sample, &self.inner.context.client_key).expect("Weight initialization failed");;
            encrypted_weights.push(encrypted_sample);
        }
        
        EncryptedTensor { data: (encrypted_weights), shape: (vec![input_size, output_size]) }
    }

    fn init_biases(&mut self, output_size:usize) -> EncryptedTensor<FheUint32> {
        let zero_enc = &self.inner.context.encrypted_zero;
        let biases = vec![zero_enc.clone(); output_size];
        EncryptedTensor::new(biases, vec![1, output_size])
    }

    fn init_gradients(&self, shape: &[usize]) -> EncryptedTensor<FheUint32> {
        let zero_enc = &self.inner.context.encrypted_zero;
        let size = shape.iter().product();
        let zeros = vec![zero_enc.clone(); size];
        let shapes: Vec<usize>;
        if(shape.to_vec().len() == 1){
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        EncryptedTensor::new(zeros, shapes)
    }

    fn init_derivatives(&self, shape: &[usize]) -> EncryptedTensor<FheUint32> {
        let zero_enc = &self.inner.context.encrypted_zero;
        let size = shape.iter().product();
        let zeros = vec![self.inner.context.encrypted_zero.clone(); size];
        let shapes: Vec<usize>;
        if(shape.to_vec().len() == 1){
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        EncryptedTensor::new(zeros, shapes)
    }

    fn encrypt_dataset(&mut self, batch_size: usize, clear_data: &[Vec<f32>]) -> EncryptedTensor<FheUint32> {
        let mut encrypted_dataset = Vec::new();
        let feature_size = clear_data[0].len();

        for vec in clear_data{
            for sample in vec{
                let u16_sample = (*sample).to_bits();
                let enc_sample = FheUint32::try_encrypt(u16_sample, &self.inner.context.client_key).unwrap();
                encrypted_dataset.push(enc_sample);
            }
        }

        EncryptedTensor { data: encrypted_dataset, shape: vec![batch_size, feature_size] }

    }

}