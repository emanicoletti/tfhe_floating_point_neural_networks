use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_utils::encrypted_context::{self, EncryptedContext};
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::encrypted_losses::loss_function::LossFunction;
use crate::encrypted_losses::loss_function::MseLoss;
use crate::encrypted_ops::*;
use crate::encrypted_nn;
use crate::encrypted_nn::EncryptedNeuralNetworkImpl;

use tfhe::prelude::FheTryEncrypt;
use tfhe::prelude::FheDecrypt;

use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64, PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use tfhe::{generate_keys, set_server_key, ClientKey, CompressedServerKey, ConfigBuilder, CudaServerKey, FheUint16, FheUint32, FheUint64, FheUint8, ServerKey};

use rand::thread_rng;
use rand_distr::{Normal, Distribution};

use half::f16;

pub trait EncryptedNeuralNetwork{
    fn create() -> Self;
    fn add_dense(&mut self, input_size: usize, output_size: usize);
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
        let encrypted_weights = self.init_weights(input_size, output_size);
        let encrypted_biases = self.init_biases(output_size);
        let encrypted_grad_weights = self.init_gradients(&[input_size, output_size]);
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
        self.inner.train(epochs, batch_size, enc_learning_rate, enc_train_inputs, enc_train_labels, enc_val_inputs, enc_val_labels);
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
    
                println!("Decrypted Weights for Layer \"{}\":", id);
                for i in 0..rows {
                    for j in 0..cols {
                        let index = i * cols + j;
                        let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                        print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                    }
                    println!();
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

                println!("Decrypted Biases for Layer \"{}\":", id);

                for i in 0..columns {
                    let decrypted: u16 = flat[i].decrypt(&self.inner.context.client_key);
                    print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                }
                println!();
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
    
                println!("Decrypted grad_weights for Layer \"{}\":", id);
                for i in 0..rows {
                    for j in 0..cols {
                        let index = i * cols + j;
                        let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                        print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                    }
                    println!();
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

                println!("Decrypted grad_biases for Layer \"{}\":", id);

                for i in 0..columns {
                    let decrypted: u16 = flat[i].decrypt(&self.inner.context.client_key);
                    print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                }
                println!();
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

        for _ in 0..(input_size * output_size) {
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