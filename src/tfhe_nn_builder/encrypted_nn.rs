use crate::tfhe_nn_builder::encrypted_utils::tensor::EncryptedTensor;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::{self, EncryptedContext};
use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::tfhe_nn_builder::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::tfhe_nn_builder::encrypted_losses::loss_function::LossFunction;
use crate::tfhe_nn_builder::encrypted_losses::loss_function::MseLoss;
use crate::tfhe_nn_builder::encrypted_ops::*;
use crate::tfhe_nn_builder::encrypted_activations::tanh_activation::*;
use crate::tfhe_nn_builder::generic_enc_nn::EncryptedNeuralNetworkImpl;

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
    fn add_max_pooling(&mut self, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize);
    fn train(
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        train_inputs: &[Vec<f32>],   
        train_labels: &[Vec<f32>],  
        input_shapes: Vec<usize>,
        label_shapes: Vec<usize>
    );
    fn inference(
        &mut self, 
        input: &[Vec<f32>], 
        input_shapes: Vec<usize>, 
        label_shapes: Vec<usize>) 
        -> Vec<f32>;
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
    (49152u16, 65535u16, 48128u16, 0u16, 0u16), // [-inf, -2], output ~ -1, derivative ≈ 0
    (47787u16, 49151u16, 13312u16, 47104u16, 13312u16), // [-2, -0.8333] slope 0.25. intercept -0.5
    (32768u16, 47786u16, 15053u16, 0u16, 15053u16), // [-0.833, -0] slope=0.85 intercept = 0
    (0u16, 15019u16, 15053u16, 0u16, 15053u16), //[0, 0.833] slope=0.85 intercept = 0
    (15020u16, 16384u16, 13312u16, 14336u16, 13312u16), //[0.833, 2] slope = 0.25 intercept 0.5
    (16385u16, 32767u16, 15360u16, 0u16, 0u16), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

pub static TANH32_PLA_RANGES: &[(u32, u32, u32, u32, u32)] = &[
    (3221225472u32, 4294967295u32, 3212836864u32, 0u32, 0u32), // [-inf, -2], output ~ -1, derivative ≈ 0
    (3210040661u32, 3221225471u32, 1048576000u32, 3204448256u32, 1048576000u32), // [-2, -0.8333] slope 0.25. intercept -0.5
    (2147483648u32, 3210040660u32, 1062836634u32, 0u32, 1062836634u32), // [-0.833, -0] slope=0.85 intercept = 0
    (0u32, 1062557013u32, 1062836634u32, 0u32, 1062836634u32), //[0, 0.833] slope=0.85 intercept = 0
    (1062557014u32, 1073741824u32, 1048576000u32, 1056964608u32, 1048576000u32), //[0.833, 2] slope = 0.25 intercept 0.5
    (1073741825u32, 2147483647u32, 1065353217u32, 0u32, 0u32), // [2.0, +inf], output ~ 1, derivative ≈ 0
];


impl EncryptedNeuralNetwork for EncryptedNeuralNetworkU16GPU {
    fn create() -> Self{
        let config =
        ConfigBuilder::with_custom_parameters(V1_2_PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let server_key = compressed_server_key.decompress_to_gpu();

        let encrypted_zero = FheUint16::try_encrypt(0u16, &client_key).unwrap();
        let encrypted_mask = FheUint16::try_encrypt(1023u16, &client_key).unwrap();

        let ranges = TANH16_PLA_RANGES
        .iter()
        .map(|&(a, b, c, d, e)| {
            (
                FheUint16::try_encrypt(a, &client_key).unwrap(),
                FheUint16::try_encrypt(b, &client_key).unwrap(),
                FheUint16::try_encrypt(c, &client_key).unwrap(),
                FheUint16::try_encrypt(d, &client_key).unwrap(),
                FheUint16::try_encrypt(e, &client_key).unwrap(),
            )
        })
        .collect();
        
        let context = EncryptedContext {
            encrypted_zero,
            encrypted_mask,
            client_key,
            server_key,
            ranges

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

    fn add_tanh_activation(&mut self, size: usize) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_tanh_activation(derivatives, self.inner.context.ranges.clone());
    }

    fn add_max_pooling(&mut self, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize) {
        self.inner.add_max_pooling(input_dim, kernel_size, stride, padding);
    }

    fn train(&mut self, epochs: usize, batch_size: usize, learning_rate: f32, train_inputs: &[Vec<f32>], train_labels: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) {
        let enc_learning_rate = FheUint16::try_encrypt(f16::from_f32(learning_rate).to_bits(), &self.inner.context.client_key).unwrap();
        let enc_train_inputs = self.encrypt_dataset(train_inputs, input_shapes.clone());
        let enc_train_labels = self.encrypt_dataset(train_labels, label_shapes.clone());
        let time = Instant::now();
        self.inner.train(epochs, batch_size, enc_learning_rate.clone(), enc_train_inputs.clone(), enc_train_labels.clone());
        println!("Training completed in {:?}", time.elapsed());
    }

    fn inference(&mut self, input: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) -> Vec<f32>{
        let inf_input = self.encrypt_dataset(input, input_shapes.clone());
        let prediction = self.inner.inference(&inf_input.clone());

        let mut prediction_f32: Vec<f32> = self.decrypt_tensor(&prediction);
        //println!("Prediction: {:?}", prediction_f32);

        let predicted_index = prediction_f32
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

        // 2. Create one-hot encoded vector
        let mut one_hot: Vec<f32> = vec![0.0; prediction_f32.len()];
        one_hot[predicted_index] = 1.0;
        
        one_hot
    }

    fn print_plain_weights(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let weights = layer.get_weights();
                let shape = &weights.shape;
    
                if shape.len() != 4 {
                    println!("Expected 4D shape for weights, got {:?}", shape);
                    return;
                }

                let rows = shape[2];
                let cols = shape[3];
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

                let columns = shape[3];
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

                if shape.len() != 4 {
                    println!("Expected 4D shape for grad_weights, got {:?}", shape);
                    return;
                }
    
                let rows = shape[2];
                let cols = shape[3];
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

                let columns = shape[3];
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

    fn decrypt_tensor(&self, encrypted_tensor: &EncryptedTensor<FheUint16>) -> Vec<f32> {
    encrypted_tensor.data.iter()
        .map(|value| f16::from_bits(EncryptableValueType::decrypt(value, &self.inner.context.client_key)).to_f32())
        .collect()
    }


    fn init_weights(&mut self, input_size: usize, output_size: usize) -> EncryptedTensor<FheUint16>{
        let mut plain_weights: Vec<u16> = vec![];
        if output_size == 16 {
            let fc1_weight: Vec<Vec<u16>> = vec![
            vec![
                f16::from_f32(0.1478_f32).to_bits(), f16::from_f32(0.2460_f32).to_bits(), f16::from_f32(-0.1902_f32).to_bits(), f16::from_f32(0.2048_f32).to_bits(),
                f16::from_f32(-0.0094_f32).to_bits(), f16::from_f32(-0.1631_f32).to_bits(), f16::from_f32(-0.0475_f32).to_bits(), f16::from_f32(-0.0258_f32).to_bits(),
                f16::from_f32(0.0352_f32).to_bits(), f16::from_f32(0.0608_f32).to_bits(), f16::from_f32(0.2474_f32).to_bits(), f16::from_f32(0.2253_f32).to_bits(),
                f16::from_f32(0.1077_f32).to_bits(), f16::from_f32(0.0669_f32).to_bits(), f16::from_f32(-0.0140_f32).to_bits(), f16::from_f32(0.1992_f32).to_bits()
            ],
            vec![
                f16::from_f32(-0.0941_f32).to_bits(), f16::from_f32(-0.1871_f32).to_bits(), f16::from_f32(-0.1587_f32).to_bits(), f16::from_f32(0.1157_f32).to_bits(),
                f16::from_f32(-0.2498_f32).to_bits(), f16::from_f32(-0.1199_f32).to_bits(), f16::from_f32(0.1795_f32).to_bits(), f16::from_f32(-0.1308_f32).to_bits(),
                f16::from_f32(0.1570_f32).to_bits(), f16::from_f32(-0.1353_f32).to_bits(), f16::from_f32(0.1298_f32).to_bits(), f16::from_f32(-0.2283_f32).to_bits(),
                f16::from_f32(0.1142_f32).to_bits(), f16::from_f32(0.0593_f32).to_bits(), f16::from_f32(0.0358_f32).to_bits(), f16::from_f32(-0.0192_f32).to_bits()
            ],
            vec![
                f16::from_f32(-0.2092_f32).to_bits(), f16::from_f32(0.2381_f32).to_bits(), f16::from_f32(-0.1979_f32).to_bits(), f16::from_f32(-0.2127_f32).to_bits(),
                f16::from_f32(-0.1407_f32).to_bits(), f16::from_f32(0.1909_f32).to_bits(), f16::from_f32(0.0236_f32).to_bits(), f16::from_f32(0.0435_f32).to_bits(),
                f16::from_f32(0.1414_f32).to_bits(), f16::from_f32(0.0379_f32).to_bits(), f16::from_f32(-0.2027_f32).to_bits(), f16::from_f32(0.1000_f32).to_bits(),
                f16::from_f32(0.1482_f32).to_bits(), f16::from_f32(-0.1996_f32).to_bits(), f16::from_f32(0.1349_f32).to_bits(), f16::from_f32(-0.0602_f32).to_bits()
            ],
            vec![
                f16::from_f32(-0.0248_f32).to_bits(), f16::from_f32(-0.0221_f32).to_bits(), f16::from_f32(-0.0468_f32).to_bits(), f16::from_f32(0.0468_f32).to_bits(),
                f16::from_f32(0.0116_f32).to_bits(), f16::from_f32(0.0588_f32).to_bits(), f16::from_f32(-0.2422_f32).to_bits(), f16::from_f32(0.0707_f32).to_bits(),
                f16::from_f32(0.1853_f32).to_bits(), f16::from_f32(-0.0841_f32).to_bits(), f16::from_f32(0.1562_f32).to_bits(), f16::from_f32(-0.0972_f32).to_bits(),
                f16::from_f32(-0.1543_f32).to_bits(), f16::from_f32(-0.0157_f32).to_bits(), f16::from_f32(0.1084_f32).to_bits(), f16::from_f32(-0.2480_f32).to_bits()
            ],
            ];
            plain_weights = fc1_weight.into_iter().flatten().collect();
        }
        if output_size == 4 {
            let fc2_weight: Vec<Vec<u16>> = vec![
            vec![f16::from_f32(-0.3546_f32).to_bits(), f16::from_f32(0.2355_f32).to_bits(), f16::from_f32(-0.2220_f32).to_bits(), f16::from_f32(-0.0288_f32).to_bits()],
            vec![f16::from_f32(-0.2830_f32).to_bits(), f16::from_f32(-0.4757_f32).to_bits(), f16::from_f32(0.1246_f32).to_bits(), f16::from_f32(0.0483_f32).to_bits()],
            ];
            plain_weights = fc2_weight.into_iter().flatten().collect();
        }
        if output_size == 2 {
            let fc3_weight: Vec<Vec<u16>> = vec![
                vec![f16::from_f32(-0.6488_f32).to_bits(), f16::from_f32(0.2701_f32).to_bits()],
                vec![f16::from_f32(0.1953_f32).to_bits(), f16::from_f32(0.1416_f32).to_bits()],
                vec![f16::from_f32(-0.1549_f32).to_bits(), f16::from_f32(-0.6687_f32).to_bits()],
            ];
            plain_weights = fc3_weight.into_iter().flatten().collect();
        }
        
        /* 
        // Xavier Initialization
        let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
        let normal = Normal::new(0.0, std_dev).unwrap();

        let mut rng = thread_rng();
        */
        
        let mut encrypted_weights = Vec::with_capacity(input_size * output_size);

        for i in 0..(input_size * output_size) {
            //let sample = normal.sample(&mut rng) as f32;
            //let sample = 0.0 as f32; 
            let sample = plain_weights[i];
            let encrypted_sample = FheUint16::try_encrypt(sample, &self.inner.context.client_key).expect("Weight initialization failed");
            encrypted_weights.push(encrypted_sample);
        }
        
        EncryptedTensor { data: (encrypted_weights), shape: (vec![1, 1, input_size, output_size]) }
    }

    fn init_biases(&mut self, output_size: usize) -> EncryptedTensor<FheUint16> {
        let mut plain_biases: Vec<u16> = vec![];
        if output_size == 4 {
            let fc1_bias: Vec<u16> = vec![
                f16::from_f32(-0.0879_f32).to_bits(), f16::from_f32(0.1680_f32).to_bits(), f16::from_f32(-0.1631_f32).to_bits(), f16::from_f32(-0.0271_f32).to_bits()
            ];
            plain_biases = fc1_bias.into_iter().map(|b| b).collect();
        }
        if output_size == 2 {
            let fc2_bias: Vec<u16> = vec![
                f16::from_f32(0.4002_f32).to_bits(), f16::from_f32(-0.0112_f32).to_bits()
            ];
            plain_biases = fc2_bias.into_iter().map(|b| b).collect();
        }
        if output_size == 3 {
            let fc3_bias: Vec<u16> = vec![
                f16::from_f32(-0.1854_f32).to_bits(), f16::from_f32(-0.2199_f32).to_bits(), f16::from_f32(-0.6619_f32).to_bits()
            ];
            plain_biases = fc3_bias.into_iter().map(|b|b).collect();
        }

        let mut encrypted_biases = Vec::with_capacity(output_size);
        for bias in plain_biases {
            let encrypted_bias = FheUint16::try_encrypt(bias, &self.inner.context.client_key)
                .expect("Bias encryption failed");
            encrypted_biases.push(encrypted_bias);
        }

        EncryptedTensor::new(encrypted_biases, vec![1, 1, 1, output_size])
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

     fn init_derivatives(&self, shape: &[usize]) -> EncryptedTensor<FheUint16> {
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

    fn encrypt_dataset(&mut self, clear_data: &[Vec<f32>], input_shapes: Vec<usize>) -> EncryptedTensor<FheUint16> {
        let mut encrypted_dataset = Vec::new();

        for vec in clear_data{
            for sample in vec{
                let u16_sample = f16::from_f32(*sample).to_bits();
                let enc_sample = FheUint16::try_encrypt(u16_sample, &self.inner.context.client_key).unwrap();
                encrypted_dataset.push(enc_sample);
            }
        }
        let num_channels = 1; 
        EncryptedTensor { data: encrypted_dataset, shape: vec![input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3]] }
    }

}

/* 
impl EncryptedNeuralNetwork for EncryptedNeuralNetworkU32GPU {
    fn create() -> Self{
        let config =
        ConfigBuilder::with_custom_parameters(V1_2_PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let server_key = compressed_server_key.decompress_to_gpu();
        rayon::broadcast(|_| set_server_key(server_key.clone()));

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

    fn add_max_pooling(&mut self, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize){
        self.inner.add_max_pooling(input_dim, kernel_size, stride, padding);
    }

    fn train(&mut self, epochs: usize, batch_size: usize, learning_rate: f32, train_inputs: &[Vec<f32>], train_labels: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) {
        let enc_learning_rate = FheUint32::try_encrypt(learning_rate.to_bits(), &self.inner.context.client_key).unwrap();
        let enc_train_inputs = self.encrypt_dataset(train_inputs, input_shapes.clone());
        let enc_train_labels = self.encrypt_dataset(train_labels, label_shapes.clone());
        let time = Instant::now();
        self.inner.train(epochs, batch_size, enc_learning_rate.clone(), enc_train_inputs.clone(), enc_train_labels.clone());
        println!("Training completed in {:?}", time.elapsed());
    }

    fn inference(&mut self, input: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) -> Vec<f32>{
        let inf_input = self.encrypt_dataset(input, input_shapes.clone());
        let prediction = self.inner.inference(&inf_input.clone());

        let mut prediction_f32: Vec<f32> = self.decrypt_tensor(&prediction);
        //println!("Prediction: {:?}", prediction_f32);

        let predicted_index = prediction_f32
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

        // 2. Create one-hot encoded vector
        let mut one_hot: Vec<f32> = vec![0.0; prediction_f32.len()];
        one_hot[predicted_index] = 1.0;
        
        one_hot
    }


    fn print_plain_weights(&self, id: String) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let weights = layer.get_weights();
                let shape = &weights.shape;
    
                if shape.len() != 4 {
                    println!("Expected 4D shape for weights, got {:?}", shape);
                    return;
                }
                
                let rows = shape[2];
                let cols = shape[3];
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

                let columns = shape[3];
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
    
                if shape.len() != 4 {
                    println!("Expected 4D shape for grad_weights, got {:?}", shape);
                    return;
                }
    
                let rows = shape[2];
                let cols = shape[3];
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

                let columns = shape[3];
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

    fn decrypt_tensor(&self, encrypted_tensor: &EncryptedTensor<FheUint32>) -> Vec<f32> {
    encrypted_tensor.data.iter()
        .map(|value| f32::from_bits(EncryptableValueType::decrypt(value, &self.inner.context.client_key)))
        .collect()
    }

    fn init_weights(&mut self, input_size: usize, output_size: usize) -> EncryptedTensor<FheUint32>{
        let mut plain_weights: Vec<u32> = vec![];
        if output_size == 16 {
            let fc1_weight: Vec<Vec<u32>> = vec![
            vec![ 0.1478_f32.to_bits(),  0.2460_f32.to_bits(), (-0.1902_f32).to_bits(),  0.2048_f32.to_bits(), (-0.0094_f32).to_bits(), (-0.1631_f32).to_bits(), (-0.0475_f32).to_bits(), (-0.0258_f32).to_bits(),
                0.0352_f32.to_bits(),  0.0608_f32.to_bits(),  0.2474_f32.to_bits(),  0.2253_f32.to_bits(),  0.1077_f32.to_bits(),  0.0669_f32.to_bits(), (-0.0140_f32).to_bits(),  0.1992_f32.to_bits()],
            vec![(-0.0941_f32).to_bits(), (-0.1871_f32).to_bits(), (-0.1587_f32).to_bits(),  0.1157_f32.to_bits(), (-0.2498_f32).to_bits(), (-0.1199_f32).to_bits(),  0.1795_f32.to_bits(), (-0.1308_f32).to_bits(),
                0.1570_f32.to_bits(), (-0.1353_f32).to_bits(),  0.1298_f32.to_bits(), (-0.2283_f32).to_bits(),  0.1142_f32.to_bits(),  0.0593_f32.to_bits(),  0.0358_f32.to_bits(), (-0.0192_f32).to_bits()],
            vec![(-0.2092_f32).to_bits(),  0.2381_f32.to_bits(), (-0.1979_f32).to_bits(), (-0.2127_f32).to_bits(), (-0.1407_f32).to_bits(),  0.1909_f32.to_bits(),  0.0236_f32.to_bits(),  0.0435_f32.to_bits(),
                0.1414_f32.to_bits(),  0.0379_f32.to_bits(), (-0.2027_f32).to_bits(),  0.1000_f32.to_bits(),  0.1482_f32.to_bits(), (-0.1996_f32).to_bits(),  0.1349_f32.to_bits(), (-0.0602_f32).to_bits()],
            vec![(-0.0248_f32).to_bits(), (-0.0221_f32).to_bits(), (-0.0468_f32).to_bits(),  0.0468_f32.to_bits(),  0.0116_f32.to_bits(),  0.0588_f32.to_bits(), (-0.2422_f32).to_bits(),  0.0707_f32.to_bits(),
                0.1853_f32.to_bits(), (-0.0841_f32).to_bits(),  0.1562_f32.to_bits(), (-0.0972_f32).to_bits(), (-0.1543_f32).to_bits(), (-0.0157_f32).to_bits(),  0.1084_f32.to_bits(), (-0.2480_f32).to_bits()],
            ];
            plain_weights = fc1_weight.into_iter().flatten().collect();
        }
        if output_size == 4 {
            let fc2_weight: Vec<Vec<u32>> = vec![
            vec![(-0.3546_f32).to_bits(),  0.2355_f32.to_bits(), (-0.2220_f32).to_bits(), (-0.0288_f32).to_bits()],
            vec![(-0.2830_f32).to_bits(), (-0.4757_f32).to_bits(),  0.1246_f32.to_bits(),  0.0483_f32.to_bits()],
            ];
            plain_weights = fc2_weight.into_iter().flatten().collect();
        }
        if output_size == 2 {
            let fc3_weight: Vec<Vec<u32>> = vec![
                vec![(-0.6488_f32).to_bits(),  0.2701_f32.to_bits()],
                vec![ 0.1953_f32.to_bits(),  0.1416_f32.to_bits()],
                vec![(-0.1549_f32).to_bits(), (-0.6687_f32).to_bits()],
            ];
            plain_weights = fc3_weight.into_iter().flatten().collect();
        }
        
        /* 
        // Xavier Initialization
        let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
        let normal = Normal::new(0.0, std_dev).unwrap();

        let mut rng = thread_rng();
        */
        
        let mut encrypted_weights = Vec::with_capacity(input_size * output_size);

        for i in 0..(input_size * output_size) {
            //let sample = normal.sample(&mut rng) as f32;
            //let sample = 0.0 as f32; 
            let sample = plain_weights[i];
            let encrypted_sample = FheUint32::try_encrypt(sample, &self.inner.context.client_key).expect("Weight initialization failed");
            encrypted_weights.push(encrypted_sample);
        }
        
        EncryptedTensor { data: (encrypted_weights), shape: (vec![1, 1, input_size, output_size]) }
    }

    fn init_biases(&mut self, output_size: usize) -> EncryptedTensor<FheUint32> {
        let mut plain_biases: Vec<u32> = vec![];
        if output_size == 4 {
            let fc1_bias: Vec<u32> = vec![
                (-0.0879_f32).to_bits(), 0.1680_f32.to_bits(), (-0.1631_f32).to_bits(), (-0.0271_f32).to_bits()
            ];
            plain_biases = fc1_bias.into_iter().map(|b| b).collect();
        }
        if output_size == 2 {
            let fc2_bias: Vec<u32> = vec![
                0.4002_f32.to_bits(), (-0.0112_f32).to_bits()
            ];
            plain_biases = fc2_bias.into_iter().map(|b| b).collect();
        }
        if output_size == 3 {
            let fc3_bias: Vec<u32> = vec![
                (-0.1854_f32).to_bits(), (-0.2199_f32).to_bits(), (-0.6619_f32).to_bits()
            ];
            plain_biases = fc3_bias.into_iter().map(|b|b).collect();
        }

        let mut encrypted_biases = Vec::with_capacity(output_size);
        for bias in plain_biases {
            let encrypted_bias = FheUint32::try_encrypt(bias, &self.inner.context.client_key)
                .expect("Bias encryption failed");
            encrypted_biases.push(encrypted_bias);
        }

        EncryptedTensor::new(encrypted_biases, vec![1, 1, 1, output_size])
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

    fn encrypt_dataset(&mut self, clear_data: &[Vec<f32>], input_shapes: Vec<usize>) -> EncryptedTensor<FheUint32> {
        let mut encrypted_dataset = Vec::new();

        for vec in clear_data{
            for sample in vec{
                let u16_sample = (*sample).to_bits();
                let enc_sample = FheUint32::try_encrypt(u16_sample, &self.inner.context.client_key).unwrap();
                encrypted_dataset.push(enc_sample);
            }
        }

        let num_channels = 1; 

        EncryptedTensor { data: encrypted_dataset, shape: vec![input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3]] }

    }

}
*/
