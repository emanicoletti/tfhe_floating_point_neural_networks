use crate::tfhe_nn_builder::encrypted_utils::tensor::EncryptedTensor;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::EncryptedContext;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_types::EncryptableValueType;
use crate::tfhe_nn_builder::encrypted_layers::EncryptedLayer;
use crate::tfhe_nn_builder::encrypted_losses::loss_function::LossFunction;
use crate::tfhe_nn_builder::encrypted_losses::loss_function::MseLoss;
use crate::tfhe_nn_builder::generic_enc_nn::EncryptedNeuralNetworkImpl;
use crate::experiment_1_2::initializations::layer_initializations::*;
use crate::plain_nn_builder::plain_utils::array2_to_vecvec;

use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;
use rand::SeedableRng;
use rand_distr::Distribution;
use half::f16;
use std::time::Instant;
use tfhe::prelude::FheTryEncrypt;
use tfhe::shortint::parameters::v1_2::*;
use tfhe::{set_server_key, ClientKey, CompressedServerKey, ConfigBuilder, CudaServerKey, FheUint16, FheUint32};


pub trait EncryptedNeuralNetwork{

    fn create(
        experiment: Option<i8>
    ) -> Self;

    fn add_dense(
        &mut self, 
        input_size: usize, 
        output_size: usize
    );

    fn add_tanh_activation(
        &mut self, 
        size: usize
    );

    fn add_relu_activation(
        &mut self, size: usize
    );

    fn add_max_pooling(
        &mut self, 
        input_dim: Vec<usize>, 
        kernel_size: usize, 
        stride: usize, 
    );

    fn add_conv(
        &mut self, 
        in_channels: usize, 
        out_channels: usize, 
        kernel_width: usize, 
        kernel_height: usize, 
        stride: usize, 
        padding: usize
    );

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

    #[allow(dead_code)]
    fn inference(
        &mut self, 
        input: &[Vec<f32>], 
        input_shapes: Vec<usize>, 
        label_shapes: Vec<usize>
    ) -> Vec<f32>;

    fn print_plain_weights(
        &self, 
        id: String
    );

    fn print_plain_biases(
        &self, 
        id:String
    );

    #[allow(dead_code)]
    fn print_plain_grad_weights(
        &self, 
        id: String
    );

    #[allow(dead_code)]
    fn print_plain_grad_biases(
        &self, 
        id:String
    );
}


pub struct EncryptedNeuralNetworkU16GPU {
    pub inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint16>,
    pub experiment: Option<i8>,
}

pub struct EncryptedNeuralNetworkU32GPU {
    inner: EncryptedNeuralNetworkImpl<CudaServerKey, FheUint32>,
    experiment: Option<i8>,
}

// Format: (min_input, max_input, a, b, derivative)
pub static TANH16_PLA_RANGES: &[(u16, u16, u16, u16, u16)] = &[
    (49152u16, 65535u16, 0u16, 48128u16, 0u16), // [-inf, -2], output ~ -1, derivative ≈ 0
    (47787u16, 49151u16, 13312u16, 47104u16, 13312u16), // [-2, -0.8333] slope 0.25. intercept -0.5
    (32768u16, 47786u16, 15053u16, 0u16, 15053u16), // [-0.833, -0] slope=0.85 intercept = 0
    (0u16, 15019u16, 15053u16, 0u16, 15053u16), //[0, 0.833] slope=0.85 intercept = 0
    (15020u16, 16384u16, 13312u16, 14336u16, 13312u16), //[0.833, 2] slope = 0.25 intercept 0.5
    (16385u16, 32767u16, 0u16, 15360u16, 0u16), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

// Format: (min_input, max_input, a, b, derivative)
pub static TANH32_PLA_RANGES: &[(u32, u32, u32, u32, u32)] = &[
    (3221225472u32, 4294967295u32, 3212836864u32, 0u32, 0u32), // [-inf, -2], output ~ -1, derivative ≈ 0
    (3210040661u32, 3221225471u32, 1048576000u32, 3204448256u32, 1048576000u32), // [-2, -0.8333] slope 0.25. intercept -0.5
    (2147483648u32, 3210040660u32, 1062836634u32, 0u32, 1062836634u32), // [-0.833, -0] slope=0.85 intercept = 0
    (0u32, 1062557013u32, 1062836634u32, 0u32, 1062836634u32), //[0, 0.833] slope=0.85 intercept = 0
    (1062557014u32, 1073741824u32, 1048576000u32, 1056964608u32, 1048576000u32), //[0.833, 2] slope = 0.25 intercept 0.5
    (1073741825u32, 2147483647u32, 1065353217u32, 0u32, 0u32), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

impl EncryptedNeuralNetwork for EncryptedNeuralNetworkU16GPU {
    fn create(
        experiment: Option<i8>
    ) -> Self {
        let config =
        ConfigBuilder::with_custom_parameters(V1_2_PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let server_key = compressed_server_key.decompress_to_gpu();

        rayon::broadcast(|_| set_server_key(server_key.clone()));

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

        let loss: Box<dyn LossFunction<CudaServerKey, FheUint16>> = Box::new(MseLoss{});

        let inner = EncryptedNeuralNetworkImpl {
            layers,
            loss,
            context,
        };

        EncryptedNeuralNetworkU16GPU { inner, experiment }
    }

    fn add_dense(
        &mut self, 
        input_size: usize, 
        output_size: usize
    ) {
        let encrypted_weights = self.init_weights(output_size, input_size, 1, 1, self.experiment);
        let encrypted_biases = self.init_biases(output_size, self.experiment);
        let encrypted_grad_weights = self.init_gradients(&[output_size, input_size]);
        let encrypted_grad_biases = self.init_gradients(&[output_size]);
        self.inner.add_dense(encrypted_weights, encrypted_biases, encrypted_grad_weights, encrypted_grad_biases);
    }

    fn add_conv(
        &mut self, 
        in_channels: usize, 
        out_channels: usize, 
        kernel_width: usize, 
        kernel_height: usize, 
        stride: usize, 
        padding: usize
    ) {
        let encrypted_weights = self.init_weights(kernel_width, kernel_height, in_channels, out_channels, self.experiment);
        let encrypted_biases = self.init_biases(out_channels, self.experiment);
        let encrypted_grad_weights = self.init_gradients(&[out_channels, in_channels * kernel_width * kernel_height]);
        let encrypted_grad_biases = self.init_gradients(&[out_channels]);
        self.inner.add_conv(encrypted_weights, encrypted_biases, encrypted_grad_weights, encrypted_grad_biases, stride, padding);
    }

    fn add_tanh_activation(
        &mut self, 
        size: usize
    ) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_tanh_activation(derivatives, self.inner.context.ranges.clone());
    }

    fn add_relu_activation(
        &mut self, 
        size: usize
    ) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_relu_activation(derivatives);
    }

    fn add_max_pooling(
        &mut self, 
        input_dim: Vec<usize>, 
        kernel_size: usize, 
        stride: usize, 
    ) {
        self.inner.add_max_pooling(input_dim, kernel_size, stride);
    }

    fn train(
        &mut self, 
        epochs: usize, 
        batch_size: usize, 
        learning_rate: f32, 
        train_inputs: &[Vec<f32>], 
        train_labels: &[Vec<f32>],
        input_shapes: Vec<usize>, 
        label_shapes: Vec<usize>
    ) {
        let enc_learning_rate = FheUint16::try_encrypt(f16::from_f32(learning_rate).to_bits(), &self.inner.context.client_key).unwrap();
        let enc_train_inputs = self.encrypt_dataset(train_inputs, input_shapes.clone());
        let enc_train_labels = self.encrypt_dataset(train_labels, label_shapes.clone());
        let time = Instant::now();
        self.inner.train(epochs, batch_size, enc_learning_rate.clone(), enc_train_inputs.clone(), enc_train_labels.clone());
        println!("Training completed in {:?}", time.elapsed());
    }

    fn inference(
        &mut self, 
        input: &[Vec<f32>], 
        input_shapes: Vec<usize>, 
        _label_shapes: Vec<usize>
    ) -> Vec<f32>{
        let inf_input = self.encrypt_dataset(input, input_shapes.clone());
        let prediction = self.inner.inference(&inf_input.clone());

        let prediction_f32: Vec<f32> = self.decrypt_tensor(&prediction);

        let predicted_index = prediction_f32
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

        let mut one_hot: Vec<f32> = vec![0.0; prediction_f32.len()];
        one_hot[predicted_index] = 1.0;
        
        one_hot
    }

    fn print_plain_weights(
        &self, 
        id: String
    ) {
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

    fn print_plain_biases(
        &self, 
        id: String
    ) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_biases();
                let shape = &biases.shape;

                let columns = shape[3];
                let flat = biases.data;

                println!("\nDecrypted Biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    let decrypted: u16 = EncryptableValueType::decrypt(&flat[i], &self.inner.context.client_key);
                    print!("{:<6} ", f16::from_bits(decrypted).to_f32());
                }
                print!("]\n");
                return;
            }
        }
    }

    fn print_plain_grad_weights(
        &self, 
        id: String
    ) {
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

    fn print_plain_grad_biases(
        &self, 
        id: String
    ) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_grad_biases();
                let shape = &biases.shape;

                let columns = shape[3];
                let flat = biases.data;

                println!("\nDecrypted grad_biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
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

    fn decrypt_tensor(
        &self, 
        encrypted_tensor: &EncryptedTensor<FheUint16>
    ) -> Vec<f32> {
        encrypted_tensor.data.iter()
            .map(|value| f16::from_bits(EncryptableValueType::decrypt(value, &self.inner.context.client_key)).to_f32())
            .collect()
    }


    fn init_weights(
        &mut self, 
        input_size: usize, 
        output_size: usize, 
        in_channels: usize, 
        out_channels: usize, 
        experiment: Option<i8>
    ) -> EncryptedTensor<FheUint16>{
        let plain_weights: Vec<u16>;
        if experiment == Some(1) {
            // Initialize weights for experiment 1
            if output_size == 16 {
                let fc1_weights = EXP1_W_FC1_32.to_vec();
                let flattened_fc1_weight: Vec<u16> = fc1_weights
                    .into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                plain_weights = flattened_fc1_weight;
            }
            else if output_size == 4 {
                let fc2_weights = EXP1_W_FC2_32.to_vec();
                let flattened_fc2_weight: Vec<u16> = fc2_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                plain_weights = flattened_fc2_weight;
            }
            else if output_size == 2 {
                let fc3_weights = EXP1_W_FC3_32.to_vec();
                let flattened_fc3_weight: Vec<u16> = fc3_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                plain_weights = flattened_fc3_weight;
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            // Initialize weights for experiment 2
            if output_size == 32 {
                let fc_weights = EXP2_W_FC_32.to_vec();
                let flattened_fc_weights: Vec<u16> = fc_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                plain_weights = flattened_fc_weights;
            }
            else if output_size == 2 {
                let conv_weights = EXP2_W_CONV_32.to_vec();
                let flattened_conv_weights: Vec<u16> = conv_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                plain_weights = flattened_conv_weights;
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
            // Initialize weights for experiment 1
            let weights_file: Array2<f32> = if output_size == 6272 && input_size == 256 {
                read_npy(Path::new("src/experiment_3/initializations/fc1_weight.npy"))
                    .expect("Failed to read fc1 weights")
            } else if output_size == 256 && input_size == 10 {
                read_npy(Path::new("src/experiment_3/initializations/fc2_weight.npy"))
                    .expect("Failed to read fc2 weights")
            } else if in_channels == 1 && out_channels == 32 {
                read_npy(Path::new("src/experiment_3/initializations/conv1_weight.npy"))
                    .expect("Failed to read conv1 weights")
            } else if in_channels == 32 && out_channels == 64 {
                read_npy(Path::new("src/experiment_3/initializations/conv2_weight.npy"))
                    .expect("Failed to read conv2 weights")
            } else if in_channels == 64 && out_channels == 128 {
                read_npy(Path::new("src/experiment_3/initializations/conv3_weight.npy"))
                    .expect("Failed to read conv3 weights")
            } else {
                panic!("No matching weight file for given layer dimensions {:?}, {:?}, {:?}, {:?}", in_channels, out_channels, input_size, output_size);
            };

            let vec_vec_weights = array2_to_vecvec(&weights_file);
            let weights: Vec<u16> = vec_vec_weights
                .into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f).to_bits())
                    .collect();

            plain_weights = weights;
        }
        else {
            // Xavier initialization for other cases
            let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
            let normal = Normal::new(0.0, std_dev).unwrap();
        
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let mut weights = Vec::with_capacity(in_channels * out_channels * input_size * output_size);

            for _ in 0..(in_channels * out_channels * input_size * output_size) {
                let sample = normal.sample(&mut rng) as f32;
                let u_sample = f16::from_f32(sample).to_bits();
                weights.push(u_sample);
            }
            
            plain_weights = weights;
        }
        let mut encrypted_weights = Vec::with_capacity(input_size * output_size * in_channels * out_channels);

        for i in 0..(input_size * output_size * in_channels * out_channels) {
            let sample = plain_weights[i];
            let encrypted_sample = FheUint16::try_encrypt(sample, &self.inner.context.client_key).expect("Weight initialization failed");
            encrypted_weights.push(encrypted_sample);
        }
        
        EncryptedTensor { data: (encrypted_weights), shape: (vec![out_channels, in_channels, input_size, output_size]) }
        
    }

    fn init_biases(
        &mut self, 
        output_size: usize, 
        experiment: Option<i8>
    ) -> EncryptedTensor<FheUint16> {
        let plain_biases: Vec<u16>;
         if experiment == Some(1) {
            // Initialize weights for experiment 1
            if output_size == 4 {
                let biases_u32 = EXP1_B_FC1_32.to_vec();
                plain_biases = biases_u32.iter().map(|&x| f16::from_f32(f32::from_bits(x)).to_bits()).collect();
            }
            else if output_size == 2 {
                let biases_u32 = EXP1_B_FC2_32.to_vec();
                plain_biases = biases_u32.iter().map(|&x| f16::from_f32(f32::from_bits(x)).to_bits()).collect();
            }
            else if output_size == 3 {
                let biases_u32 = EXP1_B_FC3_32.to_vec();
                plain_biases = biases_u32.iter().map(|&x| f16::from_f32(f32::from_bits(x)).to_bits()).collect();
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            // Initialize weights for experiment 2
            if output_size == 3 {
                let biases_u32 = EXP2_B_FC_32.to_vec();
                plain_biases = biases_u32.iter().map(|&x| f16::from_f32(f32::from_bits(x)).to_bits()).collect();
            }
            else if output_size == 2 {
                let biases_u32 = EXP2_B_CONV_32.to_vec();
                plain_biases = biases_u32.iter().map(|&x| f16::from_f32(f32::from_bits(x)).to_bits()).collect();
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
            // Initialize weights for experiment 3
            let biases_file: Array2<f32> = if output_size == 256 {
            read_npy(Path::new("src/experiment_3/initializations/fc1_bias.npy"))
                    .expect("Failed to read fc1 biases")
            } else if output_size == 10 {
                read_npy(Path::new("src/experiment_3/initializations/fc2_bias.npy"))
                    .expect("Failed to read fc2 biases")
            } else if output_size == 32 {
                read_npy(Path::new("src/experiment_3/initializations/conv1_bias.npy"))
                    .expect("Failed to read conv1 biases")
            } else if output_size == 64 {
                read_npy(Path::new("src/experiment_3/initializations/conv2_bias.npy"))
                    .expect("Failed to read conv2 biases")
            } else if output_size == 128 {
                read_npy(Path::new("src/experiment_3/initializations/conv3_bias.npy"))
                    .expect("Failed to read conv3 biases")
            } else {
                panic!("No matching weight file for given layer dimensions");
            };

            let vec_vec_biases = array2_to_vecvec(&biases_file);
            let biases: Vec<u16> = vec_vec_biases
                .into_iter()
                .flatten()
                .map(|f| f16::from_f32(f).to_bits())
                .collect();
            plain_biases = biases;
        } 
        else {
            // Zero initialization for other cases
            plain_biases = vec![0u16; output_size];
        }
        let mut encrypted_biases = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let sample = plain_biases[i];
            let encrypted_sample = FheUint16::try_encrypt(sample, &self.inner.context.client_key).expect("Weight initialization failed");
            encrypted_biases.push(encrypted_sample);
        }

        return EncryptedTensor::new(encrypted_biases, vec![1, 1, 1, output_size]);
    }

    fn init_gradients(
        &self, 
        shape: &[usize]
    ) -> EncryptedTensor<FheUint16> {
        let zero_enc = &self.inner.context.encrypted_zero;
        let size = shape.iter().product();
        let zeros = vec![zero_enc.clone(); size];
        let shapes: Vec<usize>;
        if shape.to_vec().len() == 1 {
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        EncryptedTensor::new(zeros, shapes)
    }

     fn init_derivatives(
        &self, 
        shape: &[usize]
    ) -> EncryptedTensor<FheUint16> {
        let size = shape.iter().product();
        let zeros = vec![self.inner.context.encrypted_zero.clone(); size];
        let shapes: Vec<usize>;
        if shape.to_vec().len() == 1{
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        EncryptedTensor::new(zeros, shapes)
    }

    fn encrypt_dataset(
        &mut self, 
        clear_data: &[Vec<f32>], 
        input_shapes: Vec<usize>
    ) -> EncryptedTensor<FheUint16> {
        let mut encrypted_dataset = Vec::new();
        for vec in clear_data{
            for sample in vec{
                let u16_sample = f16::from_f32(*sample).to_bits();
                let enc_sample = FheUint16::try_encrypt(u16_sample, &self.inner.context.client_key).unwrap();
                encrypted_dataset.push(enc_sample);
            }
        }
        EncryptedTensor { data: encrypted_dataset, shape: vec![input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3]] }
    }

}

impl EncryptedNeuralNetwork for EncryptedNeuralNetworkU32GPU {
    
    fn create(
        experiment: Option<i8>
    ) -> Self{
        let config =
        ConfigBuilder::with_custom_parameters(V1_2_PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let server_key = compressed_server_key.decompress_to_gpu();
        rayon::broadcast(|_| set_server_key(server_key.clone()));

        let encrypted_zero = FheUint32::try_encrypt(0u32, &client_key).unwrap();
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

        let loss: Box<dyn LossFunction<CudaServerKey, FheUint32>> = Box::new(MseLoss{});

        let inner = EncryptedNeuralNetworkImpl {
            layers,
            loss,
            context,
        };

        EncryptedNeuralNetworkU32GPU { inner, experiment }
    }

    fn add_dense(
        &mut self, 
        input_size: usize, 
        output_size: usize
    ) {
        let encrypted_weights = self.init_weights(output_size, input_size, 1, 1, self.experiment);
        let encrypted_biases = self.init_biases(output_size, self.experiment);
        let encrypted_grad_weights = self.init_gradients(&[output_size, input_size]);
        let encrypted_grad_biases = self.init_gradients(&[output_size]);
        self.inner.add_dense(encrypted_weights, encrypted_biases, encrypted_grad_weights, encrypted_grad_biases);
    }

    fn add_conv(
        &mut self, 
        in_channels: usize, 
        out_channels: usize, 
        kernel_width: usize, 
        kernel_height: usize, 
        stride: usize, 
        padding: usize
    ) {
        let encrypted_weights = self.init_weights(kernel_width, kernel_height, in_channels, out_channels, self.experiment);
        let encrypted_biases = self.init_biases(out_channels, self.experiment);
        let encrypted_grad_weights = self.init_gradients(&[out_channels, in_channels * kernel_width * kernel_height]);
        let encrypted_grad_biases = self.init_gradients(&[out_channels]);
        self.inner.add_conv(encrypted_weights, encrypted_biases, encrypted_grad_weights, encrypted_grad_biases, stride, padding);
    }

    fn add_tanh_activation(
        &mut self, 
        size: usize
    ) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_tanh_activation(derivatives, self.inner.context.ranges.clone());
    }

    fn add_relu_activation(
        &mut self, 
        size: usize
    ) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_relu_activation(derivatives);
    }

    fn add_max_pooling(
        &mut self, 
        input_dim: Vec<usize>, 
        kernel_size: usize, 
        stride: usize, 
    ){
        self.inner.add_max_pooling(input_dim, kernel_size, stride);
    }

    fn train(
        &mut self, 
        epochs: usize, 
        batch_size: usize, 
        learning_rate: f32, 
        train_inputs: &[Vec<f32>], 
        train_labels: &[Vec<f32>], 
        input_shapes: Vec<usize>, 
        label_shapes: Vec<usize>
    ) {
        let enc_learning_rate = FheUint32::try_encrypt(learning_rate.to_bits(), &self.inner.context.client_key).unwrap();
        let enc_train_inputs = self.encrypt_dataset(train_inputs, input_shapes.clone());
        let enc_train_labels = self.encrypt_dataset(train_labels, label_shapes.clone());
        let time = Instant::now();
        self.inner.train(epochs, batch_size, enc_learning_rate.clone(), enc_train_inputs.clone(), enc_train_labels.clone());
        println!("Training completed in {:?}", time.elapsed());
    }

    fn inference(
        &mut self, 
        input: &[Vec<f32>], 
        input_shapes: Vec<usize>, 
        _label_shapes: Vec<usize>
    ) -> Vec<f32>{
        let inf_input = self.encrypt_dataset(input, input_shapes.clone());
        let prediction = self.inner.inference(&inf_input.clone());

        let prediction_f32: Vec<f32> = self.decrypt_tensor(&prediction);

        let predicted_index = prediction_f32
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

        let mut one_hot: Vec<f32> = vec![0.0; prediction_f32.len()];
        one_hot[predicted_index] = 1.0;
        
        one_hot
    }


    fn print_plain_weights(
        &self, 
        id: String
    ) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let weights = layer.get_weights();
                let shape = &weights.shape;
    
                if shape.len() != 4 {
                    println!("Expected 4D shape for weights, got {:?}", shape);
                    return;
                }
                
                let in_channels = shape[0];
                let out_channels = shape[1];
                let rows = shape[2];
                let cols = shape[3];
                let flat = weights.data;
    
                if flat.len() != rows * cols {
                    println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                    return;
                }
    
                println!("\nDecrypted Weights for Layer \"{}\":", id);
                for c in 0..out_channels {
                    for r in 0..in_channels {
                        for i in 0..rows {
                            print!("\n[");
                            for j in 0..cols {
                                let index = c * in_channels * rows * cols + r * rows * cols + i * cols + j;
                                let decrypted: u32 = EncryptableValueType::decrypt(&flat[index], &self.inner.context.client_key);
                                print!("{:<6} ", f32::from_bits(decrypted));
                            }
                            print!("]\n");
                        }
                        print!("\n");
                    }
                }
                return;
            }
        }
    
        println!("No layer found with id: {}", id);
    }

    fn print_plain_biases(
        &self, 
        id: String
    ) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_biases();
                let shape = &biases.shape;

                let columns = shape[3];
                let flat = biases.data;

                println!("\nDecrypted Biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    let decrypted: u32 = EncryptableValueType::decrypt(&flat[i], &self.inner.context.client_key);
                    print!("{:<6} ", f32::from_bits(decrypted));
                }
                print!("]\n");
                return;
            }
        }
    }

    fn print_plain_grad_weights(
        &self, 
        id: String
    ) {
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

    fn print_plain_grad_biases(
        &self, 
        id: String
    ) {
        for layer in &self.inner.layers {
            if layer.get_id() == id {
                let biases = layer.get_grad_biases();
                let shape = &biases.shape;

                let columns = shape[3];
                let flat = biases.data;

                println!("\nDecrypted grad_biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
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

    fn decrypt_tensor(
        &self, 
        encrypted_tensor: &EncryptedTensor<FheUint32>
    ) -> Vec<f32> {
        encrypted_tensor.data.iter()
            .map(|value| f32::from_bits(EncryptableValueType::decrypt(value, &self.inner.context.client_key)))
            .collect()
    }

    fn init_weights(
        &mut self, 
        input_size: usize, 
        output_size: usize, 
        in_channels: usize, 
        out_channels: usize, 
        experiment: Option<i8>
    ) -> EncryptedTensor<FheUint32>{
        let plain_weights: Vec<u32>;
        if experiment == Some(1) {
            // Initialize weights for experiment 1
            if output_size == 16 {
                let fc1_weights = EXP1_W_FC1_32.to_vec();
                let flattened_fc1_weight: Vec<u32> = fc1_weights.into_iter().flatten().collect();
                plain_weights = flattened_fc1_weight;
            }
            else if output_size == 4 {
                let fc2_weights = EXP1_W_FC2_32.to_vec();
                let flattened_fc2_weight: Vec<u32> = fc2_weights.into_iter().flatten().collect();
                plain_weights = flattened_fc2_weight;
            }
            else if output_size == 2 {
                let fc3_weights = EXP1_W_FC3_32.to_vec();
                let flattened_fc3_weight: Vec<u32> = fc3_weights.into_iter().flatten().collect();
                plain_weights = flattened_fc3_weight;
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            // Initialize weights for experiment 2
            if output_size == 32 {
                let fc_weights = EXP2_W_FC_32.to_vec();
                let flattened_fc_weights: Vec<u32> = fc_weights.into_iter().flatten().collect();
                plain_weights = flattened_fc_weights;
            }
            else if output_size == 2 {
                let conv_weights = EXP2_W_CONV_32.to_vec();
                let flattened_conv_weights: Vec<u32> = conv_weights.into_iter().flatten().collect();
                plain_weights = flattened_conv_weights;
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
            // Initialize weights for experiment 3
            let weights_file: Array2<f32> = if output_size == 6272 && input_size == 256 {
                read_npy(Path::new("src/experiment_3/initializations/fc1_weight.npy"))
                    .expect("Failed to read fc1 weights")
            } else if output_size == 256 && input_size == 10 {
                read_npy(Path::new("src/experiment_3/initializations/fc2_weight.npy"))
                    .expect("Failed to read fc2 weights")
            } else if in_channels == 1 && out_channels == 32 {
                read_npy(Path::new("src/experiment_3/initializations/conv1_weight.npy"))
                    .expect("Failed to read conv1 weights")
            } else if in_channels == 32 && out_channels == 64 {
                read_npy(Path::new("src/experiment_3/initializations/conv2_weight.npy"))
                    .expect("Failed to read conv2 weights")
            } else if in_channels == 64 && out_channels == 128 {
                read_npy(Path::new("src/experiment_3/initializations/conv3_weight.npy"))
                    .expect("Failed to read conv3 weights")
            } else {
                panic!("No matching weight file for given layer dimensions {:?}, {:?}, {:?}, {:?}", in_channels, out_channels, input_size, output_size);
            };

            let vec_vec_weights = array2_to_vecvec(&weights_file);
            let weights: Vec<u32> = vec_vec_weights
                .into_iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect();

            plain_weights = weights;
        }
        else {
            // Xavier initialization for other cases
            let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
            let normal = Normal::new(0.0, std_dev).unwrap();
        
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let mut weights = Vec::with_capacity(in_channels * out_channels * input_size * output_size);

            for _ in 0..(in_channels * out_channels * input_size * output_size) {
                let sample = normal.sample(&mut rng) as f32;
                let u_sample = sample.to_bits();
                weights.push(u_sample);
            }
            
            plain_weights = weights;
        }
        let mut encrypted_weights = Vec::with_capacity(input_size * output_size * in_channels * out_channels);

        for i in 0..(input_size * output_size * in_channels * out_channels) {
            let sample = plain_weights[i];
            let encrypted_sample = FheUint32::try_encrypt(sample, &self.inner.context.client_key).expect("Weight initialization failed");
            encrypted_weights.push(encrypted_sample);
        }
        
        EncryptedTensor { data: (encrypted_weights), shape: (vec![out_channels, in_channels, input_size, output_size]) }
    }

    fn init_biases(
        &mut self, 
        output_size: usize, 
        experiment: Option<i8>
    ) -> EncryptedTensor<FheUint32> {
        let plain_biases: Vec<u32>;
         if experiment == Some(1) {
            // Initialize weights for experiment 1
            if output_size == 4 {
                plain_biases = EXP1_B_FC1_32.to_vec();
            }
            else if output_size == 2 {
                plain_biases = EXP1_B_FC2_32.to_vec();
            }
            else if output_size == 3 {
                plain_biases = EXP1_B_FC3_32.to_vec();
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            // Initialize weights for experiment 2
            if output_size == 3 {
                plain_biases = EXP2_B_FC_32.to_vec();
            }
            else if output_size == 2 {
                plain_biases = EXP2_B_CONV_32.to_vec();
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
            // Initialize weights for experiment 3
            let biases_file: Array2<f32> = if output_size == 256 {
            read_npy(Path::new("src/experiment_3/initializations/fc1_bias.npy"))
                    .expect("Failed to read fc1 biases")
            } else if output_size == 10 {
                read_npy(Path::new("src/experiment_3/initializations/fc2_bias.npy"))
                    .expect("Failed to read fc2 biases")
            } else if output_size == 32 {
                read_npy(Path::new("src/experiment_3/initializations/conv1_bias.npy"))
                    .expect("Failed to read conv1 biases")
            } else if output_size == 64 {
                read_npy(Path::new("src/experiment_3/initializations/conv2_bias.npy"))
                    .expect("Failed to read conv2 biases")
            } else if output_size == 128 {
                read_npy(Path::new("src/experiment_3/initializations/conv3_bias.npy"))
                    .expect("Failed to read conv3 biases")
            } else {
                panic!("No matching weight file for given layer dimensions");
            };

            let vec_vec_biases = array2_to_vecvec(&biases_file);
            let biases: Vec<u32> = vec_vec_biases
                .into_iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect();
            plain_biases = biases;
        } 
        else {
            // Zero initialization for other cases
            plain_biases = vec![0u32; output_size];
        }
        let mut encrypted_biases = Vec::with_capacity(output_size);

        for i in 0..output_size {
            let sample = plain_biases[i];
            let encrypted_sample = FheUint32::try_encrypt(sample, &self.inner.context.client_key).expect("Weight initialization failed");
            encrypted_biases.push(encrypted_sample);
        }

        return EncryptedTensor::new(encrypted_biases, vec![1, 1, 1, output_size]);
    }

    fn init_gradients(
        &self, 
        shape: &[usize]
    ) -> EncryptedTensor<FheUint32> {
        let zero_enc = &self.inner.context.encrypted_zero;
        let size = shape.iter().product();
        let zeros = vec![zero_enc.clone(); size];
        let shapes: Vec<usize>;
        if shape.to_vec().len() == 1{
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        EncryptedTensor::new(zeros, shapes)
    }

    fn init_derivatives(
        &self, 
        shape: &[usize]
    ) -> EncryptedTensor<FheUint32> {
        let size = shape.iter().product();
        let zeros = vec![self.inner.context.encrypted_zero.clone(); size];
        let shapes: Vec<usize>;
        if shape.to_vec().len() == 1{
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        EncryptedTensor::new(zeros, shapes)
    }

    fn encrypt_dataset(
        &mut self, 
        clear_data: &[Vec<f32>], 
        input_shapes: Vec<usize>
    ) -> EncryptedTensor<FheUint32> {
        let mut encrypted_dataset = Vec::new();

        for vec in clear_data{
            for sample in vec{
                let u16_sample = (*sample).to_bits();
                let enc_sample = FheUint32::try_encrypt(u16_sample, &self.inner.context.client_key).unwrap();
                encrypted_dataset.push(enc_sample);
            }
        }

        EncryptedTensor { data: encrypted_dataset, shape: vec![input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3]] }
    }

}

