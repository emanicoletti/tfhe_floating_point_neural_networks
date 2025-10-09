use crate::plain_nn_builder::generic_nn::*;
use crate::plain_nn_builder::plain_activations::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::*;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_losses::*;
use crate::experiment_1_2::initializations::layer_initializations::*;
use crate::plain_nn_builder::plain_utils::load_from_file::array2_to_vecvec;

// for seed_from_u64
use rand::SeedableRng;      // <- import SeedableRng trait
use rand_chacha::ChaCha8Rng;
use rand_distr::{Normal, Distribution};
use core::panic;
use std::time::Instant;
use half::f16;

use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;

pub trait PlainNeuralNetwork {
    fn create(experiment: Option<i8>) -> Self;
    fn add_dense(&mut self, input_size: usize, output_size: usize);
    fn add_tanh_activation(&mut self, size: usize);
    fn add_relu_activation(&mut self, size: usize);
    fn add_max_pooling(&mut self, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize);
    fn add_conv(&mut self, in_channels: usize, out_channels: usize, kernel_width: usize, kernel_height: usize, stride: usize, padding: usize);
    fn add_batch_norm(&mut self, size:usize);
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
        inputs: &[Vec<f32>], 
        input_shapes: Vec<usize>,
        label_shapes: Vec<usize>,
    ) -> Vec<f32>;
    fn train_and_validate(
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        train_inputs: &[Vec<f32>],
        train_labels: &[Vec<f32>],
        val_inputs: &[Vec<f32>],
        val_labels: &[Vec<f32>],
        input_shapes: Vec<usize>,
        label_shapes: Vec<usize>,
        val_input_shapes: Vec<usize>,
        val_label_shapes: Vec<usize>,
    );
    fn print_plain_weights(&self, id: String);
    fn print_plain_biases(&self, id:String);
    fn print_plain_grad_weights(&self, id: String);
    fn print_plain_grad_biases(&self, id:String);
}

pub struct PlainNeuralNetworkU32 {
    inner: PlainNeuralNetworkImpl<u32>,
    experiment: Option<i8>,
}

pub struct PlainNeuralNetworkU16 {
    inner: PlainNeuralNetworkImpl<u16>,
    experiment: Option<i8>,
}

pub static TANH32_PLA_RANGES: &[(u32, u32, u32, u32, u32)] = &[
    (3221225472u32, 4294967295u32, 3212836864u32, 0u32, 0u32), // [-inf, -2], output ~ -1, derivative ≈ 0
    (3210040661u32, 3221225471u32, 1048576000u32, 3204448256u32, 1048576000u32), // [-2, -0.8333] slope 0.25. intercept -0.5
    (2147483648u32, 3210040660u32, 1062836634u32, 0u32, 1062836634u32), // [-0.833, -0] slope=0.85 intercept = 0
    (0u32, 1062557013u32, 1062836634u32, 0u32, 1062836634u32), //[0, 0.833] slope=0.85 intercept = 0
    (1062557014u32, 1073741824u32, 1048576000u32, 1056964608u32, 1048576000u32), //[0.833, 2] slope = 0.25 intercept 0.5
    (1073741825u32, 2147483647u32, 1065353217u32, 0u32, 0u32), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

pub static TANH16_PLA_RANGES: &[(u16, u16, u16, u16, u16)] = &[
    (49152u16, 65535u16, 48128u16, 0u16, 0u16), // [-inf, -2], output ~ -1, derivative ≈ 0
    (47787u16, 49151u16, 13312u16, 47104u16, 13312u16), // [-2, -0.8333] slope 0.25. intercept -0.5
    (32768u16, 47786u16, 15053u16, 0u16, 15053u16), // [-0.833, -0] slope=0.85 intercept = 0
    (0u16, 15019u16, 15053u16, 0u16, 15053u16), //[0, 0.833] slope=0.85 intercept = 0
    (15020u16, 16384u16, 13312u16, 14336u16, 13312u16), //[0.833, 2] slope = 0.25 intercept 0.5
    (16385u16, 32767u16, 15360u16, 0u16, 0u16), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

impl PlainNeuralNetwork for PlainNeuralNetworkU32 {
    fn create(experiment: Option<i8>) -> Self{
        let layers: Vec<Box<dyn PlainLayer<u32>>> = vec![];

        // Step 4: Create the loss function
        let loss: Box<dyn PlainLossFunction<u32>> = Box::new(MseLoss{});

        // Step 5: Build the inner EncryptedNeuralNetworkImpl
        let inner = PlainNeuralNetworkImpl {
            layers,
            loss,
        };

        // Step 6: Wrap in the public struct
        PlainNeuralNetworkU32 { inner, experiment }
    }

    fn add_dense(&mut self, input_size: usize, output_size: usize) {
        let weights = self.init_weights(output_size, input_size, 1, 1, self.experiment);
        let biases = self.init_biases(output_size, self.experiment);
        let grad_weights = self.init_gradients(&[output_size, input_size]);
        let grad_biases = self.init_gradients(&[output_size]);
        self.inner.add_dense(weights, biases, grad_weights, grad_biases);
    }

    fn add_tanh_activation(&mut self, size: usize) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_tanh_activation(derivatives, TANH32_PLA_RANGES.to_vec());
    }

    fn add_relu_activation(&mut self, size: usize) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_relu_activation(derivatives);
    }
    
    fn add_max_pooling(&mut self, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize){
        self.inner.add_max_pooling(input_dim, kernel_size, stride, padding);
    }

    fn add_conv(&mut self,in_channels: usize, out_channels: usize, kernel_width: usize, kernel_height: usize, stride: usize, padding: usize) {
        let weights = self.init_weights(kernel_width, kernel_height, in_channels, out_channels, self.experiment);
        let biases = self.init_biases(out_channels, self.experiment);
        let grad_weights = self.init_gradients(&[out_channels, in_channels * kernel_width * kernel_height]);
        let grad_biases = self.init_gradients(&[out_channels]);
        self.inner.add_conv(weights, biases, grad_weights, grad_biases, stride, padding);
    }

    fn add_batch_norm(&mut self, size:usize) {
        let (x_hat, mean, variance, gamma, beta) = self.init_batch_norm(&[size]);
        self.inner.add_batch_norm(x_hat, mean, variance, gamma, beta);
    }

    fn train(&mut self, epochs: usize, batch_size: usize, learning_rate: f32, train_inputs: &[Vec<f32>], train_labels: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) {
        let u32_learning_rate = learning_rate.to_bits();
        let train_inputs = self.dataset(train_inputs, input_shapes.clone());
        let train_labels = self.dataset(train_labels, label_shapes.clone());
        self.inner.train(epochs, batch_size, u32_learning_rate.clone(), train_inputs.clone(), train_labels.clone());
    }

    fn inference(&mut self, input: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) -> Vec<f32>{
        let inf_input = self.dataset(input, input_shapes.clone());
        let prediction = self.inner.inference(&inf_input.clone());
        let mut prediction_f32: Vec<f32> = vec![0.0; label_shapes[3]];
        for i in 0..prediction.data.len() {
            prediction_f32[i] = f32::from_bits(prediction.data[i]); 
        }

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

    fn train_and_validate(
            &mut self,
            epochs: usize,
            batch_size: usize,
            learning_rate: f32,
            train_inputs: &[Vec<f32>],
            train_labels: &[Vec<f32>],
            val_inputs: &[Vec<f32>],
            val_labels: &[Vec<f32>],
            input_shapes: Vec<usize>,
            label_shapes: Vec<usize>,
            val_input_shapes: Vec<usize>,
            val_label_shapes: Vec<usize>,
        ) 
    {
        let train_inputs = self.dataset(train_inputs, input_shapes.clone());
        let train_labels = self.dataset(train_labels, label_shapes.clone());
        let val_inputs = self.dataset(val_inputs, val_input_shapes.clone());
        let val_labels = self.dataset(val_labels, val_label_shapes.clone());
        self.inner.train_and_validate(epochs, batch_size, learning_rate.to_bits(), train_inputs, train_labels, val_inputs, val_labels);
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

                println!("Weights shape: {:?}", shape);

                let in_channels = shape[0];
                let out_channels = shape[1];
                let rows = shape[2];
                let cols = shape[3];
                let flat = weights.data;

                if flat.len() != in_channels * out_channels * rows * cols {
                    println!("Shape mismatch: expected {} elements, got {}", in_channels * out_channels * rows * cols, flat.len());
                    return;
                }
    
                println!("\nPlain Weights for Layer \"{}\":", id);
                for c in 0..out_channels {
                    for k in 0..in_channels {
                        for i in 0..rows {
                            print!("\n[");
                            for j in 0..cols {
                                let index = c * in_channels * rows * cols + k * rows * cols + i * cols + j;
                                print!("{:<6} ", f32::from_bits(flat[index]));
                            }
                            print!("]\n");
                        }
                    }
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

                println!("\nPlain Biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    print!("{:<6} ", f32::from_bits(flat[i]));
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
    
                println!("\nPlain grad_weights for Layer \"{}\":", id);
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        print!("{:<6} ", f32::from_bits(flat[index]));
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

                println!("\nPlain grad_biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    print!("{:<6} ", f32::from_bits(flat[i]));
                }
                print!("]\n");
                return;
            }
        }
    }

}
impl PlainNeuralNetworkU32 {

    fn init_weights(&mut self, input_size: usize, output_size: usize, in_channels: usize, out_channels: usize, experiment: Option<i8>) -> PlainTensor<u32> {

        if experiment == Some(1) {
            // Initialize weights for experiment 1
            if output_size == 16 {
                let fc1_weights = EXP1_W_FC1_32.to_vec();
                let flattened_fc1_weight: Vec<u32> = fc1_weights.into_iter().flatten().collect();
                return PlainTensor::new(flattened_fc1_weight, vec![1, 1, input_size, output_size]);
            }
            else if output_size == 4 {
                let fc2_weights = EXP1_W_FC2_32.to_vec();
                let flattened_fc2_weight: Vec<u32> = fc2_weights.into_iter().flatten().collect();
                return PlainTensor::new(flattened_fc2_weight, vec![1, 1, input_size, output_size]);
            }
            else if output_size == 2 {
                let fc3_weights = EXP1_W_FC3_32.to_vec();
                let flattened_fc3_weight: Vec<u32> = fc3_weights.into_iter().flatten().collect();
                return PlainTensor::new(flattened_fc3_weight, vec![1, 1, input_size, output_size]);
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            if output_size == 32 {
                let fc_weights = EXP2_W_FC_32.to_vec();
                let flattened_fc_weights: Vec<u32> = fc_weights.into_iter().flatten().collect();
                return PlainTensor::new(flattened_fc_weights, vec![1, 1, input_size, output_size]);
            }
            else if output_size == 2 {
                let conv_weights = EXP2_W_CONV_32.to_vec();
                let flattened_conv_weights: Vec<u32> = conv_weights.into_iter().flatten().collect();
                return PlainTensor::new(flattened_conv_weights, vec![out_channels, in_channels, input_size, output_size]);
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
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

            return PlainTensor::new(weights, vec![out_channels, in_channels, input_size, output_size]);
        }
        else {
            // Compute std_dev like PyTorch (Glorot / Xavier normal)
            let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;

            // Create a normal distribution with mean=0 and std=std_dev
            let normal = Normal::new(0.0, std_dev).unwrap();

            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let mut weights = Vec::with_capacity(in_channels * out_channels * input_size * output_size);

            for _ in 0..(in_channels * out_channels * input_size * output_size) {
                let sample = normal.sample(&mut rng) as f32;
                let u_sample = sample.to_bits();
                weights.push(u_sample);
            }
            
            return PlainTensor {
                data: weights,
                shape: vec![out_channels, in_channels, input_size, output_size],
            }
        }
    }

    fn init_biases(&mut self, output_size:usize, experiment: Option<i8>) -> PlainTensor<u32> {

        if experiment == Some(1) {
            if output_size == 4 {
                let fc_bias = EXP1_B_FC1_32.to_vec();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else if output_size == 2 {
                let fc_bias = EXP1_B_FC2_32.to_vec();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else if output_size == 3 {
                let fc_bias = EXP1_B_FC3_32.to_vec();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            if output_size == 3 {
                let fc_bias = EXP2_B_FC_32.to_vec();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else if output_size == 2 {
                let conv_bias = EXP2_B_CONV_32.to_vec();
                return PlainTensor::new(conv_bias, vec![1, 1, 1, output_size]);
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
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
            PlainTensor::new(biases, [1, 1, 1, output_size].to_vec())
        } 
        else {
            let biases = vec![0u32; output_size];
            PlainTensor::new(biases, vec![1, 1, 1, output_size])
        }
    }

    fn init_gradients(&self, shape: &[usize]) ->PlainTensor<u32> {
        let size = shape.iter().product();
        let zeros = vec![0u32; size];
        let shapes: Vec<usize>;
        if(shape.to_vec().len() == 1){
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        PlainTensor::new(zeros, shapes)
    }

    fn init_derivatives(&self, shape: &[usize]) -> PlainTensor<u32> {
        let size = shape.iter().product();
        let zeros = vec![0u32; size];
        let shapes: Vec<usize>;
        if(shape.to_vec().len() == 1){
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        PlainTensor::new(zeros, shapes)
    }

    fn init_batch_norm(&self, shape: &[usize]) -> (PlainTensor<u32>, PlainTensor<u32>, PlainTensor<u32>, PlainTensor<u32>, PlainTensor<u32>) {
        let size = shape.iter().product();
        let zeros = vec![0u32; size];
        let ones = vec![1065353216u32; size];
        let x_hat = PlainTensor::new(zeros.clone(), [1, 1, 1, shape[0]].to_vec());
        let mean = PlainTensor::new(zeros.clone(), [1, 1, 1, shape[0]].to_vec());
        let variance = PlainTensor::new(ones.clone(), [1, 1, 1, shape[0]].to_vec());

        let (beta_file, gamma_file): (Array2<f32>, Array2<f32>) = if shape[0] == 32{
            (read_npy(Path::new("src/experiment_3/initializations/bn1_bias.npy")).expect("Failed to read batch norm 32"), 
            read_npy(Path::new("src/experiment_3/initializations/bn1_weight.npy")).expect("Failed to read batch norm 32"))
        } else if shape[0] == 64 {
            (read_npy(Path::new("src/experiment_3/initializations/bn2_bias.npy")).expect("Failed to read batch norm 64"), 
            read_npy(Path::new("src/experiment_3/initializations/bn2_weight.npy")).expect("Failed to read batch norm 64"))
        } else if shape[0] == 128 {
            (read_npy(Path::new("src/experiment_3/initializations/bn3_bias.npy")).expect("Failed to read batch norm 128"), 
            read_npy(Path::new("src/experiment_3/initializations/bn3_weight.npy")).expect("Failed to read batch norm 128"))
        } else {
            panic!("No matching weight file for given layer dimensions");
        };
        let vec_vec_beta = array2_to_vecvec(&beta_file);
        let vec_vec_gamma = array2_to_vecvec(&gamma_file);
        let beta_vec: Vec<u32> = vec_vec_beta
            .into_iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect();
        let gamma_vec: Vec<u32> = vec_vec_gamma
            .into_iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect();
        let beta = PlainTensor::new(beta_vec, [1, 1, 1, shape[0]].to_vec());
        let gamma = PlainTensor::new(gamma_vec, [1, 1, 1, shape[0]].to_vec());
        (x_hat, mean, variance, gamma, beta)
    }


    fn dataset(&mut self, clear_data: &[Vec<f32>], input_shapes: Vec<usize>) -> PlainTensor<u32> {
        let mut dataset = Vec::new();

        for vec in clear_data{
            for sample in vec{
                let sample = (*sample).to_bits();
                dataset.push(sample);
            }
        }

        let num_channels = 1; 

       PlainTensor { data: dataset, shape: vec![input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3]] }

    }

}

////////////////////////////////////////////////////
/// 
impl PlainNeuralNetwork for PlainNeuralNetworkU16 {
    fn create(experiment: Option<i8>) -> Self{
        let layers: Vec<Box<dyn PlainLayer<u16>>> = vec![];

        // Step 4: Create the loss function
        let loss: Box<dyn PlainLossFunction<u16>> = Box::new(MseLoss{});

        // Step 5: Build the inner EncryptedNeuralNetworkImpl
        let inner = PlainNeuralNetworkImpl {
            layers,
            loss,
        };

        // Step 6: Wrap in the public struct
        PlainNeuralNetworkU16 { inner, experiment }
    }

    
    fn add_dense(&mut self, input_size: usize, output_size: usize) {
        let weights = self.init_weights(output_size, input_size, 1, 1, self.experiment);
        let biases = self.init_biases(output_size, self.experiment);
        let grad_weights = self.init_gradients(&[output_size, input_size]);
        let grad_biases = self.init_gradients(&[output_size]);
        self.inner.add_dense(weights, biases, grad_weights, grad_biases);
    }

    fn add_tanh_activation(&mut self, size: usize) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_tanh_activation(derivatives, TANH16_PLA_RANGES.to_vec());
    }

    fn add_relu_activation(&mut self, size: usize) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_relu_activation(derivatives);
    }

    fn add_max_pooling(&mut self, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize){
        self.inner.add_max_pooling(input_dim, kernel_size, stride, padding);
    }

    fn add_conv(&mut self, in_channels: usize, out_channels: usize, kernel_width: usize, kernel_height: usize, stride: usize, padding: usize) {
        let weights = self.init_weights(kernel_height, kernel_width, in_channels, out_channels, self.experiment);
        let biases = self.init_biases(out_channels, self.experiment);
        let grad_weights = self.init_gradients(&[out_channels, in_channels * kernel_width * kernel_height]);
        let grad_biases = self.init_gradients(&[out_channels]);
        self.inner.add_conv(weights, biases, grad_weights, grad_biases, stride, padding);
    }

    fn add_batch_norm(&mut self, size:usize) {
        let (x_hat, mean, variance, gamma, beta) = self.init_batch_norm(&[size]);
        self.inner.add_batch_norm(x_hat, mean, variance, gamma, beta);
    }

    
    fn train(&mut self, epochs: usize, batch_size: usize, learning_rate: f32, train_inputs: &[Vec<f32>], train_labels: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) {
        let u32_learning_rate = f16::from_f32(learning_rate).to_bits();
        let train_inputs = self.dataset(train_inputs, input_shapes.clone());
        let train_labels = self.dataset(train_labels, label_shapes.clone());
        let time = Instant::now();
        self.inner.train(epochs, batch_size, u32_learning_rate.clone(), train_inputs.clone(), train_labels.clone());
    }

    fn inference(&mut self, input: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) -> Vec<f32>{
        let inf_input = self.dataset(input, input_shapes.clone());
        let prediction = self.inner.inference(&inf_input.clone());
        let mut prediction_f16: Vec<f16> = vec![f16::from_f32(0.0 as f32); label_shapes[3]];
        for i in 0..prediction.data.len() {
            prediction_f16[i] = f16::from_bits(prediction.data[i]); 
        }

        let predicted_index = prediction_f16
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

        // 2. Create one-hot encoded vector
        let mut one_hot: Vec<f32> = vec![0.0; prediction_f16.len()];
        one_hot[predicted_index] = 1.0;
        
        one_hot
    }

    fn train_and_validate(
            &mut self,
            epochs: usize,
            batch_size: usize,
            learning_rate: f32,
            train_inputs: &[Vec<f32>],
            train_labels: &[Vec<f32>],
            val_inputs: &[Vec<f32>],
            val_labels: &[Vec<f32>],
            input_shapes: Vec<usize>,
            label_shapes: Vec<usize>,
            val_input_shapes: Vec<usize>,
            val_label_shapes: Vec<usize>,
        ) 
    {
        let train_inputs = self.dataset(train_inputs, input_shapes.clone());
        let train_labels = self.dataset(train_labels, label_shapes.clone());
        let val_inputs = self.dataset(val_inputs, val_input_shapes.clone());
        let val_labels = self.dataset(val_labels, val_label_shapes.clone());
        self.inner.train_and_validate(epochs, batch_size, f16::from_f32(learning_rate).to_bits(), train_inputs, train_labels, val_inputs, val_labels);
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
    
                println!("\nPlain Weights for Layer \"{}\":", id);
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        print!("{:<6} ", f16::from_bits(flat[index]));
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

                println!("\nPlain Biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    print!("{:<6} ", f16::from_bits(flat[i]));
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
    
                println!("\nPlain grad_weights for Layer \"{}\":", id);
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        print!("{:<6} ", f16::from_bits(flat[index]));
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

                println!("\nPlain grad_biases for Layer \"{}\":", id);
                print!("\n[");
                for i in 0..columns {
                    print!("{:<6} ", f16::from_bits(flat[i]));
                }
                print!("]\n");
                return;
            }
        }
    }

}
impl PlainNeuralNetworkU16 {
    fn init_weights(&mut self, input_size: usize, output_size: usize, in_channels: usize, out_channels: usize, experiment: Option<i8>) -> PlainTensor<u16> {

       if experiment == Some(1) {
            // Initialize weights for experiment 1
            if output_size == 16 {
                let fc1_weights = EXP1_W_FC1_32.to_vec();
                let flattened_fc1_weight: Vec<u16> = fc1_weights
                    .into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(flattened_fc1_weight, vec![1, 1, input_size, output_size]);
            }
            else if output_size == 4 {
                let fc2_weights = EXP1_W_FC2_32.to_vec();
                let flattened_fc2_weight: Vec<u16> = fc2_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(flattened_fc2_weight, vec![1, 1, input_size, output_size]);
            }
            else if output_size == 2 {
                let fc3_weights = EXP1_W_FC3_32.to_vec();
                let flattened_fc3_weight: Vec<u16> = fc3_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(flattened_fc3_weight, vec![1, 1, input_size, output_size]);
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            if output_size == 32 {
                let fc_weights = EXP2_W_FC_32.to_vec();
                let flattened_fc_weights: Vec<u16> = fc_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();         
                return PlainTensor::new(flattened_fc_weights, vec![1, 1, input_size, output_size]);
            }
            else if output_size == 2 {
                let conv_weights = EXP2_W_CONV_32.to_vec();
                let flattened_conv_weights: Vec<u16> = conv_weights.into_iter()
                    .flatten()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(flattened_conv_weights, vec![out_channels, in_channels, input_size, output_size]);
            }
            else {
                panic!("No matching weight file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
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

            return PlainTensor::new(weights, vec![out_channels, in_channels, input_size, output_size]);
        }
        else {
        
            let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
            let normal = Normal::new(0.0, 0.5).unwrap();
        
            // Use a fixed seed for deterministic results
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let mut weights = Vec::with_capacity(in_channels * out_channels * input_size * output_size);

            for _ in 0..(in_channels * out_channels * input_size * output_size) {
                let sample = f16::from_f32(normal.sample(&mut rng) as f32);
                let u_sample = sample.to_bits();
                weights.push(u_sample);
            }
        
            PlainTensor {
                data: weights,
                shape: vec![in_channels, out_channels, input_size, output_size],
            }
        }
    }

    fn init_biases(&mut self, output_size:usize, experiment: Option<i8>) -> PlainTensor<u16> {

        if experiment == Some(1) {
            if output_size == 4 {
                let fc_bias = EXP1_B_FC1_32.to_vec().into_iter()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else if output_size == 2 {
                let fc_bias = EXP1_B_FC2_32.to_vec().into_iter()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else if output_size == 3 {
                let fc_bias = EXP1_B_FC3_32.to_vec().into_iter()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(2) {
            if output_size == 3 {
                let fc_bias = EXP2_B_FC_32.to_vec().into_iter()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(fc_bias, vec![1, 1, 1, output_size]);
            }
            else if output_size == 2 {
                let conv_bias = EXP2_B_CONV_32.to_vec().into_iter()
                    .map(|f| f16::from_f32(f32::from_bits(f)).to_bits())
                    .collect();
                return PlainTensor::new(conv_bias, vec![1, 1, 1, output_size]);
            }
            else {
                panic!("No matching bias file for given layer dimensions");
            }
        }
        else if experiment == Some(3) {
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
                .map(|x| f16::from_f32(x).to_bits())
                .collect();
            PlainTensor::new(biases, [1, 1, 1, output_size].to_vec())
        } 
        else {
            let biases = vec![0u16; output_size];
            PlainTensor::new(biases, vec![1, 1, 1, output_size])
        }
    }

    fn init_gradients(&self, shape: &[usize]) ->PlainTensor<u16> {
        let size = shape.iter().product();
        let zeros = vec![0u16; size];
        let shapes: Vec<usize>;
        if(shape.to_vec().len() == 1){
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        PlainTensor::new(zeros, shapes)
    }

    fn init_derivatives(&self, shape: &[usize]) -> PlainTensor<u16> {
        let size = shape.iter().product();
        let zeros = vec![0u16; size];
        let shapes: Vec<usize>;
        if(shape.to_vec().len() == 1){
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        PlainTensor::new(zeros, shapes)
    }

    fn dataset(&mut self, clear_data: &[Vec<f32>], input_shapes: Vec<usize>) -> PlainTensor<u16> {
        let mut dataset = Vec::new();

        for vec in clear_data{
            for sample in vec{
                let sample = f16::from_f32((*sample)).to_bits();
                dataset.push(sample);
            }
        }

        let num_channels = 1; 

       PlainTensor { data: dataset, shape: vec![input_shapes[0], input_shapes[1], input_shapes[2], input_shapes[3]] }

    }

    fn init_batch_norm(&self, shape: &[usize]) -> (PlainTensor<u16>, PlainTensor<u16>, PlainTensor<u16>, PlainTensor<u16>, PlainTensor<u16>) {
        let size = shape.iter().product();
        let zeros = vec![0u16; size];
        let ones = vec![15360u16; size];
        let x_hat = PlainTensor::new(zeros.clone(), [1, 1, 1, shape[0]].to_vec());
        let mean = PlainTensor::new(zeros.clone(), [1, 1, 1, shape[0]].to_vec());
        let variance = PlainTensor::new(ones.clone(), [1, 1, 1, shape[0]].to_vec());

        let (beta_file, gamma_file): (Array2<f32>, Array2<f32>) = if shape[0] == 32{
            (read_npy(Path::new("src/experiment_3/initializations/bn1_bias.npy")).expect("Failed to read batch norm 32"), 
            read_npy(Path::new("src/experiment_3/initializations/bn1_weight.npy")).expect("Failed to read batch norm 32"))
        } else if shape[0] == 64 {
            (read_npy(Path::new("src/experiment_3/initializations/bn2_bias.npy")).expect("Failed to read batch norm 64"), 
            read_npy(Path::new("src/experiment_3/initializations/bn2_weight.npy")).expect("Failed to read batch norm 64"))
        } else if shape[0] == 128 {
            (read_npy(Path::new("src/experiment_3/initializations/bn3_bias.npy")).expect("Failed to read batch norm 128"), 
            read_npy(Path::new("src/experiment_3/initializations/bn3_weight.npy")).expect("Failed to read batch norm 128"))
        } else {
            panic!("No matching weight file for given layer dimensions");
        };
        let vec_vec_beta = array2_to_vecvec(&beta_file);
        let vec_vec_gamma = array2_to_vecvec(&gamma_file);
        let beta_vec: Vec<u16> = vec_vec_beta
            .into_iter()
            .flatten()
            .map(|x| f16::from_f32(x).to_bits())
            .collect();
        let gamma_vec: Vec<u16> = vec_vec_gamma
            .into_iter()
            .flatten()
            .map(|x| f16::from_f32(x).to_bits())
            .collect();
        let beta = PlainTensor::new(beta_vec, [1, 1, 1, shape[0]].to_vec());
        let gamma = PlainTensor::new(gamma_vec, [1, 1, 1, shape[0]].to_vec());
        (x_hat, mean, variance, gamma, beta)
    }

}