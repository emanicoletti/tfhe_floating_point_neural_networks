use crate::plain_nn_builder::generic_nn::*;
use crate::plain_nn_builder::plain_activations::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::*;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_losses::*;

use rand::thread_rng;
use rand_distr::{Normal, Distribution};
use std::time::Instant;

pub trait PlainNeuralNetwork {
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
    fn print_plain_weights(&self, id: String);
    fn print_plain_biases(&self, id:String);
    fn print_plain_grad_weights(&self, id: String);
    fn print_plain_grad_biases(&self, id:String);
}

pub struct PlainNeuralNetworkU32 {
    inner: PlainNeuralNetworkImpl<u32>,
}

pub static TANH32_PLA_RANGES: &[(u32, u32, u32, u32, u32)] = &[
    (3221225472u32, 4294967295u32, 3212836864u32, 0u32, 0u32), // [-inf, -2], output ~ -1, derivative ≈ 0
    (3210040661u32, 3221225471u32, 1048576000u32, 3204448256u32, 1048576000u32), // [-2, -0.8333] slope 0.25. intercept -0.5
    (2147483648u32, 3210040660u32, 1062836634u32, 0u32, 1062836634u32), // [-0.833, -0] slope=0.85 intercept = 0
    (0u32, 1062557013u32, 1062836634u32, 0u32, 1062836634u32), //[0, 0.833] slope=0.85 intercept = 0
    (1062557014u32, 1073741824u32, 1048576000u32, 1056964608u32, 1048576000u32), //[0.833, 2] slope = 0.25 intercept 0.5
    (1073741825u32, 2147483647u32, 1065353217u32, 0u32, 0u32), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

impl PlainNeuralNetwork for PlainNeuralNetworkU32 {
    fn create() -> Self{
        let layers: Vec<Box<dyn PlainLayer<u32>>> = vec![];

        // Step 4: Create the loss function
        let loss: Box<dyn PlainLossFunction<u32>> = Box::new(MseLoss{});

        // Step 5: Build the inner EncryptedNeuralNetworkImpl
        let inner = PlainNeuralNetworkImpl {
            layers,
            loss,
        };

        // Step 6: Wrap in the public struct
        PlainNeuralNetworkU32 { inner }
    }

    
    fn add_dense(&mut self, input_size: usize, output_size: usize) {
        let weights = self.init_weights(output_size, input_size);
        let biases = self.init_biases(output_size);
        let grad_weights = self.init_gradients(&[output_size, input_size]);
        let grad_biases = self.init_gradients(&[output_size]);
        self.inner.add_dense(weights, biases, grad_weights, grad_biases);
    }

    fn add_tanh_activation(&mut self, size: usize) {
        let derivatives = self.init_derivatives(&[size]);
        self.inner.add_tanh_activation(derivatives, TANH32_PLA_RANGES.to_vec());
    }

    
    fn add_max_pooling(&mut self, input_dim: Vec<usize>, kernel_size: usize, stride: usize, padding: usize){
        self.inner.add_max_pooling(input_dim, kernel_size, stride, padding);
    }

    
    fn train(&mut self, epochs: usize, batch_size: usize, learning_rate: f32, train_inputs: &[Vec<f32>], train_labels: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) {
        let u32_learning_rate = learning_rate.to_bits();
        let train_inputs = self.dataset(train_inputs, input_shapes.clone());
        let train_labels = self.dataset(train_labels, label_shapes.clone());
        let time = Instant::now();
        self.inner.train(epochs, batch_size, u32_learning_rate.clone(), train_inputs.clone(), train_labels.clone());
        println!("Training completed in {:?}", time.elapsed());
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
                        print!("{:<6} ", f32::from_bits(flat[index]));
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
    
                println!("\nDecrypted grad_weights for Layer \"{}\":", id);
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

                println!("\nDecrypted grad_biases for Layer \"{}\":", id);
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
    fn init_weights(&mut self, input_size: usize, output_size: usize) -> PlainTensor<u32>{
        
        // Xavier Initialization
        let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
        let normal = Normal::new(0.0, std_dev).unwrap();

        let mut rng = thread_rng();
        
        let mut weights = Vec::with_capacity(input_size * output_size);

        for i in 0..(input_size * output_size) {
            let sample = normal.sample(&mut rng) as f32;
            let sample = 0.0 as f32;
            let u_sample = sample.to_bits();
           weights.push(u_sample);
        }
        
        PlainTensor { data: (weights), shape: (vec![1, 1, input_size, output_size]) }
    }

    fn init_biases(&mut self, output_size:usize) -> PlainTensor<u32> {
        let biases = vec![0u32; output_size];
        PlainTensor::new(biases, vec![1, 1, 1, output_size])
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