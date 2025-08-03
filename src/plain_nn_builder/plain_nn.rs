use crate::plain_nn_builder::generic_nn::*;
use crate::plain_nn_builder::plain_activations::*;
use crate::plain_nn_builder::plain_utils::*;
use crate::plain_nn_builder::plain_layers::*;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_losses::*;

// for seed_from_u64
use rand::SeedableRng;      // <- import SeedableRng trait
use rand_chacha::ChaCha8Rng;
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
    fn inference(
        &mut self,
        inputs: &[Vec<f32>], 
        input_shapes: Vec<usize>,
        label_shapes: Vec<usize>,
    ) -> Vec<f32>;
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
    }

    fn inference(&mut self, input: &[Vec<f32>], input_shapes: Vec<usize>, label_shapes: Vec<usize>) -> Vec<f32>{
        let inf_input = self.dataset(input, input_shapes.clone());
        let prediction = self.inner.inference(&inf_input.clone());
        let mut prediction_f32: Vec<f32> = vec![0.0; label_shapes[3]];
        for i in 0..prediction.data.len() {
            prediction_f32[i] = f32::from_bits(prediction.data[i]); 
        }
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
    fn init_weights(&mut self, input_size: usize, output_size: usize) -> PlainTensor<u32> {
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
            let flattened_fc1_weight: Vec<u32> = fc1_weight.into_iter().flatten().collect();
            return PlainTensor::new(flattened_fc1_weight, vec![1, 1, input_size, output_size]);
        }
        if output_size == 4 {
            let fc2_weight: Vec<Vec<u32>> = vec![
            vec![(-0.3546_f32).to_bits(),  0.2355_f32.to_bits(), (-0.2220_f32).to_bits(), (-0.0288_f32).to_bits()],
            vec![(-0.2830_f32).to_bits(), (-0.4757_f32).to_bits(),  0.1246_f32.to_bits(),  0.0483_f32.to_bits()],
            ];
            let flattened_fc2_weight: Vec<u32> = fc2_weight.into_iter().flatten().collect();
            return PlainTensor::new(flattened_fc2_weight, vec![1, 1, input_size, output_size]);
        }
        if output_size == 2 {
            let fc3_weight: Vec<Vec<u32>> = vec![
                vec![(-0.6488_f32).to_bits(),  0.2701_f32.to_bits()],
                vec![ 0.1953_f32.to_bits(),  0.1416_f32.to_bits()],
                vec![(-0.1549_f32).to_bits(), (-0.6687_f32).to_bits()],
            ];
            let flattened_fc3_weight: Vec<u32> = fc3_weight.into_iter().flatten().collect();
            return PlainTensor::new(flattened_fc3_weight, vec![1, 1, input_size, output_size]);
        }

        // Xavier Initialization standard deviation
        let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
        let normal = Normal::new(0.0, 0.5).unwrap();
    
        // Use a fixed seed for deterministic results
        let mut rng = ChaCha8Rng::seed_from_u64(42);
    
        let mut weights = Vec::with_capacity(input_size * output_size);
    
        for _ in 0..(input_size * output_size) {
            let sample = normal.sample(&mut rng) as f32;
            let u_sample = sample.to_bits();
            weights.push(u_sample);
        }
    
        PlainTensor {
            data: weights,
            shape: vec![1, 1, input_size, output_size],
        }
    }

    fn init_biases(&mut self, output_size:usize) -> PlainTensor<u32> {

        if output_size == 4 {
            let fc1_bias: Vec<u32> = vec![
                (-0.0879_f32).to_bits(), 0.1680_f32.to_bits(), (-0.1631_f32).to_bits(), (-0.0271_f32).to_bits()
            ];
            return PlainTensor::new(fc1_bias, vec![1, 1, 1, 4]);
        }
        if output_size == 2 {
            let fc2_bias: Vec<u32> = vec![
                0.4002_f32.to_bits(), (-0.0112_f32).to_bits()
            ];
            return PlainTensor::new(fc2_bias, vec![1, 1, 1, 2]);
        }
        if output_size == 3 {
            let fc3_bias: Vec<u32> = vec![
                (-0.1854_f32).to_bits(), (-0.2199_f32).to_bits(), (-0.6619_f32).to_bits()
            ];
            return PlainTensor::new(fc3_bias, vec![1, 1, 1, 3]);
        }
        
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