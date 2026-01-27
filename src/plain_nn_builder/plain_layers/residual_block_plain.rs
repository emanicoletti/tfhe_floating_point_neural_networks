use crate::plain_nn_builder::plain_utils::{PlainElement, PlainTensor};
use crate::plain_nn_builder::plain_layers::{PlainBatchNormLayer, PlainConv2DLayer, PlainLayer};
use crate::plain_nn_builder::plain_activations::PlainReLUActivation;
use crate::plain_nn_builder::plain_ops::*;
use crate::plain_nn_builder::plain_utils::load_from_file::array2_to_vecvec;

use rand::SeedableRng;      
use rand_chacha::ChaCha8Rng;
use rand_distr::{Normal, Distribution};
use ndarray::Array2;
use ndarray_npy::read_npy;
use std::path::Path;

pub struct ResidualBlock<T> 
where T: PlainElement 
{
    // Il percorso principale (es. Conv3x3 -> ReLU -> Conv3x3)
    main_layers: Vec<Box<dyn PlainLayer<T>>>,
    
    // Il percorso skip (es. vuoto per Identity, oppure Conv1x1 per Projection)
    skip_layers: Vec<Box<dyn PlainLayer<T>>>,

    // Cache separate per i due percorsi
    main_cache: Vec<PlainTensor<T>>,
    skip_cache: Vec<PlainTensor<T>>, 

    id: String,
    experiment: Option<i8>
}

impl<T> ResidualBlock<T>
where T: PlainElement 
{
    pub fn new(
        id: String,
        experiment: Option<i8>
    ) -> Self {
        Self { 
            main_layers: Vec::new(),
            skip_layers: Vec::new(),
            main_cache: Vec::new(),
            skip_cache: Vec::new(),
            id,
            experiment,
        }
    }
}

impl<T> PlainLayer<T> for ResidualBlock<T>
where T: PlainElement + PlainAdd + Default
{
    fn forward(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        // === 1. Main Path ===
        self.main_cache.clear();
        let mut x_main = input.clone();
        self.main_cache.push(x_main.clone()); // Cache input iniziale

        for layer in &mut self.main_layers {
            x_main = layer.forward(&x_main);
            self.main_cache.push(x_main.clone());
        }
        self.main_cache.pop(); // Rimuovi l'ultimo output (non serve come input)

        // === 2. Skip Path ===
        self.skip_cache.clear();
        let mut x_skip = input.clone();
        
        // Se skip_layers è vuoto, x_skip rimane uguale a input (Identity)
        if !self.skip_layers.is_empty() {
             self.skip_cache.push(x_skip.clone());
             for layer in &mut self.skip_layers {
                 x_skip = layer.forward(&x_skip);
                 self.skip_cache.push(x_skip.clone());
             }
             self.skip_cache.pop();
        }

        // === 3. Somma ===
        x_main.add(&x_skip)
    }

    fn backward(
        &mut self,
        _input: &PlainTensor<T>, 
        grad_output: &PlainTensor<T>, // dL/dy
    ) -> PlainTensor<T> {
        
        // === Path A: Backward Main Path ===
        let mut grad_main = grad_output.clone();
        for (i, layer) in self.main_layers.iter_mut().rev().enumerate() {
            let cached_in = &self.main_cache[self.main_cache.len() - 1 - i];
            grad_main = layer.backward(cached_in, &grad_main);
        }

        // === Path B: Backward Skip Path ===
        let mut grad_skip = grad_output.clone();
        
        if self.skip_layers.is_empty() {
            // Identity Shortcut: Il gradiente passa intatto (moltiplicato per 1)
            // grad_skip rimane uguale a grad_output
        } else {
            // Projection Shortcut: Il gradiente attraversa i layer della skip (es. Conv1x1)
            for (i, layer) in self.skip_layers.iter_mut().rev().enumerate() {
                let cached_in = &self.skip_cache[self.skip_cache.len() - 1 - i];
                grad_skip = layer.backward(cached_in, &grad_skip);
            }
        }

        // === Somma dei Gradienti rispetto all'Input ===
        // L'input originale è stato inviato sia a Main che a Skip.
        // La regola della catena dice che dobbiamo sommare i gradienti di ritorno.
        grad_main.add(&grad_skip)
    }

    fn update_parameters(&mut self, learning_rate: T, weight_decay: T, momentum: T) {
        // Aggiorniamo i pesi del path principale
        for layer in &mut self.main_layers {
            layer.update_parameters(learning_rate.clone(), weight_decay.clone(), momentum.clone());
        }
        // Aggiorniamo i pesi della skip connection (se ne ha, es. Conv1x1)
        for layer in &mut self.skip_layers {
            layer.update_parameters(learning_rate.clone(), weight_decay.clone(), momentum.clone());
        }
    }

    // ... Inference, Getters ...
    fn inference(&mut self, input: &PlainTensor<T>) -> PlainTensor<T> {
        // Simile al forward ma chiamando inference
        let mut x_main = input.clone();
        for layer in &mut self.main_layers { x_main = layer.inference(&x_main); }
        
        let mut x_skip = input.clone();
        for layer in &mut self.skip_layers { x_skip = layer.inference(&x_skip); }
        
        x_main.add(&x_skip)
    }
    
    fn get_id(&self) -> String { self.id.clone() }
    fn approximate_inference(&mut self, _input: &PlainTensor<T>) -> PlainTensor<T> { todo!() }
    fn get_weights(&self) -> PlainTensor<T> { todo!() }
    fn get_biases(&self) -> PlainTensor<T> { todo!() }
    fn get_grad_weights(&self) -> PlainTensor<T> { todo!() }
    fn get_grad_biases(&self) -> PlainTensor<T> { todo!() }
}

impl ResidualBlock<u32>
{
    pub fn add_conv(&mut self, is_in_skip_path: bool, in_channels: usize, out_channels: usize, kernel_width: usize, kernel_height: usize, stride: usize, padding: usize, id: String){
        let conv_layer = PlainConv2DLayer {
            id: id.clone(),
            weights: self.init_weights(kernel_width, kernel_height, in_channels, out_channels, id.clone(), self.experiment),
            biases: self.init_biases(out_channels, id.clone(), self.experiment),
            grad_weights: Some(self.init_gradients(&[out_channels, in_channels * kernel_width * kernel_height])),
            grad_biases: Some(self.init_gradients(&[out_channels])),
            stride,
            padding,
            velocity_weights: None,
            velocity_biases: None,
        };
        if is_in_skip_path{
            self.skip_layers.push(Box::new(conv_layer));
        }
        else {
            self.main_layers.push(Box::new(conv_layer));
        }
    }

    pub fn add_batch_norm(&mut self, is_in_skip_path: bool, size:usize, id:String){
        let (x_hat, mean, variance, gamma, beta) = self.init_batch_norm(&[size]);
        let batch_norm_layer = PlainBatchNormLayer::new(id, x_hat, mean, variance, gamma, beta);
        if is_in_skip_path{
            self.skip_layers.push(Box::new(batch_norm_layer));
        }
        else {
            self.main_layers.push(Box::new(batch_norm_layer));
        }
    }

    pub fn add_relu_activation(&mut self, is_in_skip_path: bool, size:usize, id: String){
        let relu_activation = PlainReLUActivation::new(id.clone(), self.init_derivatives(&[size]));
        if is_in_skip_path{
            self.skip_layers.push(Box::new(relu_activation));
        }
        else {
            self.main_layers.push(Box::new(relu_activation));
        }
    }

    fn init_weights(&mut self, input_size: usize, output_size: usize, in_channels: usize, out_channels: usize, id: String, experiment: Option<i8>) -> PlainTensor<u32> {

        if experiment == Some(9) {
            let weights_file: Array2<f32> = match id.as_str() {
                "Rb1_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_0_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb1_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_0_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb2_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_1_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb2_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_1_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb3_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_2_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb3_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_2_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb4_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_0_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb4_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_0_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb4_skip_conv" => {    
                    read_npy(Path::new("src/cifar10/initializations/layer2_0_shortcut_0_weight.npy")).expect("Failed to read skip conv weights")
                },
                "Rb5_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_1_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb5_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_1_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb6_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_2_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb6_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_2_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb7_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_0_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb7_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_0_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb7_skip_conv" => {    
                    read_npy(Path::new("src/cifar10/initializations/layer3_0_shortcut_0_weight.npy")).expect("Failed to read skip conv weights")
                },
                "Rb8_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_1_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb8_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_1_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                "Rb9_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_2_conv1_weight.npy")).expect("Failed to read conv1 weights")
                },
                "Rb9_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_2_conv2_weight.npy")).expect("Failed to read conv2 weights")
                },
                _ => panic!("Unknown layer ID for experiment 9"),
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
            let std_dev = ((2.0 / (input_size + output_size) as f64).sqrt()) as f32;
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

    fn init_biases(&mut self, output_size:usize, id: String, experiment: Option<i8>) -> PlainTensor<u32> {
        if experiment == Some(9) {
            let biases_file: Array2<f32> = match id.as_str() {
                "Rb1_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_0_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb1_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_0_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb2_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_1_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb2_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_1_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb3_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_2_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb3_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer1_2_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb4_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_0_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb4_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_0_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb4_skip_conv" => {    
                    read_npy(Path::new("src/cifar10/initializations/layer2_0_shortcut_0_bias.npy")).expect("Failed to read skip conv biases")
                },
                "Rb5_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_1_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb5_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_1_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb6_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_2_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb6_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer2_2_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb7_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_0_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb7_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_0_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb7_skip_conv" => {    
                    read_npy(Path::new("src/cifar10/initializations/layer3_0_shortcut_0_bias.npy")).expect("Failed to read skip conv biases")
                },
                "Rb8_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_1_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb8_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_1_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                "Rb9_conv1" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_2_conv1_bias.npy")).expect("Failed to read conv1 biases")
                },
                "Rb9_conv2" => {
                    read_npy(Path::new("src/cifar10/initializations/layer3_2_conv2_bias.npy")).expect("Failed to read conv2 biases")
                },
                _ => panic!("Unknown layer ID for experiment 9"),
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

    fn init_gradients(&self, shape: &[usize]) -> PlainTensor<u32> {
        let size = shape.iter().product();
        let zeros = vec![0u32; size];
        let shapes: Vec<usize>;
        if shape.to_vec().len() == 1 {
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        PlainTensor::new(zeros, shapes)
    }

    fn init_batch_norm(&self, shape: &[usize]) -> (PlainTensor<u32>, PlainTensor<u32>, PlainTensor<u32>, PlainTensor<u32>, PlainTensor<u32>) {
        let size = shape[0];
        let zeros = vec![0u32; size]; 
        let ones = vec![1.0f32.to_bits(); size]; 

        let x_hat = PlainTensor::new(zeros.clone(), [1, 1, 1, size].to_vec());
        let mean = PlainTensor::new(zeros.clone(), [1, 1, 1, size].to_vec());
        let variance = PlainTensor::new(ones.clone(), [1, 1, 1, size].to_vec());

        let beta = PlainTensor::new(zeros.clone(), [1, 1, 1, size].to_vec());
        let gamma = PlainTensor::new(ones.clone(), [1, 1, 1, size].to_vec());

        (x_hat, mean, variance, gamma, beta)
    }

    fn init_derivatives(&self, shape: &[usize]) -> PlainTensor<u32> {
        let size = shape.iter().product();
        let zeros = vec![0u32; size];
        let shapes: Vec<usize>;
        if shape.to_vec().len() == 1{
            shapes = [1, shape[0]].to_vec();
        }
        else{
            shapes = shape.to_vec();
        }
        PlainTensor::new(zeros, shapes)
    }
}