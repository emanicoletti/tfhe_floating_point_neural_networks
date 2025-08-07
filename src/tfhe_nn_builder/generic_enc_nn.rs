use crate::tfhe_nn_builder::encrypted_utils::encrypted_context::{self, EncryptedContext};
use crate::tfhe_nn_builder::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::tfhe_nn_builder::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::tfhe_nn_builder::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer, EncryptedMaxPoolingLayer};
use crate::tfhe_nn_builder::encrypted_activations::{EncryptedTanhActivation, EncryptedReLUActivation};
use crate::tfhe_nn_builder::encrypted_losses::loss_function::LossFunction;
use crate::tfhe_nn_builder::encrypted_utils::tensor::EncryptedTensor;
use crate::tfhe_nn_builder::encrypted_ops::*;

use half::f16;

use std::time::Instant;

use tfhe::set_server_key;

/// Core generic implementation of an encrypted neural network
pub struct EncryptedNeuralNetworkImpl<K: ServerKeyTrait, T: EncryptedElement> {
    pub layers: Vec<Box<dyn EncryptedLayer<K, T>>>,
    pub loss: Box<dyn LossFunction<K, T>>,
    pub context: EncryptedContext<K, T>,
}
impl<K, T> EncryptedNeuralNetworkImpl<K, T> 
where
    K: ServerKeyTrait
    + EncryptedAdd<K, T>
    + EncryptedMul<K, T>
    + EncryptedDiv<K, T>
    + EncryptedNegate<K, T>
    + EncryptedTanh<K, T>
    + EncryptedMax<K, T> 
    + EncryptedReLU<K, T>
    + EncryptedBackwardRelu<K, T>
    + EncryptedGradIfEqual<K, T>,
    T: EncryptedElement + Clone + EncryptableValueType<Plain=u16> + 'static,
{
    pub fn add_dense(&mut self, weights: EncryptedTensor<T>, biases: EncryptedTensor<T>, grad_weights: EncryptedTensor<T>, grad_biases: EncryptedTensor<T>) {
        let id = format!("Dense{}", self.layers.len() + 1);
        let dense_layer = EncryptedDenseLayer {
            id: id, 
            weights: weights,
            biases: biases,
            grad_weights: Some(grad_weights),
            grad_biases: Some(grad_biases),
        };
        self.layers.push(Box::new(dense_layer));
    }

    pub fn add_tanh_activation(&mut self, derivatives: EncryptedTensor<T>, ranges: Vec<(T, T, T, T, T)>) {
        let id = format!("Tanh{}", self.layers.len() + 1);
        let tanh_layer = EncryptedTanhActivation {
            id: id,
            derivatives: derivatives,
            ranges: ranges,
        };
        self.layers.push(Box::new(tanh_layer));
    }

     pub fn add_relu_activation(&mut self, derivatives: EncryptedTensor<T>) {
        let id = format!("ReLU{}", self.layers.len() + 1);
        let relu_layer = EncryptedReLUActivation {
            id: id,
            derivatives: derivatives,
        };
        self.layers.push(Box::new(relu_layer));
    }

    pub fn add_max_pooling(
        &mut self,
        input_dim: Vec<usize>,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) {
        let id = format!("MaxPooling{}", self.layers.len() + 1);
        let max_pooling_layer = EncryptedMaxPoolingLayer::new(id, input_dim, kernel_size, stride, padding);
        self.layers.push(Box::new(max_pooling_layer));
    }
   
    pub fn train( 
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: T,
        train_inputs: EncryptedTensor<T>,
        train_labels: EncryptedTensor<T>,
    ) 
    {
        for epoch in 0..epochs{
            println!("Epoch {}/{}", epoch + 1, epochs);
            let time = Instant::now();
            let forward_time = Instant::now();
            println!("input_shapes: {:?}", train_inputs.shape);
            println!("labels shapes: {:?}", train_labels.shape);
            for (input_batch, label_batch) in self.iter_batches(&train_inputs, &train_labels, batch_size){
                let mut activations = vec![input_batch.clone()];
                println!("Forward started...");
                for layer in &mut self.layers {
                    let output = layer.forward(activations.last().unwrap(), &self.context);
                    activations.push(output.clone());
                    println!("Layer passed");
                    let prediction = activations.last().unwrap();
                    /* 
                    let size = prediction.shape[0];
                    let rows = prediction.shape[2];
                    let cols = prediction.shape[3];
                    let flat = &prediction.data;
        
                    if flat.len() != size * rows * cols {
                        println!("Shape mismatch: expected {} elements, got {}", size * rows * cols, flat.len());
                        return;
                    }
        
                    for b in 0..size {
                        println!("\nBatch {}", b);
                        for i in 0..rows {
                            print!("[");
                            for j in 0..cols {
                                let index = b * rows * cols + i * cols + j;
                                let decrypted: u16 = EncryptableValueType::decrypt(&flat[index], &self.context.client_key);
                                print!("{:<6} ", f16::from_bits(decrypted));
                            }
                            print!("]\n");
                        }
                    }
                    */
                }
                let prediction = activations.last().unwrap();
                /* 
                let loss_val = self.loss.compute_loss(&prediction, &label_batch, &self.context);
                let decrypted: u16 = EncryptableValueType::decrypt(&loss_val.data[0], &self.context.client_key);
                println!("Batch Loss: {:<6} ", f16::from_bits(decrypted).to_f32());
                */
                println!("Forward pass time: {:?}", forward_time.elapsed());
                println!("Backward started...");
                let mut grad = self.loss.gradient(&prediction, &label_batch, &self.context);
                for (i, layer) in self.layers.iter_mut().rev().enumerate() {
                    let input_to_layer = &activations[activations.len() - 2 - i];
                    grad = layer.backward(input_to_layer, &grad, &self.context);
                    /* 
                    let size = grad.shape[0];
                    let rows = grad.shape[2];
                    let cols = grad.shape[3];
                    let flat = &grad.data;
        
                    if flat.len() != size * rows * cols {
                        println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                        return;
                    }
        
                    for b in 0..size {
                        println!("\nBatch {}", b);
                        for i in 0..rows {
                            print!("[");
                            for j in 0..cols {
                                let index = b * rows * cols + i * cols + j;
                                let decrypted: u16 = EncryptableValueType::decrypt(&flat[index], &self.context.client_key);
                                print!("{:<6} ", f16::from_bits(decrypted));
                            }
                            print!("]\n");
                        }
                    }
                    */
                }
                println!("Backward ended...");
                for layer in &mut self.layers{
                    layer.update_parameters(learning_rate.clone(), &self.context);
                }
            }
            println!("Epoch {} completed in {:?}", epoch + 1, time.elapsed());
        }
    }

    pub fn inference(
        &mut self,
        input: &EncryptedTensor<T>
    ) -> EncryptedTensor<T>{
        let mut activations = vec![input.clone()];
        for layer in &mut self.layers {
            let output = layer.forward(activations.last().unwrap(), &self.context);
            activations.push(output.clone());
        }
        let prediction = EncryptedTensor{
            data: activations.last().unwrap().data.clone(),
            shape: activations.last().unwrap().shape.clone(),
        };

        prediction
    }

    fn iter_batches(
        &self,
        inputs: &EncryptedTensor<T>,
        labels: &EncryptedTensor<T>,
        batch_size: usize,
    ) -> Vec<(EncryptedTensor<T>, EncryptedTensor<T>)> {
        assert!(
            inputs.shape.len() == 4,
            "Expected 4D input tensor, got shape {:?}",
            inputs.shape
        );
        assert!(
            labels.shape.len() == 4 || labels.shape.len() == 2,
            "Expected 2D or 4D label tensor, got shape {:?}",
            labels.shape
        );
    
        let num_samples = inputs.shape[0];
        let input_sample_size = inputs.shape[1] * inputs.shape[2] * inputs.shape[3];
        let label_sample_size: usize = labels.shape.iter().skip(1).product();
    
        let mut batches = Vec::new();
        let mut start = 0;
    
        while start < num_samples {
            let end = usize::min(start + batch_size, num_samples);
    
            // Slice input batch
            let input_start = start * input_sample_size;
            let input_end = end * input_sample_size;
            let input_batch_data = inputs.data[input_start..input_end].to_vec();
            let input_batch_shape = vec![
                end - start,
                inputs.shape[1],
                inputs.shape[2],
                inputs.shape[3],
            ];
            let input_batch = EncryptedTensor {
                data: input_batch_data,
                shape: input_batch_shape,
            };
    
            // Slice label batch
            let label_start = start * label_sample_size;
            let label_end = end * label_sample_size;
            let label_batch_data = labels.data[label_start..label_end].to_vec();
            let mut label_batch_shape = labels.shape.clone();
            label_batch_shape[0] = end - start;
            let label_batch = EncryptedTensor {
                data: label_batch_data,
                shape: label_batch_shape,
            };
    
            batches.push((input_batch, label_batch));
            start = end;
        }
    
        batches
    }
}