use crate::encrypted_utils::encrypted_context::{self, EncryptedContext};
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::activations::{EncryptedTanhActivation};
use crate::encrypted_losses::loss_function::LossFunction;
use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_ops::*;

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
    + EncryptedTanh<K, T>,
    T: EncryptedElement + Clone + EncryptableValueType<Plain=u32> + 'static,
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

    pub fn train( 
        &mut self,
        epochs: usize,
        batch_size: usize,
        learning_rate: T,
        train_inputs: EncryptedTensor<T>,
        train_labels: EncryptedTensor<T>,
        val_inputs: EncryptedTensor<T>,
        val_labels: EncryptedTensor<T>,
    ) 
    {
        for epoch in 0..epochs{
            println!("Epoch {}/{}", epoch + 1, epochs);
            let time = Instant::now();
            let forward_time = Instant::now();
            for (input_batch, label_batch) in self.iter_batches(&train_inputs, &train_labels, batch_size){
                let mut activations = vec![input_batch.clone()];
                println!("Forward started...");
                for layer in &mut self.layers {
                    let output = layer.forward(activations.last().unwrap(), &self.context);
                    activations.push(output.clone());
                    println!("Layer passed");
                    let prediction = activations.last().unwrap();
                    let rows = prediction.shape[0];
                    let cols = prediction.shape[1];
                    let flat = &prediction.data;
        
                    if flat.len() != rows * cols {
                        println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                        return;
                    }
        
                    for i in 0..rows {
                        print!("\n[");
                        for j in 0..cols {
                            let index = i * cols + j;
                            //let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                            let decrypted: u32 = EncryptableValueType::decrypt(&flat[index], &self.context.client_key);
                            print!("{:<6} ", f32::from_bits(decrypted));
                        }
                        print!("]\n");
                    }
                }
                let prediction = activations.last().unwrap();
                let rows = prediction.shape[0];
                let cols = prediction.shape[1];
                let flat = &prediction.data;
    
                if flat.len() != rows * cols {
                    println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                    return;
                }
    
                for i in 0..rows {
                    print!("\n[");
                    for j in 0..cols {
                        let index = i * cols + j;
                        //let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                        let decrypted: u32 = EncryptableValueType::decrypt(&flat[index], &self.context.client_key);
                        print!("{:<6} ", f32::from_bits(decrypted));
                    }
                    print!("]\n");
                }
                let loss_val = self.loss.compute_loss(&prediction, &label_batch, &self.context);
                let decrypted: u32 = EncryptableValueType::decrypt(&loss_val.data[0], &self.context.client_key);
                println!("Batch Loss: {:<6} ", f32::from_bits(decrypted));
                println!("Forward pass time: {:?}", forward_time.elapsed());
                println!("Backward started...");
                let mut grad = self.loss.gradient(&prediction, &label_batch, &self.context);
                for (i, layer) in self.layers.iter_mut().rev().enumerate() {
                    let input_to_layer = &activations[activations.len() - 2 - i];
                    grad = layer.backward(input_to_layer, &grad, &self.context);
                    let rows = grad.shape[0];
                    let cols = grad.shape[1];
                    let flat = &grad.data;
        
                    if flat.len() != rows * cols {
                        println!("Shape mismatch: expected {} elements, got {}", rows * cols, flat.len());
                        return;
                    }
        
                    for i in 0..rows {
                        print!("\n[");
                        for j in 0..cols {
                            let index = i * cols + j;
                            //let decrypted: u16 = flat[index].decrypt(&self.inner.context.client_key);
                            let decrypted: u32 = EncryptableValueType::decrypt(&flat[index], &self.context.client_key);
                            print!("{:<6} ", f32::from_bits(decrypted));
                        }
                        print!("]\n");
                    }
                }
                println!("Backward ended...");
                for layer in &mut self.layers{
                    layer.update_parameters(learning_rate.clone(), &self.context);
                }
            }
            println!("Epoch {} completed in {:?}", epoch + 1, time.elapsed());
        }
    }

    fn iter_batches(
        &self,
        inputs: &EncryptedTensor<T>,
        labels: &EncryptedTensor<T>,
        batch_size: usize,
    ) -> Vec<(EncryptedTensor<T>, EncryptedTensor<T>)> {
        let feature_size = inputs.shape[1];
        let label_size = labels.shape[1];

        let num_samples = inputs.shape[0];
        let mut batches = Vec::new();

        let mut start = 0;
        while start < num_samples {
            let end = usize::min(start + batch_size, num_samples);
            let input_batch_data = inputs.data[start * feature_size..end * feature_size].to_vec();
            let label_batch_data = labels.data[start * label_size..end * label_size].to_vec();
            let input_batch = EncryptedTensor {
                data: input_batch_data,
                shape: vec![end - start, feature_size],
            };
            let label_batch = EncryptedTensor {
                data: label_batch_data,
                shape: vec![end - start, label_size],
            };

            batches.push((input_batch, label_batch));
            start = end;
        }

        batches
    }
}