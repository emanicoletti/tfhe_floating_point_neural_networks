use crate::encrypted_utils::encrypted_context::{self, EncryptedContext};
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::encrypted_losses::loss_function::LossFunction;
use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_ops::*;

use half::f16;

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
    + EncryptedNegate<K, T>,
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
            for (input_batch, label_batch) in self.iter_batches(&train_inputs, &train_labels, batch_size){
                let mut activations = input_batch.clone();
                for layer in &self.layers{
                    activations = layer.forward(&activations, &self.context)
                }
                let loss_val = self.loss.compute_loss(&activations, &label_batch, &self.context);
                let decrypted: u16 = EncryptableValueType::decrypt(&loss_val.data[0], &self.context.client_key);
                println!("Batch Loss:{:<6} ", f16::from_bits(decrypted).to_f32());
                let mut grad = self.loss.gradient(&activations, &label_batch, &self.context);
                let flat = &grad.data;
                let rows = grad.shape[0];
                let cols = grad.shape[1];
                for layer in self.layers.iter_mut().rev() {
                    grad = layer.backward(&input_batch, &grad, &self.context);
                }
                for layer in &mut self.layers{
                    layer.update_parameters(learning_rate.clone(), &self.context);
                }
            }
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