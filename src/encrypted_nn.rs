use crate::encrypted_utils::encrypted_context::{self, EncryptedContext};
use crate::encrypted_utils::server_key_trait::ServerKeyTrait;
use crate::encrypted_utils::encrypted_types::{EncryptedElement, EncryptableValueType};
use crate::encrypted_layers::{EncryptedLayer, EncryptedDenseLayer};
use crate::encrypted_losses::loss_function::LossFunction;
use crate::encrypted_utils::tensor::EncryptedTensor;
use crate::encrypted_ops::*;

/// Core generic implementation of an encrypted neural network
pub struct EncryptedNeuralNetworkImpl<K: ServerKeyTrait, T: EncryptedElement> {
    pub layers: Vec<Box<dyn EncryptedLayer<K, T>>>,
    pub loss: Box<dyn LossFunction<K, T>>,
    pub context: EncryptedContext<K, T>,
}
impl<K, T> EncryptedNeuralNetworkImpl<K, T> 
where
    K: ServerKeyTrait + EncryptedAdd<K, T> + EncryptedMul<K, T> + EncryptedDiv<K, T> + EncryptedNegate<K, T>,
    T: Clone + EncryptedElement + 'static,
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
}